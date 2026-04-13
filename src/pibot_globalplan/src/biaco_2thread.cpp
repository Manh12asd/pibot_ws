#include <memory>
#include <thread>
#include <vector>
#include <cmath>
#include <mutex>
#include <random>
#include <algorithm>
#include <limits>
#include <array>
#include <chrono> 
#include <atomic> 

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "pibot_msg/action/aco_plan.hpp"

using std::placeholders::_1;
using std::placeholders::_2;

// --- 1. Helper Structs ---

struct GraphNode {
    int x;
    int y;

    bool operator==(const GraphNode& other) const {
        return x == other.x && y == other.y;
    }

    bool operator!=(const GraphNode& other) const {
        return !(*this == other);
    }

    GraphNode operator+(const GraphNode& other) const {
        return {x + other.x, y + other.y};
    }
};

// --- 2. Main Class ---

class ACOPlannerActionServer : public rclcpp::Node {
public:
    using AcoPlan = pibot_msg::action::AcoPlan;
    using GoalHandleAcoPlan = rclcpp_action::ServerGoalHandle<AcoPlan>;

    ACOPlannerActionServer() : Node("aco_planner_action_server"), has_last_goal_(false) {
        alpha_ = 10.0;
        beta_ = 1.0;
        rho_ = 0.5;
        num_ants_ = 20;
        iterations_ = 6;
        cost_post = 99;

        max_pheromone_ = 50.0; 

        c_w_len_ = 0.5;   
        c_w_wp_ = 0.25;   
        c_w_theta_ = 0.25;

        q1_ = 1.0; 
        q2_ = 2.0; 
        q3_ = 2.5; 
        adaptive_weight_factor_ = 0.05;

        callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        auto map_qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local();
        rclcpp::SubscriptionOptions sub_opts;
        sub_opts.callback_group = callback_group_;
        
        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/costmap/costmap", map_qos,
            std::bind(&ACOPlannerActionServer::map_callback, this, _1), sub_opts);

        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/improve_aco3/path", 10);
        waypoint_pub_ = this->create_publisher<nav_msgs::msg::Path>("/improve_aco3/waypoints", 10);
        debug_failed_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/debug/failed_path", 10);
        raw_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/improve_aco3/raw_path", 10);

        action_server_ = rclcpp_action::create_server<AcoPlan>(
            this,
            "aco_plan",
            std::bind(&ACOPlannerActionServer::handle_goal, this, _1, _2),
            std::bind(&ACOPlannerActionServer::handle_cancel, this, _1),
            std::bind(&ACOPlannerActionServer::handle_accepted, this, _1),
            rcl_action_server_get_default_options(),
            callback_group_
        );

        std::random_device rd;
        rng_ = std::mt19937(rd());

        RCLCPP_INFO(this->get_logger(), "ACO Planner Ready.");
    }

    ~ACOPlannerActionServer() {
        preempt_requested_ = true;
        if (execution_thread_.joinable()) {
            execution_thread_.join();
        }
    }

private:
    std::mutex map_lock_;
    nav_msgs::msg::OccupancyGrid map_;
    std::vector<double> pheromone_grid_; 
    
    // THÊM: Các biến quản lý luồng tính toán
    std::mutex thread_mutex_;
    std::thread execution_thread_;
    std::atomic<bool> preempt_requested_{false};
    
    geometry_msgs::msg::Pose last_goal_pose_;
    bool has_last_goal_;

    double alpha_, beta_, rho_;
    double c_w_len_, c_w_wp_, c_w_theta_; 
    double q1_, q2_, q3_; 
    double adaptive_weight_factor_; 
    double max_pheromone_;
    int num_ants_, iterations_, cost_post;
    
    rclcpp::CallbackGroup::SharedPtr callback_group_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_, waypoint_pub_, debug_failed_path_pub_, raw_path_pub_;
    rclcpp_action::Server<AcoPlan>::SharedPtr action_server_;
    
    std::mt19937 rng_;

    struct PathResult {
        std::vector<GraphNode> full_path;
        std::vector<GraphNode> chain;    
        std::vector<GraphNode> raw_path; 
        bool success;
    };

    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(map_lock_);
        map_ = *msg;
        size_t size = map_.info.width * map_.info.height;
        if (pheromone_grid_.size() != size) {
            pheromone_grid_.assign(size, 0.1);
        }
    }

    rclcpp_action::GoalResponse handle_goal(const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const AcoPlan::Goal> goal) {
        (void)uuid; (void)goal;
        std::lock_guard<std::mutex> lock(map_lock_);
        if (map_.data.empty()) { return rclcpp_action::GoalResponse::REJECT; }

        try {
            geometry_msgs::msg::TransformStamped t;
            t = tf_buffer_->lookupTransform(map_.header.frame_id, "base_footprint", tf2::TimePointZero);
            GraphNode robot_node = world_to_grid(t.transform.translation.x, t.transform.translation.y, map_);

            if (robot_node.x >= 0 && robot_node.x < (int)map_.info.width &&
                robot_node.y >= 0 && robot_node.y < (int)map_.info.height) {
                int idx = robot_node.y * map_.info.width + robot_node.x;
                cost_post = static_cast<int>(map_.data[idx]);
            }
        } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN(this->get_logger(), "Không lấy được TF trong handle_goal: %s", ex.what());
        }

        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    }

    rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandleAcoPlan> goal_handle) {
        (void)goal_handle;
        return rclcpp_action::CancelResponse::ACCEPT;
    }

    // CẬP NHẬT: Quản lý luồng bằng cơ chế chèn ngang (Preemption)
    void handle_accepted(const std::shared_ptr<GoalHandleAcoPlan> goal_handle) {
        std::lock_guard<std::mutex> lock(thread_mutex_);
        
        // Nếu đang có luồng chạy -> Bật cờ hủy và chờ nó dừng
        if (execution_thread_.joinable()) {
            RCLCPP_WARN(this->get_logger(), "Nhận Goal mới! Đang hủy Goal cũ để tính lại (Replan)...");
            preempt_requested_ = true;  
            execution_thread_.join();   // Chờ thuật toán cũ thoát an toàn
        }
        
        // Reset cờ để luồng mới bắt đầu chạy
        preempt_requested_ = false;
        
        execution_thread_ = std::thread([this, goal_handle]() {
            this->execute_callback(goal_handle);
        });
    }

    GraphNode world_to_grid(double x, double y, const nav_msgs::msg::OccupancyGrid& m) {
        int gx = static_cast<int>((x - m.info.origin.position.x) / m.info.resolution);
        int gy = static_cast<int>((y - m.info.origin.position.y) / m.info.resolution);
        return {gx, gy};
    }

    geometry_msgs::msg::Pose grid_to_world(GraphNode n, const nav_msgs::msg::OccupancyGrid& m) {
        geometry_msgs::msg::Pose p;
        p.position.x = n.x * m.info.resolution + m.info.origin.position.x;
        p.position.y = n.y * m.info.resolution + m.info.origin.position.y;
        p.orientation.w = 1.0;
        return p;
    }

    bool is_physically_safe(const GraphNode& n, const nav_msgs::msg::OccupancyGrid& m) {
        if (n.x >= 0 && n.x < (int)m.info.width && n.y >= 0 && n.y < (int)m.info.height) {
            int idx = n.y * m.info.width + n.x;
            int8_t cost = m.data[idx];
            if (cost >= 99 || cost == -1) return false;
            if (cost > cost_post) return false;
            return true;
        }
        return false;
    }

    bool start_safe(const GraphNode& n, const nav_msgs::msg::OccupancyGrid& m) {
        if (n.x >= 0 && n.x < (int)m.info.width && n.y >= 0 && n.y < (int)m.info.height) {
            int idx = n.y * m.info.width + n.x;
            int8_t cost = m.data[idx];
            if (cost >= 99 || cost == -1) return false; 
            return true;
        }
        return false;
    }

    double dist_sq(const GraphNode& a, const GraphNode& b) {
        double dx = static_cast<double>(a.x - b.x);
        double dy = static_cast<double>(a.y - b.y);
        return dx * dx + dy * dy;
    }

    double calculate_cos_angle(const GraphNode& curr, const GraphNode& neighbor, const GraphNode& target) {
        double v1_x = static_cast<double>(neighbor.x - curr.x);
        double v1_y = static_cast<double>(neighbor.y - curr.y);
        double v2_x = static_cast<double>(target.x - curr.x);
        double v2_y = static_cast<double>(target.y - curr.y);

        double dot = v1_x * v2_x + v1_y * v2_y;
        double mag1_sq = v1_x * v1_x + v1_y * v1_y;
        double mag2_sq = v2_x * v2_x + v2_y * v2_y;

        if (mag1_sq == 0 || mag2_sq == 0) return 0.0;
        return dot / std::sqrt(mag1_sq * mag2_sq);
    }

    double calculate_path_length(const std::vector<GraphNode>& path, double resolution) {
        double len = 0.0;
        for (size_t i = 0; i < path.size() - 1; ++i) {
            len += std::hypot(path[i+1].x - path[i].x, path[i+1].y - path[i].y);
        }
        return len * resolution;
    }

    double calculate_mean_sq_angle(const std::vector<GraphNode>& chain) {
        if (chain.size() < 3) return 0.0;
        double total_sq_angle = 0.0;
        for (size_t i = 1; i < chain.size() - 1; ++i) {
            double v1x = static_cast<double>(chain[i].x - chain[i-1].x);
            double v1y = static_cast<double>(chain[i].y - chain[i-1].y);
            double v2x = static_cast<double>(chain[i+1].x - chain[i].x);
            double v2y = static_cast<double>(chain[i+1].y - chain[i].y);
            
            double dot = v1x * v2x + v1y * v2y;
            double mag1 = std::hypot(v1x, v1y);
            double mag2 = std::hypot(v2x, v2y);
            
            if (mag1 > 0 && mag2 > 0) {
                double val = dot / (mag1 * mag2);
                val = std::max(-1.0, std::min(1.0, val));
                double angle = std::acos(val);
                total_sq_angle += (angle * angle); 
            }
        }
        return total_sq_angle / static_cast<double>(chain.size());
    }

    std::vector<GraphNode> bresenham(GraphNode start, GraphNode end) {
        std::vector<GraphNode> line;
        int x0 = start.x, y0 = start.y;
        int x1 = end.x, y1 = end.y;
        int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx - dy;

        while (true) {
            line.push_back({x0, y0});
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
        return line;
    }

    bool fast_check_line_of_sight(GraphNode start, GraphNode end, const nav_msgs::msg::OccupancyGrid& m) {
        int x0 = start.x, y0 = start.y;
        int x1 = end.x, y1 = end.y;
        int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx - dy;

        while (true) {
            if (!is_physically_safe({x0, y0}, m)) return false;
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
        return true;
    }

    std::vector<GraphNode> generate_optimal_chain(const std::vector<GraphNode>& path, const nav_msgs::msg::OccupancyGrid& m) {
        if (path.size() < 2) return path;

        std::vector<GraphNode> tp;
        tp.push_back(path[0]);
        
        for (size_t i = 1; i < path.size() - 1; ++i) {
            int dx1 = path[i].x - path[i-1].x;
            int dy1 = path[i].y - path[i-1].y;
            int dx2 = path[i+1].x - path[i].x;
            int dy2 = path[i+1].y - path[i].y;
            
            if (dx1 != dx2 || dy1 != dy2) { tp.push_back(path[i]); }
        }
        tp.push_back(path.back());

        std::vector<GraphNode> optimized;
        size_t curr = 0;
        while (curr < tp.size() - 1) {
            optimized.push_back(tp[curr]);
            size_t next_pt = curr + 1;
            
            for (int j = tp.size() - 1; j > (int)curr; --j) {
                if (fast_check_line_of_sight(tp[curr], tp[j], m)) {
                    next_pt = j;
                    break;
                }
            }
            curr = next_pt;
        }
        optimized.push_back(tp.back());
        return optimized;
    }

    std::vector<GraphNode> reconstruct_full_path(const std::vector<GraphNode>& chain) {
        std::vector<GraphNode> full;
        for (size_t i = 0; i < chain.size() - 1; ++i) {
            auto segment = bresenham(chain[i], chain[i+1]);
            full.insert(full.end(), segment.begin(), segment.end() - 1);
        }
        full.push_back(chain.back());
        return full;
    }

    PathResult construct_solution(const GraphNode& start, const GraphNode& goal, 
                                  const nav_msgs::msg::OccupancyGrid& m) {
        
        GraphNode curr_f = start;
        GraphNode curr_b = goal; 
        
        std::vector<GraphNode> path_f = {curr_f};
        std::vector<GraphNode> path_b = {curr_b};
        
        int map_size = m.info.width * m.info.height;
        std::vector<uint8_t> visited_f(map_size, 0);
        std::vector<uint8_t> visited_b(map_size, 0);
        
        auto get_idx = [&](const GraphNode& n) { return n.y * m.info.width + n.x; };

        visited_f[get_idx(curr_f)] = 1;
        visited_b[get_idx(curr_b)] = 1;
        
        int max_steps = 250000;
        const int dx[] = {-1, 1, 0, 0, -1, 1, -1, 1};
        const int dy[] = {0, 0, -1, 1, -1, 1, 1, -1};

        std::array<GraphNode, 8> neighbors;
        std::array<double, 8> probs;

        for (int step = 0; step < max_steps; ++step) {
            // CẬP NHẬT: THOÁT KHẨN CẤP nếu có lệnh hủy (để nhận Goal mới)
            if (preempt_requested_) return {path_f, {}, {}, false}; 
            
            bool meet_in_middle = (dist_sq(curr_f, curr_b) <= 2.0);
            bool forward_reached = (curr_f == goal);
            bool backward_reached = (curr_b == start);

            if (meet_in_middle || forward_reached || backward_reached) {
                std::vector<GraphNode> raw_path;
                
                if (forward_reached) {
                    raw_path = path_f;
                } 
                else if (backward_reached) {
                    raw_path = path_b;
                    std::reverse(raw_path.begin(), raw_path.end());
                } 
                else { 
                    raw_path = path_f;
                    for (auto it = path_b.rbegin(); it != path_b.rend(); ++it) {
                        if (raw_path.empty() || *it != raw_path.back()) {
                            raw_path.push_back(*it);
                        }
                    }
                }

                auto optimized_chain = generate_optimal_chain(raw_path, m);
                auto smoothed_full_path = reconstruct_full_path(optimized_chain);
                
                return {smoothed_full_path, optimized_chain, raw_path, true};
            }

            // DI CHUYỂN KIẾN TIẾN (FORWARD ANT)
            double curr_dist_T = std::hypot(curr_f.x - goal.x, curr_f.y - goal.y);
            double curr_dist_ib = std::hypot(curr_f.x - curr_b.x, curr_f.y - curr_b.y);
            double delta_d_f = curr_dist_T - curr_dist_ib;
            double w_goal_f = std::exp(-adaptive_weight_factor_ * delta_d_f);
            double w_backward_f = std::exp(adaptive_weight_factor_ * delta_d_f);

            int valid_count_f = 0;
            double denom_f = 0.0;

            for (int i = 0; i < 8; ++i) {
                GraphNode nb = {curr_f.x + dx[i], curr_f.y + dy[i]};

                if (!is_physically_safe(nb, m)) continue;
                
                int idx = get_idx(nb);
                if (visited_f[idx]) continue; 

                double tau = pheromone_grid_[idx];
                
                double cos_T = calculate_cos_angle(curr_f, nb, goal);
                double dist_T = std::hypot(nb.x - goal.x, nb.y - goal.y);
                double eta_static = std::exp(cos_T) / (dist_T + 0.1);

                double cos_ib = calculate_cos_angle(curr_f, nb, curr_b);
                double dist_ib = std::hypot(nb.x - curr_b.x, nb.y - curr_b.y);
                double eta_dynamic = std::exp(cos_ib) / (dist_ib + 0.1);
                
                double eta = w_goal_f * eta_static + w_backward_f * eta_dynamic;
                int8_t cell_cost = m.data[idx];
                double cost_penalty = (cell_cost > 0) ? static_cast<double>(cell_cost) : 0.0;
                double v = std::pow(tau, alpha_) * std::pow(eta * std::exp(-cost_penalty), beta_);
                
                neighbors[valid_count_f] = nb;
                probs[valid_count_f] = v;
                denom_f += v;
                valid_count_f++;
            }

            if (valid_count_f == 0 || denom_f == 0.0) {
                if (path_f.size() > 1) {
                    path_f.pop_back(); 
                    curr_f = path_f.back(); 
                } else { return {path_f, {}, {}, false}; }
            } else {
                std::discrete_distribution<> d(probs.begin(), probs.begin() + valid_count_f);
                curr_f = neighbors[d(rng_)];
                path_f.push_back(curr_f);
                visited_f[get_idx(curr_f)] = 1;
            }

            // DI CHUYỂN KIẾN LÙI (BACKWARD ANT)
            double curr_dist_S = std::hypot(curr_b.x - start.x, curr_b.y - start.y);
            double curr_dist_if = std::hypot(curr_b.x - curr_f.x, curr_b.y - curr_f.y); 
            double delta_d_b = curr_dist_S - curr_dist_if;
            double w_start_b = std::exp(-adaptive_weight_factor_ * delta_d_b);
            double w_forward_b = std::exp(adaptive_weight_factor_ * delta_d_b);

            int valid_count_b = 0;
            double denom_b = 0.0;

            for (int i = 0; i < 8; ++i) {
                GraphNode nb = {curr_b.x + dx[i], curr_b.y + dy[i]};

                if (!is_physically_safe(nb, m)) continue;
                
                int idx = get_idx(nb);
                if (visited_b[idx]) continue;

                double tau = pheromone_grid_[idx];
                
                double cos_S = calculate_cos_angle(curr_b, nb, start);
                double dist_S = std::hypot(nb.x - start.x, nb.y - start.y);
                double eta_static = std::exp(cos_S) / (dist_S + 0.1);

                double cos_if = calculate_cos_angle(curr_b, nb, curr_f);
                double dist_if = std::hypot(nb.x - curr_f.x, nb.y - curr_f.y);
                double eta_dynamic = std::exp(cos_if) / (dist_if + 0.1);
                
                double eta = w_start_b * eta_static + w_forward_b * eta_dynamic;
                int8_t cell_cost = m.data[idx];
                double cost_penalty = (cell_cost > 0) ? static_cast<double>(cell_cost) : 0.0;
                double v = std::pow(tau, alpha_) * std::pow(eta * std::exp(-cost_penalty), beta_);
                
                neighbors[valid_count_b] = nb;
                probs[valid_count_b] = v;
                denom_b += v;
                valid_count_b++;
            }

            if (valid_count_b == 0 || denom_b == 0.0) {
                if (path_b.size() > 1) {
                    path_b.pop_back(); 
                    curr_b = path_b.back(); 
                } else { return {path_b, {}, {}, false}; }
            } else {
                std::discrete_distribution<> d(probs.begin(), probs.begin() + valid_count_b);
                curr_b = neighbors[d(rng_)];
                path_b.push_back(curr_b);
                visited_b[get_idx(curr_b)] = 1;
            }
        }

        return {path_f, {}, {}, false}; 
    }

    void update_pheromones(const std::vector<std::vector<GraphNode>>& paths, 
                           const std::vector<double>& Fit_list, 
                           const std::vector<int>& best_indices, 
                           const std::vector<int>& worst_indices) 
    {
        for (auto& val : pheromone_grid_) {
            val *= (1.0 - rho_);
        }
        
        for (size_t k = 0; k < paths.size(); ++k) {
            double delta_tau = 0.0;
            
            bool is_best = std::find(best_indices.begin(), best_indices.end(), k) != best_indices.end();
            bool is_worst = std::find(worst_indices.begin(), worst_indices.end(), k) != worst_indices.end();
            
            if (is_best) {
                delta_tau = q1_ * Fit_list[k] + q2_ * Fit_list[k];
            } 
            else if (is_worst) {
                delta_tau = q1_ * Fit_list[k] - q3_ * Fit_list[k];
            } 
            else {
                delta_tau = q1_ * Fit_list[k];
            }

            for (const auto& n : paths[k]) {
                int idx = n.y * map_.info.width + n.x;
                if (idx >= 0 && idx < (int)pheromone_grid_.size()) {
                    pheromone_grid_[idx] += delta_tau;
                    if (pheromone_grid_[idx] < 0.01) {
                        pheromone_grid_[idx] = 0.01;
                    }
                    if (pheromone_grid_[idx] > max_pheromone_) {
                        pheromone_grid_[idx] = max_pheromone_;
                    }
                }
            }
        }
    }

    void publish_debug_path(const std::vector<GraphNode>& nodes, const nav_msgs::msg::OccupancyGrid& local_map) {
        if (nodes.empty()) return;
        auto msg = create_path_msg(nodes, local_map);
        debug_failed_path_pub_->publish(msg);
    }

    nav_msgs::msg::Path create_path_msg(const std::vector<GraphNode>& nodes, const nav_msgs::msg::OccupancyGrid& local_map) {
        nav_msgs::msg::Path path_msg;
        path_msg.header.frame_id = local_map.header.frame_id;
        path_msg.header.stamp = this->now();
        for (const auto& n : nodes) {
            geometry_msgs::msg::PoseStamped ps;
            ps.pose = grid_to_world(n, local_map);
            ps.header = path_msg.header;
            path_msg.poses.push_back(ps);
        }
        return path_msg;
    }

    void execute_callback(const std::shared_ptr<GoalHandleAcoPlan> goal_handle) {
        nav_msgs::msg::OccupancyGrid local_map;
        {
            std::lock_guard<std::mutex> lock(map_lock_);
            local_map = map_;
        }

        auto result = std::make_shared<AcoPlan::Result>();
        auto feedback = std::make_shared<AcoPlan::Feedback>();
        
        std::vector<GraphNode> final_iteration_best_chain;
        std::vector<GraphNode> final_iteration_best_raw_path; 
        double final_best_length = std::numeric_limits<double>::infinity();

        GraphNode start_node{0,0}, goal_node{0,0};

        try {
            geometry_msgs::msg::TransformStamped t;
            t = tf_buffer_->lookupTransform(local_map.header.frame_id, "base_footprint", tf2::TimePointZero);
            start_node = world_to_grid(t.transform.translation.x, t.transform.translation.y, local_map);
            
            auto goal_pose = goal_handle->get_goal()->goal_pose.pose;
            goal_node = world_to_grid(goal_pose.position.x, goal_pose.position.y, local_map);

            if (has_last_goal_) {
                double dist_to_old_goal = std::hypot(
                    goal_pose.position.x - last_goal_pose_.position.x,
                    goal_pose.position.y - last_goal_pose_.position.y
                );

                if (dist_to_old_goal > 1.0) {
                    std::fill(pheromone_grid_.begin(), pheromone_grid_.end(), 0.1);
                }
            } else {
                has_last_goal_ = true;
            }
            last_goal_pose_ = goal_pose;

        } catch (const tf2::TransformException & ex) {
            RCLCPP_ERROR(this->get_logger(), "TF Error: %s", ex.what());
            goal_handle->abort(result);
            return;
        }

        if (!start_safe(start_node, local_map)) {
            RCLCPP_ERROR(this->get_logger(), "Start is in a static obstacle");
            goal_handle->abort(result);
            return;
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        double epsilon = 0.001; 

        for (int i = 0; i < iterations_; ++i) {
            // CẬP NHẬT: Kiểm tra Hủy từ phía ROS 2 hoặc do có Goal mới chèn vào
            if (goal_handle->is_canceling() || preempt_requested_) {
                if (preempt_requested_) {
                    goal_handle->abort(result); // Báo Goal này bị loại bỏ để nhường chỗ
                } else {
                    goal_handle->canceled(result);
                }
                return; // Thoát luồng ngay lập tức
            }

            std::vector<std::vector<GraphNode>> optimized_paths;
            std::vector<std::vector<GraphNode>> optimized_chains;
            std::vector<std::vector<GraphNode>> raw_paths; 
            
            std::vector<double> L_list;
            std::vector<double> W_list;
            std::vector<double> Theta_list;

            for (int ant = 0; ant < num_ants_; ++ant) {
                // CẬP NHẬT: Thoát vòng lặp kiến sớm nếu có yêu cầu Hủy
                if (preempt_requested_) break;

                PathResult pr = construct_solution(start_node, goal_node, local_map);
                if (pr.success) {
                    double smoothed_len = calculate_path_length(pr.chain, local_map.info.resolution);
                    double wp_count = static_cast<double>(pr.chain.size());
                    double mean_sq_angle = calculate_mean_sq_angle(pr.chain);

                    optimized_paths.push_back(pr.full_path);
                    optimized_chains.push_back(pr.chain);
                    raw_paths.push_back(pr.raw_path); 
                    
                    L_list.push_back(smoothed_len);
                    W_list.push_back(wp_count);
                    Theta_list.push_back(mean_sq_angle);
                }
            }
            
            if (preempt_requested_) return; // Kiểm tra thêm lần nữa trước khi update Pheromone

            if (!optimized_paths.empty()) {
                double L_max = *std::max_element(L_list.begin(), L_list.end());
                double W_max = *std::max_element(W_list.begin(), W_list.end());
                double Theta_max = *std::max_element(Theta_list.begin(), Theta_list.end());

                L_max = (L_max > 0.0) ? L_max : 1.0;
                W_max = (W_max > 0.0) ? W_max : 1.0;
                Theta_max = (Theta_max > 0.0) ? Theta_max : 1.0;

                std::vector<double> Fit_list; 
                for (size_t k = 0; k < L_list.size(); ++k) {
                    double cost = c_w_len_ * (L_list[k] / L_max) + 
                                  c_w_wp_ * (W_list[k] / W_max) + 
                                  c_w_theta_ * (Theta_list[k] / Theta_max);
                    Fit_list.push_back(1.0 / (cost + 1e-6)); 
                }

                auto it_min_len = std::min_element(L_list.begin(), L_list.end());
                int current_iter_best_idx = std::distance(L_list.begin(), it_min_len);
                double current_iter_best_len = L_list[current_iter_best_idx];

                if (current_iter_best_len < final_best_length - epsilon) {
                    final_best_length = current_iter_best_len;
                    final_iteration_best_chain = optimized_chains[current_iter_best_idx];
                    final_iteration_best_raw_path = raw_paths[current_iter_best_idx];
                }

                std::vector<int> best_indices;
                std::vector<int> worst_indices;
                double max_fit = -1.0;
                double min_fit = std::numeric_limits<double>::infinity();
                double fit_epsilon = 1e-6; 

                for(size_t k = 0; k < Fit_list.size(); ++k) {
                    if (Fit_list[k] < min_fit) min_fit = Fit_list[k];
                    if (Fit_list[k] > max_fit) max_fit = Fit_list[k];
                }

                for(size_t k = 0; k < Fit_list.size(); ++k) {
                    if (std::abs(Fit_list[k] - max_fit) <= fit_epsilon) {
                        best_indices.push_back(k);
                    }
                    else if (std::abs(Fit_list[k] - min_fit) <= fit_epsilon) {
                        worst_indices.push_back(k);
                    }
                }

                update_pheromones(optimized_paths, Fit_list, best_indices, worst_indices);
                
                feedback->iteration = i;
                feedback->current_best_length = final_best_length; 
                goal_handle->publish_feedback(feedback);
            } else {
                RCLCPP_WARN(this->get_logger(), " Iteration %d:path not found.", i);
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        if (!final_iteration_best_chain.empty()) {
            auto final_full_nodes = reconstruct_full_path(final_iteration_best_chain);
            auto full_msg = create_path_msg(final_full_nodes, local_map);
            auto wp_msg = create_path_msg(final_iteration_best_chain, local_map);
            auto raw_msg = create_path_msg(final_iteration_best_raw_path, local_map); 
            
            path_pub_->publish(full_msg);
            waypoint_pub_->publish(wp_msg);
            raw_path_pub_->publish(raw_msg); 
            
            result->path = full_msg; 
            result->total_length = final_best_length;
            goal_handle->succeed(result);
        } else {
            RCLCPP_ERROR(this->get_logger(), " failed after %ld ms", total_duration_ms);
            goal_handle->abort(result);
        }
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ACOPlannerActionServer>();
    
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    
    rclcpp::shutdown();
    return 0;
}