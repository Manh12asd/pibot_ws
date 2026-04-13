cmake_minimum_required(VERSION 3.8)
project(pibot_localplan)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pibot_msg REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(obstacle_detector REQUIRED)
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  ${PROJECT_NAME}/dwa.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch 
  DESTINATION share/${PROJECT_NAME}
)

add_executable(standalone_costmap_node src/standalone_costmap_node.cpp)
ament_target_dependencies(standalone_costmap_node
  rclcpp
  rclcpp_lifecycle
  nav2_costmap_2d
  nav2_util
)

add_executable(local_costmap src/local_costmap.cpp)
ament_target_dependencies(local_costmap
  rclcpp
  rclcpp_lifecycle
  nav2_costmap_2d
  nav2_util
)

add_executable(dwa_cpp src/dwa.cpp)
ament_target_dependencies(dwa_cpp
  rclcpp rclcpp_action tf2 tf2_ros
  geometry_msgs nav_msgs visualization_msgs
  obstacle_detector pibot_msg
  tf2_geometry_msgs
)

add_executable(dwa_cpp_replan src/dwa_cpp_replan.cpp)
ament_target_dependencies(dwa_cpp_replan
  rclcpp rclcpp_action tf2 tf2_ros
  geometry_msgs nav_msgs visualization_msgs
  obstacle_detector pibot_msg
  tf2_geometry_msgs
)


install(TARGETS
  local_costmap
  dwa_cpp
  dwa_cpp_replan
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
