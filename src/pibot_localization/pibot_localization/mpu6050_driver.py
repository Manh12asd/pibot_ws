#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from rclpy.qos import qos_profile_sensor_data 

PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT_H = 0x3B
DEVICE_ADDRESS = 0x68

class MPU6050_Driver(Node):

    def __init__(self):
        super().__init__("mpu6050_driver")
        
        self.A_OFF_X = 816
        self.A_OFF_Y = -293
        self.A_OFF_Z = 1123
        self.G_OFF_X = 36
        self.G_OFF_Y = 10
        self.G_OFF_Z = -44
        
        self.is_connected_ = False
        self.init_i2c()

        self.imu_pub_ = self.create_publisher(Imu, "/imu/data_raw", qos_profile_sensor_data)
        
        self.imu_msg_ = Imu()
        self.imu_msg_.header.frame_id = "imu_link"
        
        self.imu_msg_.orientation_covariance = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.imu_msg_.linear_acceleration_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        self.imu_msg_.angular_velocity_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]

        self.frequency_ = 0.01  
        self.timer_ = self.create_timer(self.frequency_, self.timerCallback)

    def timerCallback(self):
        try:
            if not self.is_connected_:
                self.init_i2c()
                return 
            
            block = self.bus_.read_i2c_block_data(DEVICE_ADDRESS, ACCEL_XOUT_H, 14)
            
            raw_acc_x = self.merge_bytes(block[0], block[1])
            raw_acc_y = self.merge_bytes(block[2], block[3])
            raw_acc_z = self.merge_bytes(block[4], block[5])
            
            raw_gyro_x = self.merge_bytes(block[8], block[9])
            raw_gyro_y = self.merge_bytes(block[10], block[11])
            raw_gyro_z = self.merge_bytes(block[12], block[13])
            
            acc_x = raw_acc_x - self.A_OFF_X
            acc_y = raw_acc_y - self.A_OFF_Y
            acc_z = raw_acc_z - self.A_OFF_Z
            
            gyro_x = raw_gyro_x - self.G_OFF_X
            gyro_y = raw_gyro_y - self.G_OFF_Y
            gyro_z = raw_gyro_z - self.G_OFF_Z
            
            self.imu_msg_.linear_acceleration.x = acc_x / 1670.13
            self.imu_msg_.linear_acceleration.y = acc_y / 1670.13
            self.imu_msg_.linear_acceleration.z = acc_z / 1670.13
            
            self.imu_msg_.angular_velocity.x = gyro_x / 939.65
            self.imu_msg_.angular_velocity.y = gyro_y / 939.65
            self.imu_msg_.angular_velocity.z = gyro_z / 939.65

            self.imu_msg_.header.stamp = self.get_clock().now().to_msg()
            
            self.imu_pub_.publish(self.imu_msg_)
            
        except OSError:
            self.get_logger().warn("I2C Bus Error. Reconnecting...")
            self.is_connected_ = False

    def merge_bytes(self, high, low):
        val = (high << 8) | low
        return val - 65536 if val > 32768 else val

    def init_i2c(self):
        try:
            self.bus_ = smbus.SMBus(1)
            self.bus_.write_byte_data(DEVICE_ADDRESS, SMPLRT_DIV, 7)
            self.bus_.write_byte_data(DEVICE_ADDRESS, PWR_MGMT_1, 1)
            self.bus_.write_byte_data(DEVICE_ADDRESS, CONFIG, 0)
            self.bus_.write_byte_data(DEVICE_ADDRESS, GYRO_CONFIG, 24)
            self.bus_.write_byte_data(DEVICE_ADDRESS, INT_ENABLE, 1)
            self.is_connected_ = True
            self.get_logger().info("MPU6050 Connected Successfully!")
        except OSError:
            self.is_connected_ = False
            self.get_logger().error("MPU6050 Connection Failed!")

def main():
    rclpy.init()
    mpu6050_driver = MPU6050_Driver()
    rclpy.spin(mpu6050_driver)
    mpu6050_driver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()