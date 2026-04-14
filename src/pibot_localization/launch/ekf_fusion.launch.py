from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os

def generate_launch_description():

    mpu6050 = Node(
        package="pibot_localization",
        executable="mpu6050_driver.py",
        name="mpu6050_node",
        output="screen",
    )

    madgwick_config_path = os.path.join(
        get_package_share_directory('pibot_localization'),
        'config',
        'madgwick_params.yaml'
    )

    madgwick_filter_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        parameters=[madgwick_config_path],
        remappings=[
            ('/imu/data_raw', '/imu/data_raw'),
            ('/imu/data', '/imu/data')
        ]
    )

    robot_localization = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[os.path.join(get_package_share_directory("pibot_localization"), "config", "ekf.yaml")],
    )


    return LaunchDescription([
        robot_localization,
        mpu6050,
        madgwick_filter_node
    ])