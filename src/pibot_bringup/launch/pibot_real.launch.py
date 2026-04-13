import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    replan_enabled_arg = DeclareLaunchArgument(
        "replan_enabled",
        default_value="true",
        description="Enable or disable the DWA replanning feature (true/false)"
    )
    replan_arg = LaunchConfiguration("replan_enabled")

    laser_driver = Node(
            package="rplidar_ros",
            executable="rplidar_node",
            name="rplidar_node",
            parameters=[os.path.join(
                get_package_share_directory("pibot_bringup"),
                "config",
                "rplidar_a1.yaml"
            )],
            output="screen"
    )

    pibot_controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("pibot_controller"),
                "launch",
                "pibot_controller.launch.py"
            )
        )
    )

    joy_controller = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("pibot_controller"),
                "launch",
                "joy_teleop.launch.py"
            )
        )
    )

    local_localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("pibot_localization"),
                "launch",
                "ekf_fusion.launch.py"
            )
        )
    )

    global_localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("pibot_localization"),
                "launch",
                "global_localization.launch.py"
            )
        )
    )
    
    local_plan = IncludeLaunchDescription(
        PythonLaunchDescriptionSource( 
            os.path.join(
                get_package_share_directory("pibot_localplan"),
                "launch",
                "dwa.launch.py"
            )
        ),
        launch_arguments={"replan_enabled": replan_arg}.items()
    )   

    global_plan = IncludeLaunchDescription(
        PythonLaunchDescriptionSource( 
            os.path.join(
                get_package_share_directory("pibot_globalplan"),
                "launch",
                "globalplan.launch.py"
            )
        ),
        launch_arguments={"replan_enabled": replan_arg}.items()
    )   

    return LaunchDescription([
        replan_enabled_arg, 
        pibot_controller,
        joy_controller,
        laser_driver,
        local_localization,
        global_localization,
    ])