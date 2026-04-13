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
        local_plan,
        global_plan
    ])