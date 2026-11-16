"""Launch llm_memory_capture with its default config and a configurable mission
name. See requirements.md's Phase 3 note: ship a ros2 launch example, not just
library usage."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("llm_memory_capture"), "config", "capture_params.yaml"]
    )
    config_path_arg = DeclareLaunchArgument("config_path", default_value=default_config)
    mission_name_arg = DeclareLaunchArgument("mission_name", default_value="session")

    node = Node(
        package="llm_memory_capture",
        executable="capture_node",
        name="llm_memory_capture",
        output="screen",
        parameters=[
            {
                "config_path": LaunchConfiguration("config_path"),
                "mission_name": LaunchConfiguration("mission_name"),
            }
        ],
    )

    return LaunchDescription([config_path_arg, mission_name_arg, node])
