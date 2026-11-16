"""Gazebo + Nav2 TurtleBot3 simulation plus the llm_memory_agent node.

    ros2 launch llm_memory_agent agent_sim.launch.py            # headless Gazebo
    ros2 launch llm_memory_agent agent_sim.launch.py headless:=False
    ros2 launch llm_memory_agent agent_sim.launch.py start_sim:=False  # sim already up

    ros2 launch llm_memory_agent agent_sim.launch.py provider:=ollama model:=qwen3:8b

provider:=anthropic (default) needs ANTHROPIC_API_KEY in the environment;
provider:=openai needs OPENAI_API_KEY; provider:=ollama needs a running Ollama
(base_url:=..., else OLLAMA_HOST, else localhost:11434). Empty model:= uses the
provider's default.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument("start_sim", default_value="True"),
        DeclareLaunchArgument("headless", default_value="True"),
        DeclareLaunchArgument("robot_config", default_value=PathJoinSubstitution(
            [FindPackageShare("llm_memory_agent"), "config", "tb3_sandbox.yaml"])),
        DeclareLaunchArgument("bundle_root", default_value="~/llm_memory_bundle"),
        DeclareLaunchArgument("memory_enabled", default_value="True"),
        DeclareLaunchArgument("provider", default_value="anthropic"),
        DeclareLaunchArgument("model", default_value=""),
        DeclareLaunchArgument("base_url", default_value=""),
        DeclareLaunchArgument("effort", default_value="medium"),
    ]
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare("nav2_bringup"), "launch", "tb3_simulation_launch.py"])),
        launch_arguments={"headless": LaunchConfiguration("headless")}.items(),
        condition=IfCondition(LaunchConfiguration("start_sim")),
    )
    agent = Node(
        package="llm_memory_agent",
        executable="agent_node",
        name="llm_memory_agent",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "robot_config": LaunchConfiguration("robot_config"),
            "bundle_root": LaunchConfiguration("bundle_root"),
            "memory_enabled": LaunchConfiguration("memory_enabled"),
            "provider": LaunchConfiguration("provider"),
            "model": LaunchConfiguration("model"),
            "base_url": LaunchConfiguration("base_url"),
            "effort": LaunchConfiguration("effort"),
        }],
    )
    return LaunchDescription([*args, sim, agent])
