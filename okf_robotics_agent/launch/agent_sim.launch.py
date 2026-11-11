"""Gazebo + Nav2 TurtleBot3 simulation plus the okf_robotics_agent node.

    ros2 launch okf_robotics_agent agent_sim.launch.py            # headless Gazebo
    ros2 launch okf_robotics_agent agent_sim.launch.py headless:=False
    ros2 launch okf_robotics_agent agent_sim.launch.py start_sim:=False  # sim already up

Needs ANTHROPIC_API_KEY (or an `ant auth login` profile) in the environment.
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
            [FindPackageShare("okf_robotics_agent"), "config", "tb3_sandbox.yaml"])),
        DeclareLaunchArgument("bundle_root", default_value="~/okf_robotics_bundle"),
        DeclareLaunchArgument("memory_enabled", default_value="True"),
        DeclareLaunchArgument("model", default_value="claude-opus-5-5"),
        DeclareLaunchArgument("effort", default_value="medium"),
    ]
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare("nav2_bringup"), "launch", "tb3_simulation_launch.py"])),
        launch_arguments={"headless": LaunchConfiguration("headless")}.items(),
        condition=IfCondition(LaunchConfiguration("start_sim")),
    )
    agent = Node(
        package="okf_robotics_agent",
        executable="agent_node",
        name="okf_robotics_agent",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "robot_config": LaunchConfiguration("robot_config"),
            "bundle_root": LaunchConfiguration("bundle_root"),
            "memory_enabled": LaunchConfiguration("memory_enabled"),
            "model": LaunchConfiguration("model"),
            "effort": LaunchConfiguration("effort"),
        }],
    )
    return LaunchDescription([*args, sim, agent])
