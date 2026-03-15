import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


def generate_launch_description():
    default_use_sim_time = "true"
    default_use_rviz = "true"
    default_cloud_topic = "/sim/point_cloud"
    default_scan_topic = "/scan"
    default_target_frame = "Leatherback"
    default_nav2_cmd_vel_topic = "/cmd_vel_nav"
    default_ackermann_cmd_topic = "/ackermann_cmd"
    default_scan_min_height = "0.05"
    default_scan_max_height = "0.80"
    default_map_yaml = os.path.join(
        get_package_share_directory("map2_navigation"), "maps", "map_2.yaml"
    )
    default_params_file = os.path.join(
        get_package_share_directory("map2_navigation"), "params", "nav2_params.yaml"
    )
    default_rviz_config = os.path.join(
        get_package_share_directory("nav2_bringup"), "rviz", "nav2_default_view.rviz"
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    use_rviz = LaunchConfiguration("use_rviz")
    cloud_topic = LaunchConfiguration("cloud_topic")
    scan_topic = LaunchConfiguration("scan_topic")
    target_frame = LaunchConfiguration("target_frame")
    nav2_cmd_vel_topic = LaunchConfiguration("nav2_cmd_vel_topic")
    ackermann_cmd_topic = LaunchConfiguration("ackermann_cmd_topic")
    scan_min_height = LaunchConfiguration("scan_min_height")
    scan_max_height = LaunchConfiguration("scan_max_height")
    wheelbase = LaunchConfiguration("wheelbase")
    max_steering_angle = LaunchConfiguration("max_steering_angle")
    map_yaml = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    rviz_config = LaunchConfiguration("rviz_config")

    nav2_bringup_launch_dir = os.path.join(
        get_package_share_directory("nav2_bringup"), "launch"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value=default_map_yaml,
                description="Full path to map yaml to load",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params_file,
                description="Full path to the ROS2 parameters file",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value=default_use_sim_time,
                description="Use simulation clock if true",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz_config,
                description="Full path to an RViz config file",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value=default_use_rviz,
                description="Whether to start RViz",
            ),
            DeclareLaunchArgument(
                "cloud_topic",
                default_value=default_cloud_topic,
                description="Input PointCloud2 topic to convert into LaserScan",
            ),
            DeclareLaunchArgument(
                "scan_topic",
                default_value=default_scan_topic,
                description="Output LaserScan topic (Nav2 AMCL/costmaps subscribe to this)",
            ),
            DeclareLaunchArgument(
                "target_frame",
                default_value=default_target_frame,
                description="TF frame to transform pointcloud into before projection",
            ),
            DeclareLaunchArgument(
                "nav2_cmd_vel_topic",
                default_value=default_nav2_cmd_vel_topic,
                description="Intermediate Twist cmd_vel topic produced by Nav2 (after remap)",
            ),
            DeclareLaunchArgument(
                "ackermann_cmd_topic",
                default_value=default_ackermann_cmd_topic,
                description="Ackermann command output topic for the vehicle",
            ),
            DeclareLaunchArgument(
                "scan_min_height",
                default_value=default_scan_min_height,
                description="Minimum Z (m) kept when projecting pointcloud to LaserScan",
            ),
            DeclareLaunchArgument(
                "scan_max_height",
                default_value=default_scan_max_height,
                description="Maximum Z (m) kept when projecting pointcloud to LaserScan",
            ),
            DeclareLaunchArgument(
                "wheelbase",
                default_value="0.32",
                description="Vehicle wheelbase in meters (used for Twist->Ackermann)",
            ),
            DeclareLaunchArgument(
                "max_steering_angle",
                default_value="0.60",
                description="Max steering angle in radians (used for Twist->Ackermann)",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(nav2_bringup_launch_dir, "rviz_launch.py")
                ),
                condition=IfCondition(use_rviz),
                launch_arguments={
                    "namespace": "",
                    "use_namespace": "False",
                    "use_sim_time": use_sim_time,
                    "rviz_config": rviz_config,
                }.items(),
            ),
            GroupAction(
                actions=[
                    SetRemap(src="cmd_vel", dst=nav2_cmd_vel_topic),
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            os.path.join(nav2_bringup_launch_dir, "bringup_launch.py")
                        ),
                        launch_arguments={
                            "map": map_yaml,
                            "use_sim_time": use_sim_time,
                            "params_file": params_file,
                        }.items(),
                    ),
                ]
            ),

            Node(
                package="map2_navigation",
                executable="twist_to_ackermann.py",
                name="twist_to_ackermann",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "use_sim_time": use_sim_time,
                        "wheelbase": wheelbase,
                        "max_steering_angle": max_steering_angle,
                    },
                ],
                remappings=[
                    ("cmd_vel", nav2_cmd_vel_topic),
                    ("ackermann_cmd", ackermann_cmd_topic),
                ],
            ),

            Node(
                package="pointcloud_to_laserscan",
                executable="pointcloud_to_laserscan_node",
                name="pointcloud_to_laserscan",
                output="screen",
                remappings=[("cloud_in", cloud_topic), ("scan", scan_topic)],
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "target_frame": target_frame,
                        "transform_tolerance": 0.01,
                        "min_height": scan_min_height,
                        "max_height": scan_max_height,
                        "angle_min": -3.14159,
                        "angle_max": 3.14159,
                        "angle_increment": 0.0087,
                        "scan_time": 0.3333,
                        "range_min": 0.05,
                        "range_max": 100.0,
                        "use_inf": True,
                        "inf_epsilon": 1.0,
                    }
                ],
            ),
        ]
    )
