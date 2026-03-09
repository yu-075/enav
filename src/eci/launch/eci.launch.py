import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('eci')
    eci_params_path = os.path.join(pkg_dir, 'config', 'eci_params.yaml')

    return LaunchDescription([
        Node(
            package='eci',
            executable='eci_monitor.py',
            name='eci_monitor',
            output='screen',
            parameters=[
                {'topic': '/eci_value'},
                {'window_seconds': 30.0},
            ],
        ),
        Node(
            package='eci',
            executable='eci_node.py',
            name='eci_node',
            output='screen',
            parameters=[
                eci_params_path,
            ],
        ),
    ])
