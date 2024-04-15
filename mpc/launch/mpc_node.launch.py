from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    # Launch the pure pursuit node
    mpc_node = Node(
        package='mpc',
        executable='scripts/mpc_node.py',
        parameters=[
            {
            }
        ]
    )

    return LaunchDescription([
        # f1tenth_gym_ros_launch,
        # safety_node,
        mpc_node
    ])