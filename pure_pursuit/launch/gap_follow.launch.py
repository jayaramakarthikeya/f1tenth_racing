from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch the f1tenth_gym_ros gym_bridge_launch.py
    f1tenth_gym_ros_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('f1tenth_gym_ros'), 
                 'launch', 
                 'gym_bridge_launch.py']
            )
        )
    )

    # # Launch the safety_node node
    # safety_node = Node(
    #     package='safety_node',
    #     executable='safety_node'
    # )

    # Launch the pure pursuit node
    pure_pursuit_node = Node(
        package='pure_pursuit',
        executable='pure_pursuit_node',
        parameters=[
            {
            }
        ]
    )

    return LaunchDescription([
        f1tenth_gym_ros_launch,
        # safety_node,
        pure_pursuit_node
    ])