from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    metronome_node = Node(
        package='data_synchronization',
        executable='metronome_node',
        name='metronode_node',
        parameters=[
            {'config_file': 'bimanual_collect_data.yaml'},
        ],
        output='screen',
        emulate_tty=True,
    )

    policy_interface = Node(
        package="rdm_policy",
        executable="policy_interface",
        name="policy_interface",
        parameters=[
            {'config_file': 'replay_policy.yaml'}
        ]
    )


    return LaunchDescription([
        metronome_node,
        policy_interface,
    ])