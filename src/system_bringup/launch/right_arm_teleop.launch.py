from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    piper_right_teacher_arm_node = Node(
        package='teacher_arm',
        executable='piper_teleop',
        name='piper_right_teacher',
        parameters=[
            {'name': 'right'},
            {'config_file': 'piper_gellos.yaml'},
        ],
        output='screen',
        emulate_tty=True,
    )

    piper_right_node = Node(
        package='piper_control',
        executable='piper_node',
        name='piper_teleop_right',
        parameters=[
            {'name': 'right'},
            {'config_file': 'piper_arm.yaml'},
        ],
        output='screen',
        emulate_tty=True,
    )


    return LaunchDescription([
        piper_right_teacher_arm_node,
        piper_right_node
    ])