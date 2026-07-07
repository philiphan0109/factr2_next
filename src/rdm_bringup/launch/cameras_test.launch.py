from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    left_wrist_camera_node = Node(
        package='cameras',
        executable='arducam_node',
        name='left_wrist_camera',
        parameters=[
            {'camera_name': 'left_wrist'},
            {'serial': 'glorbot2umi2wristleft'},
            {'config_file': "camera_config.yaml"}
        ],
        output='screen',
        emulate_tty=True,
    )

    right_wrist_camera_node = Node(
        package='cameras',
        executable='arducam_node',
        name='right_wrist_camera',
        parameters=[
            {'camera_name': 'right_wrist'},
            {'serial': 'glorbot2umi2wristright'},
            {'config_file': "camera_config.yaml"}
        ],
        output='screen',
        emulate_tty=True,
    )

    lower_scene_camera_node = Node(
        package='cameras',
        executable='arducam_node',
        name='lower_scene_camera',
        parameters=[
            {'camera_name': 'lower_scene'},
            {'serial': '00006'},
            {'config_file': "camera_config.yaml"}
        ],
        output='screen',
        emulate_tty=True,
    )


    return LaunchDescription([
        left_wrist_camera_node,
        right_wrist_camera_node,
        lower_scene_camera_node
    ])