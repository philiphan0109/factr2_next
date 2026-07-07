import os
import time
import numpy as np
import yaml

from piper_control.piper import PiperArm

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

from ament_index_python.packages import get_package_share_directory

class PiperNode(Node):
    def __init__(self):
        super().__init__("piper_node")


        self.name = self.declare_parameter("name", "right").get_parameter_value().string_value
        self.config_file_name = self.declare_parameter("config_file", "piper_arm.yaml").get_parameter_value().string_value

        share_dir = get_package_share_directory('piper_control')

        self.config_file_path = os.path.join(share_dir, 'configs', self.config_file_name)
        self.urdf_model_dir = os.path.join(share_dir, 'urdf')
        
        with open(self.config_file_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)
        
        self.can_port = self.config[self.name]["can_port"]
        self.gripper_exist = self.config[self.name]["gripper_exist"]
        self.publish_hz = self.config["shared"]["publish_hz"]
        
        self.piper = PiperArm(urdf_dir=self.urdf_model_dir, name=self.name, config=self.config)
        self.piper.go_to_home_config()

        self._set_up_communication()
        self.obs_publish_timer = self.create_timer(1.0/self.publish_hz, self._observation_publish_callback)
    
    def _create_joint_state_message(self, data, timestamp = None):
        msg = JointState()
        if timestamp is not None:
            msg.header.stamp = timestamp
        else:
            msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = [float(x) for x in data]
        return msg

    def _create_stamped_pose_message(self, position, orientation = None, timestamp = None):
        msg = PoseStamped()
        if timestamp is not None:
            msg.header.stamp = timestamp
        else:
            msg.header.stamp = self.get_clock().now().to_msg()
        
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]

        if orientation is not None:
            msg.pose.orientation.x = orientation[0]
            msg.pose.orientation.y = orientation[1]
            msg.pose.orientation.z = orientation[2]
            msg.pose.orientation.w = orientation[3]
        
        return msg
    
    def _set_up_communication(self):
        
        self.joint_pos_cmd_sub = self.create_subscription(JointState, f"/piper/{self.name}/joint_pos_cmd", self._joint_pos_cmd_callback, qos_profile=qos_profile_sensor_data)
        self.gripper_pos_cmd_sub = self.create_subscription(JointState, f"/piper/{self.name}/gripper_pos_cmd", self._gripper_pos_cmd_callback, qos_profile=qos_profile_sensor_data)

        self.joint_pos_obs_pub = self.create_publisher(JointState, f"/piper/{self.name}/joint_pos_obs", qos_profile=qos_profile_sensor_data)
        self.joint_vel_obs_pub = self.create_publisher(JointState, f"/piper/{self.name}/joint_vel_obs", qos_profile=qos_profile_sensor_data)
        self.joint_effort_obs_pub = self.create_publisher(JointState, f"/piper/{self.name}/joint_effort_obs", qos_profile=qos_profile_sensor_data)

        self.gripper_pos_obs_pub = self.create_publisher(JointState, f"/piper/{self.name}/gripper_pos_obs", qos_profile=qos_profile_sensor_data)
        self.gripper_eff_obs_pub = self.create_publisher(JointState, f"/piper/{self.name}/gripper_eff_obs", qos_profile=qos_profile_sensor_data)

    def _joint_pos_cmd_callback(self, msg):
        joint_pos_target = np.array(msg.position)
        self.piper.set_arm_joint_target(joint_pos_target=joint_pos_target)

    def _gripper_pos_cmd_callback(self, msg):
        gripper_pos_target = msg.position[0]
        gripper_effort_target = msg.position[1]

        self.piper.set_gripper_target(gripper_pos_target=gripper_pos_target, gripper_effort_target=gripper_effort_target, normalized=True)

    def _shutdown(self):
        self.piper.disable()

    def _observation_publish_callback(self):
        curr_time = self.get_clock().now().to_msg()
        joint_position = self.piper.get_joint_position()
        joint_velocity = self.piper.get_joint_arithmetic_velocity()
        joint_effort = self.piper.get_joint_current()
        gripper_position = self.piper.get_gripper_position()
        gripper_effort = self.piper.get_gripper_effort()

        self.joint_pos_obs_pub.publish(self._create_joint_state_message(data=joint_position, timestamp=curr_time))
        self.joint_vel_obs_pub.publish(self._create_joint_state_message(data=joint_velocity, timestamp=curr_time))
        self.joint_effort_obs_pub.publish(self._create_joint_state_message(data=joint_effort, timestamp=curr_time))

        self.gripper_pos_obs_pub.publish(self._create_joint_state_message(data=[gripper_position], timestamp=curr_time))
        self.gripper_eff_obs_pub.publish(self._create_joint_state_message(data=[gripper_effort], timestamp=curr_time))

def main(args=None):
    rclpy.init(args=args)
    node = PiperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("[SHUTTING DOWN] Piper Node received KeyboardInterrupt.")
    finally:
        node._shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
