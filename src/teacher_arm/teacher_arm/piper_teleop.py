import os
import time 
import numpy as np
import yaml
 

import rclpy
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from teacher_arm.base_teleop import BaseTeleopController


class PiperTeleop(BaseTeleopController):
    def __init__(self):
        super().__init__()
        self.gripper_exist = self.declare_parameter("gripper_exist", True).get_parameter_value().bool_value
        self.gripper_external_torque = 0.0

    def _create_joint_state_message(self, data, timestamp = None):
        msg = JointState()
        if timestamp is not None:
            msg.header.stamp = timestamp
        else:
            msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = [float(x) for x in data]
        return msg

    def _gripper_eff_obs_callback(self, msg):
        self.gripper_external_torque = msg.position[0]

    def set_up_communication(self):
        self.arm_pos_command_pub = self.create_publisher(JointState, f"/piper/{self.name}/joint_pos_cmd", qos_profile=qos_profile_sensor_data)
        self.gripper_pos_command_pub = self.create_publisher(JointState, f"/piper/{self.name}/gripper_pos_cmd", qos_profile=qos_profile_sensor_data)

        self.matched_pub = self.create_publisher(Bool, f"/piper/{self.name}/matched", qos_profile=qos_profile_sensor_data)

        self.gripper_eff_obs_sub = self.create_subscription(JointState, f"/piper/{self.name}/gripper_eff_obs", self._gripper_eff_obs_callback, qos_profile=qos_profile_sensor_data)
    
    def compute_gripper_feedback_torque(self, leader_gripper_pos, leader_gripper_vel):
        kp = self.gripper_feedback_kp
        kd = self.gripper_feedback_kd
        
        target = self.gripper_command_range
        error = target - leader_gripper_pos
        
        # constant spring back force
        p_term = kp* error
        d_term = -kd * leader_gripper_vel
        ff_term = self.gripper_feedback_kff * np.sign(error) if abs(error) > self.gripper_feedback_ff_deadband else 0.0

        # external contact feedback
        follower_gripper_effort = self.gripper_external_torque
        contact_effort = max(0.0, -follower_gripper_effort - self.gripper_feedback_contact_threshold)
        contact_term = self.gripper_feedback_contact_gain * contact_effort
        contact_term = np.clip(contact_term, 0.0, self.gripper_feedback_contact_max)


        gripper_torque = p_term + d_term + ff_term + contact_term
        # if self.step % 100 == 0:
        #     self.get_logger().info(
        #         f"grip pos={leader_gripper_pos:.3f} "
        #         f"err={error:.3f} "
        #         f"vel={leader_gripper_vel:.3f} "
        #         f"follower_eff={follower_gripper_effort:.3f} "
        #         f"C={contact_term:.3f} tau={gripper_torque:.3f}"
        #     )
        return gripper_torque

    
    def update_communication(self, leader_arm_pos, leader_gripper_pos):
        self.arm_pos_command_pub.publish(self._create_joint_state_message(leader_arm_pos))

        gripper_cmd_normalized = np.clip(leader_gripper_pos / self.gripper_command_range, 0.0, 1.0)
        if gripper_cmd_normalized < self.gripper_quantization_lower_threshold:
            gripper_cmd_normalized = 0.0
        elif gripper_cmd_normalized > self.gripper_quantization_upper_threshold:
            gripper_cmd_normalized = 1.0
        gripper_cmd = gripper_cmd_normalized * self.gripper_limit_max
        self.gripper_pos_command_pub.publish(self._create_joint_state_message([gripper_cmd, 2.0]))
        
def main(args = None):
    rclpy.init()
    node = PiperTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt Recevied.")
    finally:
        node.shut_down()
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()