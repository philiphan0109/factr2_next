import math
import threading
import time
import numpy as np

from piper_sdk import *
from piper_sdk import C_PiperInterface_V2
import yaml

class Rate:
    def __init__(self, hz):
        self.dt = 1.0 / hz
        self.next = time.perf_counter()
    
    def sleep(self):
        self.next += self.dt
        sleep_time = self.next - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            self.next = time.perf_counter()

class PiperArm:
    def __init__(self, urdf_dir, name, config):
        self.urdf_dir = urdf_dir
        self.name = name
        self.command_hz = config["shared"]["command_hz"]

        self.can_port = config[self.name]["can_port"]
        self.gripper_exist = config[self.name]["gripper_exist"]
        self.robot_base_pose = config[self.name]["robot_base_pose"]
        self.modifier = config[self.name]["modifier"]
        self.home_pos = config[self.name]["home_pos"]
        self.rest_pos = config[self.name]["rest_pos"]

        self.auto_enable = config["shared"]["auto_enable"]
        self.auto_read = config["shared"]["auto_read"]
        self.read_hz = config["shared"]["read_hz"]

        self.arm_enabled = False

        self.piper = C_PiperInterface_V2(can_name=self.can_port)
        self.piper.ConnectPort()

        self._prepare_buffers()

        if self.auto_enable:
            self.enable_arm()
        
        if self.auto_read:
            self.start_read_loop()

    def _prepare_buffers(self):
        # joint limits
        self.joint_pos_min = np.array([-2.6179, 0.0, -2.967, -1.745, -1.22, -2.09439])
        self.joint_pos_max = np.array([2.6179, 3.14, 0.0, 1.745, 1.22, 2.09439])
        self.gripper_pos_max = 0.095
        self.gripper_effort_max = 2.0

        # joint states
        self.joint_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.joint_prev_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.joint_arithmetic_velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.joint_motor_velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.joint_current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.joint_effort = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # joint gains
        self.kp = np.array([0.7, 0.5, 0.5, 1.0, 1.0, 1.0])*12
        self.kd = np.array([0.07, 0.05, 0.05, 0.1, 0.1, 0.1])*12

        # gripper states
        self.gripper_pos = 0.0
        self.gripper_prev_pos = 0.0
        self.gripper_vel = 0.0
        self.gripper_effort = 0.0

        # arm and gripper target states
        self.joint_pos_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.joint_vel_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.gripper_pos_target = 0.0
        self.gripper_effort_target = 0.0

        self.joint_state_dict = {
            "joint_pos": self.joint_pos.copy(),
            "joint_motor_vel": self.joint_motor_velocity.copy(),
            "joint_current": self.joint_current.copy(),
            "gripper_pos": self.gripper_pos,
            "gripper_vel": self.gripper_vel,
            "gripper_effort": self.gripper_effort
        }
    
    def get_joint_position(self):
        return self.joint_pos

    def get_joint_arithmetic_velocity(self):
        return self.joint_arithmetic_velocity

    def get_joint_motor_velocity(self):
        return self.joint_motor_velocity

    def get_joint_current(self):
        return self.joint_current

    def get_joint_effort(self):
        return self.joint_effort

    def get_gripper_position(self):
        return self.gripper_pos
    
    def get_gripper_velocity(self):
        return self.gripper_vel
    
    def get_gripper_effort(self):
        return self.gripper_effort

    def enable_arm(self):
        start_time = time.time()
        while True:
            if self.gripper_exist:
                self.piper.GripperCtrl(0, 1000, 0x01, 0)
            
            motors_enabled = all([
                self.piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status, 
                self.piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status, 
                self.piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status, 
                self.piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status, 
                self.piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status, 
                self.piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status, 
            ])
            if motors_enabled: 
                self.arm_enabled = True
                return True
            
            # 7 means to enable all joints
            self.piper.EnableArm(7)

            elapsed_time = time.time() - start_time
            if elapsed_time > 5:
                print("[ERROR] Piper Arm Enable Arm Timeout.")
                return False

            time.sleep(0.01)
            
    def disable(self):
        print(f"[DISABLING] Piper Arm [{self.can_port}] Disabling.")
        if self.gripper_exist:
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
        
        # go to home position
        self.go_to_home_config()

        # go to rest position
        self.go_to_rest_config()

        # hold at rest position
        self.hold_at_joint_config(self.rest_pos, hold_time=1.0)


        self.arm_enabled = False
        self.piper.DisableArm(7)
        if self.gripper_exist:
            self.piper.GripperCtrl(0, 1000, 0x00, 0)
        
        print(f"[DISABLING] Piper Arm [{self.can_port}] Disabled.")

    def go_to_joint_config(self, joint_pos_target, steps = 250):
        current_joint_pos = self.joint_pos.copy()
        joint_position_targets = np.linspace(current_joint_pos, joint_pos_target, steps)
        for joint_target in joint_position_targets:
            self.set_arm_joint_target(joint_target)
            time.sleep(1/self.command_hz)
    
    def go_to_home_config(self):
        self.go_to_joint_config(self.home_pos, steps=250)

    def go_to_rest_config(self):
        self.go_to_joint_config(self.rest_pos, steps=250)

    def hold_at_joint_config(self, joint_pos_target, hold_time = 1.0):
        steps = round(self.command_hz * hold_time)
        for _ in range(steps):
            self.set_arm_joint_target(joint_pos_target)
            time.sleep(1 / self.command_hz)

    def read_joint_state(self):
        # NOTE: raw data is in units deg * 1000 (0.001 deg). this converts it into radians
        def conv(deg):
            return (deg / 1000) * (math.pi / 180)
        
        self.joint_prev_pos = self.joint_pos.copy()

        self.joint_pos = np.asarray([
            conv(self.piper.GetArmJointMsgs().joint_state.joint_1),
            conv(self.piper.GetArmJointMsgs().joint_state.joint_2),
            conv(self.piper.GetArmJointMsgs().joint_state.joint_3),
            conv(self.piper.GetArmJointMsgs().joint_state.joint_4),
            conv(self.piper.GetArmJointMsgs().joint_state.joint_5),
            conv(self.piper.GetArmJointMsgs().joint_state.joint_6),
        ])

        self.joint_arithmetic_velocity = (self.joint_pos - self.joint_prev_pos) / (1 / self.read_hz)

        self.joint_motor_velocity = np.asarray([
            self.piper.GetArmHighSpdInfoMsgs().motor_1.motor_speed / 1000,
            self.piper.GetArmHighSpdInfoMsgs().motor_2.motor_speed / 1000,
            self.piper.GetArmHighSpdInfoMsgs().motor_3.motor_speed / 1000,
            self.piper.GetArmHighSpdInfoMsgs().motor_4.motor_speed / 1000,
            self.piper.GetArmHighSpdInfoMsgs().motor_5.motor_speed / 1000,
            self.piper.GetArmHighSpdInfoMsgs().motor_6.motor_speed / 1000,
        ])

        self.joint_current = np.asarray([
            self.piper.GetArmHighSpdInfoMsgs().motor_1.current / 1000,
            self.piper.GetArmHighSpdInfoMsgs().motor_2.current / 1000,
            self.piper.GetArmHighSpdInfoMsgs().motor_3.current / 1000,
            self.piper.GetArmHighSpdInfoMsgs().motor_4.current / 1000,
            self.piper.GetArmHighSpdInfoMsgs().motor_5.current / 1000,
            self.piper.GetArmHighSpdInfoMsgs().motor_6.current / 1000,
        ])

        self.joint_current = np.asarray([
            self.piper.GetArmHighSpdInfoMsgs().motor_1.effort / 1000,
            self.piper.GetArmHighSpdInfoMsgs().motor_2.effort / 1000,
            self.piper.GetArmHighSpdInfoMsgs().motor_3.effort / 1000,
            self.piper.GetArmHighSpdInfoMsgs().motor_4.effort / 1000,
            self.piper.GetArmHighSpdInfoMsgs().motor_5.effort / 1000,
            self.piper.GetArmHighSpdInfoMsgs().motor_6.effort / 1000,
        ])

        self.joint_state_dict["joint_pos"] = self.joint_pos.copy()
        self.joint_state_dict["joint_motor_vel"] = self.joint_motor_velocity.copy()
        self.joint_state_dict["joint_current"] = self.joint_current.copy()

        if self.gripper_exist:
            self.gripper_prev_pos = self.gripper_pos
            self.gripper_pos = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1e6
            self.gripper_vel = (self.gripper_pos - self.gripper_prev_pos) / (1 / self.read_hz)  
            self.gripper_effort = self.piper.GetArmGripperMsgs().gripper_state.grippers_effort/ 1000

            self.joint_state_dict["gripper_pos"] = self.gripper_pos
            self.joint_state_dict["gripper_vel"] = self.gripper_vel
            self.joint_state_dict["gripper_effort"] = self.gripper_effort

        return self.joint_state_dict

    def read_states_callback(self):
        read_rate = Rate(self.read_hz)
        while True:
            self.read_joint_state()
            read_rate.sleep()

    def start_read_loop(self):
        self.thread = threading.Thread(target=self.read_states_callback)
        self.thread.daemon = True
        self.thread.start()


    def set_arm_joint_target(self, joint_pos_target, joint_vel_target=np.zeros(6), gravity_comp=False, kp=None, kd=None):
        if self.arm_enabled:
            self.joint_pos_target = joint_pos_target
            self.joint_vel_target = joint_vel_target
            self.piper.MotionCtrl_2(0x01, 0x04, 0, 0xAD)

            # MITCtrl
            # pos_ref: Desired target position, unit: rad, range [-12.5, 12.5] (seems like unit is rad)
            # vel_ref: Desired motor speed, range [-45.0, 45.0] (seems like unit is rad/s)
            # kp: Proportional gain, controls the influence of position error on output torque, reference value: 10, range [0.0, 500.0] (seems like unit is N/rad)
            # kd: Derivative gain, controls the influence of speed error on output torque, reference value: 0.8, range [-5.0, 5.0] (seems like unit is N*s/rad)
            # t_ref: Target torque reference, controls the torque applied by the motor, range [-18.0, 18.0] (seems like 18.0 corresponds to 8.0 Nm)
            # self.piper.JointMitCtrl(int(motor_id), pos, 0, self.kp[int(motor_id-1)], self.kd[int(motor_id-1)], tau)

            # gravity comp
            if gravity_comp:
                # TO BE IMPLEMENTED
                torque_ff = np.zeros(6)
            else:
                torque_ff = np.zeros(6)

            if kp is None:
                kp = self.kp
            if kd is None:
                kd = self.kd
            
            for joint_idx in range(6):
                self.piper.JointMitCtrl(joint_idx+1, self.joint_pos_target[joint_idx], self.joint_vel_target[joint_idx], kp[joint_idx], kd[joint_idx], torque_ff[joint_idx])
        else:
            print(f"[IGNORING] Piper Arm {self.can_port} Not Enabled. Ignoring Joint Position Command.")
            

    def set_gripper_target(self, gripper_pos_target, gripper_effort_target=0, normalized=False):
        # normalized assumes that the gripper position was passed in [0, 1]
        if self.arm_enabled:
            if self.gripper_exist:

                self.gripper_pos_target = gripper_pos_target * self.gripper_pos_max if normalized else gripper_pos_target
                self.gripper_effort_target = gripper_effort_target

                self.gripper_pos_target = np.clip(self.gripper_pos_target, 0.0, self.gripper_pos_max)
                self.gripper_effort_target = np.clip(self.gripper_effort_target, 0.0, self.gripper_effort_max)

                self.piper.GripperCtrl(round(self.gripper_pos_target * 1e6), round(self.gripper_effort_target * 1000), 0x01, 0)
            else:
                print(f"[IGNORING] Piper Arm {self.can_port} Has No Gripper. Ignoring Gripper Position Command.")
        else:
            print(f"[IGNORING] Piper Arm {self.can_port} Not Enabled. Ignoring Gripper Position Command.")

def main():

    with open("/home/kshaw/rdm-deploy/src/piper_control/piper_control/configs/piper_arm.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    arm = PiperArm(urdf_dir="/home/kshaw/rdm-deploy/src/piper_control/piper_control/urdf", name = "right", config = config)
    arm.enable_arm()

    start_time = time.time()

    at_hold_pose = False
    
    try: 
        while True:
            joint_states = arm.read_joint_state()
            joint_positions = joint_states["joint_pos"]

            t = time.time() - start_time

            if not at_hold_pose:
                arm.go_to_joint_config([0.0, 0.35, -0.35, 0.0, 0.0, 0.0])
                time.sleep(0.1)
                at_hold_pose = True
            else:
                wiggle1 = 0.05 * math.sin(2 * math.pi * 0.25 * t)
                wiggle2 = 0.5 * math.sin(2 * math.pi * 0.15 * t)
                wiggle3 = 0.5 * math.sin(2 * math.pi * 0.35 * t)
                arm.set_arm_joint_target([0.0, 0.35, -0.35, 0.0, 0.0, 0.0])
                arm.set_gripper_target(abs(wiggle1), 2.0)

            time.sleep(1 / arm.command_hz)
    
    except KeyboardInterrupt:
        arm.disable()

if __name__ == "__main__":
    main()