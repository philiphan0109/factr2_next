
from abc import ABC, abstractmethod
import os
import subprocess
import time
import numpy as np
import pinocchio as pin
import yaml

from rclpy.node import Node
from std_msgs.msg import Bool

from ament_index_python.packages import get_package_share_directory
from teacher_arm.dynamixel.driver import DynamixelDriver
from teacher_arm.factr2_feedback import Factr2TorqueFeedback

from scipy.spatial.transform import Rotation

class BaseTeleopController(Node, ABC):
    def __init__(self):
        super().__init__('base_teleop_controller')

        self.name = self.declare_parameter('name', 'right').get_parameter_value().string_value
        self.config_file_name = self.declare_parameter('config_file', 'piper_gellos.yaml').get_parameter_value().string_value

        self.share_dir = get_package_share_directory('teacher_arm')

        self.config_file_path = os.path.join(self.share_dir, 'configs', self.config_file_name)
        with open(self.config_file_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

        self.control_hz = self.config["shared"]["controller"]["frequency"]
        self.dt = 1.0 / self.control_hz

        self.step = 0
        self.skip = 5

        self._shutting_down = False

        self._prepare_inverse_dynamics()
        self._prepare_dynamixels()

        # leader arm parameters
        self.num_arm_joints = self.config["shared"]["arm_teleop"]["num_arm_joints"]
        self.safety_margin = self.config["shared"]["arm_teleop"]["arm_joint_limit_safety_margin"]
        self.arm_joint_limits_min = np.array(self.config["shared"]["arm_teleop"]["arm_joint_limits_min"]) + self.safety_margin
        self.arm_joint_limits_max = np.array(self.config["shared"]["arm_teleop"]["arm_joint_limits_max"]) - self.safety_margin

        self.calibration_joint_pos = np.array(self.config[self.name]["calibration_joint_pos"])
        self.arm_initial_match_pos = np.array(self.config["shared"]["arm_teleop"]["initial_arm_match_pos"])

        assert self.num_arm_joints == len(self.arm_joint_limits_min)
        assert self.num_arm_joints == len(self.arm_joint_limits_max)
        assert self.num_arm_joints == len(self.calibration_joint_pos)
        assert self.num_arm_joints == len(self.arm_initial_match_pos)

        # leader gripper parameters
        self.gripper_limit_min = 0.0
        self.gripper_limit_max = self.config["shared"]["gripper_teleop"]["actuation_range"]
        self.gripper_prev_pos = 0.0
        self.gripper_pos = 0.0
        self.gripper_command_range = self.config["shared"]["gripper_teleop"]["gripper_command_range"]

        ### low level controller

        # gravity compensation
        self.gravity_compenation_enabled = self.config["shared"]["controller"]["gravity_comp"]["enable"]
        self.gravity_compensation_modifier = np.array(self.config["shared"]["controller"]["gravity_comp"]["modifier"], dtype=float)
        self.tau_gravity = np.zeros(self.num_arm_joints)
        self.gello_base_orientation = self.config[self.name]["gello_base_orientation"]

        # static friction compensation
        self.static_friction_dither_enabled = self.config["shared"]["controller"]["static_friction_dither"]["enable"]
        self.tau_static_friction_dither = np.zeros(self.num_arm_joints)
        self.static_friction_compensation_enable_speed = self.config["shared"]["controller"]["static_friction_dither"]["enable_speed"]
        self.static_friction_compensation_modifier = self.config["shared"]["controller"]["static_friction_dither"]["modifier"]
        self.static_friction_dither_flag = np.ones(self.num_arm_joints, dtype = bool)

        # joint limit repulsion controller
        self.joint_limit_repulsion_kp = self.config["shared"]["controller"]["joint_limit_barrier_repulsion"]["kp"]
        self.joint_limit_repulsion_kd = self.config["shared"]["controller"]["joint_limit_barrier_repulsion"]["kd"]

        # null space regulation
        self.null_space_regulation_enabled = self.config["shared"]["controller"]["null_space_regulation"]["enable"]
        self.null_space_regulation_joint_targets = self.config["shared"]["controller"]["null_space_regulation"]["null_space_joint_targets"]
        self.null_space_regulation_modifier = self.config["shared"]["controller"]["null_space_regulation"]["null_space_regulation_modifier"]
        self.null_space_regulation_kp = self.config["shared"]["controller"]["null_space_regulation"]["kp"]
        self.null_space_regulation_kd = self.config["shared"]["controller"]["null_space_regulation"]["kd"]

        # torque feedback
        # TODO: implement

        # gripper feedback
        self.enable_gripper_feedback = self.config["shared"]["controller"]["gripper_feedback"]["enable"]
        self.gripper_feedback_kp = self.config["shared"]["controller"]["gripper_feedback"]["kp"]
        self.gripper_feedback_kd = self.config["shared"]["controller"]["gripper_feedback"]["kd"]
        self.gripper_feedback_ff_deadband = self.config["shared"]["controller"]["gripper_feedback"]["ff_deadband"]
        self.gripper_feedback_kff = self.config["shared"]["controller"]["gripper_feedback"]["kff"]
        self.gripper_feedback_contact_threshold = self.config["shared"]["controller"]["gripper_feedback"]["contact_threshold"]
        self.gripper_feedback_contact_gain = self.config["shared"]["controller"]["gripper_feedback"]["contact_gain"]
        self.gripper_feedback_contact_max = self.config["shared"]["controller"]["gripper_feedback"]["contact_feedback_max_torque"]
        self.gripper_quantization_lower_threshold = self.config["shared"]["controller"]["gripper_quantization"]["lower_threshold"]
        self.gripper_quantization_upper_threshold = self.config["shared"]["controller"]["gripper_quantization"]["upper_threshold"]
        

        # position control
        self.joint_position_control_kp  = np.array(self.config["shared"]["controller"]["joint_position_control"]["kp"])
        self.joint_position_control_kd  = np.array(self.config["shared"]["controller"]["joint_position_control"]["kd"])


        ### main set up loop
        self._set_gravity_vector(self.gello_base_orientation)
        self.set_up_communication()
        self._setup_factr2_feedback()
        self._get_dynamixel_offsets()
        self._match_start_pos()

        self.timer = self.create_timer(self.dt, self.control_loop_callback)

    def _setup_factr2_feedback(self):
        cfg = self.config["shared"]["controller"].get("factr2_torque_feedback", {})
        self.factr2_feedback = Factr2TorqueFeedback(
            self,
            self.name,
            cfg,
            self.num_arm_joints,
        )

    def _prepare_inverse_dynamics(self):
        self.urdf_name = self.config["shared"]["arm_teleop"]["teacher_arm_urdf"]
        self.urdf_dir = os.path.join(self.share_dir, "urdf")
        self.urdf_path = os.path.join(self.urdf_dir, self.urdf_name)

        self.pin_model, _, _ = pin.buildModelsFromUrdf(filename=self.urdf_path, package_dirs=self.urdf_dir)
        self.pin_data = self.pin_model.createData()
        
    
    def find_ttyusb(self, port_name):
        """
        This function is used to locate the underlying ttyUSB device.
        """
        base_path = "/dev/serial/by-id/"
        full_path = os.path.join(base_path, port_name)
        if not os.path.exists(full_path):
            raise Exception(f"Port '{port_name}' does not exist in {base_path}.")
        try:
            resolved_path = os.readlink(full_path)
            actual_device = os.path.basename(resolved_path)
            if actual_device.startswith("ttyUSB"):
                return actual_device
            else:
                raise Exception(
                    f"The port '{port_name}' does not correspond to a ttyUSB device. It links to {resolved_path}."
                )
        except Exception as e:
            raise Exception(f"Unable to resolve the symbolic link for '{port_name}'. {e}")


    def _prepare_dynamixels(self):
        """
        sets up dyanimxel drivers for teacher arm
        """
        self.servo_types = self.config["shared"]["dynamixel"]["servo_types"]
        self.num_motors = len(self.servo_types)
        self.joint_signs = np.array(self.config[self.name]["dynamixel"]["joint_signs"], dtype = float)

        assert self.num_motors == len(self.servo_types)
        assert self.num_motors == len(self.joint_signs)

        self.dynamixel_port = "/dev/serial/by-id/" + self.config[self.name]["dynamixel"]["port"]

                # checks of the latency timer on ttyUSB of the corresponding port is 1
        # if it is not 1, the control loop cannot run at above 200 Hz, which will 
        # cause extremely undesirable behaviour for the leader arm. If the latency 
        # timer is not 1, one can set it to 1 as follows:
        # echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB{NUM}/latency_timer
        ttyUSBx = self.find_ttyusb(self.dynamixel_port)
        command = f"cat /sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"        
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        ttyUSB_latency_timer = int(result.stdout)
        if ttyUSB_latency_timer != 1:
            print(f"[INFO] Setting latency_timer of {ttyUSBx} to 1...")
            set_cmd = f"echo 1 | sudo tee /sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"
            subprocess.run(set_cmd, shell=True, check=True)
        
        self.joint_ids = np.arange(self.num_motors) + 1
        try:
            self.driver = DynamixelDriver(self.joint_ids, self.servo_types, self.dynamixel_port)
        except FileNotFoundError:
            self.get_logger().info(f"Port {self.dynamixel_port} not found.")
            return
        
        self.driver.set_torque_mode(False)
        self.driver.set_operating_mode(0) # current mode
        self.driver.set_torque_mode(True)

    def _get_dynamixel_offsets(self, verbose = False):
        """
        calibrates the dynamixel motors with respect to arm to ensure that the joint position readings of the leader arm correspond to those of the follower arm

        before launching, place the leader arm manually into a position roughly corresponding to the follower's calibration position described in self.calibration_joint_pos

        this method is a macro-calibration method, it makes sure that rotations of the dynamixel motors off by pi/2 radians are handled properly
        """
        for i in range(50):
            self.driver.get_positions_and_velocities()
        
        def _get_error(calibration_joint_pos, offset, index, joint_state):
            joint_sign_i = self.joint_signs[index]
            joint_i_pos = joint_sign_i * (joint_state[index] - offset)
            calibration_i_pos = calibration_joint_pos[index]

            return np.abs(joint_i_pos - calibration_i_pos)
    
        # loop through possible joint offsets (one step every pi/2 rads)
        self.joint_offsets = []
        curr_joint_positions, _ = self.driver.get_positions_and_velocities()
        for i in range(self.num_arm_joints):
            best_offset = 0.0
            best_error = 1e9
            for offset in np.linspace(-20.0 * np.pi , 20.0 * np.pi, 40 * 2 + 1):
                error = _get_error(self.calibration_joint_pos, offset, i, curr_joint_positions)
                if error < best_error:
                    best_error = error
                    best_offset = offset
            self.joint_offsets.append(best_offset)
        
        curr_gripper_position = curr_joint_positions[-1] + 0.5 * self.joint_signs[-1] # understand that 0.5 is a hardcoded value
        self.joint_offsets.append(curr_gripper_position)

        self.joint_offsets = np.asarray(self.joint_offsets)

        if verbose:
            print(f"Best offsets:")
            print([f"{x:.3f}" for x in self.joint_offsets])
            print(f"Rounded as a function of pi:")
            print([f"{int(np.round(x/(np.pi/2)))}*np.pi/2" for x in self.joint_offsets])

    def get_leader_joint_states(self):
        """
        returns:
            - current leader arm joint positions
            - current leader arm joint velocities
            - current leader arm gripper position
            - current leader arm gripper velocity
        
        return values are aligned (range and direction) with those of the follower arm
        """
        self.gripper_prev_pos = self.gripper_pos
        motor_pos, motor_vel = self.driver.get_positions_and_velocities()
        arm_joint_pos = self.joint_signs[0:self.num_arm_joints]  * (motor_pos[0:self.num_arm_joints] - self.joint_offsets[0:self.num_arm_joints])
        self.gripper_pos = self.joint_signs[-1] * (motor_pos[-1] - self.joint_offsets[-1])

        arm_joint_vel = motor_vel[0:self.num_arm_joints] * self.joint_signs[0:self.num_arm_joints]
        gripper_vel = (self.gripper_pos - self.gripper_prev_pos) / self.dt

        return arm_joint_pos, arm_joint_vel, self.gripper_pos, gripper_vel

    def _match_start_pos(self):
        """
        blocks execution and waits for the leader arm to be "paired" with the follower arm before mirroring movements
        """
        arm_joint_pos, _, _, _ = self.get_leader_joint_states()
        norm_difference = np.linalg.norm(arm_joint_pos - self.arm_initial_match_pos[0:self.num_arm_joints])

        while(norm_difference > 0.3):
            arm_joint_pos, _, _, _ = self.get_leader_joint_states()
            norm_difference = np.linalg.norm(arm_joint_pos - self.arm_initial_match_pos[0:self.num_arm_joints])
            
            time.sleep(self.dt)

        self.get_logger().info(f"[TELEOP] {self.name}:  joint position matched.")
        self.matched_pub.publish(Bool(data=True))

    def shut_down(self):
        """
        terminates control loop immediately
        """
        self._shutting_down = True
        try:
            self.timer.cancel()
        except:
            pass

        try:
            self.driver.safe_disable()
        except Exception as e :
            try: self.get_logger().error(f"Torque disable failed: {e}")
            except Exception: print(f"Torque disable failed: {e}")
        
        time.sleep(0.5)

        try:
            self.driver.close()
        except:
            pass

    def _power_gello(self, enable: bool):
        """
        enables gello torque
        """
        try:
            self.driver.safe_set_torque_mode(enable)
        except Exception as e:
            self.get_logger().error(f"Gello power toggle failed: {e}")

    def get_commanded_joint_torque(self, leader_arm_pos, leader_arm_vel, commanded_pos, kp=None, kd=None):
        """
        returns the joint torque responsible for moving the leader arm to the configuration specified by commanded_pos
        """
        if commanded_pos is None:
            return
        
        kp = kp if kp is not None else self.joint_position_control_kp
        kd = kd if kd is not None else self.joint_position_control_kd

        if isinstance(kp, (int, float)):
            commanded_torque = kp * (commanded_pos - leader_arm_pos) - kd * leader_arm_vel
            return commanded_torque
        else:
            difference = np.array(commanded_pos - leader_arm_pos)
            p_term = np.multiply(kp, difference)
            d_term = np.multiply(kd, np.array(leader_arm_vel))
            
            commanded_torque = p_term - d_term
            return commanded_torque

    
    def get_joint_limit_barrier_repulsion_torque(self, arm_joint_pos, arm_joint_vel, gripper_joint_pos, gripper_joint_vel):
        """
        computes the repulsive torques to prevent the leader arm joints and gripper from exceeding the physical joint limites of the follower arm
        """
        tau_limit_arm = np.zeros(self.num_arm_joints)
        tau_limit_gripper = 0.0

        exceed_max_mask = arm_joint_pos > self.arm_joint_limits_max
        exceed_min_mask = arm_joint_pos < self.arm_joint_limits_min

        max_limit_error = arm_joint_pos - self.arm_joint_limits_max
        max_limit_repulsion = -self.joint_limit_repulsion_kp * max_limit_error -self.joint_limit_repulsion_kd * arm_joint_vel
        tau_limit_arm += exceed_max_mask * max_limit_repulsion

        min_limit_error = arm_joint_pos - self.arm_joint_limits_min
        min_limit_repulsion = -self.joint_limit_repulsion_kp * min_limit_error -self.joint_limit_repulsion_kd * arm_joint_vel
        tau_limit_arm += exceed_min_mask * min_limit_repulsion

        if gripper_joint_pos > self.gripper_command_range:
            gripper_max_error = gripper_joint_pos - self.gripper_command_range
            tau_limit_gripper = -self.joint_limit_repulsion_kp * gripper_max_error -self.joint_limit_repulsion_kd * gripper_joint_vel
        elif gripper_joint_pos < self.gripper_limit_min:
            gripper_min_error = gripper_joint_pos - self.gripper_limit_min
            tau_limit_gripper = -self.joint_limit_repulsion_kp * gripper_min_error -self.joint_limit_repulsion_kd * gripper_joint_vel
        
        return tau_limit_arm, tau_limit_gripper

    def set_leader_joint_torque(self, arm_torque, gripper_torque):
        """
        applies a specified toeque to the leader arm joints and gripper
        """
        arm_gripper_torque = np.append(arm_torque, gripper_torque)
        self.driver.set_torque(arm_gripper_torque * self.joint_signs)

    def set_leader_joint_pos(self, goal_arm_joint_pos):
        """
        moves the leader arm and gripper to a specified joint configuration using a PD control loop

        calculates multiple torque terms (commanded torque, limit barrier repulsion, etc.)

        this method only computes and sends one torque-control step towards the target joint configuration
        """
        leader_arm_joint_pos, leader_arm_joint_vel, leader_gripper_pos, leader_gripper_vel = self.get_leader_joint_states()

        arm_commanded_torque = np.zeros(self.num_arm_joints)
        joint_limit_barrier_torque, torque_gripper = self.get_joint_limit_barrier_repulsion_torque(
                                                                leader_arm_joint_pos, 
                                                                leader_arm_joint_vel, 
                                                                leader_gripper_pos, 
                                                                leader_gripper_vel)
        arm_commanded_torque += joint_limit_barrier_torque

        ### TODO: Implement later
        # arm_commanded_torque += self.null_space_regulation()
        # arm_commanded_torque += self.gravity_compensation()
        arm_commanded_torque += self.get_commanded_joint_torque(leader_arm_joint_pos, leader_arm_joint_vel, goal_arm_joint_pos)

        self.set_leader_joint_torque(arm_commanded_torque, torque_gripper)



    def go_to_joint_config(self, joint_target: np.ndarray, seconds: float = 1.0):
        """
        interpolates from currnt leader arm joint posiitons to the target position

        executes over 'seconds'

        blocks calling thread, and uses set_leader_joint_pos() each step
        """
        joint_target = np.asarray(joint_target)
        joint_target = np.clip(joint_target, self.arm_joint_limits_min, self.arm_joint_limits_max)

        current_arm_joint_pose, _, _, _ = self.get_leader_joint_states()
        num_steps = max(1, int(seconds / self.dt))

        for inter_target in np.linspace(current_arm_joint_pose, joint_target, num_steps):
            if self._shutting_down:
                break

            self.set_leader_joint_pos(inter_target)
            time.sleep(self.dt)

    def _go_to_home_pose(self):
        """
        sends the teacher arm to the home pose
        """
        self.go_to_joint_config(self.arm_initial_match_pos[0:self.num_arm_joints])

    def _set_gravity_vector(self, gello_base_orientation):
        """
        sets gravity vector for all joints given the orientation of the gello base
        """
        gravity_world = np.array([0.0, 0.0, -9.81])
        R_world_base = Rotation.from_euler("xyz", gello_base_orientation).as_matrix()
        gravity_base = R_world_base.T @ gravity_world
        self.pin_model.gravity.linear = gravity_base

    def gravity_compensation(self, arm_joint_pos, arm_joint_vel):
        """
        computes joint torque for gravity compensation using inverse dynamics

        uses the Recursive Newton-Euler Alogorithm (RNEA) provided inthe pinnochio library
        """
        self.tau_gravity = pin.rnea(self.pin_model, self.pin_data, arm_joint_pos, arm_joint_vel, np.zeros_like(arm_joint_vel))
        self.tau_gravity *= self.gravity_compensation_modifier
        return self.tau_gravity
        

    def static_friction_dither(self, arm_joint_vel):
        self.tau_static_friction_dither = np.zeros(self.num_arm_joints)
        for i in range(self.num_arm_joints):
            if abs(arm_joint_vel[i]) < self.static_friction_compensation_enable_speed:
                if self.static_friction_dither_flag[i]:
                    self.tau_static_friction_dither[i] = self.static_friction_compensation_modifier * abs(self.tau_gravity[i])
                else:
                    self.tau_static_friction_dither[i] = -self.static_friction_compensation_modifier * abs(self.tau_gravity[i])

                self.static_friction_dither_flag[i] = ~self.static_friction_dither_flag[i]
        
        return self.tau_static_friction_dither

    def null_space_regulation(self, curr_arm_joint_pos, curr_arm_joint_vel):
        """
        computes joint torques to perform null-space regulation for redundancy resolution of leader arm
        """
        self.null_space_projector = np.zeros((self.num_arm_joints, self.num_arm_joints))
        self.null_space_projector[3, 3] = 0.5
        self.null_space_projector[2, 2] = 0.5

        # J = pin.computeJointJacobian(self.pin_model, self.pin_data, curr_arm_joint_pos, self.num_arm_joints)
        # J = J[:3, :]    # keep only translational rows of the jacobian, corresponds to joint motion to end-effector linear velocity mapping
        #                 # the projector keeps the posture torques that does not translate the end effector
        # J_dagger = np.linalg.pinv(J)
        # self.null_space_projector = np.eye(self.num_arm_joints) - (J_dagger @ J)

        q_error = curr_arm_joint_pos - self.null_space_regulation_joint_targets[0:self.num_arm_joints]
        tau_n = self.null_space_projector @ (-(self.null_space_regulation_kp * q_error + self.null_space_regulation_kd* curr_arm_joint_vel))
        tau_n *= self.null_space_regulation_modifier
        return tau_n


    def control_loop_callback(self):
        """
        runs the main control loop of the leader arm
        """
        if self._shutting_down:
            return
        
        leader_arm_joint_pos, leader_arm_joint_vel, leader_gripper_pos, leader_gripper_vel = self.get_leader_joint_states()
        arm_torque_command = np.zeros(self.num_arm_joints)

        joint_limit_barrier_torque, gripper_torque = self.get_joint_limit_barrier_repulsion_torque(
                                                                leader_arm_joint_pos, 
                                                                leader_arm_joint_vel, 
                                                                leader_gripper_pos, 
                                                                leader_gripper_vel)
        arm_torque_command += joint_limit_barrier_torque

        if self.gravity_compenation_enabled:
            arm_torque_command += self.gravity_compensation(leader_arm_joint_pos, leader_arm_joint_vel)
        
        if self.null_space_regulation_enabled:
            arm_torque_command += self.null_space_regulation(leader_arm_joint_pos, leader_arm_joint_vel)

        if self.static_friction_dither_enabled:
            arm_torque_command += self.static_friction_dither(leader_arm_joint_vel)

        arm_torque_command += self.factr2_feedback.torque(leader_arm_joint_vel)

        if self.enable_gripper_feedback:
            gripper_torque += self.compute_gripper_feedback_torque(leader_gripper_pos, leader_gripper_vel)
        
        if self.step % self.skip == 0:
            self.update_communication(leader_arm_joint_pos, leader_gripper_pos)
        
        self.set_leader_joint_torque(arm_torque_command, gripper_torque)
        self.step += 1

    @abstractmethod
    def set_up_communication(self):
        """
        this method should be implemented to set up communication between the leader arm and the follower arm for mirrored teleoperation

        this method is called once in the __init__ method

        usage:
            - a subscriber can be set up to receive external joint torque from the leader arm 
            - a publisher can be set up to publish joint position target commands for the follower arm
            - publishers and subscriber can also be used to setup to record the follower arm's joint states
        """
        pass

    @abstractmethod
    def compute_gripper_feedback_torque(self):
        """
        this method processes feedback signals for the gripper and returns teh force-feedback input for the leader gripper

        this is called at every iteration of the control loop if self.enable_gripper_feedback is set to true
        """
        pass

    @abstractmethod
    def update_communication(self):
        """
        this method is intended to be called at every iteration of the control loop to transmit relevant data

        this can include:
            - joint position targets
        """
        pass
