import time

import numpy as np
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32


class Factr2TorqueFeedback:
    def __init__(self, node, arm, cfg, num_joints):
        self.node = node
        self.arm = arm
        self.num_joints = int(num_joints)
        self.enabled = bool(cfg.get("enable", False))
        self.gains = self._vec(cfg.get("gains", [0.0] * self.num_joints))
        self.damping = self._vec(cfg.get("damping_gains", [0.0] * self.num_joints))
        self.max_torque = np.abs(self._vec(cfg.get("max_torque", [0.0] * self.num_joints)))
        self.ramp_alpha = float(np.clip(cfg.get("ramp_alpha", 0.35), 0.0, 1.0))
        self.stale_timeout = float(cfg.get("stale_timeout_seconds", 0.2))

        self.tau_ext = np.zeros(self.num_joints, dtype=float)
        self.contact = False
        self.gate = 0.0
        self.last_torque_time = None
        self.last_contact_time = None
        self.subs = []
        self.torque_pub = node.create_publisher(
            JointState,
            self._topic(cfg.get("feedback_torque_topic", "/factr2_feedback/{arm}/torque")),
            10,
        )
        self.gate_pub = node.create_publisher(
            Float32,
            self._topic(cfg.get("feedback_gate_topic", "/factr2_feedback/{arm}/gate")),
            10,
        )

        if not self.enabled:
            node.get_logger().info("FACTR2 feedback disabled; publishing zero diagnostics")
            return

        torque_topic = self._topic(
            cfg.get("external_joint_torque_topic", "/next/{arm}/external_joint_torque")
        )
        contact_topic = self._topic(
            cfg.get("contact_state_topic", "/next/{arm}/contact_state")
        )
        self.subs = [
            node.create_subscription(
                JointState,
                torque_topic,
                self._torque_callback,
                qos_profile_sensor_data,
            ),
            node.create_subscription(
                Bool,
                contact_topic,
                self._contact_callback,
                10,
            ),
        ]
        node.get_logger().info(
            f"FACTR2 feedback enabled: torque={torque_topic}, contact={contact_topic}"
        )

    def torque(self, leader_joint_vel):
        if not self.enabled:
            tau = np.zeros(self.num_joints, dtype=float)
            self._publish(tau)
            return tau

        target_gate = 1.0 if self.contact and self._fresh() else 0.0
        self.gate = (1.0 - self.ramp_alpha) * self.gate + self.ramp_alpha * target_gate

        tau = self.gate * self.gains * self.tau_ext
        tau -= self.damping * np.asarray(leader_joint_vel, dtype=float)[: self.num_joints]
        tau = np.clip(tau, -self.max_torque, self.max_torque)
        self._publish(tau)
        return tau

    def _torque_callback(self, msg):
        values = np.asarray(msg.position, dtype=float)
        if values.size < self.num_joints:
            self.node.get_logger().warn(
                f"FACTR2 torque len {values.size} < {self.num_joints}; ignoring"
            )
            return
        self.tau_ext = values[: self.num_joints].copy()
        self.last_torque_time = time.monotonic()

    def _contact_callback(self, msg):
        self.contact = bool(msg.data)
        self.last_contact_time = time.monotonic()

    def _fresh(self):
        now = time.monotonic()
        if self.last_torque_time is None or self.last_contact_time is None:
            return False
        return (
            now - self.last_torque_time <= self.stale_timeout
            and now - self.last_contact_time <= self.stale_timeout
        )

    def _publish(self, tau):
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.position = tau.tolist()
        self.torque_pub.publish(msg)
        self.gate_pub.publish(Float32(data=float(self.gate)))

    def _topic(self, topic):
        return str(topic).format(arm=self.arm)

    def _vec(self, values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            arr = np.zeros(self.num_joints, dtype=float)
        if arr.size < self.num_joints:
            arr = np.pad(arr, (0, self.num_joints - arr.size), mode="edge")
        return arr[: self.num_joints]
