from pathlib import Path

import numpy as np
import rclpy
import torch
import yaml
from ament_index_python.packages import get_package_share_directory
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.inference.history_buffer import HistoryBuffer


class InferenceNode(Node):
    def __init__(self):
        super().__init__("factr2_next_inference")
        default_config = (
            Path(get_package_share_directory("factr2_next")) / "config" / "inference.yaml"
        )
        config_file = self.declare_parameter("config_file", str(default_config)).value
        self.cfg = self._load_config(config_file)
        self.arm = str(self.cfg.get("arm", "robot"))

        self.loaded = load_checkpoint(
            self._format_path(self.cfg["checkpoint_dir"]),
            device=self.cfg.get("device", "cpu"),
        )
        self.buffer = HistoryBuffer(self.loaded.history)
        self.topic_cfg = self.cfg["topics"]
        self.keys = ["joint_pos", "joint_vel", "joint_cmd", "measured_joint_torque"]

        self.subscribers = [
            Subscriber(
                self,
                JointState,
                self._format_topic(self.topic_cfg[key]["topic"]),
                qos_profile=qos_profile_sensor_data,
            )
            for key in self.keys
        ]
        sync_cfg = self.cfg.get("sync", {})
        self.sync = ApproximateTimeSynchronizer(
            self.subscribers,
            queue_size=int(sync_cfg.get("queue_size", 20)),
            slop=float(sync_cfg.get("slop_seconds", 0.03)),
        )
        self.sync.registerCallback(self._callback)

        outputs = self.cfg["outputs"]
        self.free_pub = self.create_publisher(
            JointState, self._format_topic(outputs["free_joint_torque_pred"]), 10
        )
        self.ext_pub = self.create_publisher(
            JointState, self._format_topic(outputs["external_joint_torque"]), 10
        )
        self.mse_pub = self.create_publisher(Float32, self._format_topic(outputs["mse"]), 10)
        self.score_pub = self.create_publisher(Float32, self._format_topic(outputs["score"]), 10)

        self.get_logger().info(f"Loaded NEXT checkpoint for arm '{self.arm}'.")

    def _callback(self, pos_msg, vel_msg, cmd_msg, torque_msg):
        joint_pos = self._extract(pos_msg, "joint_pos")
        joint_vel = self._extract(vel_msg, "joint_vel")
        joint_cmd = self._extract(cmd_msg, "joint_cmd")
        measured = self._extract(torque_msg, "measured_joint_torque")

        self.buffer.append(joint_pos, joint_vel, joint_cmd)
        if not self.buffer.ready:
            return

        tau_free = self._predict_free_torque()
        tau_ext = measured - tau_free
        mse = float(np.mean((tau_free - measured) ** 2))
        score = mse  # First public slice: score is the same scalar error as MSE.

        stamp = torque_msg.header.stamp
        self.free_pub.publish(self._joint_state(tau_free, stamp))
        self.ext_pub.publish(self._joint_state(tau_ext, stamp))
        self.mse_pub.publish(Float32(data=mse))
        self.score_pub.publish(Float32(data=score))

    def _predict_free_torque(self):
        norm = self.loaded.normalization
        x = (self.buffer.array() - norm["x_mean"]) / norm["x_std"]
        x = torch.from_numpy(x).unsqueeze(0).to(self.loaded.device)
        with torch.no_grad():
            y = self.loaded.model(x).cpu().numpy()[0]
        return (y * norm["y_std"] + norm["y_mean"]).astype(np.float32)

    def _extract(self, msg, key):
        field = self.topic_cfg[key].get("field", "position")
        if field == "position":
            return np.asarray(msg.position, dtype=np.float32)
        if field == "velocity":
            return np.asarray(msg.velocity, dtype=np.float32)
        if field == "effort":
            return np.asarray(msg.effort, dtype=np.float32)
        raise ValueError(f"Unsupported JointState field: {field}")

    def _joint_state(self, values, stamp):
        msg = JointState()
        msg.header.stamp = stamp
        msg.position = [float(x) for x in values]
        return msg

    def _format_topic(self, topic):
        return str(topic).format(arm=self.arm)

    def _format_path(self, path):
        return Path(str(path).format(arm=self.arm)).expanduser()

    def _load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
