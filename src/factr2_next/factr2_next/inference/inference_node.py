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
from std_msgs.msg import Bool, Float32

from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.inference.history_buffer import HistoryBuffer


class TorqueFilter:
    def __init__(self, cfg):
        self.mode = str(cfg.get("mode", "none")).lower()
        if not bool(cfg.get("enabled", self.mode != "none")):
            self.mode = "none"
        self.ema_alpha = float(cfg.get("ema_alpha", 0.2))
        self.cutoff_hz = float(cfg.get("cutoff_hz", 5.0))
        self.sample_hz = float(cfg.get("sample_hz", 50.0))
        self.y = None

        valid = {"none", "ema", "lowpass"}
        if self.mode not in valid:
            raise ValueError(f"Unsupported smoothing mode '{self.mode}'. Use: {sorted(valid)}")

    def update(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.mode == "none" or self.y is None:
            self.y = x.copy()
            return x

        alpha = self._alpha()
        self.y = (1.0 - alpha) * self.y + alpha * x
        return self.y.astype(np.float32)

    def _alpha(self):
        if self.mode == "ema":
            return float(np.clip(self.ema_alpha, 0.0, 1.0))
        if self.cutoff_hz <= 0.0:
            return 1.0
        dt = 1.0 / max(self.sample_hz, 1.0)
        rc = 1.0 / (2.0 * np.pi * self.cutoff_hz)
        return float(np.clip(dt / (rc + dt), 0.0, 1.0))


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
        self.torque_filter = TorqueFilter(self.cfg.get("smoothing", {}))
        self.score_cfg = self.cfg.get("score", {})
        self.contact_cfg = self.cfg.get("contact", {})
        self.contact_state = False
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
        external_topic = self._format_topic(outputs["external_joint_torque"])
        external_raw_topic = self._format_topic(
            outputs.get("external_joint_torque_raw", outputs["external_joint_torque"] + "/raw")
        )
        self.ext_pub = self.create_publisher(
            JointState, external_topic, 10
        )
        self.ext_raw_pub = self.create_publisher(
            JointState, external_raw_topic, 10
        )
        self.mse_pub = self.create_publisher(Float32, self._format_topic(outputs["mse"]), 10)
        self.score_pub = self.create_publisher(Float32, self._format_topic(outputs["score"]), 10)
        self.contact_pub = self.create_publisher(
            Bool,
            self._format_topic(outputs.get("contact_state", "/next/{arm}/contact_state")),
            10,
        )

        self.get_logger().info(f"Loaded NEXT checkpoint for arm '{self.arm}'.")
        self.get_logger().info(
            f"External torque smoothing: {self.torque_filter.mode}; "
            f"filtered={external_topic}, raw={external_raw_topic}"
        )
        self.get_logger().info(
            "Score: "
            f"{self.score_cfg.get('norm', 'l1')} norm of "
            f"{self.score_cfg.get('source', 'filtered_external_joint_torque')}"
        )
        self.get_logger().info(
            "Contact hysteresis: "
            f"low={self.contact_cfg.get('low_threshold', 1.0)}, "
            f"high={self.contact_cfg.get('high_threshold', 2.0)}"
        )

    def _callback(self, pos_msg, vel_msg, cmd_msg, torque_msg):
        joint_pos = self._extract(pos_msg, "joint_pos")
        joint_vel = self._extract(vel_msg, "joint_vel")
        joint_cmd = self._extract(cmd_msg, "joint_cmd")
        measured = self._extract(torque_msg, "measured_joint_torque")

        self.buffer.append(joint_pos, joint_vel, joint_cmd)
        if not self.buffer.ready:
            return

        tau_free = self._predict_free_torque()
        tau_ext_raw = measured - tau_free
        tau_ext = self.torque_filter.update(tau_ext_raw)
        mse = float(np.mean((tau_free - measured) ** 2))
        score = self._torque_score(tau_ext, tau_ext_raw)
        contact = self._update_contact_state(score)

        stamp = torque_msg.header.stamp
        self.free_pub.publish(self._joint_state(tau_free, stamp))
        self.ext_raw_pub.publish(self._joint_state(tau_ext_raw, stamp))
        self.ext_pub.publish(self._joint_state(tau_ext, stamp))
        self.mse_pub.publish(Float32(data=mse))
        self.score_pub.publish(Float32(data=score))
        self.contact_pub.publish(Bool(data=contact))

    def _predict_free_torque(self):
        norm = self.loaded.normalization
        x = (self.buffer.array() - norm["x_mean"]) / norm["x_std"]
        x = torch.from_numpy(x).unsqueeze(0).to(self.loaded.device)
        with torch.no_grad():
            y = self.loaded.model(x).cpu().numpy()[0]
        return (y * norm["y_std"] + norm["y_mean"]).astype(np.float32)

    def _torque_score(self, tau_ext, tau_ext_raw):
        source = str(self.score_cfg.get("source", "filtered_external_joint_torque")).lower()
        values = tau_ext_raw if source == "raw_external_joint_torque" else tau_ext
        values = np.asarray(values, dtype=np.float32)
        norm = str(self.score_cfg.get("norm", "l1")).lower()

        if norm == "l1":
            score = float(np.sum(np.abs(values)))
        elif norm == "l2":
            score = float(np.linalg.norm(values))
        elif norm in ("linf", "inf", "max"):
            score = float(np.max(np.abs(values))) if values.size else 0.0
        else:
            raise ValueError(f"Unsupported score norm '{norm}'. Use: l1, l2, linf.")

        scale = max(float(self.score_cfg.get("scale", 1.0)), 1e-6)
        return score / scale

    def _update_contact_state(self, score):
        if not bool(self.contact_cfg.get("enabled", True)):
            self.contact_state = False
            return False

        low = float(self.contact_cfg.get("low_threshold", 1.0))
        high = float(self.contact_cfg.get("high_threshold", 2.0))
        if high < low:
            high = low

        if score >= high:
            self.contact_state = True
        elif score <= low:
            self.contact_state = False
        return self.contact_state

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
