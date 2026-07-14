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


INPUT_KEYS = ("joint_pos", "joint_vel", "joint_cmd", "measured_joint_torque")


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
            raise ValueError(
                f"Unsupported smoothing mode '{self.mode}'. Use: {sorted(valid)}"
            )

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
        self.robot_topic_root = str(self.cfg.get("robot_topic_root", "/robot"))
        self.next_topic_root = str(self.cfg.get("next_topic_root", "/next"))
        self._check_sign_convention()

        self.loaded = load_checkpoint(
            Path(str(self.cfg["checkpoint_dir"])).expanduser(),
            device=self.cfg.get("device", "cpu"),
        )
        self.buffer = HistoryBuffer(self.loaded.history)
        self.torque_filter = TorqueFilter(self.cfg.get("smoothing", {}))
        self.score_cfg = self.cfg.get("score", {})
        self.contact_cfg = self.cfg.get("contact", {})
        self.contact_state = False
        self.topic_cfg = self.cfg["topics"]

        self.subscribers = [
            Subscriber(
                self,
                JointState,
                self._format_topic(self.topic_cfg[key]["topic"]),
                qos_profile=qos_profile_sensor_data,
            )
            for key in INPUT_KEYS
        ]
        sync_cfg = self.cfg.get("sync", {})
        self.sync = ApproximateTimeSynchronizer(
            self.subscribers,
            queue_size=int(sync_cfg.get("queue_size", 20)),
            slop=float(sync_cfg.get("slop_seconds", 0.03)),
        )
        self.sync.registerCallback(self._callback)

        self.outputs = self._output_topics(self.cfg["outputs"])
        self.pubs = {
            "free": self._pub(JointState, "free_joint_torque_pred"),
            "external": self._pub(JointState, "external_joint_torque"),
            "external_raw": self._pub(JointState, "external_joint_torque_raw"),
            "mse": self._pub(Float32, "mse"),
            "score": self._pub(Float32, "score"),
            "contact": self._pub(Bool, "contact_state"),
        }

        self.get_logger().info(
            f"Loaded NEXT checkpoint: robot={self.robot_topic_root}, next={self.next_topic_root}"
        )
        self.get_logger().info(
            f"External torque smoothing: {self.torque_filter.mode}; "
            f"filtered={self.outputs['external_joint_torque']}, "
            f"raw={self.outputs['external_joint_torque_raw']}"
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

    def _callback(self, *msgs):
        joint_pos, joint_vel, joint_cmd, measured = [
            self._extract(msg, key) for msg, key in zip(msgs, INPUT_KEYS)
        ]

        self.buffer.append(joint_pos, joint_vel, joint_cmd)
        # Paper Sec. 4, Eq. (3): f_theta consumes a full history window.
        if not self.buffer.ready:
            return

        tau_free = self._predict_free_torque()
        # Paper Sec. 4, Eq. (2): tau_ext_hat = tau_m - tau_free_hat.
        tau_ext_raw = measured - tau_free
        # Smoothing, scoring, and hysteresis are runtime/demo post-processing,
        # not part of the learned NEXT free-space torque model.
        tau_ext = self.torque_filter.update(tau_ext_raw)
        mse = float(np.mean((tau_free - measured) ** 2))
        score = self._torque_score(tau_ext, tau_ext_raw)
        contact = self._update_contact_state(score)

        stamp = msgs[-1].header.stamp
        self.pubs["free"].publish(self._joint_state(tau_free, stamp))
        self.pubs["external_raw"].publish(self._joint_state(tau_ext_raw, stamp))
        self.pubs["external"].publish(self._joint_state(tau_ext, stamp))
        self.pubs["mse"].publish(Float32(data=mse))
        self.pubs["score"].publish(Float32(data=score))
        self.pubs["contact"].publish(Bool(data=contact))

    def _predict_free_torque(self):
        norm = self.loaded.normalization
        x = (self.buffer.array() - norm["x_mean"]) / norm["x_std"]
        x = torch.from_numpy(x).unsqueeze(0).to(self.loaded.device)
        with torch.inference_mode():
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
        return str(topic).format(
            robot_topic_root=self.robot_topic_root,
            next_topic_root=self.next_topic_root,
        )

    def _pub(self, msg_type, key):
        return self.create_publisher(msg_type, self.outputs[key], 10)

    def _output_topics(self, outputs):
        outputs = dict(outputs)
        outputs.setdefault(
            "external_joint_torque_raw",
            outputs["external_joint_torque"] + "/raw",
        )
        outputs.setdefault("contact_state", "{next_topic_root}/contact_state")
        return {key: self._format_topic(topic) for key, topic in outputs.items()}

    def _check_sign_convention(self):
        sign = self.cfg.get("next", {}).get(
            "sign_convention",
            "measured_minus_predicted",
        )
        if sign != "measured_minus_predicted":
            raise ValueError("Only sign_convention=measured_minus_predicted is supported.")

    def _load_config(self, path):
        with open(Path(path).expanduser(), "r") as f:
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
