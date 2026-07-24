import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

try:
    import rclpy
    from ament_index_python.packages import get_package_share_directory
    from message_filters import ApproximateTimeSynchronizer, Subscriber
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool, Float32
except ImportError:  # Keep row assembly importable for offline tests/tools.
    rclpy = None
    Node = object

from factr2_next.data_collection.h5_writer import H5Writer


RAW_KEYS = ("joint_pos", "joint_vel", "joint_cmd", "measured_joint_torque")
VECTOR_KEYS = (
    "free_joint_torque_pred", "external_joint_torque_raw",
    "external_joint_torque_filtered", "feedback_torque",
)
SCALAR_KEYS = ("score", "contact_state", "feedback_gate")
ALL_KEYS = RAW_KEYS + VECTOR_KEYS + SCALAR_KEYS
EXPECTED = {key: (8 if key in RAW_KEYS[:3] else 7) for key in RAW_KEYS + VECTOR_KEYS}


class FreshSamples:
    """Small timestamped cache used to assemble one complete evaluation row."""

    def __init__(self, timeout):
        self.timeout = float(timeout)
        self.values = {}

    def update(self, key, value, received=None):
        self.values[key] = (np.asarray(value, dtype=np.float32),
                            time.monotonic() if received is None else float(received))

    def assemble(self, raw, now=None):
        now = time.monotonic() if now is None else float(now)
        missing = [key for key in VECTOR_KEYS + SCALAR_KEYS
                   if key not in self.values or now - self.values[key][1] > self.timeout]
        if missing:
            return None, missing
        row = {key: np.asarray(raw[key], dtype=np.float32) for key in RAW_KEYS}
        row.update({key: self.values[key][0] for key in VECTOR_KEYS + SCALAR_KEYS})
        bad = [key for key, size in EXPECTED.items()
               if key in row and row[key].size != size]
        return (None, bad) if bad else (row, [])


class EvaluationRecorder(Node):
    def __init__(self):
        super().__init__("factr2_next_evaluation_recorder")
        default = Path(get_package_share_directory("factr2_next")) / "config" / "eval_glorbot2.yaml"
        config_path = self.declare_parameter("config_file", str(default)).value
        with open(Path(config_path).expanduser(), "r") as stream:
            self.cfg = yaml.safe_load(stream) or {}
        self.config_path = str(config_path)
        self.arm = str(self.cfg.get("arm", "right"))
        if self.arm not in ("left", "right"):
            raise ValueError("arm must be 'left' or 'right'")
        self.robot = str(self.cfg.get("robot_name", "glorbot2_1"))
        self.topics = {key: str(value).format(robot_name=self.robot, arm=self.arm)
                       for key, value in self.cfg["topics"].items()}
        rec = self.cfg.get("recording", {})
        self.target_hz = float(rec.get("target_hz", 60.0))
        self.min_hz = float(rec.get("min_hz", 50.0))
        self.warn_after = float(rec.get("warn_after_seconds", 2.0))
        self.cache = FreshSamples(rec.get("aux_timeout_seconds", 0.1))
        self.writer = None
        self.recording = False
        self.lock = threading.Lock()
        self.last_warn = 0.0
        self.rate_start = time.monotonic()
        self.rate_rows = 0

        raw_subs = [Subscriber(self, JointState, self.topics[key],
                               qos_profile=qos_profile_sensor_data) for key in RAW_KEYS]
        sync = self.cfg.get("sync", {})
        self.sync = ApproximateTimeSynchronizer(raw_subs, int(sync.get("queue_size", 20)),
                                                float(sync.get("slop_seconds", 0.03)))
        self.sync.registerCallback(self._raw_callback)
        self.aux_subs = []
        for key in VECTOR_KEYS:
            self.aux_subs.append(self.create_subscription(
                JointState, self.topics[key], lambda msg, k=key: self._vector(k, msg),
                qos_profile_sensor_data))
        for key, msg_type in (("score", Float32), ("contact_state", Bool),
                              ("feedback_gate", Float32)):
            self.aux_subs.append(self.create_subscription(
                msg_type, self.topics[key], lambda msg, k=key: self._scalar(k, msg), 10))
        self._stop = threading.Event()
        self._tty = None
        self.keyboard = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.keyboard.start()
        self.get_logger().info("Press 'r' to start/stop an evaluation episode.")

    def _vector(self, key, msg):
        self.cache.update(key, msg.position)

    def _scalar(self, key, msg):
        self.cache.update(key, [float(msg.data)])

    def _raw_callback(self, *msgs):
        raw = {}
        for key, msg in zip(RAW_KEYS, msgs):
            field = "position"
            raw[key] = np.asarray(getattr(msg, field), dtype=np.float32)
        now = time.monotonic()
        with self.lock:
            if not self.recording or self.writer is None:
                return
            row, unavailable = self.cache.assemble(raw, now)
            if row is None:
                self._warn("Skipping row; missing, stale, or incorrectly shaped: " +
                           ", ".join(unavailable))
                return
            stamp = min(m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
                        for m in msgs)
            self.writer.append(stamp, row)
            self.rate_rows += 1
            self._check_rate(now)

    def _toggle(self):
        with self.lock:
            if self.recording:
                self.recording = False
                self.writer.flush()
                self.get_logger().info("Stopped recording; data flushed.")
                return
            missing = [topic for topic in self.topics.values()
                       if self.count_publishers(topic) == 0]
            if missing:
                self.get_logger().warn("No publishers yet for: " + ", ".join(missing))
            if self.writer is None:
                output = Path(self.cfg.get("output_dir", "data/eval")).expanduser()
                name = str(self.cfg.get("session_name", "contact_pushes"))
                path = output / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.h5"
                meta = self._metadata()
                self.writer = H5Writer(path, name, ALL_KEYS, metadata=meta)
                self.get_logger().info(f"Writing: {path}")
            episode = self.writer.start_episode()
            self.recording = True
            self.rate_start = time.monotonic()
            self.rate_rows = 0
            self.get_logger().info(f"Started recording {episode}.")

    def _metadata(self):
        plot = self.cfg.get("plot", {})
        return {"arm": self.arm, "robot_name": self.robot,
                "target_hz": self.target_hz, "config_path": self.config_path,
                "topics_yaml": yaml.safe_dump(self.topics),
                "checkpoint_path": str(self.cfg.get("checkpoint_path", "")),
                "online_ema_alpha": float(plot.get("online_ema_alpha", 0.2)),
                "contact_low_threshold": float(plot.get("contact_low_threshold", 1.0)),
                "contact_high_threshold": float(plot.get("contact_high_threshold", 1.5))}

    def _check_rate(self, now):
        dt = now - self.rate_start
        if dt >= self.warn_after and dt >= 1.0:
            hz = self.rate_rows / dt
            if hz < self.min_hz:
                self._warn(f"Recording rate low: {hz:.1f} Hz < {self.min_hz:.1f} Hz")
            self.rate_start, self.rate_rows = now, 0

    def _warn(self, text):
        now = time.monotonic()
        if now - self.last_warn >= 1.0:
            self.get_logger().warn(text)
            self.last_warn = now

    def _keyboard_loop(self):
        if not sys.stdin.isatty():
            self.get_logger().warn("Keyboard toggle disabled: stdin is not a TTY.")
            return
        self._tty = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not self._stop.is_set() and rclpy.ok():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready and sys.stdin.read(1).lower() == "r":
                    self._toggle()
        finally:
            self._restore_terminal()

    def _restore_terminal(self):
        if sys.stdin.isatty() and self._tty is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._tty)

    def close(self):
        self._stop.set()
        self.keyboard.join(timeout=1.0)
        self._restore_terminal()
        with self.lock:
            if self.writer is not None:
                path = self.writer.path
                self.writer.close()
                self.writer = None
                self.get_logger().info(f"Saved evaluation data to {path}")


def main(args=None):
    if rclpy is None:
        raise RuntimeError("ROS 2 Python packages are required for next_eval_record")
    rclpy.init(args=args)
    node = EvaluationRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
