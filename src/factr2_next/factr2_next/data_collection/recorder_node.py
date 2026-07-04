import select
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from termcolor import colored

from factr2_next.data_collection.h5_writer import H5Writer


class RecorderNode(Node):
    def __init__(self):
        super().__init__("factr2_next_recorder")
        default_config = (
            Path(get_package_share_directory("factr2_next")) / "config" / "record.yaml"
        )
        config_file = self.declare_parameter("config_file", str(default_config)).value
        self.cfg = self._load_config(config_file)

        self.output_dir = Path(self.cfg.get("output_dir", "data")).expanduser()
        self.session_name = str(self.cfg.get("session_name", "free_motion"))
        self.robot_topic_root = str(self.cfg.get("robot_topic_root", "/robot"))
        self.topic_cfg = self.cfg.get("topics", {})
        if not self.topic_cfg:
            raise ValueError("record.yaml must define at least one topic.")

        self.keys = list(self.topic_cfg.keys())
        self.topics = [
            self._format_topic(spec["topic"]) for spec in self.topic_cfg.values()
        ]
        self.writer = None
        self.recording = False
        self.episode_count = 0
        self.latest_samples = None
        self.latest_timestamp_ns = None
        self.latest_sync_time = None
        self.last_written_timestamp_ns = None
        self.record_start_time = None
        self.last_rate_check_time = None
        self.rows_written_since_check = 0
        self.last_warn_time = 0.0
        self.lock = threading.Lock()

        rec_cfg = self.cfg.get("recording", {})
        self.target_hz = float(rec_cfg.get("target_hz", 50.0))
        self.min_hz = float(rec_cfg.get("min_hz", 0.9 * self.target_hz))
        self.sample_timeout = float(rec_cfg.get("sample_timeout_seconds", 0.2))
        self.warn_after = float(rec_cfg.get("warn_after_seconds", 2.0))

        self.subscribers = [
            Subscriber(
                self,
                JointState,
                topic,
                qos_profile=qos_profile_sensor_data,
            )
            for topic in self.topics
        ]

        sync_cfg = self.cfg.get("sync", {})
        self.sync = ApproximateTimeSynchronizer(
            self.subscribers,
            queue_size=int(sync_cfg.get("queue_size", 20)),
            slop=float(sync_cfg.get("slop_seconds", 0.03)),
        )
        self.sync.registerCallback(self._sync_callback)
        self.record_timer = self.create_timer(
            1.0 / max(self.target_hz, 1e-6),
            self._record_timer_callback,
        )

        self._stop_keyboard = threading.Event()
        self._tty_settings = None
        self.keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self.keyboard_thread.start()

        self.get_logger().info("Press 'r' to start/stop recording.")

    def _load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _sync_callback(self, *msgs):
        with self.lock:
            samples = {}
            for key, msg in zip(self.keys, msgs):
                field = self.topic_cfg[key].get("field", "position")
                samples[key] = self._extract_field(msg, field)

            self.latest_samples = samples
            self.latest_timestamp_ns = min(self._stamp_ns(msg) for msg in msgs)
            self.latest_sync_time = time.monotonic()

    def _record_timer_callback(self):
        with self.lock:
            if not self.recording or self.writer is None:
                return

            now = time.monotonic()
            elapsed = now - self.record_start_time
            self._check_rate(now)
            if self.latest_samples is None:
                if elapsed >= self.warn_after:
                    self._warn("Recording active, but no synchronized samples received.")
                return

            if now - self.latest_sync_time > self.sample_timeout:
                self._warn("Latest synchronized sample is stale; not writing rows.")
                return

            if self.latest_timestamp_ns == self.last_written_timestamp_ns:
                return

            self.writer.append(self.latest_timestamp_ns, self.latest_samples)
            self.last_written_timestamp_ns = self.latest_timestamp_ns
            self.rows_written_since_check += 1

    def _extract_field(self, msg, field):
        if field == "position":
            return np.asarray(msg.position, dtype=np.float32)
        if field == "velocity":
            return np.asarray(msg.velocity, dtype=np.float32)
        if field == "effort":
            return np.asarray(msg.effort, dtype=np.float32)
        raise ValueError(f"Unsupported JointState field: {field}")

    def _stamp_ns(self, msg):
        return int(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)

    def _toggle_recording(self):
        with self.lock:
            if self.recording:
                self.recording = False
                if self.writer is not None:
                    self.writer.flush()
                self.get_logger().info("Stopped recording.")
                return

            missing = self._missing_publishers()
            if missing:
                self.get_logger().warn(
                    "No publishers yet for: " + ", ".join(missing)
                )

            if self.writer is None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = self.output_dir / f"{self.session_name}_{stamp}.h5"
                self.writer = H5Writer(path, self.session_name, self.keys)
                self.get_logger().info(f"Writing: {path}")

            episode = self.writer.start_episode()
            self.recording = True
            self.episode_count += 1
            now = time.monotonic()
            self.record_start_time = now
            self.last_rate_check_time = now
            self.rows_written_since_check = 0
            self.last_written_timestamp_ns = None
            self.get_logger().info(f"Started recording {episode}.")

    def _missing_publishers(self):
        missing = []
        for topic in self.topics:
            if self.count_publishers(topic) == 0:
                missing.append(topic)
        return missing

    def _format_topic(self, topic):
        return str(topic).format(robot_topic_root=self.robot_topic_root)

    def _check_rate(self, now):
        if now - self.record_start_time < self.warn_after:
            return
        dt = now - self.last_rate_check_time
        if dt < 1.0:
            return
        hz = self.rows_written_since_check / dt
        if hz < self.min_hz:
            self.get_logger().warn(
                f"Recording rate low: {hz:.1f} Hz < {self.min_hz:.1f} Hz"
            )
        self.last_rate_check_time = now
        self.rows_written_since_check = 0

    def _warn(self, message):
        now = time.monotonic()
        if now - self.last_warn_time >= 1.0:
            self.get_logger().warn(message)
            self.last_warn_time = now

    def _keyboard_loop(self):
        if not sys.stdin.isatty():
            self.get_logger().warn("Keyboard toggle disabled: stdin is not a TTY.")
            return

        self._tty_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not self._stop_keyboard.is_set() and rclpy.ok():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready and sys.stdin.read(1).lower() == "r":
                    self._toggle_recording()
        finally:
            self._restore_terminal()

    def close(self):
        self._stop_keyboard.set()
        if self.keyboard_thread.is_alive():
            self.keyboard_thread.join(timeout=1.0)
        self._restore_terminal()
        with self.lock:
            was_recording = self.recording
            self.recording = False
            if self.writer is not None:
                path = self.writer.path
                self.writer.close()
                self.writer = None
                if was_recording:
                    print(
                        colored(
                            f"Stopped during active episode; saved partial data to {path}",
                            "yellow",
                        )
                    )
                else:
                    print(colored(f"Saved {self.episode_count} episode(s) to {path}", "green"))

    def _restore_terminal(self):
        if sys.stdin.isatty() and self._tty_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._tty_settings)
            print("\033[?25h", end="", flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
