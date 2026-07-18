from collections import deque

import numpy as np


class HistoryBuffer:
    def __init__(self, history):
        self.history = int(history)
        self.rows = deque(maxlen=self.history)

    @property
    def ready(self):
        return len(self.rows) == self.history

    def append(self, joint_pos, joint_vel, joint_cmd):
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        joint_vel = np.asarray(joint_vel, dtype=np.float32)
        joint_cmd = np.asarray(joint_cmd, dtype=np.float32)
        if joint_pos.shape != joint_vel.shape or joint_pos.shape != joint_cmd.shape:
            raise ValueError(
                "NEXT position, velocity, and command shapes must match: "
                f"{joint_pos.shape}, {joint_vel.shape}, {joint_cmd.shape}"
            )
        # Match the NEXT training input exactly: [q, qdot, q_cmd - q].
        self.rows.append(np.concatenate([joint_pos, joint_vel, joint_cmd - joint_pos]))

    def array(self):
        if not self.ready:
            raise ValueError(f"HistoryBuffer needs {self.history} rows, has {len(self.rows)}.")
        return np.stack(self.rows).astype(np.float32)
