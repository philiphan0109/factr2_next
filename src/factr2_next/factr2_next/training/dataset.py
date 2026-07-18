from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class NextTorqueDataset(Dataset):
    def __init__(self, h5_paths, key_templates, history, arm=None, episodes="all"):
        self.history = int(history)
        if self.history < 1:
            raise ValueError("history must be >= 1.")
        self.arm = arm
        self.keys = {
            name: template.format(arm=arm) if arm is not None else template
            for name, template in key_templates.items()
        }
        self.x, self.y = self._load(h5_paths, episodes)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])

    @property
    def input_size(self):
        return self.x.shape[-1]

    @property
    def output_size(self):
        return self.y.shape[-1]

    def _load(self, h5_paths, episodes):
        xs, ys = [], []
        for path in h5_paths:
            with h5py.File(Path(path).expanduser(), "r") as h5:
                for ep in self._episode_names(h5, episodes):
                    x_step, y_step = self._episode_arrays(h5[ep])
                    if len(x_step) < self.history:
                        continue
                    xs.append(self._windows(x_step))
                    # Paper Sec. 4, Eq. (3): each history window predicts tau_m
                    # at its final timestep, not at a future timestep.
                    ys.append(y_step[self.history - 1 :])

        if not xs:
            raise ValueError(
                "No training windows found. Check H5 paths, episodes, keys, and history."
            )
        return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)

    def _episode_arrays(self, episode):
        pos = self._read(episode, "joint_pos")
        vel = self._read(episode, "joint_vel")
        cmd = self._read(episode, "joint_cmd")
        torque = self._read(episode, "measured_joint_torque")
        widths = {
            "position": pos.shape[1],
            "velocity": vel.shape[1],
            "command": cmd.shape[1],
        }
        if len(set(widths.values())) != 1:
            raise ValueError(f"NEXT input widths must match: {widths}")
        n = min(len(pos), len(vel), len(cmd), len(torque))
        # Paper Sec. 4, Eq. (3): x_i = [q, qdot, q_cmd - q] over history.
        x_step = np.concatenate([pos[:n], vel[:n], cmd[:n] - pos[:n]], axis=1)
        # Paper Sec. 4: in D_free, tau_m supervises tau_free_hat.
        return x_step, torque[:n]

    def _read(self, episode, name):
        key = self.keys[name]
        if key not in episode:
            raise KeyError(f"Missing H5 key '{key}' for dataset field '{name}'.")
        return np.asarray(episode[key]["data"], dtype=np.float32)

    def _windows(self, x_step):
        return np.stack(
            [x_step[i : i + self.history] for i in range(len(x_step) - self.history + 1)]
        )

    def _episode_names(self, h5, episodes):
        if episodes == "all":
            return sorted(h5.keys())
        return list(episodes)
