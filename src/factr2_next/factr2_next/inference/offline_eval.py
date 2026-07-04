import argparse

import h5py
import numpy as np
import torch

from factr2_next.inference.checkpoint import load_checkpoint
from factr2_next.inference.history_buffer import HistoryBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--arm", required=True, choices=["left", "right"])
    parser.add_argument("--episode", default="ep_0000")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    loaded = load_checkpoint(args.run_dir, args.device)
    arrays = read_episode(args.h5_path, args.episode, args.arm, args.max_steps)
    pred, measured = predict_sequence(loaded, arrays)
    residual = measured - pred
    err = pred - measured

    print(f"predictions: {len(pred)}")
    print(f"mean_mse: {np.mean(err ** 2):.6f}")
    print_array("per_joint_rmse", np.sqrt(np.mean(err ** 2, axis=0)))
    print_pair("measured", measured)
    print_pair("predicted", pred)
    print_pair("residual", residual)


def read_episode(h5_path, episode, arm, max_steps):
    names = {
        "joint_pos": f"{arm}_joint_pos",
        "joint_vel": f"{arm}_joint_vel",
        "joint_cmd": f"{arm}_joint_cmd",
        "measured_joint_torque": f"{arm}_measured_joint_torque",
    }
    with h5py.File(h5_path, "r") as h5:
        ep = h5[episode]
        out = {key: np.asarray(ep[name]["data"], dtype=np.float32) for key, name in names.items()}
    n = min(len(v) for v in out.values())
    if max_steps is not None:
        n = min(n, max_steps)
    return {key: value[:n] for key, value in out.items()}


def predict_sequence(loaded, arrays):
    buf = HistoryBuffer(loaded.history)
    pred, measured = [], []
    norm = loaded.normalization

    with torch.no_grad():
        for pos, vel, cmd, torque in zip(
            arrays["joint_pos"],
            arrays["joint_vel"],
            arrays["joint_cmd"],
            arrays["measured_joint_torque"],
        ):
            buf.append(pos, vel, cmd)
            if not buf.ready:
                continue

            x = (buf.array() - norm["x_mean"]) / norm["x_std"]
            x = torch.from_numpy(x).unsqueeze(0).to(loaded.device)
            y = loaded.model(x).cpu().numpy()[0]
            y = y * norm["y_std"] + norm["y_mean"]
            pred.append(y.astype(np.float32))
            measured.append(torque.astype(np.float32))

    if not pred:
        raise ValueError("No predictions produced. Increase max-steps or check history length.")
    return np.stack(pred), np.stack(measured)


def print_pair(name, values):
    print_array(f"{name}_mean", values.mean(axis=0))
    print_array(f"{name}_std", values.std(axis=0))


def print_array(name, values):
    text = " ".join(f"{x:.4f}" for x in values)
    print(f"{name}: [{text}]")


if __name__ == "__main__":
    main()
