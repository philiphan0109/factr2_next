import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def ema(values, alpha):
    result = np.asarray(values, dtype=np.float32).copy()
    for i in range(1, len(result)):
        result[i] = (1.0 - alpha) * result[i - 1] + alpha * result[i]
    return result


def _read(ep, key):
    return np.asarray(ep[key]["data"])


def plot_episode(h5_path, episode="ep_0000", output="figures/eval_episode.png",
                 ema_alpha=None, window=None, ylim=None):
    with h5py.File(h5_path, "r") as h5:
        ep = h5[episode]
        stamps = np.asarray(ep["joint_pos"]["timestamps"])
        t = (stamps - stamps[0]) * 1e-9
        raw = _read(ep, "external_joint_torque_raw")
        filtered = _read(ep, "external_joint_torque_filtered")

    mask = np.ones(len(t), dtype=bool)
    if window is not None:
        mask &= (t >= window[0]) & (t <= window[1])
    t = t[mask]
    torque = ema(raw, float(ema_alpha)) if ema_alpha is not None else filtered
    signed_sum = torque[mask].sum(axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    ax.plot(t, signed_sum, color="tab:blue", lw=1.4)
    ax.axhline(0.0, color="black", lw=0.7, alpha=0.45)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signed Σ external torque (N·m)")
    ax.set_title(f"External torque — {episode}")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.2)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot a FACTR2 evaluation episode.")
    parser.add_argument("h5_path")
    parser.add_argument("--episode", default="ep_0000")
    parser.add_argument("--output", default="figures/eval_episode.png")
    parser.add_argument("--ema-alpha", type=float)
    parser.add_argument("--window", type=float, nargs=2, metavar=("START", "END"))
    parser.add_argument("--ylim", type=float, nargs=2, metavar=("MIN", "MAX"))
    args = parser.parse_args(argv)
    path = plot_episode(args.h5_path, args.episode, args.output, args.ema_alpha,
                        args.window, args.ylim)
    print(path)


if __name__ == "__main__":
    main()
