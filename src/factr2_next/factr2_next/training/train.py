import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from factr2_next.training.dataset import NextTorqueDataset
from factr2_next.training.models import build_model


def main():
    parser = argparse.ArgumentParser()
    default_config = Path(__file__).resolve().parents[1] / "config" / "train.yaml"
    parser.add_argument("--config", default=str(default_config))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["train"].get("seed", 0)))
    for arm in resolve_arms(cfg["data"]):
        train_arm(cfg, arm)


def train_arm(cfg, arm):
    label = arm if arm is not None else "single"
    data_cfg = cfg["data"]
    train_dataset, val_dataset = make_datasets(data_cfg, arm)
    train_x, train_y = dataset_arrays(train_dataset)
    # Fit normalization on training data only, then reuse it for validation/inference.
    norm = fit_normalization(train_x, train_y)
    apply_normalization_pair(train_dataset, val_dataset, norm)

    device = torch.device(cfg["train"].get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model_cfg = normalized_model_cfg(cfg["model"])
    base_dataset = unwrap_dataset(train_dataset)
    model = build_model(
        model_cfg,
        base_dataset.input_size,
        base_dataset.output_size,
        base_dataset.history,
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"].get("learning_rate", 1e-3)),
    )
    batch_size = int(cfg["train"].get("batch_size", 2048))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    metrics = {"arm": label, "train_loss": [], "val_loss": []}
    epochs = int(cfg["train"].get("epochs", 20))
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, device, opt)
        val_loss = run_epoch(model, val_loader, device)
        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        print(f"{label}: epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")

    out_dir = make_run_dir(cfg["save"], arm)
    save_run(out_dir, model, cfg, norm, metrics, base_dataset, model_cfg)
    print(f"{label}: saved {out_dir}")


def make_datasets(data_cfg, arm):
    keys = data_cfg["keys"]
    history = data_cfg["history"]
    train_paths = data_cfg.get("train_h5_paths", data_cfg.get("h5_paths", []))
    val_paths = data_cfg.get("val_h5_paths", [])
    train_episodes = data_cfg.get("train_episodes", data_cfg.get("episodes", "all"))
    val_episodes = data_cfg.get("val_episodes", "all")

    if val_paths:
        train_dataset = NextTorqueDataset(
            train_paths,
            keys,
            history,
            arm=arm,
            episodes=train_episodes,
        )
        val_dataset = NextTorqueDataset(
            val_paths,
            keys,
            history,
            arm=arm,
            episodes=val_episodes,
        )
        return train_dataset, val_dataset

    dataset = NextTorqueDataset(
        train_paths,
        keys,
        history,
        arm=arm,
        episodes=train_episodes,
    )
    train_idx, val_idx = split_indices(
        len(dataset),
        data_cfg.get("val_fraction", 0.1),
        data_cfg.get("val_buffer_windows", history),
        data_cfg.get("val_split", "contiguous"),
    )
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Validation split produced an empty train or validation set.")
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def run_epoch(model, loader, device, opt=None):
    model.train(opt is not None)
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        # Paper Sec. 4, Eq. (4): L2 regression trains f_theta(x_i) -> tau_m,i.
        loss = F.mse_loss(pred, y)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        total += loss.item() * len(x)
        n += len(x)
    return total / max(n, 1)


def resolve_arms(data_cfg):
    mode = data_cfg.get("arm_mode", "single")
    if mode == "both":
        # NEXT is robot/arm-specific; bimanual data trains one model per arm.
        return list(data_cfg.get("arms", ["left", "right"]))
    if mode in ("left", "right"):
        return [mode]
    if mode in ("single", "none"):
        return [None]
    raise ValueError(
        f"Unsupported arm_mode '{mode}'. Use one of: both, left, right, single, none."
    )


def split_indices(n, val_fraction, buffer_windows=0, mode="contiguous"):
    if mode == "random":
        indices = np.random.permutation(n)
        val_n = max(1, int(n * float(val_fraction)))
        return indices[val_n:], indices[:val_n]

    val_n = max(1, int(n * float(val_fraction)))
    val_start = n - val_n
    train_end = max(0, val_start - int(buffer_windows))
    return np.arange(train_end), np.arange(val_start, n)


def fit_normalization(x, y):
    return {
        "x_mean": x.mean(axis=(0, 1)),
        "x_std": x.std(axis=(0, 1)) + 1e-6,
        "y_mean": y.mean(axis=0),
        "y_std": y.std(axis=0) + 1e-6,
    }


def apply_normalization(dataset, norm):
    if isinstance(dataset, Subset):
        apply_normalization(dataset.dataset, norm)
        return
    dataset.x = (dataset.x - norm["x_mean"]) / norm["x_std"]
    dataset.y = (dataset.y - norm["y_mean"]) / norm["y_std"]


def apply_normalization_pair(train_dataset, val_dataset, norm):
    train_base = unwrap_dataset(train_dataset)
    val_base = unwrap_dataset(val_dataset)
    apply_normalization(train_base, norm)
    if val_base is not train_base:
        apply_normalization(val_base, norm)


def dataset_arrays(dataset):
    if isinstance(dataset, Subset):
        return dataset.dataset.x[dataset.indices], dataset.dataset.y[dataset.indices]
    return dataset.x, dataset.y


def unwrap_dataset(dataset):
    return dataset.dataset if isinstance(dataset, Subset) else dataset


def make_run_dir(save_cfg, arm):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = arm if arm is not None else "single"
    out = (
        Path(save_cfg.get("output_dir", "runs")).expanduser()
        / f"{save_cfg.get('run_name', 'next')}_{label}_{stamp}"
    )
    out.mkdir(parents=True, exist_ok=False)
    return out


def save_run(out_dir, model, cfg, norm, metrics, dataset, model_cfg):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": dataset.input_size,
            "output_size": dataset.output_size,
            "history": dataset.history,
            "model": model_cfg,
        },
        out_dir / "model.pt",
    )
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    np.savez(out_dir / "normalization.npz", **norm)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def normalized_model_cfg(model_cfg):
    model_cfg = dict(model_cfg)
    model_cfg["type"] = str(model_cfg.get("type", "lstm")).lower()
    return model_cfg


def load_yaml(path):
    with open(Path(path).expanduser(), "r") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
