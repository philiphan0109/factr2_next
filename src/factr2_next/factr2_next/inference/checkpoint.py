from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from factr2_next.training.models import build_model


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    normalization: dict
    config: dict
    history: int
    input_size: int
    output_size: int
    device: torch.device


def load_checkpoint(run_dir, device="cpu"):
    run_dir = Path(run_dir)
    device = torch.device(device)
    ckpt = torch.load(run_dir / "model.pt", map_location=device)
    config = _load_yaml(run_dir / "config.yaml")
    model_cfg = ckpt.get("model", config.get("model", {}))

    model = build_model(
        model_cfg,
        ckpt["input_size"],
        ckpt["output_size"],
        ckpt["history"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm = np.load(run_dir / "normalization.npz")
    normalization = {key: norm[key].astype(np.float32) for key in norm.files}

    return LoadedCheckpoint(
        model=model,
        normalization=normalization,
        config=config,
        history=int(ckpt["history"]),
        input_size=int(ckpt["input_size"]),
        output_size=int(ckpt["output_size"]),
        device=device,
    )


def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}
