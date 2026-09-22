"""Convert an original PyTorch checkpoint to the from-scratch JAX weight format.

Example from the repository root:

    python scripts/convert_checkpoint.py \
        --checkpoint final_report/OptimalModel_check_n64item10PI1gamma0.2l2/models/model_iteration11650.pth \
        --output weights/optimal_model_iteration11650.npz

The output contains B, W, F, tau, and dale_sign arrays and can be selected in
run_experiment.py with:

    WEIGHTS_SOURCE = "file"
    WEIGHTS_PATH = "weights/optimal_model_iteration11650.npz"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Original YAML config paired with the checkpoint")
    parser.add_argument("--label", help="Human-readable label for this pretrained run")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required = ("B", "W", "F", "tau", "dales_sign")
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f"Checkpoint is missing parameters: {', '.join(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "B": state["B"].detach().cpu().numpy(),
        "W": state["W"].detach().cpu().numpy(),
        "F": state["F"].detach().cpu().numpy(),
        "tau": state["tau"].detach().cpu().numpy(),
        "dale_sign": state["dales_sign"].detach().cpu().numpy(),
    }
    np.savez(args.output, **arrays)

    metadata = {
        "label": args.label or args.output.stem,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_sha256": sha256(args.checkpoint),
        "output": str(args.output.resolve()),
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
    }
    if args.config:
        with args.config.open() as file_handle:
            config = yaml.safe_load(file_handle)
        metadata["source_config"] = str(args.config.resolve())
        metadata["conditions"] = {
            "model": config.get("model_params", {}),
            "training": config.get("training_params", {}),
            "logging": config.get("model_and_logging_params", {}),
        }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
