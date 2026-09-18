"""Configuration loading and derived calibration values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .equations import gamma_rate_constant, phase_indices, positive_orientation_encoding, simulation_steps


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as file_handle:
        config = yaml.safe_load(file_handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    experiment = config["experiment"]
    if model["max_items"] < 1 or model["neurons"] < 1:
        raise ValueError("max_items and neurons must be positive")
    if model["dt_ms"] <= 0:
        raise ValueError("dt_ms must be positive")
    if experiment["spike_noise_type"] == "gamma" and experiment["spike_noise_factor"] <= 0:
        raise ValueError("gamma noise requires a positive noise factor")
    simulation_steps(
        experiment["init_ms"], experiment["stimulus_ms"], experiment["delay_ms"],
        experiment["decode_ms"], model["dt_ms"],
    )


def derive_calibration(config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    experiment = config["experiment"]
    steps = simulation_steps(
        experiment["init_ms"], experiment["stimulus_ms"], experiment["delay_ms"],
        experiment["decode_ms"], model["dt_ms"],
    )
    result = {
        "input_dim": model["max_items"] * 3 if model["positive_input"] else model["max_items"] * 2,
        "output_dim": model["max_items"] * 2,
        "simulation_steps": steps,
        "phase_indices": phase_indices(
            experiment["init_ms"], experiment["stimulus_ms"], experiment["delay_ms"],
            experiment["decode_ms"], model["dt_ms"],
        ),
        "gamma_rate_constant": gamma_rate_constant(
            model["dt_ms"], experiment["spike_noise_factor"],
        ) if experiment["spike_noise_type"] == "gamma" else None,
        "positive_encoding_min": float(positive_orientation_encoding(
            [-3.141592653589793, 0.0, 3.141592653589793]
        ).min()),
        "positive_encoding_max": float(positive_orientation_encoding(
            [-3.141592653589793, 0.0, 3.141592653589793]
        ).max()),
    }
    return result
