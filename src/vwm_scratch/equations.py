"""Pure reference equations for the VWM task."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def simulation_steps(init_ms: float, stimulus_ms: float, delay_ms: float, decode_ms: float, dt_ms: float) -> int:
    total_ms = init_ms + stimulus_ms + delay_ms + decode_ms
    steps = total_ms / dt_ms
    if not steps.is_integer():
        raise ValueError("Total duration must be divisible by dt_ms")
    return int(steps)


def phase_indices(init_ms: float, stimulus_ms: float, delay_ms: float, decode_ms: float, dt_ms: float) -> dict[str, Tuple[int, int]]:
    init_end = int(init_ms / dt_ms)
    stimulus_end = int((init_ms + stimulus_ms) / dt_ms)
    delay_end = int((init_ms + stimulus_ms + delay_ms) / dt_ms)
    decode_end = int((init_ms + stimulus_ms + delay_ms + decode_ms) / dt_ms)
    return {
        "init": (0, init_end),
        "stimulus": (init_end, stimulus_end),
        "delay": (stimulus_end, delay_end),
        "decode": (delay_end, decode_end),
    }


def positive_orientation_encoding(theta: np.ndarray) -> np.ndarray:
    """Map angles to the three-channel non-negative representation used by the model."""
    theta = np.asarray(theta)
    return np.stack(
        (
            1.0 + np.cos(theta) / math.sqrt(2.0) + np.sin(theta) / math.sqrt(6.0),
            1.0 - np.cos(theta) / math.sqrt(2.0) + np.sin(theta) / math.sqrt(6.0),
            1.0 - 2.0 * np.sin(theta) / math.sqrt(6.0),
        ),
        axis=-1,
    )


def decode_orientation(cos_sin: np.ndarray) -> np.ndarray:
    cos_sin = np.asarray(cos_sin)
    if cos_sin.shape[-1] != 2:
        raise ValueError("The final dimension must contain cosine and sine")
    return np.arctan2(cos_sin[..., 1], cos_sin[..., 0])


def circular_difference(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return (np.asarray(target) - np.asarray(predicted) + np.pi) % (2.0 * np.pi) - np.pi


def gamma_rate_constant(dt_ms: float, spike_noise_factor: float) -> float:
    if spike_noise_factor <= 0.0:
        raise ValueError("spike_noise_factor must be positive for gamma noise")
    return (dt_ms / 1000.0) / spike_noise_factor**2
