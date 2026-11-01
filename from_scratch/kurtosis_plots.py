from __future__ import annotations

import sys
import secrets
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_experiment import (  # noqa: E402
    DELAY_MS,
    DT_MS,
    INIT_MS,
    MAX_ITEMS,
    NOISE_FACTOR,
    NOISE_TYPE,
    RESULTS_DIR,
    SATURATION_RATE_HZ,
    STIMULUS_MS,
    _jax,
    _trial_data,
    _weights,
)
from vwm_scratch.trial import compiled_trial, decode_readouts  # noqa: E402

TRIALS_PER_SET_SIZE = 1000
DENSITY_BINS = 48
DENSITY_SMOOTH_SIGMA_BINS = 1.5
def pearson_kurtosis(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).ravel()
    centered = values - values.mean()
    if values.size < 2:
        return 0.0
    sample_std = np.std(values, ddof=1)
    if sample_std == 0.0:
        return 0.0
    return float(np.sum(centered**4) / (values.size * sample_std**4))


def smooth_circular_density(density: np.ndarray, sigma_bins: float) -> np.ndarray:
    radius = int(np.ceil(4.0 * sigma_bins))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(density, (radius, radius), mode="wrap")
    return np.convolve(padded, kernel, mode="valid")


def kurtosis_by_set_size(trials_per_set_size: int = TRIALS_PER_SET_SIZE):
    jax, jnp, _ = _jax()
    run_seed = secrets.randbits(32)
    print(f"Graph run seed: {run_seed}")
    weights = _weights()
    neuron_count = int(weights["B"].shape[0])
    delay_start = int((INIT_MS + STIMULUS_MS) / DT_MS)
    delay_end = int((INIT_MS + STIMULUS_MS + DELAY_MS) / DT_MS)
    set_sizes = np.arange(1, MAX_ITEMS + 1)
    kurtosis_values = []
    angle_errors_by_set_size = {}
    decode_start = int((INIT_MS + STIMULUS_MS + DELAY_MS) / DT_MS)

    trial_offset = 0
    for set_size in set_sizes:
        delay_activations = []
        angle_errors = []
        for trial_index in range(trials_per_set_size):
            seed = (run_seed + trial_offset + trial_index) % (2**32)
            key, theta, presence, inputs, _ = _trial_data(
                jax, jnp, int(set_size), seed
            )
            result = compiled_trial(
                weights,
                inputs,
                jnp.zeros(neuron_count),
                DT_MS,
                SATURATION_RATE_HZ,
                NOISE_TYPE,
                NOISE_FACTOR,
                key,
                True,
            )
            delay_activations.append(np.asarray(result["states"][delay_start:delay_end]))
            decoded = np.asarray(decode_readouts(result["readouts"], decode_start))
            theta = np.asarray(theta)
            presence = np.asarray(presence, dtype=bool)
            wrapped_error = (theta - decoded + np.pi) % (2.0 * np.pi) - np.pi
            angle_errors.extend(wrapped_error[presence].tolist())
        trial_offset += trials_per_set_size
        kurtosis = pearson_kurtosis(np.concatenate(delay_activations, axis=0))
        kurtosis_values.append(kurtosis)
        angle_errors_by_set_size[int(set_size)] = np.asarray(angle_errors)
        print(f"set_size={set_size}, delay_activation_kurtosis={kurtosis:.6g}")

    return set_sizes, np.asarray(kurtosis_values), angle_errors_by_set_size


def main():
    set_sizes, kurtosis_values, angle_errors_by_set_size = kurtosis_by_set_size()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    kurtosis_path = RESULTS_DIR / "kurtosis_vs_set_size.png"
    angle_error_path = RESULTS_DIR / "angle_error_distribution_by_set_size.png"

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(set_sizes, kurtosis_values, marker="o")
    axis.axhline(3.0, color="black", linewidth=0.8, alpha=0.6, label="normal reference")
    axis.set_xticks(set_sizes)
    axis.set_xlabel("Set size (items)")
    axis.set_ylabel("Pearson kurtosis")
    axis.set_title("Delay-period neural activity Pearson kurtosis by set size")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(kurtosis_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 6))
    bin_edges = np.linspace(-np.pi, np.pi, DENSITY_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    for set_size in set_sizes:
        density, _ = np.histogram(
            angle_errors_by_set_size[int(set_size)], bins=bin_edges, density=True
        )
        density = smooth_circular_density(density, DENSITY_SMOOTH_SIGMA_BINS)
        axis.plot(bin_centers, density, linewidth=1.5, label=f"{set_size} item(s)")
    axis.set_xlim(-np.pi, np.pi)
    axis.set_xticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
    axis.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    axis.set_xlabel("Signed wrapped angle error (rad)")
    axis.set_ylabel("Probability density")
    axis.set_title("Angle error probability density by set size")
    axis.grid(alpha=0.25)
    axis.legend(title="Set size", ncol=2)
    figure.tight_layout()
    figure.savefig(angle_error_path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(figure)
    print(f"Saved plot to {kurtosis_path}")
    print(f"Saved plot to {angle_error_path}")


if __name__ == "__main__":
    main()
