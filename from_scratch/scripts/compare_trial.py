"""Compare one deterministic 3-item trial between PyTorch and JAX.

Run both stages from the repository root:

    python from_scratch/scripts/compare_trial.py --backend torch
    .venv\\Scripts\\python.exe from_scratch/scripts/compare_trial.py --backend jax

The first command loads the original pretrained checkpoint and saves its
weights, input, and activation trace. The second command loads those exact
arrays and compares the from-scratch JAX rollout timestep by timestep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FROM_SCRATCH = ROOT / "from_scratch"
CHECKPOINT = ROOT / "final_report/OptimalModel_check_n64item10PI1gamma0.2l2/models/model_iteration11650.pth"
ARTIFACT_DIR = FROM_SCRATCH / "comparison_results"
ARTIFACT = ARTIFACT_DIR / "three_item_trial.npz"
REPORT = ARTIFACT_DIR / "three_item_comparison.json"
PLOT = ARTIFACT_DIR / "three_item_activations.png"


def fixed_inputs():
    theta = np.array([-1.4, 0.2, 1.7] + [0.0] * 7, dtype=np.float32)
    presence = np.array([1.0, 1.0, 1.0] + [0.0] * 7, dtype=np.float32)
    encoded = np.stack(
        (
            1.0 + np.cos(theta) / np.sqrt(2.0) + np.sin(theta) / np.sqrt(6.0),
            1.0 - np.cos(theta) / np.sqrt(2.0) + np.sin(theta) / np.sqrt(6.0),
            1.0 - 2.0 * np.sin(theta) / np.sqrt(6.0),
        ),
        axis=-1,
    ) * presence[:, None]
    encoded = encoded.reshape(1, -1)
    times = np.arange(202, dtype=np.float32) * 10.0
    mask = ((times >= 20.0) & (times < 520.0)).astype(np.float32)
    inputs = (encoded[:, None, :] * mask[None, :, None] * 120.0 / 10.0).astype(np.float32)
    return theta, presence, inputs


def run_torch():
    import torch

    sys.path.insert(0, str(ROOT))
    from rnn import RNNMemoryModel

    theta, presence, inputs = fixed_inputs()
    model = RNNMemoryModel(10, 64, 10.0, 50.0, 300.0, "gamma", 0.2, 60.0, "cpu", True, True)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=False))
    model.spike_noise_factor = 0.0
    model.eval()
    input_tensor = torch.from_numpy(inputs)
    with torch.no_grad():
        states, _ = model(input_tensor)
        readouts = model.readout(states.reshape(-1, 64)).reshape(1, 202, 20)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        ARTIFACT,
        B=model.B.detach().numpy(), W=model.W.detach().numpy(), F=model.F.detach().numpy(),
        tau=model.tau.detach().numpy(), dale_sign=model.dales_sign.detach().numpy(),
        inputs=inputs[0], initial_state=np.zeros(64, dtype=np.float32),
        torch_states=states[0].numpy(), torch_readouts=readouts[0].numpy(),
        theta=theta, presence=presence,
    )
    print(json.dumps({"backend": "torch", "artifact": str(ARTIFACT), "states": list(states.shape), "readouts": list(readouts.shape)}, indent=2))


def run_jax():
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(FROM_SCRATCH / "src"))
    from vwm_scratch.trial import compiled_trial

    data = np.load(ARTIFACT)
    weights = {name: jnp.asarray(data[name]) for name in ("B", "W", "F", "tau", "dale_sign")}
    result = compiled_trial(
        weights,
        jnp.asarray(data["inputs"]),
        jnp.asarray(data["initial_state"]),
        10.0,
        saturation_rate_hz=60.0,
        noise_type="none",
        noise_factor=0.0,
        key=jax.random.PRNGKey(0),
        store_states=True,
    )
    jax_states = np.asarray(result["states"])
    jax_readouts = np.asarray(result["readouts"])
    torch_states = data["torch_states"]
    torch_readouts = data["torch_readouts"]
    state_difference = np.abs(jax_states - torch_states)
    readout_difference = np.abs(jax_readouts - torch_readouts)
    report = {
        "backend": "jax",
        "checkpoint": str(CHECKPOINT),
        "conditions": {"items": 3, "dt_ms": 10.0, "noise_type": "none", "noise_factor": 0.0, "input_strength": 120.0},
        "state_shape": list(jax_states.shape),
        "readout_shape": list(jax_readouts.shape),
        "max_state_abs_difference": float(state_difference.max()),
        "mean_state_abs_difference": float(state_difference.mean()),
        "max_readout_abs_difference": float(readout_difference.max()),
        "mean_readout_abs_difference": float(readout_difference.mean()),
        "equivalent_within_1e-4": bool(state_difference.max() < 1e-4 and readout_difference.max() < 1e-4),
        "plot": str(PLOT),
    }
    selected_neurons = [0, 1, 32, 33]
    time_ms = np.arange(jax_states.shape[0]) * 10.0
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for neuron in selected_neurons:
        axes[0].plot(time_ms, torch_states[:, neuron], label=f"neuron {neuron}")
        axes[1].plot(time_ms, jax_states[:, neuron], "--", label=f"neuron {neuron}")
    for axis, title in zip(axes, ("Original PyTorch pretrained model", "From-scratch JAX model")):
        axis.axvspan(20, 520, color="gold", alpha=0.18, label="stimulus" if axis is axes[0] else None)
        axis.axvspan(1520, 2020, color="lightgreen", alpha=0.18, label="decode" if axis is axes[0] else None)
        axis.set_ylabel("Firing rate")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend(ncol=3)
    axes[1].set_xlabel("Time (ms)")
    figure.suptitle("Three-item trial: identical pretrained weights and input", fontsize=14)
    figure.tight_layout()
    figure.savefig(PLOT, dpi=160)
    plt.close(figure)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "jax"), required=True)
    args = parser.parse_args()
    if args.backend == "torch":
        run_torch()
    else:
        run_jax()


if __name__ == "__main__":
    main()
