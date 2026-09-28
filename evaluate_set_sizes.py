"""Evaluate pretrained VWM-RNN accuracy as memory-set size increases."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from rnn import RNNMemoryModel


def generate_report_input(presence, theta, params):
    max_item_num = params["max_item_num"]
    dt = params["dt"]
    steps = int((params["T_init"] + params["T_stimi"] + params["T_delay"] + params["T_decode"]) / dt)
    theta_noisy = theta + params["ILC_noise"] * torch.randn_like(theta)
    theta_noisy = (theta_noisy + torch.pi) % (2 * torch.pi) - torch.pi
    theta_cos = torch.cos(theta_noisy)
    theta_sin = torch.sin(theta_noisy)
    encoded = torch.stack(
        (
            1 + theta_cos / np.sqrt(2) + theta_sin / np.sqrt(6),
            1 - theta_cos / np.sqrt(2) + theta_sin / np.sqrt(6),
            1 - 2 * theta_sin / np.sqrt(6),
        ),
        dim=-1,
    ) * presence.unsqueeze(-1)
    encoded = encoded.reshape(presence.shape[0], -1)
    times = torch.arange(steps, device=presence.device) * dt
    stimulus_mask = (times >= params["T_init"]) & (times < params["T_init"] + params["T_stimi"])
    return encoded.unsqueeze(1) * stimulus_mask.unsqueeze(0).unsqueeze(-1) * params["input_strength"] / max_item_num


def evaluate(model, params, set_size, trials, device):
    presence = torch.zeros(trials, params["max_item_num"], device=device)
    for trial in range(trials):
        indices = torch.randperm(params["max_item_num"], device=device)[:set_size]
        presence[trial, indices] = 1
    theta = (torch.rand(trials, params["max_item_num"], device=device) * 2 - 1) * torch.pi
    inputs = generate_report_input(presence, theta, params)
    with torch.no_grad():
        activity, _ = model(inputs)
        decode_start = int((params["T_init"] + params["T_stimi"] + params["T_delay"]) / params["dt"])
        decode_activity = activity[:, decode_start:, :]
        readout = model.readout(decode_activity.reshape(-1, params["num_neurons"]))
        readout = readout.reshape(trials, -1, 2 * params["max_item_num"])
        decoded = model.decode(readout.transpose(0, 1))
    difference = (theta - decoded + torch.pi) % (2 * torch.pi) - torch.pi
    trial_errors = (difference.abs() * presence).sum(dim=1) / presence.sum(dim=1)
    return trial_errors.mean().item(), trial_errors.std(unbiased=True).item()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="final_report/OptimalModel_check_n64item10PI1gamma0.2l2/config.yaml")
    parser.add_argument("--checkpoint", default="final_report/OptimalModel_check_n64item10PI1gamma0.2l2/models/model_iteration11650.pth")
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20250918)
    parser.add_argument("--output", default="verification_set_size.json")
    parser.add_argument("--plot", default="verification_set_size.png")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    with config_path.open() as file_handle:
        params = yaml.safe_load(file_handle)["model_params"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNMemoryModel(
        params["max_item_num"], params["num_neurons"], params["dt"],
        params["tau_min"], params["tau_max"], params["spike_noise_type"],
        params["spike_noise_factor"], params["saturation_firing_rate"],
        str(device), params["positive_input"], params["dales_law"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.eval()

    set_sizes = list(range(1, params["max_item_num"] + 1))
    results = [evaluate(model, params, set_size, args.trials, device) for set_size in set_sizes]
    means = np.array([result[0] for result in results])
    standard_deviations = np.array([result[1] for result in results])
    degrees = np.degrees(means)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].errorbar(set_sizes, means, yerr=standard_deviations, marker="o", capsize=4, color="#176b87")
    axes[0].axhline(np.pi / 2, linestyle="--", color="#d1495b", label="random guessing = 1.571 rad")
    axes[0].set_title("Prediction error vs memory-set size")
    axes[0].set_xlabel("Number of items")
    axes[0].set_ylabel("Mean angular error (rad)")
    axes[0].set_xticks(set_sizes)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].errorbar(set_sizes, degrees, yerr=np.degrees(standard_deviations), marker="o", capsize=4, color="#edae49")
    axes[1].axhline(90, linestyle="--", color="#d1495b", label="random guessing = 90 deg")
    axes[1].set_title("Prediction error in degrees")
    axes[1].set_xlabel("Number of items")
    axes[1].set_ylabel("Mean angular error (degrees)")
    axes[1].set_xticks(set_sizes)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("Optimal pretrained VWM-RNN under report conditions", fontsize=14)
    figure.tight_layout()
    figure.savefig(args.plot, dpi=160)
    plt.close(figure)

    report = {
        "status": "PASS",
        "device": str(device),
        "seed": args.seed,
        "trials_per_set_size": args.trials,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "conditions": {
            "ILC_noise": params["ILC_noise"],
            "spike_noise_type": params["spike_noise_type"],
            "spike_noise_factor": params["spike_noise_factor"],
            "input_strength": params["input_strength"],
            "T_init": params["T_init"],
            "T_stimi": params["T_stimi"],
            "T_delay": params["T_delay"],
            "T_decode": params["T_decode"],
            "dt": params["dt"],
        },
        "set_sizes": set_sizes,
        "mean_error_rad": means.tolist(),
        "std_error_rad": standard_deviations.tolist(),
        "mean_error_degrees": degrees.tolist(),
        "random_guess_baseline_rad": float(np.pi / 2),
        "plot": str(Path(args.plot).resolve()),
    }
    with Path(args.output).open("w") as file_handle:
        json.dump(report, file_handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
