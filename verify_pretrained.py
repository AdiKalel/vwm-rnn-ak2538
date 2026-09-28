"""Reproduce a pretrained VWM-RNN result from an archived config and checkpoint."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from rnn import RNNMemoryModel


def make_input(presence, theta, params):
    max_item_num = params["max_item_num"]
    dt = params["dt"]
    steps = int((params["T_init"] + params["T_stimi"] + params["T_delay"] + params["T_decode"]) / dt)
    theta_cos = torch.cos(theta)
    theta_sin = torch.sin(theta)
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
    stimulus_mask = ((times >= params["T_init"]) & (times < params["T_init"] + params["T_stimi"]))
    inputs = encoded.unsqueeze(1) * stimulus_mask.unsqueeze(0).unsqueeze(-1)
    return inputs * params["input_strength"] / max_item_num


def angular_error(decoded, target, presence):
    difference = (target - decoded + torch.pi) % (2 * torch.pi) - torch.pi
    return ((difference.abs() * presence).sum(dim=1) / presence.sum(dim=1)).mean()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_result_plot(decoded, theta, presence, output_path):
    angular_errors = ((theta - decoded + torch.pi) % (2 * torch.pi) - torch.pi).abs()
    trial_errors = (angular_errors * presence).sum(dim=1) / presence.sum(dim=1)
    target_values = theta[:, 0].cpu().numpy()
    decoded_values = decoded[:, 0].cpu().numpy()
    trial_errors = trial_errors.cpu().numpy()
    trial_numbers = np.arange(1, len(target_values) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(target_values, decoded_values, color="#176b87", alpha=0.85)
    axes[0].plot([-np.pi, np.pi], [-np.pi, np.pi], "--", color="#d1495b", label="perfect reconstruction")
    axes[0].set_title("Pretrained model: target vs decoded")
    axes[0].set_xlabel("Target orientation (rad)")
    axes[0].set_ylabel("Decoded orientation (rad)")
    axes[0].set_xlim(-np.pi, np.pi)
    axes[0].set_ylim(-np.pi, np.pi)
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].bar(trial_numbers, trial_errors, color="#edae49")
    axes[1].axhline(trial_errors.mean(), color="#d1495b", linestyle="--", label=f"mean = {trial_errors.mean():.3f} rad")
    axes[1].set_title("Angular error by trial")
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Absolute angular error (rad)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    figure.suptitle("Reproduction from archived pretrained weights", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="final_report/check_l2_n64item10PI1gamma0.2l2/config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="final_report/check_l2_n64item10PI1gamma0.2l2/models/model_iteration8450.pth",
    )
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--set-size", type=int, default=1)
    parser.add_argument("--disable-spike-noise", action="store_true")
    parser.add_argument("--seed", type=int, default=20250917)
    parser.add_argument("--output", default="verification_pretrained.json")
    parser.add_argument("--plot", default="verification_pretrained.png")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    with config_path.open() as file_handle:
        config = yaml.safe_load(file_handle)
    params = config["model_params"]

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
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    if not 1 <= args.set_size <= params["max_item_num"]:
        raise ValueError(f"--set-size must be between 1 and {params['max_item_num']}")
    presence = torch.zeros(args.trials, params["max_item_num"], device=device)
    presence[:, :args.set_size] = 1
    if args.disable_spike_noise:
        model.spike_noise_factor = 0.0
    theta = (torch.rand(args.trials, params["max_item_num"], device=device) * 2 - 1) * torch.pi
    inputs = make_input(presence, theta, params)
    with torch.no_grad():
        activity, _ = model(inputs)
        decode_start = int((params["T_init"] + params["T_stimi"] + params["T_delay"]) / params["dt"])
        decode_activity = activity[:, decode_start:, :]
        readout = model.readout(decode_activity.reshape(-1, params["num_neurons"]))
        readout = readout.reshape(args.trials, -1, 2 * params["max_item_num"])
        decoded = model.decode(readout.transpose(0, 1))
        mean_error = angular_error(decoded, theta, presence).item()
        save_result_plot(decoded, theta, presence, args.plot)

    report = {
        "status": "PASS",
        "device": str(device),
        "seed": args.seed,
        "trials": args.trials,
        "set_size": args.set_size,
        "spike_noise_factor": model.spike_noise_factor,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input_shape": list(inputs.shape),
        "activity_shape": list(activity.shape),
        "mean_angular_error_rad": mean_error,
        "finite_outputs": bool(torch.isfinite(activity).all() and torch.isfinite(decoded).all()),
        "plot": str(Path(args.plot).resolve()),
    }
    with Path(args.output).open("w") as file_handle:
        json.dump(report, file_handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()