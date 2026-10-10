"""Matched single-GPU training-step benchmark for Derek's PyTorch and JAX.

Run each backend in a fresh process from the repository root, so one framework
releases GPU memory before the other starts:

    python from_scratch/scripts/benchmark_training.py --backend torch --warmup 2 --steps 10
    .venv/Scripts/python.exe from_scratch/scripts/benchmark_training.py --backend jax --warmup 2 --steps 10
    python from_scratch/scripts/benchmark_training.py --backend summary

On CBL, run the equivalent commands from the checkout. Use a longer measured
run (e.g. --steps 50) after the short check. This measures optimizer steps on
the same fixed, config-matched batch; JAX compile time is reported separately.
It does not start a full 40,000-step training job.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
FROM_SCRATCH = SCRIPT.parents[1]
REPO_ROOT = SCRIPT.parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"
DEFAULT_WEIGHTS = FROM_SCRATCH / "weights" / "derek_rad_n256_gamma03.npz"
RESULTS = FROM_SCRATCH / "results"


def load_settings(config_path: Path):
    import yaml

    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    model = config["model_params"]
    training = config["training_params"]
    stage_count = training.get("multi_stage_training", {}).get("num_stage", 1)
    maximum_noise = model.get("final_spike_noise_factor", model.get("spike_noise_factor", 0.0))
    return {
        "max_items": int(model["max_item_num"]),
        "neurons": int(model["num_neurons"]),
        "dt_ms": float(model["dt"]),
        "tau_min_ms": float(model["tau_min"]),
        "tau_max_ms": float(model["tau_max"]),
        "positive_input": bool(model["positive_input"]),
        "dale_law": bool(model["dales_law"]),
        "saturation_rate_hz": float(model["saturation_firing_rate"]),
        "input_strength": float(model["input_strength"]),
        "sensory_noise_rad": float(model.get("ILC_noise", 0.0)),
        "noise_type": str(model["spike_noise_type"]).lower(),
        "final_noise_factor": float(maximum_noise),
        "init_ms": float(model["T_init"]),
        "stimulus_ms": float(model["T_stimi"]),
        "delay_ms": float(model["T_delay"]),
        "decode_ms": float(model["T_decode"]),
        "loss_type": str(training["error_def"]).lower(),
        "learning_rate": float(training["eta"]),
        "lambda_err": float(training["lambda_err"]),
        "lambda_reg": float(training["lambda_reg"]),
        "num_trials": int(training["num_trials"]),
        "configured_steps": int(training["num_iterations"]),
        "logging_period": int(training["logging_period"]),
        "early_stop_patience": int(training["early_stop_patience"]),
        "adaptive_lr_patience": int(training["adaptive_lr_patience"]),
        "num_stages": int(stage_count),
    }


def make_fixed_batch(settings, seed):
    rng = np.random.default_rng(seed)
    batch = settings["num_trials"]
    items = settings["max_items"]
    input_channels = 3 if settings["positive_input"] else 2
    set_sizes = np.resize(np.arange(1, items + 1, dtype=np.int32), batch)
    presence = np.zeros((batch, items), dtype=np.float32)
    for trial, count in enumerate(set_sizes):
        presence[trial, rng.choice(items, size=count, replace=False)] = 1.0
    theta = rng.uniform(-np.pi, np.pi, (batch, items)).astype(np.float32)
    theta_input = theta + settings["sensory_noise_rad"] * rng.standard_normal(theta.shape).astype(np.float32)
    if settings["positive_input"]:
        encoded = np.stack((
            1 + np.cos(theta_input) / np.sqrt(2) + np.sin(theta_input) / np.sqrt(6),
            1 - np.cos(theta_input) / np.sqrt(2) + np.sin(theta_input) / np.sqrt(6),
            1 - 2 * np.sin(theta_input) / np.sqrt(6),
        ), axis=-1)
    else:
        encoded = np.stack((np.cos(theta_input), np.sin(theta_input)), axis=-1)
    encoded = (encoded * presence[:, :, None]).reshape(batch, items * input_channels)
    dt = settings["dt_ms"]
    durations = (settings["init_ms"], settings["stimulus_ms"], settings["delay_ms"], settings["decode_ms"])
    steps = int(sum(durations) / dt)
    time = np.arange(steps, dtype=np.float32) * dt
    stimulus = ((time >= settings["init_ms"]) & (time < settings["init_ms"] + settings["stimulus_ms"])).astype(np.float32)
    inputs = encoded[:, None, :] * stimulus[None, :, None] * settings["input_strength"] / items
    return {"inputs": inputs.astype(np.float32), "theta": theta, "presence": presence, "set_sizes": set_sizes}


def load_npz_weights(path, device_converter):
    archive = np.load(path)
    return {name: device_converter(archive[name]) for name in ("B", "W", "F", "tau", "dale_sign")}


def _numpy_training_loss_jax(jnp, readouts, theta, presence, error_type):
    from vwm_scratch.losses import training_loss

    target = jnp.stack((jnp.cos(theta), jnp.sin(theta)), axis=-1).reshape(theta.shape[0], -1)
    return training_loss(readouts, target, theta, presence, error_type)[0]


def benchmark_jax(settings, weights_path, warmup, steps, gpu_index):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    import jax
    import jax.numpy as jnp
    import optax

    sys.path.insert(0, str(FROM_SCRATCH / "src"))
    from vwm_scratch.trial import run_trial
    from vwm_scratch.losses import training_loss

    devices = jax.devices()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX backend is {jax.default_backend()}, not GPU; devices={devices}")
    batch_np = make_fixed_batch(settings, seed=128)
    batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
    weights = load_npz_weights(weights_path, jnp.asarray)
    trainable = {key: weights[key] for key in ("B", "W", "F")}
    fixed = {key: weights[key] for key in ("tau", "dale_sign")}
    optimizer = optax.scale_by_adam()
    optimizer_state = optimizer.init(trainable)
    decode_start = int((settings["init_ms"] + settings["stimulus_ms"] + settings["delay_ms"]) / settings["dt_ms"])
    stage_noise = settings["final_noise_factor"]
    keys = jax.random.split(jax.random.PRNGKey(219), settings["num_trials"])

    def step(params, opt_state):
        def objective(candidate):
            full_params = {**candidate, **fixed}

            def one_trial(input_seq, init, key):
                return run_trial(full_params, input_seq, init, settings["dt_ms"], settings["saturation_rate_hz"], settings["noise_type"], stage_noise, key, True, decode_start)
            outputs = jax.vmap(one_trial)(batch["inputs"], jnp.zeros((settings["num_trials"], settings["neurons"])), keys)
            readouts = outputs["readouts"].transpose(1, 0, 2)[decode_start:]
            states = outputs["states"]
            prediction_loss = training_loss(
                readouts,
                jnp.stack((jnp.cos(batch["theta"]), jnp.sin(batch["theta"])), axis=-1).reshape(settings["num_trials"], -1),
                batch["theta"], batch["presence"], settings["loss_type"],
            )[0]
            return settings["lambda_err"] * prediction_loss + settings["lambda_reg"] * jnp.mean(jnp.abs(states))
        loss, gradient = jax.value_and_grad(objective)(params)
        updates, opt_state = optimizer.update(gradient, opt_state, params)
        params = optax.apply_updates(params, jax.tree.map(lambda value: -settings["learning_rate"] * value, updates))
        params = dict(params)
        if settings["positive_input"]:
            params["B"] = jnp.maximum(params["B"], 0)
        if settings["dale_law"]:
            params["W"] = jnp.maximum(params["W"], 0)
        return params, opt_state, loss

    step = jax.jit(step)
    start_compile = time.perf_counter()
    trainable, optimizer_state, loss = step(trainable, optimizer_state)
    loss.block_until_ready()
    compile_and_first_step = time.perf_counter() - start_compile
    for _ in range(warmup - 1):
        trainable, optimizer_state, loss = step(trainable, optimizer_state)
    loss.block_until_ready()

    timings = []
    for _ in range(steps):
        start = time.perf_counter()
        trainable, optimizer_state, loss = step(trainable, optimizer_state)
        loss.block_until_ready()
        timings.append(time.perf_counter() - start)
    del fixed
    return {
        "backend": "jax",
        "devices": [str(device) for device in devices],
        "compile_plus_first_step_seconds": compile_and_first_step,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "seconds_per_step": timings,
        "mean_seconds_per_step": statistics.mean(timings),
        "median_seconds_per_step": statistics.median(timings),
        "final_loss": float(loss),
    }


def _torch_loss(torch, readouts, theta, presence, error_type):
    import torch.nn.functional as F

    steps, batch, _ = readouts.shape
    predictions = readouts.reshape(steps, batch, -1, 2)
    target = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
    if error_type == "l2":
        per_item = ((target - predictions.mean(dim=0)) ** 2).sum(dim=-1)
    elif error_type == "sqrtl2":
        per_item = 10.0 * torch.linalg.norm(target - predictions.mean(dim=0), dim=-1).sqrt()
    elif error_type == "norml2":
        p = F.normalize(predictions, dim=-1)
        t = F.normalize(target, dim=-1)
        per_item = torch.linalg.norm(t - p.mean(dim=0), dim=-1)
    elif error_type in ("rad", "exp"):
        p = F.normalize(predictions, dim=-1)
        delta = torch.acos(torch.clamp((target * p.mean(dim=0)).sum(dim=-1), -1.0, 1.0))
        per_item = delta if error_type == "rad" else -10.0 * torch.exp(-3.0 * delta / torch.pi)
    else:
        raise ValueError(f"Unsupported benchmark loss {error_type}")
    per_trial = (per_item * presence).sum(dim=-1) / presence.sum(dim=-1)
    return per_trial.mean()


def benchmark_torch(settings, weights_path, warmup, steps, gpu_index, batch_np):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    import torch
    sys.path.insert(0, str(REPO_ROOT))
    from rnn import RNNMemoryModel

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA in this environment")
    device = torch.device("cuda:0")
    weights_np = {key: value for key, value in np.load(weights_path).items()}
    model = RNNMemoryModel(
        settings["max_items"], settings["neurons"], settings["dt_ms"],
        settings["tau_min_ms"], settings["tau_max_ms"], settings["noise_type"],
        settings["final_noise_factor"], settings["saturation_rate_hz"], str(device),
        settings["positive_input"], settings["dale_law"],
    ).to(device)
    model.load_state_dict({
        "B": torch.as_tensor(weights_np["B"], device=device),
        "W": torch.as_tensor(weights_np["W"], device=device),
        "F": torch.as_tensor(weights_np["F"], device=device),
        "tau": torch.as_tensor(weights_np["tau"], device=device),
        "dales_sign": torch.as_tensor(weights_np["dale_sign"], device=device),
    })
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    inputs = torch.as_tensor(batch_np["inputs"], device=device)
    theta = torch.as_tensor(batch_np["theta"], device=device)
    presence = torch.as_tensor(batch_np["presence"], device=device)
    decode_start = int((settings["init_ms"] + settings["stimulus_ms"] + settings["delay_ms"]) / settings["dt_ms"])

    def step():
        model.spike_noise_factor = settings["final_noise_factor"]
        activity, _ = model(inputs)
        decode_states = activity[:, decode_start:, :]
        readouts = model.readout(decode_states.reshape(-1, settings["neurons"]))
        readouts = readouts.reshape(decode_states.shape[1], inputs.shape[0], -1)
        readouts = readouts.permute(1, 0, 2)
        error = _torch_loss(torch, readouts, theta, presence, settings["loss_type"])
        activation = activity.abs().mean()
        total = settings["lambda_err"] * error + settings["lambda_reg"] * activation
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        with torch.no_grad():
            if settings["positive_input"]:
                model.B.clamp_(min=0)
            if settings["dale_law"]:
                model.W.clamp_(min=0)
        return total.detach()

    for _ in range(warmup):
        loss = step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    return {
        "backend": "torch",
        "device": torch.cuda.get_device_name(0),
        "warmup_steps": warmup,
        "measured_steps": steps,
        "seconds_per_step": timings,
        "mean_seconds_per_step": statistics.mean(timings),
        "median_seconds_per_step": statistics.median(timings),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "final_loss": float(loss),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("jax", "torch", "summary"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.backend == "summary":
        reports = [json.loads(path.read_text()) for path in (RESULTS / "benchmark_torch.json", RESULTS / "benchmark_jax.json")]
        seconds = {report["backend"]: report["mean_seconds_per_step"] for report in reports}
        fastest = min(seconds, key=seconds.get)
        print(json.dumps({"seconds_per_step": seconds, "faster_backend": fastest,
                          "torch_speedup_vs_jax": seconds["jax"] / seconds["torch"],
                          "40000_step_hours_steady_state": {key: value * 40000 / 3600 for key, value in seconds.items()},
                          "jax_compile_plus_first_step_seconds": reports[1].get("compile_plus_first_step_seconds")}, indent=2))
        return

    settings = load_settings(args.config.resolve())
    if args.weights.resolve().is_file() is False:
        raise FileNotFoundError(args.weights.resolve())
    if np.load(args.weights).get("B").shape[0] != settings["neurons"]:
        raise ValueError("Weights neuron dimension does not match benchmark config")
    batch_np = make_fixed_batch(settings, seed=128)
    if args.backend == "jax":
        report = benchmark_jax(settings, args.weights.resolve(), args.warmup, args.steps, args.gpu)
    else:
        report = benchmark_torch(settings, args.weights.resolve(), args.warmup, args.steps, args.gpu, batch_np)
    report.update({"config": str(args.config.resolve()), "weights": str(args.weights.resolve()),
                   "gpu_index": args.gpu, "batch_size": settings["num_trials"],
                   "sequence_steps": batch_np["inputs"].shape[1], "neurons": settings["neurons"],
                   "loss_type": settings["loss_type"], "noise_type": settings["noise_type"],
                   "noise_factor": settings["final_noise_factor"]})
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"benchmark_{args.backend}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"saved report: {path}")


if __name__ == "__main__":
    main()
