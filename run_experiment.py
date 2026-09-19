"""One-file control panel for the from-scratch JAX VWM RNN.

Edit the KNOBS section, then run from the repository root:

    python from_scratch/run_experiment.py

This file is intentionally explicit. It is the place to choose whether the
run calibrates, simulates, trains, evaluates performance, inspects weights, or
saves artifacts. The lower-level modules contain the reusable mathematics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# =============================================================================
# KNOBS: edit this section before running
# =============================================================================

# Choose exactly one primary action.
#   "calibrate"       derive and save dimensions/timing/noise constants
#   "trial"           run one trial and report its prediction and loss
#   "train"           optimise all differentiable weights on repeated trials
#   "performance"     evaluate set sizes 1..MAX_ITEMS and save a JSON table
#   "weight_analysis" save matrix shapes, norms, and Dale-law diagnostics
MODE = "trial"

# Choose where weights come from.
#   "initialize"      make deterministic weights from INITIALIZATION_SEED
#   "file"            load a converted/trained .npz file from WEIGHTS_PATH
WEIGHTS_SOURCE = "file"
# To use the committed pretrained report weights, change the two lines to:
#   WEIGHTS_SOURCE = "file"
#   WEIGHTS_PATH = "from_scratch/weights/derek_optimal_l2_n64_gamma02.npz"
WEIGHTS_PATH = "from_scratch/weights/derek_rad_n256_gamma03.npz"
INITIALIZATION_SEED = 7

# Choose one loss. These are minimised by training:
#   "angular"          wrapped absolute angle error, in radians
#   "euclidean"        squared distance between predicted and target cos/sin
#   "rooted_euclidean" square-root Euclidean distance (less sensitive to outliers)
#   "exponential"      negative psychophysical similarity; lower is better
#   "custom"           use CUSTOM_LOSS below
LOSS_TYPE = "angular"

# Custom loss instructions:
# 1. Set LOSS_TYPE = "custom".
# 2. Replace the body of CUSTOM_LOSS.
# 3. Keep this signature and return one scalar JAX value:
#       CUSTOM_LOSS(predicted_output, target_output, presence)
#    predicted_output and target_output have shape [max_items * 2], ordered
#    [cos(item_1), sin(item_1), cos(item_2), sin(item_2), ...].
def CUSTOM_LOSS(predicted_output, target_output, presence):
    import jax.numpy as jnp

    difference = predicted_output - target_output
    return jnp.sum((difference.reshape(-1, 2) ** 2).sum(axis=1) * presence) / jnp.sum(presence)


# Report/model calibration knobs.
MAX_ITEMS = 10
NEURONS = 256
DT_MS = 10.0
TAU_MIN_MS = 50.0
TAU_MAX_MS = 300.0
POSITIVE_INPUT = True
DALE_LAW = True
SATURATION_RATE_HZ = 60.0
INPUT_STRENGTH = 120.0
INIT_MS = 20.0
STIMULUS_MS = 500.0
DELAY_MS = 1000.0
DECODE_MS = 500.0

# Noise knobs. NOISE_FACTOR is the end/final spike-noise level.
# Set NOISE_TYPE = "none" and NOISE_FACTOR = 0.0 for deterministic dynamics.
# "gamma" matches the report. Other equivalent legacy choices are "gaussian",
# "puregauss", and "csnr"; "none" disables neuron noise.
NOISE_TYPE = "gamma"
NOISE_FACTOR = 0.3
SENSORY_NOISE_RAD = 0.0
RANDOM_SEED = 20250918

# Trial/performance knobs.
SET_SIZE = 1
TRIALS_PER_SET_SIZE = 64
ANALYSIS_SET_SIZES = range(1, MAX_ITEMS + 1)
STORE_STATES = True  # False saves memory when only readouts/final state are needed.

# Training knobs. Used only when MODE = "train".
TRAIN_STEPS = 100
LEARNING_RATE = 1e-4
LAMBDA_ERR = 1.0       # Weight applied to the prediction/error loss.
LAMBDA_REG = 1e-5      # Weight applied to mean absolute neural activation.
TRAIN_SET_SIZE = 1
TRAIN_NOISE_TYPE = "gamma"
TRAIN_NOISE_FACTOR = 0.3
TRAIN_STORE_STATES = True  # Required for the activation regularizer; disable only if LAMBDA_REG = 0.

# Artifact knobs. Set any path to None to skip that artifact.
RESULTS_DIR = Path("from_scratch/results")
SAVE_WEIGHTS = False
SAVE_TRIAL = True
SAVE_RESULTS = True

# =============================================================================
# ANALYSIS API
# =============================================================================
# Use this function from another script or an interactive Python session to
# sweep one condition while keeping every unspecified trial condition at the
# defaults above. Example:
#
#   noise_levels = [0.02 * i for i in range(6)]
#   activation_loss = [
#       run_analysis_trial(noise_factor=noise)["activation_loss"]
#       for noise in noise_levels
#   ]
#   plt.plot(noise_levels, activation_loss)
#
# The function name is `run_analysis_trial`. Supported condition overrides are
# `set_size`, `seed`, `noise_type`, `noise_factor`, `sensory_noise_rad`,
# `input_strength`, `loss_type`, and `store_states`. Pass `weights=` to reuse
# already-loaded weights during a large sweep.

# =============================================================================
# Runner implementation
# =============================================================================


def _jax():
    try:
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError as error:
        raise SystemExit("Install dependencies with: pip install -e 'from_scratch[jax,test]'") from error
    return jax, jnp, optax


def _weights():
    import jax.numpy as jnp
    from vwm_scratch.weights import initialize_weights

    input_dim = MAX_ITEMS * (3 if POSITIVE_INPUT else 2)
    output_dim = MAX_ITEMS * 2
    if WEIGHTS_SOURCE == "initialize":
        source = initialize_weights(
            input_dim, NEURONS, output_dim, TAU_MIN_MS, TAU_MAX_MS,
            dale_law=DALE_LAW, positive_input=POSITIVE_INPUT,
            seed=INITIALIZATION_SEED,
        )
        return {name: jnp.asarray(getattr(source, name)) for name in ("B", "W", "F", "tau", "dale_sign")}
    if WEIGHTS_SOURCE == "file":
        data = np.load(WEIGHTS_PATH)
        weights = {name: jnp.asarray(data[name]) for name in ("B", "W", "F", "tau", "dale_sign")}
        expected_input_dim = MAX_ITEMS * (3 if POSITIVE_INPUT else 2)
        expected_output_dim = MAX_ITEMS * 2
        if weights["B"].shape != (NEURONS, expected_input_dim):
            raise ValueError(
                f"{WEIGHTS_PATH} has B shape {weights['B'].shape}; set NEURONS={weights['B'].shape[0]} "
                f"and MAX_ITEMS={weights['B'].shape[1] // (3 if POSITIVE_INPUT else 2)}"
            )
        if weights["F"].shape != (expected_output_dim, NEURONS):
            raise ValueError(f"{WEIGHTS_PATH} has F shape {weights['F'].shape}; check NEURONS and MAX_ITEMS")
        return weights
    raise ValueError("WEIGHTS_SOURCE must be 'initialize' or 'file'")


def _steps() -> int:
    total = INIT_MS + STIMULUS_MS + DELAY_MS + DECODE_MS
    if total % DT_MS:
        raise ValueError("Total duration must be divisible by DT_MS")
    return int(total / DT_MS)


def _inputs(jax, jnp, theta, presence, key, sensory_noise_rad=SENSORY_NOISE_RAD, input_strength=INPUT_STRENGTH):
    theta_noisy = theta + sensory_noise_rad * jax.random.normal(key, theta.shape)
    if POSITIVE_INPUT:
        encoded = jnp.stack(
            (
                1.0 + jnp.cos(theta_noisy) / jnp.sqrt(2.0) + jnp.sin(theta_noisy) / jnp.sqrt(6.0),
                1.0 - jnp.cos(theta_noisy) / jnp.sqrt(2.0) + jnp.sin(theta_noisy) / jnp.sqrt(6.0),
                1.0 - 2.0 * jnp.sin(theta_noisy) / jnp.sqrt(6.0),
            ),
            axis=-1,
        )
    else:
        encoded = jnp.stack((jnp.cos(theta_noisy), jnp.sin(theta_noisy)), axis=-1)
    encoded = encoded * presence[:, None]
    encoded = encoded.reshape(-1)
    times = jnp.arange(_steps()) * DT_MS
    mask = (times >= INIT_MS) & (times < INIT_MS + STIMULUS_MS)
    return encoded.reshape(1, -1) * mask[:, None] * input_strength / MAX_ITEMS


def _target_output(jnp, theta, presence):
    output = jnp.stack((jnp.cos(theta), jnp.sin(theta)), axis=-1).reshape(-1)
    return output * jnp.repeat(presence, 2)


def _trial_data(jax, jnp, set_size, seed, sensory_noise_rad=SENSORY_NOISE_RAD, input_strength=INPUT_STRENGTH):
    key = jax.random.PRNGKey(seed)
    theta = jax.random.uniform(key, (MAX_ITEMS,), minval=-jnp.pi, maxval=jnp.pi)
    presence = jnp.concatenate((jnp.ones(set_size), jnp.zeros(MAX_ITEMS - set_size)))
    inputs = _inputs(jax, jnp, theta, presence, key, sensory_noise_rad, input_strength)
    initial_state = jnp.zeros(NEURONS)
    return key, theta, presence, inputs, initial_state


def _loss_from_result(jnp, result, theta, presence, loss_fn, loss_type=LOSS_TYPE, lambda_err=LAMBDA_ERR, lambda_reg=LAMBDA_REG):
    from vwm_scratch.trial import decode_readouts

    decode_start = int((INIT_MS + STIMULUS_MS + DELAY_MS) / DT_MS)
    mean_output = result["readouts"][decode_start:].mean(axis=0)
    target_output = _target_output(jnp, theta, presence)
    decoded = decode_readouts(result["readouts"], decode_start)
    if loss_type == "angular":
        error_loss = loss_fn(decoded, theta, presence)
    else:
        error_loss = loss_fn(mean_output, target_output, presence)
    if result["states"] is None:
        activation_penalty = jnp.nan
        activation_loss = jnp.nan
        total_loss = jnp.nan
    else:
        activation_penalty = jnp.mean(jnp.abs(result["states"]))
        activation_loss = lambda_reg * activation_penalty
        total_loss = lambda_err * error_loss + activation_loss
    return total_loss, error_loss, activation_penalty, activation_loss, decoded, mean_output


def _run_trial(
    weights,
    set_size=SET_SIZE,
    seed=RANDOM_SEED,
    noise_type=NOISE_TYPE,
    noise_factor=NOISE_FACTOR,
    sensory_noise_rad=SENSORY_NOISE_RAD,
    input_strength=INPUT_STRENGTH,
    loss_type=LOSS_TYPE,
    store_states=STORE_STATES,
):
    jax, jnp, _ = _jax()
    from vwm_scratch.losses import get_loss
    from vwm_scratch.trial import compiled_trial

    key, theta, presence, inputs, initial_state = _trial_data(
        jax, jnp, set_size, seed, sensory_noise_rad, input_strength
    )
    result = compiled_trial(
        weights, inputs, initial_state, DT_MS,
        saturation_rate_hz=SATURATION_RATE_HZ,
        noise_type=noise_type, noise_factor=noise_factor, key=key,
        store_states=store_states,
    )
    loss_fn = get_loss(loss_type, CUSTOM_LOSS if loss_type == "custom" else None)
    total_loss, error_loss, activation_penalty, activation_loss, decoded, mean_output = _loss_from_result(
        jnp, result, theta, presence, loss_fn, loss_type, LAMBDA_ERR, LAMBDA_REG
    )
    present_theta = np.asarray(theta)[np.asarray(presence).astype(bool)]
    present_decoded = np.asarray(decoded)[np.asarray(presence).astype(bool)]
    return result, {
        "total_loss": float(total_loss),
        "error_loss": float(error_loss),
        "activation_penalty": float(activation_penalty),
        "activation_loss": float(activation_loss),
        "theta_all_slots": np.asarray(theta).tolist(),
        "theta_present_slots": present_theta.tolist(),
        "presence": np.asarray(presence).tolist(),
        "decoded_all_slots": np.asarray(decoded).tolist(),
        "decoded_present_slots": present_decoded.tolist(),
        "mean_output": np.asarray(mean_output).tolist(),
        "conditions": {
            "set_size": set_size,
            "seed": seed,
            "noise_type": noise_type,
            "noise_factor": noise_factor,
            "sensory_noise_rad": sensory_noise_rad,
            "input_strength": input_strength,
            "loss_type": loss_type,
            "store_states": store_states,
        },
    }


def run_analysis_trial(
    *,
    weights=None,
    set_size=SET_SIZE,
    seed=RANDOM_SEED,
    noise_type=NOISE_TYPE,
    noise_factor=NOISE_FACTOR,
    sensory_noise_rad=SENSORY_NOISE_RAD,
    input_strength=INPUT_STRENGTH,
    loss_type=LOSS_TYPE,
    store_states=STORE_STATES,
):
    """Run one analysis trial with optional condition overrides.

    Every argument defaults to the trial knobs above. Override only the
    condition being analysed, for example:

        result = run_analysis_trial(noise_factor=0.02 * i)
        activation_loss[i] = result["activation_loss"]

    Set ``store_states=True`` (the default) to calculate activation loss. Set
    it to false for a lower-memory prediction-only run; activation losses then
    return NaN because the state trajectory was intentionally not retained.
    Pass a preloaded ``weights`` pytree to avoid loading/initialising weights
    inside every iteration of a sweep.
    """
    if weights is None:
        weights = _weights()
    _, report = _run_trial(
        weights,
        set_size=set_size,
        seed=seed,
        noise_type=noise_type,
        noise_factor=noise_factor,
        sensory_noise_rad=sensory_noise_rad,
        input_strength=input_strength,
        loss_type=loss_type,
        store_states=store_states,
    )
    return report


def _train(weights):
    jax, jnp, optax = _jax()
    from vwm_scratch.losses import get_loss
    from vwm_scratch.trial import run_trial

    loss_fn = get_loss(LOSS_TYPE, CUSTOM_LOSS if LOSS_TYPE == "custom" else None)
    optimizer = optax.adam(LEARNING_RATE)
    state = optimizer.init(weights)

    def objective(current_weights, seed):
        key, theta, presence, inputs, initial_state = _trial_data(jax, jnp, TRAIN_SET_SIZE, seed)
        result = run_trial(current_weights, inputs, initial_state, DT_MS, SATURATION_RATE_HZ, TRAIN_NOISE_TYPE, TRAIN_NOISE_FACTOR, key, TRAIN_STORE_STATES)
        total_loss, _, _, _, _, _ = _loss_from_result(jnp, result, theta, presence, loss_fn)
        return total_loss

    objective = jax.jit(objective)
    history = []
    for step in range(TRAIN_STEPS):
        value, gradients = jax.value_and_grad(objective)(weights, RANDOM_SEED + step)
        updates, state = optimizer.update(gradients, state, weights)
        weights = optax.apply_updates(weights, updates)
        history.append(float(value))
    return weights, {"training_loss": history}


def _save_weights(weights, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{name: np.asarray(value) for name, value in weights.items()})


def main():
    from vwm_scratch.calibration import derive_calibration

    if MODE == "calibrate":
        config = {
            "model": {"max_items": MAX_ITEMS, "neurons": NEURONS, "dt_ms": DT_MS, "tau_min_ms": TAU_MIN_MS, "tau_max_ms": TAU_MAX_MS, "positive_input": POSITIVE_INPUT},
            "experiment": {"init_ms": INIT_MS, "stimulus_ms": STIMULUS_MS, "delay_ms": DELAY_MS, "decode_ms": DECODE_MS, "spike_noise_type": NOISE_TYPE, "spike_noise_factor": NOISE_FACTOR},
        }
        report = derive_calibration(config)
        print(json.dumps(report, indent=2))
        return

    jax, jnp, _ = _jax()
    weights = _weights()
    report = {"mode": MODE, "loss_type": LOSS_TYPE, "noise_type": NOISE_TYPE, "noise_factor": NOISE_FACTOR}

    if MODE == "trial":
        result, trial_report = _run_trial(weights)
        report.update(trial_report)
        if SAVE_TRIAL:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(RESULTS_DIR / "trial.npz", **{name: np.asarray(value) for name, value in result.items() if value is not None})
    elif MODE == "train":
        weights, training_report = _train(weights)
        report.update(training_report)
    elif MODE == "performance":
        values = []
        for set_size in ANALYSIS_SET_SIZES:
            losses = [
                _run_trial(weights, set_size=set_size, seed=RANDOM_SEED + trial)[1]["total_loss"]
                for trial in range(TRIALS_PER_SET_SIZE)
            ]
            values.append({"set_size": set_size, "mean_total_loss": float(np.mean(losses)), "std_total_loss": float(np.std(losses))})
        report["set_size_results"] = values
    elif MODE == "weight_analysis":
        report["weights"] = {name: {"shape": list(value.shape), "norm": float(jnp.linalg.norm(value))} for name, value in weights.items()}
        report["effective_W_norm"] = float(jnp.linalg.norm(weights["W"] * weights["dale_sign"][None, :]))
    elif MODE == "save_weights":
        pass
    else:
        raise ValueError("MODE must be calibrate, trial, train, performance, weight_analysis, or save_weights")

    if SAVE_WEIGHTS or MODE == "save_weights":
        _save_weights(weights, RESULTS_DIR / "weights.npz")
        report["weights_path"] = str(RESULTS_DIR / "weights.npz")
    if SAVE_RESULTS:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
