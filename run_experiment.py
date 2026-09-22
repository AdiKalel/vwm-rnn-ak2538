"""Compact callable control surface for the from-scratch JAX model.

Use from the repository root:

    import sys
    sys.path.insert(0, "from_scratch")
    from run_experiment import train, save_weights, run_trial

    weights, history = train()
    save_weights(weights, "weights/my_run.npz")
    error = run_trial.eloss(weights=weights, set_size=3)

Edit the KNOBS block below. There are no mode switches: call the function you
want. All trial functions accept keyword overrides and otherwise use defaults.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# =============================================================================
# EDITABLE KNOBS: keep experiment choices together here
# =============================================================================

# Use "file" for trained weights or "initialize" for new untrained weights.
WEIGHTS_SOURCE = "file"
WEIGHTS_PATH = ROOT / "weights" / "derek_rad_n256_gamma03.npz"
INITIALIZATION_SEED = 7

# Architecture and timing.
MAX_ITEMS = 10
NEURONS = 256
DT_MS = 10.0
TAU_MIN_MS, TAU_MAX_MS = 50.0, 300.0
POSITIVE_INPUT = True
DALE_LAW = True
SATURATION_RATE_HZ = 60.0
INPUT_STRENGTH = 120.0
INIT_MS, STIMULUS_MS, DELAY_MS, DECODE_MS = 20.0, 500.0, 1000.0, 500.0

# Default trial conditions; override any of these in run_trial calls.
SET_SIZE = 1
NOISE_TYPE = "gamma"  # none, gamma, gaussian, puregauss, or csnr
NOISE_FACTOR = 0.3
SENSORY_NOISE_RAD = 0.0
RANDOM_SEED = 20250918

# Training defaults.
LOSS_TYPE = "angular"  # angular, euclidean, rooted_euclidean, exponential, custom
TRAIN_STEPS = 100
LEARNING_RATE = 1e-4
LAMBDA_ERR, LAMBDA_REG = 1.0, 1e-5
TRAIN_SET_SIZE = 1
TRAIN_NOISE_TYPE, TRAIN_NOISE_FACTOR = "gamma", 0.3
TRAIN_NUM_TRIALS = 300
TRAIN_ITEM_NUM = tuple(range(1, MAX_ITEMS + 1))
TRAIN_LOGGING_PERIOD = 10
TRAIN_EARLY_STOP_PATIENCE = 150
TRAIN_ADAPTIVE_LR_PATIENCE = 100
TRAIN_LR_FACTOR = 0.5
TRAIN_NUM_STAGES = 5
TRAIN_MIN_NOISE_FACTOR = 1e-3

RESULTS_DIR = ROOT / "results"
WEIGHTS_DIR = ROOT / "weights"
TRAIN_SAVE_DIR = RESULTS_DIR / "training"


def CUSTOM_LOSS(predicted_output, target_output, presence):
    """Edit this function when LOSS_TYPE='custom'; return one JAX scalar."""
    import jax.numpy as jnp
    difference = predicted_output - target_output
    return jnp.sum((difference.reshape(-1, 2) ** 2).sum(axis=1) * presence) / jnp.sum(presence)


# =============================================================================
# PUBLIC FUNCTIONS: call these manually
# =============================================================================


def train(*, steps=TRAIN_STEPS, weights=None, loss_type=LOSS_TYPE,
          learning_rate=LEARNING_RATE, set_size=TRAIN_SET_SIZE,
          noise_type=TRAIN_NOISE_TYPE, noise_factor=TRAIN_NOISE_FACTOR,
        show_plot=True, num_trials=TRAIN_NUM_TRIALS,
        item_numbers=TRAIN_ITEM_NUM, logging_period=TRAIN_LOGGING_PERIOD,
        early_stop_patience=TRAIN_EARLY_STOP_PATIENCE,
        adaptive_lr_patience=TRAIN_ADAPTIVE_LR_PATIENCE,
        num_stages=TRAIN_NUM_STAGES, resume_history=None):
    """Train with Derek's mixed-set-size curriculum and checkpoint behavior.

    The returned history contains overall and per-set-size evaluation errors,
    standard deviations, activation, learning rate, stage/noise, and best-step
    metadata. Checkpoints and ``training_history.json`` are written under
    ``TRAIN_SAVE_DIR`` every ``logging_period`` iterations.
    """
    jax, jnp, optax = _jax()
    from vwm_scratch.losses import training_loss
    from vwm_scratch.trial import run_trial as jax_run_trial

    current = _weights() if weights is None else weights
    if loss_type == "custom":
        raise ValueError("Custom loss training requires adding it to training_loss in losses.py")
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(current)

    if noise_factor <= 0.0:
        stage_levels = [0.0]
    elif num_stages == 1:
        stage_levels = [noise_factor]
    else:
        stage_levels = np.geomspace(TRAIN_MIN_NOISE_FACTOR, noise_factor, num_stages).tolist()
    current_stage = 0
    current_step = 0
    current_lr = learning_rate
    history = resume_history or _new_history(item_numbers, stage_levels)
    if resume_history:
        current_step = history["iterations"][-1] if history["iterations"] else 0
        current_stage = history.get("current_stage", 0)
        current_lr = history.get("lr", [learning_rate])[-1]
        optimizer = optax.adam(current_lr)
        optimizer_state = optimizer.init(current)

    training_error_type = {
        "angular": "rad",
        "euclidean": "l2",
        "rooted_euclidean": "sqrtl2",
        "exponential": "exp",
    }.get(loss_type, loss_type)

    def objective(parameters, batch, keys, stage_noise):
        results = jax.vmap(
            lambda inputs, initial, key: jax_run_trial(
                parameters, inputs, initial, DT_MS, SATURATION_RATE_HZ,
                noise_type, stage_noise, key, True
            )
        )(batch["inputs"], batch["initial"], keys)
        readouts = results["readouts"].transpose(1, 0, 2)
        states = results["states"].transpose(1, 0, 2)
        target_output = _target_output(jnp, batch["theta"], batch["presence"])
        train_mean, train_var, eval_mean, eval_var = training_loss(
            readouts, target_output, batch["theta"], batch["presence"], training_error_type
        )
        activation = jnp.mean(jnp.abs(states))
        total = LAMBDA_ERR * train_mean + LAMBDA_REG * activation
        return total, (train_mean, train_var, eval_mean, eval_var, activation)

    best_weights = current
    best_value = np.inf
    steps_without_improvement = 0
    plateau_steps = 0
    for step in range(current_step, steps):
        if current_stage >= len(stage_levels):
            break
        batch = _training_batch(jax, jnp, num_trials, item_numbers, RANDOM_SEED + step, input_strength=INPUT_STRENGTH)
        keys = jax.random.split(jax.random.PRNGKey(RANDOM_SEED + 100000 + step), num_trials)
        compiled_objective = jax.jit(objective, static_argnames=("stage_noise",))
        value_aux, gradients = jax.value_and_grad(compiled_objective, has_aux=True)(current, batch, keys, stage_levels[current_stage])
        value, aux = value_aux
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, current)
        current = optax.apply_updates(current, updates)
        value = float(value)
        train_mean, train_var, eval_mean, eval_var, activation = [float(x) for x in aux]
        history["iterations"].append(step)
        history["overall_errors"].append(eval_mean)
        history["overall_std"].append(float(np.sqrt(eval_var)))
        history["overall_activ"].append(activation)
        history["total_losses"].append(value)
        history["lr"].append(current_lr)
        history["stage"].append(current_stage)
        history["noise_level"].append(stage_levels[current_stage])
        if value < best_value:
            best_value = value
            best_weights = current
            history["best_model_iter"] = step
            history["best_model_loss"] = value
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
        plateau_steps += 1
        if plateau_steps >= adaptive_lr_patience:
            current_lr *= TRAIN_LR_FACTOR
            optimizer = optax.adam(current_lr)
            optimizer_state = optimizer.init(current)
            history["lr_reductions"].append(step)
            plateau_steps = 0
        if steps_without_improvement >= early_stop_patience:
            if current_stage + 1 < len(stage_levels):
                current_stage += 1
                steps_without_improvement = 0
                current_lr = learning_rate
                optimizer = optax.adam(current_lr)
                optimizer_state = optimizer.init(current)
                history["stage_switch_iters"].append(step)
            else:
                history["training_completed"] = True
                break
        if step % logging_period == 0 or step == steps - 1:
            TRAIN_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(TRAIN_SAVE_DIR / f"weights_iteration{step}.npz", **{name: np.asarray(x) for name, x in current.items()})
            (TRAIN_SAVE_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
        print(f"training step {step + 1}/{steps}: total={value:.6g} eval={eval_mean:.6g} stage={current_stage + 1}/{len(stage_levels)} noise={stage_levels[current_stage]:.4g}", end="\r")
    print()
    if show_plot:
        _plot_training(history["total_losses"])
    history["training_completed"] = history.get("training_completed", False) or current_step + len(history["iterations"]) >= steps
    history["best_weights_path"] = str(TRAIN_SAVE_DIR / "weights_best.npz")
    TRAIN_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(TRAIN_SAVE_DIR / "weights_best.npz", **{name: np.asarray(x) for name, x in best_weights.items()})
    (TRAIN_SAVE_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
    return best_weights, history


def save_weights(weights, path=None):
    """Save explicitly supplied weights as `.npz` and return the path."""
    if weights is None:
        raise ValueError("Pass weights explicitly, normally the first result from train()")
    path = Path(path or WEIGHTS_DIR / "weights.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{name: np.asarray(value) for name, value in weights.items()})
    print(f"saved weights to {path}")
    return path


class _TrialAPI:
    """Use run_trial(), run_trial.tloss(), run_trial.aloss(), or run_trial.eloss()."""

    def __call__(self, **kwargs):
        """Return the full report dictionary for one trial."""
        return _trial_report(**kwargs)

    def tloss(self, **kwargs):
        """Return weighted total loss = error loss + activation loss."""
        return _trial_report(**kwargs)["total_loss"]

    def aloss(self, **kwargs):
        """Return activation regularization loss only."""
        return _trial_report(**kwargs)["activation_loss"]

    def eloss(self, **kwargs):
        """Return prediction/error loss only."""
        return _trial_report(**kwargs)["error_loss"]


run_trial = _TrialAPI()

# Example analysis loop (copy into a script/notebook):
# weights = _weights()
# noise = [0.02 * i for i in range(6)]
# activation_loss = [run_trial.aloss(weights=weights, noise_factor=x) for x in noise]
# error_loss = [run_trial.eloss(weights=weights, noise_factor=x) for x in noise]
# plt.plot(noise, activation_loss, label="activation loss")
# plt.plot(noise, error_loss, label="error loss")


# =============================================================================
# PRIVATE HELPERS
# =============================================================================


def _jax():
    try:
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError as error:
        raise SystemExit("Install with: pip install -e 'from_scratch[jax,test]'") from error
    return jax, jnp, optax


def _weights():
    import jax.numpy as jnp
    from vwm_scratch.weights import initialize_weights

    input_dim, output_dim = MAX_ITEMS * (3 if POSITIVE_INPUT else 2), MAX_ITEMS * 2
    if WEIGHTS_SOURCE == "initialize":
        source = initialize_weights(input_dim, NEURONS, output_dim, TAU_MIN_MS, TAU_MAX_MS,
                                    DALE_LAW, POSITIVE_INPUT, INITIALIZATION_SEED)
        return {name: jnp.asarray(getattr(source, name)) for name in ("B", "W", "F", "tau", "dale_sign")}
    if WEIGHTS_SOURCE == "file":
        data = np.load(WEIGHTS_PATH)
        weights = {name: jnp.asarray(data[name]) for name in ("B", "W", "F", "tau", "dale_sign")}
        if weights["B"].shape != (NEURONS, input_dim) or weights["F"].shape != (output_dim, NEURONS):
            raise ValueError(f"Weight shapes do not match NEURONS={NEURONS}, MAX_ITEMS={MAX_ITEMS}")
        return weights
    raise ValueError("WEIGHTS_SOURCE must be 'initialize' or 'file'")


def _steps():
    total = INIT_MS + STIMULUS_MS + DELAY_MS + DECODE_MS
    if total % DT_MS:
        raise ValueError("Total duration must be divisible by DT_MS")
    return int(total / DT_MS)


def _new_history(item_numbers, stage_levels):
    """Create a JSON-serializable history with Derek's main fields."""
    return {
        "iterations": [],
        "overall_errors": [],
        "overall_std": [],
        "overall_activ": [],
        "total_losses": [],
        "lr": [],
        "stage": [],
        "noise_level": [],
        "stage_noise_levels": list(stage_levels),
        "stage_switch_iters": [],
        "group_errors": {str(item): [] for item in item_numbers},
        "group_std": {str(item): [] for item in item_numbers},
        "group_activ": {str(item): [] for item in item_numbers},
        "lr_reductions": [],
        "best_model_iter": None,
        "best_model_loss": None,
        "training_completed": False,
        "current_stage": 0,
    }


def _training_batch(jax, jnp, num_trials, item_numbers, seed, input_strength):
    """Generate Derek's balanced 1..10 set-size batch without Python loops."""
    item_numbers = np.asarray(item_numbers, dtype=np.int32)
    counts = np.full(len(item_numbers), num_trials // len(item_numbers), dtype=np.int32)
    counts[: num_trials % len(item_numbers)] += 1
    set_sizes = np.repeat(item_numbers, counts)
    if len(set_sizes) != num_trials:
        raise ValueError("num_trials must be at least the number of item groups")
    key = jax.random.PRNGKey(seed)
    theta = jax.random.uniform(key, (num_trials, MAX_ITEMS), minval=-jnp.pi, maxval=jnp.pi)
    permutation_keys = jax.random.split(key, num_trials)

    def make_presence(row_key, size):
        permutation = jax.random.permutation(row_key, MAX_ITEMS)
        selected = (jnp.arange(MAX_ITEMS) < size).astype(jnp.float32)
        return jnp.zeros(MAX_ITEMS).at[permutation].set(selected)

    presence = jax.vmap(make_presence)(permutation_keys, jnp.asarray(set_sizes))
    theta_input = theta + SENSORY_NOISE_RAD * jax.random.normal(key, theta.shape)
    if POSITIVE_INPUT:
        encoded = jnp.stack((
            1 + jnp.cos(theta_input) / jnp.sqrt(2) + jnp.sin(theta_input) / jnp.sqrt(6),
            1 - jnp.cos(theta_input) / jnp.sqrt(2) + jnp.sin(theta_input) / jnp.sqrt(6),
            1 - 2 * jnp.sin(theta_input) / jnp.sqrt(6),
        ), axis=-1)
    else:
        encoded = jnp.stack((jnp.cos(theta_input), jnp.sin(theta_input)), axis=-1)
    encoded = (encoded * presence[:, :, None]).reshape(num_trials, -1)
    times = jnp.arange(_steps()) * DT_MS
    mask = ((times >= INIT_MS) & (times < INIT_MS + STIMULUS_MS))[None, :, None]
    inputs = encoded[:, None, :] * mask * input_strength / MAX_ITEMS
    return {"inputs": inputs, "initial": jnp.zeros((num_trials, NEURONS)), "theta": theta, "presence": presence, "set_sizes": jnp.asarray(set_sizes)}


def _trial_data(jax, jnp, set_size, seed, sensory_noise_rad=SENSORY_NOISE_RAD,
                input_strength=INPUT_STRENGTH):
    key = jax.random.PRNGKey(seed)
    theta = jax.random.uniform(key, (MAX_ITEMS,), minval=-jnp.pi, maxval=jnp.pi)
    presence = jnp.concatenate((jnp.ones(set_size), jnp.zeros(MAX_ITEMS - set_size)))
    theta_input = theta + sensory_noise_rad * jax.random.normal(key, theta.shape)
    if POSITIVE_INPUT:
        encoded = jnp.stack((
            1 + jnp.cos(theta_input) / jnp.sqrt(2) + jnp.sin(theta_input) / jnp.sqrt(6),
            1 - jnp.cos(theta_input) / jnp.sqrt(2) + jnp.sin(theta_input) / jnp.sqrt(6),
            1 - 2 * jnp.sin(theta_input) / jnp.sqrt(6),
        ), axis=-1)
    else:
        encoded = jnp.stack((jnp.cos(theta_input), jnp.sin(theta_input)), axis=-1)
    encoded = (encoded * presence[:, None]).reshape(-1)
    times = jnp.arange(_steps()) * DT_MS
    mask = ((times >= INIT_MS) & (times < INIT_MS + STIMULUS_MS))[:, None]
    return key, theta, presence, encoded[None, :] * mask * input_strength / MAX_ITEMS, jnp.zeros(NEURONS)


def _target_output(jnp, theta, presence):
    """Encode target angles for either one trial ``[items]`` or a batch.

    The training batch has ``theta=[trials, items]``, but the public
    ``run_trial`` interface intentionally uses a single ``theta=[items]``.
    Keeping the item axis last avoids treating the item count as a batch axis.
    """
    pairs = jnp.stack((jnp.cos(theta), jnp.sin(theta)), axis=-1)
    target = pairs.reshape(*theta.shape[:-1], theta.shape[-1] * 2)
    return target * jnp.repeat(presence, 2, axis=-1)


def _loss_terms(jnp, result, theta, presence, loss_fn, loss_type):
    from vwm_scratch.trial import decode_readouts
    decode_start = int((INIT_MS + STIMULUS_MS + DELAY_MS) / DT_MS)
    mean_output = result["readouts"][decode_start:].mean(axis=0)
    decoded = decode_readouts(result["readouts"], decode_start)
    target = _target_output(jnp, theta, presence)
    error = loss_fn(decoded, theta, presence) if loss_type == "angular" else loss_fn(mean_output, target, presence)
    penalty = jnp.mean(jnp.abs(result["states"]))
    activation = LAMBDA_REG * penalty
    return LAMBDA_ERR * error + activation, error, penalty, activation, decoded, mean_output


def _trial_report(weights=None, set_size=SET_SIZE, seed=RANDOM_SEED, noise_type=NOISE_TYPE,
                  noise_factor=NOISE_FACTOR, sensory_noise_rad=SENSORY_NOISE_RAD,
                  input_strength=INPUT_STRENGTH, loss_type=LOSS_TYPE, store_states=True):
    jax, jnp, _ = _jax()
    from vwm_scratch.losses import get_loss
    from vwm_scratch.trial import compiled_trial

    weights = _weights() if weights is None else weights
    key, theta, presence, inputs, initial = _trial_data(jax, jnp, set_size, seed, sensory_noise_rad, input_strength)
    result = compiled_trial(weights, inputs, initial, DT_MS, SATURATION_RATE_HZ,
                            noise_type, noise_factor, key, store_states)
    # Training names deliberately match Derek (e.g. ``norml2``), while the
    # one-trial reporting API exposes its corresponding evaluation loss.
    evaluation_loss_type = {
        "rad": "angular", "l2": "euclidean", "sqrtl2": "rooted_euclidean",
        "exp": "exponential", "norml2": "euclidean",
    }.get(loss_type, loss_type)
    loss_fn = get_loss(evaluation_loss_type, CUSTOM_LOSS if loss_type == "custom" else None)
    total, error, penalty, activation, decoded, mean_output = _loss_terms(
        jnp, result, theta, presence, loss_fn, evaluation_loss_type
    )
    active = np.asarray(presence).astype(bool)
    return {
        "total_loss": float(total), "error_loss": float(error),
        "activation_penalty": float(penalty), "activation_loss": float(activation),
        "theta_all_slots": np.asarray(theta).tolist(),
        "theta_present_slots": np.asarray(theta)[active].tolist(),
        "presence": np.asarray(presence).tolist(),
        "decoded_all_slots": np.asarray(decoded).tolist(),
        "decoded_present_slots": np.asarray(decoded)[active].tolist(),
        "mean_output": np.asarray(mean_output).tolist(),
        "conditions": {"set_size": set_size, "seed": seed, "noise_type": noise_type,
                       "noise_factor": noise_factor, "sensory_noise_rad": sensory_noise_rad,
                       "input_strength": input_strength, "loss_type": loss_type,
                       "store_states": store_states},
    }


def _plot_training(history):
    import matplotlib.pyplot as plt
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.plot(history)
    plt.xlabel("Training step")
    plt.ylabel("Total loss")
    plt.title("JAX training progress")
    plt.grid(alpha=0.25)
    plt.savefig(RESULTS_DIR / "training_progress.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Running this file directly performs one visible 1-item demonstration.
    # In your own code, use `print(run_trial.eloss(...))`: the loss methods
    # return numbers and do not print automatically.
    print("Running 1-item trial with the current KNOBS...")
    weights = _weights()
    report = run_trial(weights=weights, set_size=1, store_states=True)
    print(f"total_loss      = {report['total_loss']:.6g}")
    print(f"error_loss      = {report['error_loss']:.6g}")
    print(f"activation_loss = {report['activation_loss']:.6g}")
    print(f"target_present  = {report['theta_present_slots']}")
    print(f"decoded_present = {report['decoded_present_slots']}")

    tloss = np.zeros(100)
    eloss = np.zeros(100)
    aloss = np.zeros(100)
    noise = np.zeros(100)

    for i in range(100):
        # Keep states because activation_loss is mean(abs(state)); setting
        # store_states=False is only valid when plotting error/total loss.
        report = run_trial(weights=weights, set_size=1, seed=i, store_states=True, noise_factor=0.3 * i * 0.02)
        tloss[i] = report["total_loss"]
        eloss[i] = report["error_loss"]
        aloss[i] = report["activation_loss"]
        noise[i] = report["conditions"]["noise_factor"]

    
    plt.plot(noise, tloss, label="total loss")
    plt.plot(noise, eloss, label="error loss")
    plt.plot(noise, aloss, label="activation loss")
    plt.xlabel("Noise factor")
    plt.ylabel("Loss")
    plt.title("Loss vs noise factor")
    plt.legend()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(RESULTS_DIR / "loss_vs_noise.png", dpi=160, bbox_inches="tight")
    plt.show()
    print(f"saved graph to {RESULTS_DIR / 'loss_vs_noise.png'}")
    print("check: total_loss - error_loss - activation_loss =", tloss - eloss - aloss)
