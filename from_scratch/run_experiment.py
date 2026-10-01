"""Compact callable control surface for the from-scratch JAX model.

Use from the repository root:

    import sys
    sys.path.insert(0, "from_scratch")
    from run_experiment import train, save_weights, run_trial

    weights, history = train()
    save_weights(weights, "from_scratch/weights/my_run.npz")
    error = run_trial.eloss(weights=weights, set_size=3)

Edit the KNOBS block below. There are no mode switches: call the function you
want. All trial functions accept keyword overrides and otherwise use defaults.
"""

from __future__ import annotations

import sys
import json
import pickle
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
# Derek's active project config uses 40,000 iterations; lower this for a smoke run.
TRAIN_STEPS = 40000
LEARNING_RATE = 1e-4
LAMBDA_ERR, LAMBDA_REG = 1.0, 1e-5
TRAIN_SET_SIZE = 1
TRAIN_NOISE_TYPE, TRAIN_NOISE_FACTOR = "gamma", 0.3
TRAIN_NUM_TRIALS = 300
TRAIN_ITEM_NUM = tuple(range(1, MAX_ITEMS + 1))
TRAIN_LOGGING_PERIOD = 10
TRAIN_CHECKPOINT_PERIOD = 100
TRAIN_VERIFY_PERIOD = 100
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
        checkpoint_period=TRAIN_CHECKPOINT_PERIOD,
        verification_period=TRAIN_VERIFY_PERIOD,
        early_stop_patience=TRAIN_EARLY_STOP_PATIENCE,
        adaptive_lr_patience=TRAIN_ADAPTIVE_LR_PATIENCE,
        num_stages=TRAIN_NUM_STAGES, stage=None, resume_from=None):
    """Train with Derek's mixed-set-size curriculum and checkpoint behavior.

    The returned history contains overall and per-set-size evaluation errors,
    standard deviations, activation, learning rate, stage/noise, and best-step
    metadata. Checkpoints and ``training_history.json`` are written under
    ``TRAIN_SAVE_DIR`` every ``logging_period`` iterations.

    Leave ``stage=None`` to use automatic stage transitions. Set ``stage=1``
    through ``stage=num_stages`` to train only that noise stage, then pass the
    returned weights into the next call. ``steps`` is the total target
    iteration for automatic/resumed training and the step budget for a fresh
    manual stage. ``verification_period`` controls a separate fixed validation
    batch; ``checkpoint_period`` controls full resumable checkpoint writes.
    """
    jax, jnp, optax = _jax()
    from vwm_scratch.losses import training_loss
    from vwm_scratch.trial import run_trial as jax_run_trial

    print(f"JAX training devices: {jax.devices()}")

    current = _weights() if weights is None else weights
    fixed_parameters = {name: current[name] for name in ("tau", "dale_sign")}
    trainable = {name: current[name] for name in ("B", "W", "F")}
    if loss_type == "custom":
        raise ValueError("Custom loss training requires adding it to training_loss in losses.py")
    optimizer = optax.scale_by_adam()
    optimizer_state = optimizer.init(trainable)

    if noise_factor <= 0.0:
        stage_levels = [0.0]
    elif num_stages == 1:
        stage_levels = [noise_factor]
    else:
        stage_levels = np.geomspace(TRAIN_MIN_NOISE_FACTOR, noise_factor, num_stages).tolist()
    train_signature = {
        "loss_type": loss_type,
        "learning_rate": learning_rate,
        "num_trials": num_trials,
        "item_numbers": list(item_numbers),
        "noise_type": noise_type,
        "stage_noise_levels": list(stage_levels),
        "lambda_err": LAMBDA_ERR,
        "lambda_reg": LAMBDA_REG,
        "dt_ms": DT_MS,
        "input_strength": INPUT_STRENGTH,
        "max_items": MAX_ITEMS,
        "neurons": NEURONS,
    }
    if stage is not None and not 1 <= stage <= len(stage_levels):
        raise ValueError(f"stage must be between 1 and {len(stage_levels)}")
    current_stage = stage - 1 if stage is not None else 0
    current_step = 0
    current_lr = learning_rate
    stage_best_value = np.inf
    global_best_value = np.inf
    steps_without_improvement = 0
    plateau_steps = 0
    history = _new_history(item_numbers, stage_levels)
    history["current_stage"] = current_stage
    history["stage_only"] = stage is not None
    history["verification_steps"] = []
    history["verification_errors"] = []
    best_weights = current
    if resume_from is not None:
        with _resolve_local_path(resume_from).open("rb") as file_handle:
            checkpoint = pickle.load(file_handle)
        if checkpoint["stage_noise_levels"] != stage_levels:
            raise ValueError("Checkpoint noise curriculum differs from current noise_factor or num_stages")
        if checkpoint.get("train_signature") != train_signature:
            raise ValueError("Checkpoint training settings differ; resume with the original loss, batch, noise, and model settings")
        current = {name: jnp.asarray(value) for name, value in checkpoint["weights"].items()}
        fixed_parameters = {name: current[name] for name in ("tau", "dale_sign")}
        trainable = {name: current[name] for name in ("B", "W", "F")}
        optimizer_state = jax.tree.map(jnp.asarray, checkpoint["optimizer_state"])
        history = checkpoint["history"]
        current_step = checkpoint["next_step"]
        checkpoint_stage = checkpoint["current_stage"]
        if stage is None or stage == checkpoint_stage + 1:
            current_stage = checkpoint_stage
            current_lr = checkpoint["current_lr"]
            stage_best_value = checkpoint["stage_best_value"]
            steps_without_improvement = checkpoint["steps_without_improvement"]
            plateau_steps = checkpoint["plateau_steps"]
        elif stage == checkpoint_stage + 2:
            current_stage = checkpoint_stage + 1
            current_lr = learning_rate
            stage_best_value = np.inf
            steps_without_improvement = 0
            plateau_steps = 0
            history["current_stage"] = current_stage
            history["stage_completed"] = False
            history["stage_budget_exhausted"] = False
            history["stage_switch_iters"].append(current_step)
        else:
            raise ValueError(
                f"Checkpoint is at stage {checkpoint_stage + 1}; resume at that stage "
                f"or promote exactly one stage to {checkpoint_stage + 2}."
            )
        global_best_value = checkpoint["global_best_value"]
        best_weights = {name: jnp.asarray(value) for name, value in checkpoint["best_weights"].items()}

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
        decoded = jnp.arctan2(readouts.mean(axis=0).reshape(num_trials, MAX_ITEMS, 2)[..., 1],
                              readouts.mean(axis=0).reshape(num_trials, MAX_ITEMS, 2)[..., 0])
        angular_delta = (batch["theta"] - decoded + jnp.pi) % (2 * jnp.pi) - jnp.pi
        trial_errors = jnp.sum(jnp.abs(angular_delta) * batch["presence"], axis=1) / jnp.sum(batch["presence"], axis=1)
        trial_activation = jnp.mean(jnp.abs(results["states"]), axis=(1, 2))
        group_errors = jnp.stack([
            jnp.sum(trial_errors * (batch["set_sizes"] == item)) / jnp.maximum(jnp.sum(batch["set_sizes"] == item), 1)
            for item in item_numbers
        ])
        group_vars = jnp.stack([
            jnp.sum((trial_errors - group_errors[i]) ** 2 * (batch["set_sizes"] == item)) /
            jnp.maximum(jnp.sum(batch["set_sizes"] == item) - 1, 1)
            for i, item in enumerate(item_numbers)
        ])
        group_activations = jnp.stack([
            jnp.sum(trial_activation * (batch["set_sizes"] == item)) /
            jnp.maximum(jnp.sum(batch["set_sizes"] == item), 1)
            for item in item_numbers
        ])
        return total, (train_mean, train_var, eval_mean, eval_var, activation, group_errors, group_vars, group_activations)

    def trainable_objective(parameters, batch, keys, stage_noise):
        return objective({**parameters, **fixed_parameters}, batch, keys, stage_noise)

    compiled_objective = jax.jit(trainable_objective, static_argnames=("stage_noise",))
    verification_batch = _training_batch(
        jax, jnp, num_trials, item_numbers, RANDOM_SEED + 900000,
        input_strength=INPUT_STRENGTH,
    )
    verification_keys = jax.random.split(jax.random.PRNGKey(RANDOM_SEED + 900001), num_trials)
    stage_call_completed = False
    starting_history_count = len(history["iterations"])
    for step in range(current_step, steps):
        if current_stage >= len(stage_levels):
            break
        batch = _training_batch(jax, jnp, num_trials, item_numbers, RANDOM_SEED + step, input_strength=INPUT_STRENGTH)
        keys = jax.random.split(jax.random.PRNGKey(RANDOM_SEED + 100000 + step), num_trials)
        value_aux, gradients = jax.value_and_grad(compiled_objective, has_aux=True)(trainable, batch, keys, stage_levels[current_stage])
        value, aux = value_aux
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, trainable)
        updates = jax.tree.map(lambda update: -current_lr * update, updates)
        trainable = optax.apply_updates(trainable, updates)
        if POSITIVE_INPUT:
            trainable["B"] = jnp.maximum(trainable["B"], 0.0)
        if DALE_LAW:
            trainable["W"] = jnp.maximum(trainable["W"], 0.0)
        current = {**trainable, **fixed_parameters}
        value = float(value)
        train_mean, train_var, eval_mean, eval_var, activation, group_errors, group_vars, group_activations = aux
        group_errors = np.asarray(group_errors).tolist()
        group_stds = np.sqrt(np.asarray(group_vars)).tolist()
        group_activations = np.asarray(group_activations).tolist()
        train_mean, train_var, eval_mean, eval_var, activation = [float(x) for x in (train_mean, train_var, eval_mean, eval_var, activation)]
        history["iterations"].append(step)
        history["overall_errors"].append(eval_mean)
        history["overall_std"].append(float(np.sqrt(eval_var)))
        history["overall_activ"].append(activation)
        history["total_losses"].append(value)
        history["lr"].append(current_lr)
        history["stage"].append(current_stage)
        history["noise_level"].append(stage_levels[current_stage])
        history["current_stage"] = current_stage
        for index, item in enumerate(item_numbers):
            history["group_errors"][str(item)].append(group_errors[index])
            history["group_std"][str(item)].append(group_stds[index])
            history["group_activ"][str(item)].append(group_activations[index])
        if value < global_best_value:
            global_best_value = value
            best_weights = current
            history["best_model_iter"] = step
            history["best_model_loss"] = value
        if value < stage_best_value:
            stage_best_value = value
            steps_without_improvement = 0
            plateau_steps = 0
        else:
            steps_without_improvement += 1
            plateau_steps += 1
        if plateau_steps >= adaptive_lr_patience:
            current_lr *= TRAIN_LR_FACTOR
            history["lr_reductions"].append(step)
            plateau_steps = 0
        if steps_without_improvement >= early_stop_patience:
            if stage is not None:
                history["stage_completed"] = True
                stage_call_completed = True
                break
            if current_stage + 1 < len(stage_levels):
                current_stage += 1
                stage_best_value = np.inf
                steps_without_improvement = 0
                current_lr = learning_rate
                history["stage_switch_iters"].append(step)
            else:
                history["training_completed"] = True
                break
        if verification_period > 0 and ((step + 1) % verification_period == 0 or step == current_step):
            _, verification_aux = compiled_objective(
                trainable, verification_batch, verification_keys, 0.0
            )
            validation_error = float(verification_aux[2])
            history["verification_steps"].append(step + 1)
            history["verification_errors"].append(validation_error)
            print()
            print(f"  verification at step {step + 1}: angular_error={validation_error:.6g}")
        if step % logging_period == 0 or step == steps - 1:
            TRAIN_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(TRAIN_SAVE_DIR / f"weights_iteration{step}.npz", **{name: np.asarray(x) for name, x in current.items()})
            (TRAIN_SAVE_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
        if checkpoint_period > 0 and ((step + 1) % checkpoint_period == 0 or step == steps - 1):
            _save_training_checkpoint(
                TRAIN_SAVE_DIR / "latest_checkpoint.pkl", current, optimizer_state,
                history, step + 1, current_stage, current_lr, stage_best_value,
                global_best_value, steps_without_improvement, plateau_steps,
                best_weights, stage_levels, train_signature,
            )
        print(f"training step {step + 1}/{steps}: total={value:.6g} eval={eval_mean:.6g} stage={current_stage + 1}/{len(stage_levels)} noise={stage_levels[current_stage]:.4g}", end="\r")
    print()
    if show_plot:
        _plot_training(history)
    history["stage_completed"] = history.get("stage_completed", False) or stage_call_completed
    history["steps_this_call"] = len(history["iterations"]) - starting_history_count
    history["requested_total_steps"] = steps
    history["stage_budget_exhausted"] = bool(
        stage is not None and not history["stage_completed"] and history["steps_this_call"] >= steps
    )
    history["step_budget_exhausted"] = bool(
        history["iterations"] and history["iterations"][-1] + 1 >= steps
    )
    # Completion means the final curriculum stage converged, not merely that
    # this invocation used its requested step budget.
    history["training_completed"] = bool(history.get("training_completed", False))
    history["best_weights_path"] = str(TRAIN_SAVE_DIR / "weights_best.npz")
    TRAIN_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(TRAIN_SAVE_DIR / "weights_best.npz", **{name: np.asarray(x) for name, x in best_weights.items()})
    (TRAIN_SAVE_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
    _save_training_checkpoint(
        TRAIN_SAVE_DIR / "latest_checkpoint.pkl", current, optimizer_state,
        history, (history["iterations"][-1] + 1) if history["iterations"] else current_step,
        current_stage, current_lr, stage_best_value, global_best_value,
        steps_without_improvement, plateau_steps, best_weights, stage_levels,
        train_signature,
    )
    return best_weights, history


def save_weights(weights, path=None):
    """Save explicitly supplied weights as `.npz` and return the path."""
    if weights is None:
        raise ValueError("Pass weights explicitly, normally the first result from train()")
    path = _resolve_local_path(path or WEIGHTS_DIR / "weights.npz")
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
        raise SystemExit(
            "JAX/Optax dependencies are missing. From inside the from_scratch directory run:\n"
            "  pip install -e '.[jax,test]'\n"
            "Or, from the repository root run:\n"
            "  pip install -e 'from_scratch[jax,test]'"
        ) from error
    return jax, jnp, optax


def _resolve_local_path(path):
    """Resolve defaults relative to this script, independent of shell cwd.

    For convenience, user-supplied paths beginning with ``from_scratch/``
    also work when the script is launched from the repository root.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    return ROOT / path


def _weights():
    import jax.numpy as jnp
    from vwm_scratch.weights import initialize_weights

    input_dim, output_dim = MAX_ITEMS * (3 if POSITIVE_INPUT else 2), MAX_ITEMS * 2
    if WEIGHTS_SOURCE == "initialize":
        source = initialize_weights(input_dim, NEURONS, output_dim, TAU_MIN_MS, TAU_MAX_MS,
                                    DALE_LAW, POSITIVE_INPUT, INITIALIZATION_SEED)
        return {name: jnp.asarray(getattr(source, name)) for name in ("B", "W", "F", "tau", "dale_sign")}
    if WEIGHTS_SOURCE == "file":
        weights_path = _resolve_local_path(WEIGHTS_PATH)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Weights not found: {weights_path}. Check WEIGHTS_PATH; "
                "relative paths are resolved from the from_scratch folder."
            )
        data = np.load(weights_path)
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


def _save_training_checkpoint(path, weights, optimizer_state, history, next_step,
                             current_stage, current_lr, stage_best_value,
                             global_best_value, steps_without_improvement,
                             plateau_steps, best_weights, stage_levels,
                             train_signature):
    """Atomically save enough state to continue the exact Optax run."""
    import jax

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": jax.tree.map(np.asarray, weights),
        "optimizer_state": jax.tree.map(np.asarray, optimizer_state),
        "history": history,
        "next_step": next_step,
        "current_stage": current_stage,
        "current_lr": current_lr,
        "stage_best_value": stage_best_value,
        "global_best_value": global_best_value,
        "steps_without_improvement": steps_without_improvement,
        "plateau_steps": plateau_steps,
        "best_weights": jax.tree.map(np.asarray, best_weights),
        "stage_noise_levels": list(stage_levels),
        "train_signature": train_signature,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file_handle:
        pickle.dump(payload, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)


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
    iterations = history["iterations"]
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(iterations, history["overall_errors"], label="evaluation angular error")
    axes[0].fill_between(
        iterations,
        np.asarray(history["overall_errors"]) - np.asarray(history["overall_std"]),
        np.asarray(history["overall_errors"]) + np.asarray(history["overall_std"]),
        alpha=0.2,
        label="error +/- std",
    )
    axes[0].set_ylabel("Error (rad)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(iterations, history["overall_activ"], color="tab:orange", label="mean absolute activation")
    axes[1].set_ylabel("Activation")
    axes[1].set_xlabel("Training iteration")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("Training progress")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "training_progress.png", dpi=160)
    plt.close(figure)

    group_figure, group_axis = plt.subplots(figsize=(9, 5))
    for item, values in history["group_errors"].items():
        group_axis.plot(iterations, values, label=f"{item} item(s)")
    group_axis.set_xlabel("Training iteration")
    group_axis.set_ylabel("Evaluation angular error (rad)")
    group_axis.set_title("Error by set size")
    group_axis.grid(alpha=0.25)
    group_axis.legend(ncol=2)
    group_figure.tight_layout()
    group_figure.savefig(RESULTS_DIR / "training_error_by_set_size.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Running this file directly performs one visible 1-item demonstration.
    # In your own code, use `print(run_trial.eloss(...))`: the loss methods
    # return numbers and do not print automatically.
    print("multi-item trial with the current KNOBS...")
    weights = _weights()
    report = run_trial(weights=weights, set_size=1, store_states=True)
    print(f"total_loss      = {report['total_loss']:.6g}")
    print(f"error_loss      = {report['error_loss']:.6g}")
    print(f"activation_loss = {report['activation_loss']:.6g}")
    print(f"target_present  = {report['theta_present_slots']}")
    print(f"decoded_present = {report['decoded_present_slots']}")

    tloss = np.zeros(10)
    eloss = np.zeros(10)
    aloss = np.zeros(10)
    items = np.zeros(10)

    for i in range(10):
        # Keep states because activation_loss is mean(abs(state)); setting
        # store_states=False is only valid when plotting error/total loss.
        total_eloss = 0
        total_aloss = 0
        for j in range(10):
            report = run_trial(weights=weights, set_size= i + 1, seed=i*10 + j, store_states=True)
            total_eloss += report["error_loss"]
            print(f"set_size={i + 1}, seed={i*10 + j}, error_loss={report['error_loss']:.6g}, activation_loss={report['activation_loss']:.6g}")
            total_aloss += report["activation_loss"]
        eloss[i] = total_eloss / 10
        aloss[i] = total_aloss / 10
        items[i] = report["conditions"]["set_size"]

    
    plt.plot(items, eloss, label="error loss")
    plt.xlabel("Set size")
    plt.ylabel("Error loss")
    plt.title("Loss vs set size")
    plt.legend()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    #plt.savefig(RESULTS_DIR / "loss_vs_set_size.png", dpi=160, bbox_inches="tight")
    plt.show()

    plt.plot(items, aloss, label="activation loss")
    plt.xlabel("Set size")
    plt.ylabel("Activation loss")
    plt.title("Loss vs set size")
    plt.legend()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    #plt.savefig(RESULTS_DIR / "loss_vs_set_size.png", dpi=160, bbox_inches="tight")
    plt.show()
    #print(f"saved graph to {RESULTS_DIR / 'loss_vs_set_size.png'}")
    print("check: total_loss - error_loss - activation_loss =", tloss - eloss - aloss)
    print('this is changed')
    train(steps=100, show_plot=True, num_trials=100, item_numbers=(1,), logging_period=10)
