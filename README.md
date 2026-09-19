# VWM RNN from scratch

This folder is an isolated rebuild of the visual working-memory RNN. The first stage contains only the mathematical specification and calibration utilities. It does not train a model, load checkpoints, or run post-hoc analysis.

## Modes

- `calibration`: derive dimensions, timing indices, encoding ranges, and noise constants.
- `data`: later stage for stimulus generation.
- `model`: later stage for recurrent dynamics and readout.
- `training`: later stage for JAX/Optax optimization.
- `evaluation`: later stage for prediction and set-size curves.
- `analysis`: later stage for weights, activity, SNR, PCA, and fixed points.

## Run calibration

From the repository root:

```powershell
python from_scratch/scripts/calibrate.py --config from_scratch/configs/report_optimal.yaml
```

The script writes a JSON record to `from_scratch/calibration_reports/` containing the derived values. The existing project is not imported.

## Structure

```text
from_scratch/
├── configs/                 # Explicit experiment values
├── src/vwm_scratch/
│   ├── calibration.py       # Derived dimensions, timing, encoding, noise
│   └── equations.py         # Pure mathematical functions
│   └── weights.py           # B, W, F matrices and node-link operations
│   └── trial.py             # Single-trial JAX scan, readout, and loss
├── scripts/calibrate.py     # Human-readable calibration entry point
├── tests/test_calibration.py
└── calibration_reports/
```

## Weight conventions

The first model-building layer is `src/vwm_scratch/weights.py`:

```text
B: [neurons, input_dim]   input -> neuron connections
W: [neurons, neurons]     recurrent neuron -> neuron connections
F: [output_dim, neurons]  neuron -> output connections
tau: [neurons]             one time constant per neuron
dale_sign: [neurons]       sign of each source neuron's output
```

For a state stored as `[batch, neurons]`, the operations are:

```text
external_input = inputs @ B.T
recurrent_input = state @ effective_W.T
readout = state @ F.T
```

`effective_W` applies Dale's law by signing recurrent matrix columns. The
`WeightMatrices` type validates these dimensions and exposes the operations as
functions, without depending on PyTorch or JAX. JAX can use the same structure
after the reference behavior is verified.

## Single-trial dynamics

`src/vwm_scratch/trial.py` provides `run_trial(...)`, which takes weights,
`inputs[time, input_dim]`, an initial state, time constants stored in the
weights, and an explicit JAX random key. It returns readouts, the final state,
and optionally the full state trajectory. `trial_loss(...)` wraps the same run
with decode-window averaging and masked circular angular error.

Example shape contract:

```text
inputs       [time, input_dim]
initial      [neurons]
states       [time, neurons]       optional
readouts     [time, output_dim]
final_state  [neurons]
decoded      [max_items]
```

The dynamics use `jax.lax.scan`, so the recurrent state is carried one step at
a time. For training, use `store_states=False` when the loss only needs the
decode readouts. Reverse-mode differentiation still needs intermediate values
somewhere, but JAX can rematerialize them with `jax.checkpoint`/`jax.remat` to
trade extra computation for lower peak memory. Start with the simple scan,
then add rematerialization or truncated backpropagation only after profiling.

The JAX step matches the original dynamics: signed recurrent columns, the
saturating `30 * (1 + tanh(0.14*x - 4.2))` activation, Euler integration
`r_next = r + dt * (-r + rate) / tau`, and noisy readout from the updated
state. Supported noise modes are `gamma`, `gaussian`, `puregauss`, `csnr`, and
`none`, with the same parameter equations as the original implementation.
`compiled_trial(...)` makes the static timing/noise settings compile-time
constants for fast repeated evaluation.

JAX should be introduced after these NumPy reference functions are tested. The reference implementation is the correctness oracle for the later `jax.numpy` implementation.

## One-file control panel

Edit the `KNOBS` section near the top of [run_experiment.py](run_experiment.py),
then import and call the function you need:

```python
from run_experiment import train, save_weights, run_trial

weights, history = train()
save_weights(weights, "from_scratch/weights/my_run.npz")
error = run_trial.eloss(weights=weights, set_size=3)
```

`train()` prints progress and saves/shows a training-loss graph. `save_weights()`
saves an explicitly supplied weight pytree. `run_trial()` returns the full
report, while `run_trial.tloss()`, `run_trial.aloss()`, and `run_trial.eloss()`
return total, activation, and prediction/error loss respectively.

`WEIGHTS_SOURCE` is either `initialize` for reproducible fresh weights or
`file` for a `.npz` produced by this runner. `LOSS_TYPE` accepts `angular`,
`euclidean`, `rooted_euclidean`, `exponential`, or `custom`. For a custom loss,
set `LOSS_TYPE = "custom"` and edit `CUSTOM_LOSS` while preserving:

```python
CUSTOM_LOSS(predicted_output, target_output, presence) -> scalar_jax_value
```

`NOISE_TYPE` and `NOISE_FACTOR` control the end/final recurrent and readout
noise. `SENSORY_NOISE_RAD` controls input-angle noise. `TRAIN_NOISE_TYPE` and
`TRAIN_NOISE_FACTOR` can differ from evaluation noise when calling `train()`.

Trial reports distinguish `theta_all_slots` from `theta_present_slots` because
the simulator stores one angle for every possible output slot but calculates
error only where `presence == 1`. They also print `error_loss`,
`activation_penalty`, `activation_loss`, and `total_loss`. The activation term
uses the original regularizer, `mean(abs(state))`, weighted by `LAMBDA_REG`.
Training keeps `TRAIN_STORE_STATES = True` by default because that regularizer
needs the state trajectory; set it to false only when `LAMBDA_REG = 0`.

## Analysis condition sweeps

Use `run_trial.aloss`, `run_trial.eloss`, or `run_trial.tloss` from
`run_experiment.py` when writing a custom analysis loop. They use the current
default trial conditions unless a keyword is overridden.

```python
import matplotlib.pyplot as plt
import run_experiment as experiment

# Load once, so the loop does not reload weights every time.
weights = experiment._weights()
noise_levels = [0.02 * i for i in range(6)]
activation_loss = []
error_loss = []

for i, noise in enumerate(noise_levels):
	activation_loss.append(experiment.run_trial.aloss(
		weights=weights, noise_factor=noise, seed=experiment.RANDOM_SEED + i))
	error_loss.append(experiment.run_trial.eloss(
		weights=weights, noise_factor=noise, seed=experiment.RANDOM_SEED + i))

plt.plot(noise_levels, activation_loss, label="activation loss")
plt.plot(noise_levels, error_loss, label="error loss")
plt.xlabel("Noise factor")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

Supported per-call overrides are `set_size`, `seed`, `noise_type`,
`noise_factor`, `sensory_noise_rad`, `input_strength`, `loss_type`, and
`store_states`. For activation loss, leave `store_states=True`.

## Selecting pretrained weights

A converted copy of the report's optimal PyTorch checkpoint is included at:

```text
weights/optimal_model_iteration11650.npz
```

To run the from-scratch code with it, change these two knobs in
`run_experiment.py`:

```python
WEIGHTS_SOURCE = "file"
WEIGHTS_PATH = "from_scratch/weights/optimal_model_iteration11650.npz"
```

The file contains `B`, `W`, `F`, `tau`, and `dale_sign`. Provenance and the
source checkpoint hash are recorded in the adjacent JSON metadata file. To
convert another original `.pth` checkpoint, run:

```powershell
python from_scratch/scripts/convert_checkpoint.py `
	--checkpoint path/to/model_iterationXXXX.pth `
	--output from_scratch/weights/my_weights.npz
```
