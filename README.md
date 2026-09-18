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

JAX should be introduced after these NumPy reference functions are tested. The reference implementation is the correctness oracle for the later `jax.numpy` implementation.

## One-file control panel

Edit [run_experiment.py](run_experiment.py) and change only the `KNOBS`
section near the top. Then run:

```powershell
python from_scratch/run_experiment.py
```

`MODE` selects the operation:

```text
calibrate       derive and print system calibration
trial           run one chosen set-size trial
train           optimise B, W, F, and tau with Optax
performance     average loss over set sizes 1..MAX_ITEMS
weight_analysis inspect shapes, norms, and effective Dale-signed W
save_weights    save the selected weights as results/weights.npz
```

`WEIGHTS_SOURCE` is either `initialize` for reproducible fresh weights or
`file` for a `.npz` produced by this runner. `LOSS_TYPE` accepts `angular`,
`euclidean`, `rooted_euclidean`, `exponential`, or `custom`. For a custom loss,
set `LOSS_TYPE = "custom"` and edit `CUSTOM_LOSS` while preserving:

```python
CUSTOM_LOSS(predicted_output, target_output, presence) -> scalar_jax_value
```

`NOISE_TYPE` and `NOISE_FACTOR` control the end/final recurrent and readout
noise. `SENSORY_NOISE_RAD` controls input-angle noise. `TRAIN_NOISE_TYPE` and
`TRAIN_NOISE_FACTOR` can differ from evaluation noise when `MODE = "train"`.
`SAVE_TRIAL`, `SAVE_WEIGHTS`, and `SAVE_RESULTS` control artifacts in
`from_scratch/results/`.
