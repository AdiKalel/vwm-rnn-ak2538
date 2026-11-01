# Derek's pretrained weights

These are converted from the original PyTorch checkpoints into `.npz` files
for the JAX implementation. Each `.npz` has an adjacent `.json` file with the
source checkpoint hash and the complete original YAML conditions.

## Available runs

| File | Checkpoint | Neurons | Training loss | Spike noise | Sensory noise | Input strength |
|---|---:|---:|---|---:|---:|---:|
| `derek_check_l2_n64_gamma02_sensory001.npz` | iteration 8450 | 64 | L2 | gamma 0.2 | 0.01 rad | 100 |
| `derek_optimal_l2_n64_gamma02.npz` | iteration 11650 | 64 | L2 | gamma 0.2 | 0.00 rad | 120 |
| `derek_rad_n256_gamma03.npz` | iteration 2300 | 256 | angular/radian | gamma 0.3 | 0.00 rad | 120 |

All three were trained for the 1-to-10 item task with positive three-channel
input, Dale's law, `dt = 10 ms`, `tau_min = 50 ms`, `tau_max = 300 ms`, and
phase durations of `20 / 500 / 1000 / 500 ms` for init/stimulus/delay/decode.
The paired JSON files contain the exact settings; the `rad` label corresponds
to the original config's `error_def: rad`.

## Select one in the runner

Edit `from_scratch/run_experiment.py`:

```python
# Fresh random weights:
WEIGHTS_SOURCE = "initialize"

# Derek's optimal 64-neuron run:
WEIGHTS_SOURCE = "file"
WEIGHTS_PATH = "from_scratch/weights/derek_optimal_l2_n64_gamma02.npz"
```

For the other runs, replace `WEIGHTS_PATH` with one of:

```python
"from_scratch/weights/derek_check_l2_n64_gamma02_sensory001.npz"
"from_scratch/weights/derek_rad_n256_gamma03.npz"
```

When using a saved run, set the runner's `NOISE_TYPE`, `NOISE_FACTOR`,
`SENSORY_NOISE_RAD`, `INPUT_STRENGTH`, and `NEURONS` to match the selected
row. In particular, the 256-neuron file requires `NEURONS = 256` and the
`derek_check_l2` file uses `INPUT_STRENGTH = 100` and sensory noise `0.01`.
