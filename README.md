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
├── scripts/calibrate.py     # Human-readable calibration entry point
├── tests/test_calibration.py
└── calibration_reports/
```

JAX should be introduced after these NumPy reference functions are tested. The reference implementation is the correctness oracle for the later `jax.numpy` implementation.
