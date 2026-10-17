import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

pytestmark = pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")

TRAIN_ARGS = dict(
    steps=1, num_trials=4, item_numbers=(1, 2), logging_period=1,
    show_plot=False, noise_type="none", noise_factor=0.0,
)


def _configure(experiment, monkeypatch, tmp_path, source):
    monkeypatch.setattr(experiment, "WEIGHTS_SOURCE", source)
    monkeypatch.setattr(experiment, "WEIGHTS_PATH", tmp_path / "missing.npz")
    monkeypatch.setattr(experiment, "NEURONS", 8)
    monkeypatch.setattr(experiment, "MAX_ITEMS", 2)
    monkeypatch.setattr(experiment, "TRAIN_SAVE_DIR", tmp_path / "training")
    monkeypatch.setattr(experiment, "RESULTS_DIR", tmp_path)


@pytest.mark.parametrize("source", ["initialize", "file"])
def test_train_neurons_sets_network_size_and_trained_weights_evaluate(tmp_path, monkeypatch, source):
    import run_experiment as experiment

    _configure(experiment, monkeypatch, tmp_path, source)
    trained, _ = experiment.train(neurons=6, **TRAIN_ARGS)
    assert trained["W"].shape == (6, 6)
    assert trained["B"].shape[0] == 6 and trained["F"].shape[1] == 6
    assert experiment.run_trial.eloss(weights=trained, set_size=1) >= 0


def test_train_neurons_must_match_supplied_weights(tmp_path, monkeypatch):
    import run_experiment as experiment

    _configure(experiment, monkeypatch, tmp_path, "initialize")
    with pytest.raises(ValueError, match="does not match"):
        experiment.train(weights=experiment._weights(), neurons=4, **TRAIN_ARGS)
