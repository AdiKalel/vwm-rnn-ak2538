import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

jax_available = importlib.util.find_spec("jax") is not None
pytestmark = pytest.mark.skipif(not jax_available, reason="JAX is not installed")


@pytest.mark.skipif(not jax_available, reason="JAX is not installed")
def test_training_updates_only_learned_matrices_and_resume_appends_steps(tmp_path, monkeypatch):
    import run_experiment as experiment

    monkeypatch.setattr(experiment, "WEIGHTS_SOURCE", "initialize")
    monkeypatch.setattr(experiment, "NEURONS", 8)
    monkeypatch.setattr(experiment, "MAX_ITEMS", 2)
    monkeypatch.setattr(experiment, "TRAIN_SAVE_DIR", tmp_path / "training")
    monkeypatch.setattr(experiment, "RESULTS_DIR", tmp_path)

    initial = experiment._weights()
    tau_before = np.asarray(initial["tau"]).copy()
    signs_before = np.asarray(initial["dale_sign"]).copy()
    trained, history = experiment.train(
        steps=2,
        weights=initial,
        num_trials=4,
        item_numbers=(1, 2),
        logging_period=1,
        show_plot=False,
        noise_type="none",
        noise_factor=0.0,
    )
    assert history["iterations"] == [0, 1]
    assert not np.array_equal(np.asarray(trained["B"]), np.asarray(initial["B"]))
    assert not np.array_equal(np.asarray(trained["F"]), np.asarray(initial["F"]))
    assert np.array_equal(np.asarray(trained["tau"]), tau_before)
    assert np.array_equal(np.asarray(trained["dale_sign"]), signs_before)

    _, resumed = experiment.train(
        steps=4,
        weights=trained,
        num_trials=4,
        item_numbers=(1, 2),
        logging_period=1,
        show_plot=False,
        noise_type="none",
        noise_factor=0.0,
        resume_from=tmp_path / "training" / "latest_checkpoint.pkl",
    )
    assert resumed["iterations"] == [0, 1, 2, 3]


@pytest.mark.skipif(not jax_available, reason="JAX is not installed")
def test_manual_stage_promotion_preserves_history_and_marks_stage(tmp_path, monkeypatch):
    import run_experiment as experiment

    monkeypatch.setattr(experiment, "WEIGHTS_SOURCE", "initialize")
    monkeypatch.setattr(experiment, "NEURONS", 8)
    monkeypatch.setattr(experiment, "MAX_ITEMS", 2)
    monkeypatch.setattr(experiment, "TRAIN_SAVE_DIR", tmp_path / "training")
    monkeypatch.setattr(experiment, "RESULTS_DIR", tmp_path)
    initial = experiment._weights()

    _, stage_one = experiment.train(
        steps=2, stage=1, num_stages=3, weights=initial, num_trials=4,
        item_numbers=(1, 2), logging_period=1, checkpoint_period=1,
        verification_period=1, show_plot=False, noise_type="none", noise_factor=0.3,
    )
    assert stage_one["stage"] == [0, 0]
    assert stage_one["stage_budget_exhausted"]
    assert not stage_one["training_completed"]
    assert stage_one["verification_steps"] == [1, 2]

    _, stage_two = experiment.train(
        steps=4, stage=2, num_stages=3, num_trials=4,
        item_numbers=(1, 2), logging_period=1, checkpoint_period=1,
        verification_period=1, show_plot=False, noise_type="none", noise_factor=0.3,
        resume_from=tmp_path / "training" / "latest_checkpoint.pkl",
    )
    assert stage_two["stage"][-2:] == [1, 1]
    assert stage_two["stage_switch_iters"] == [2]
    assert stage_two["iterations"] == [0, 1, 2, 3]
    assert not stage_two["training_completed"]
