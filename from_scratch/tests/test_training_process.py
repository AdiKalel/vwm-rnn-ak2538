import importlib.util
import json
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
    monkeypatch.setattr(experiment, "TRAINING_DATA_DIR", tmp_path / "training_data")
    monkeypatch.setattr(experiment, "RESULTS_DIR", tmp_path)

    initial = experiment._weights()
    tau_before = np.asarray(initial["tau"]).copy()
    signs_before = np.asarray(initial["dale_sign"]).copy()
    trained, history = experiment.train(
        steps=3,
        weights=initial,
        num_trials=4,
        item_numbers=(1, 2),
        logging_period=1,
        checkpoint_period=2,
        show_plot=False,
        noise_type="none",
        noise_factor=0.0,
    )
    assert history["iterations"] == [0, 1, 2]
    assert not np.array_equal(np.asarray(trained["B"]), np.asarray(initial["B"]))
    assert not np.array_equal(np.asarray(trained["F"]), np.asarray(initial["F"]))
    assert np.array_equal(np.asarray(trained["tau"]), tau_before)
    assert np.array_equal(np.asarray(trained["dale_sign"]), signs_before)
    assert len(history["overall_train_errors"]) == len(history["iterations"])
    assert len(history["group_train_errors"]["1"]) == len(history["iterations"])
    run_directory = Path(history["run_directory"])
    assert run_directory.is_dir()
    assert not list(run_directory.parent.glob("*.zip"))
    run_files = {path.name for path in run_directory.iterdir()}
    assert {"training_history.json", "weights_final.npz", "latest_checkpoint.pkl",
            "training_summary.png", "weights_checkpoint2.npz"}.issubset(
                run_files
            )
    assert "weights_best.npz" not in run_files
    assert "weights_checkpoint1.npz" not in run_files
    assert "weights_checkpoint3.npz" not in run_files
    assert not list(run_directory.glob("weights_iteration*.npz"))
    saved_history = json.loads((run_directory / "training_history.json").read_text())
    assert saved_history["group_train_errors"]["1"] == history["group_train_errors"]["1"]

    _, resumed = experiment.train(
        steps=5,
        weights=trained,
        num_trials=4,
        item_numbers=(1, 2),
        logging_period=1,
        show_plot=False,
        noise_type="none",
        noise_factor=0.0,
        resume_from=run_directory,
    )
    assert resumed["iterations"] == [0, 1, 2, 3, 4]
    assert resumed["run_directory"] != history["run_directory"]


@pytest.mark.skipif(not jax_available, reason="JAX is not installed")
def test_manual_stage_promotion_preserves_history_and_marks_stage(tmp_path, monkeypatch):
    import run_experiment as experiment

    monkeypatch.setattr(experiment, "WEIGHTS_SOURCE", "initialize")
    monkeypatch.setattr(experiment, "NEURONS", 8)
    monkeypatch.setattr(experiment, "MAX_ITEMS", 2)
    monkeypatch.setattr(experiment, "TRAIN_SAVE_DIR", tmp_path / "training")
    monkeypatch.setattr(experiment, "TRAINING_DATA_DIR", tmp_path / "training_data")
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
        resume_from=stage_one["run_directory"],
    )
    assert stage_two["stage"][-2:] == [1, 1]
    assert stage_two["stage_switch_iters"] == [2]
    assert stage_two["iterations"] == [0, 1, 2, 3]
    assert not stage_two["training_completed"]
