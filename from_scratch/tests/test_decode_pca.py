import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

pytestmark = pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX is not installed")


def test_decode_pca_saves_first_slot_decode_samples(tmp_path, monkeypatch):
    import run_experiment as experiment
    from PCA_analysis import decode_pca

    weights = experiment._weights(8, "initialize")
    original_trial_data = experiment._trial_data
    seen_presence = []

    def trial_data(jax, jnp, set_size, seed):
        result = original_trial_data(jax, jnp, set_size, seed)
        seen_presence.append(np.asarray(result[2]))
        return result

    monkeypatch.setattr(experiment, "_trial_data", trial_data)
    data_path, plot_path = decode_pca(
        weights=weights, num_trials=3, samples_per_trial=3,
        seed=42, output_dir=tmp_path, noise_type="none",
    )
    assert plot_path.is_file()
    assert all(presence.tolist() == [1.0] + [0.0] * (experiment.MAX_ITEMS - 1)
               for presence in seen_presence)
    with np.load(data_path) as data:
        assert data["components"].shape == (3, 8)
        assert data["projections"].shape == (3, 3, 3)
        assert data["sample_indices"].tolist() == [152, 176, 201]
        assert np.allclose(data["components"] @ data["components"].T, np.eye(3))
        assert np.all(np.isfinite(data["explained_variance_ratio"]))


def test_decode_pca_rejects_too_many_decode_samples():
    from PCA_analysis import decode_pca

    with pytest.raises(ValueError, match="timesteps in the requested time range"):
        decode_pca(num_trials=1, samples_per_trial=51)


def test_decode_pca_custom_slots_and_time_window(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import jax
    import run_experiment as experiment
    import PCA_analysis
    import jax.numpy as jnp

    weights = experiment._weights(8, "initialize")
    actual_rollout = PCA_analysis.jax_run_trial
    observed_inputs = []

    def rollout(weights, inputs, *args):
        observed_inputs.append(np.asarray(inputs))
        return actual_rollout(weights, inputs, *args)

    monkeypatch.setattr(experiment, "_jax", lambda: (SimpleNamespace(random=jax.random, jit=lambda function: function), jnp, None))
    monkeypatch.setattr(PCA_analysis, "jax_run_trial", rollout)
    data_path, plot_path = PCA_analysis.decode_pca(
        weights=weights, num_trials=2, samples_per_trial=3, seed=13,
        output_dir=tmp_path, noise_type="none", slots=(4, 2),
        time_range_ms=(520, 620), weight_name="my_model.npz",
    )
    assert data_path.name == "pca_my_model_slots2-4_520-620ms_seed13.npz"
    assert plot_path.name == "pca_my_model_slots2-4_520-620ms_seed13.png"
    assert plot_path.is_file()
    with np.load(data_path) as data:
        assert data["sample_indices"].tolist() == [52, 56, 61]
        assert data["active_slots"].tolist() == [2, 4]
        assert data["target_angles"].shape == (2, 2)
        assert data["time_range_ms"].tolist() == [520, 620]
        assert str(data["weight_name"]) == "my_model"
    assert len(observed_inputs) == 2
    stimulus = observed_inputs[0][int(experiment.INIT_MS / experiment.DT_MS)].reshape(experiment.MAX_ITEMS, 3)
    assert np.all(stimulus[[0, 2, *range(4, experiment.MAX_ITEMS)]] == 0)
    assert np.any(stimulus[1] != 0) and np.any(stimulus[3] != 0)


@pytest.mark.parametrize("slots", [(), (0,), (11,), (1, 1)])
def test_decode_pca_rejects_invalid_slots(slots):
    from PCA_analysis import decode_pca

    with pytest.raises(ValueError, match="slots"):
        decode_pca(slots=slots, num_trials=3)


@pytest.mark.parametrize("window", [(-1, 50), (100, 100), (0, 3000)])
def test_decode_pca_rejects_invalid_windows(window):
    from PCA_analysis import decode_pca

    with pytest.raises(ValueError, match="time_range_ms"):
        decode_pca(time_range_ms=window, num_trials=3)


def test_plot_pca_trajectories_colours_each_active_slot(tmp_path, monkeypatch):
    import jax.numpy as jnp
    import run_experiment as experiment
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from PCA_analysis import plot_pca_trajectories
    from vwm_scratch.trial import run_trial

    source = tmp_path / "pca_model_slots2-4_520-620ms_seed13.npz"
    np.savez(source, components=np.eye(3, 8), mean=np.zeros(8),
             active_slots=np.array([1]), noise_type="none", noise_factor=0.0,
             explained_variance_ratio=np.array([0.6, 0.3, 0.1]))
    weights = experiment._weights(8, "initialize")
    jax, jnp, _ = experiment._jax()
    original_trial_data = experiment._trial_data
    recorded_trials = []
    recorded_segments = []
    recorded_colours = []
    original_init = Line3DCollection.__init__

    def capture_trial(jax, jnp, set_size, seed):
        result = original_trial_data(jax, jnp, set_size, seed)
        recorded_trials.append((seed, result))
        return result

    def capture_lines(collection, segments, *args, **kwargs):
        if "colors" in kwargs:
            recorded_segments.append(np.asarray(segments))
            recorded_colours.append(np.asarray(kwargs["colors"]))
        return original_init(collection, segments, *args, **kwargs)

    monkeypatch.setattr(experiment, "_trial_data", capture_trial)
    monkeypatch.setattr(Line3DCollection, "__init__", capture_lines)
    paths = plot_pca_trajectories(
        source, slots=[4, 2], weights=weights, num_trials=2, seed=13, sample_stride=4,
    )
    assert [path.name for path in paths] == [
        f"{source.stem}_trajectories_slots2-4_trials2_stride4_seed13_color_slot4.png",
        f"{source.stem}_trajectories_slots2-4_trials2_stride4_seed13_color_slot2.png",
    ]
    assert all(path.is_file() for path in paths)
    assert [trial[0] for trial in recorded_trials] == [13, 14]
    assert recorded_segments[0].shape == (100, 2, 3)
    assert np.allclose(recorded_colours[0][0, :3], 0)
    angle = float(recorded_trials[0][1][1][3])
    assert np.allclose(recorded_colours[0][49, :3], colormaps["hsv"](Normalize(-np.pi, np.pi)(angle))[:3])
    key, _, _, inputs, _ = recorded_trials[0][1]
    mask = np.repeat(np.array([0, 1, 0, 1] + [0] * (experiment.MAX_ITEMS - 4)), 3)
    states = np.asarray(run_trial(weights, inputs * jnp.asarray(mask), jnp.zeros(8),
                                  experiment.DT_MS, experiment.SATURATION_RATE_HZ,
                                  "none", 0.0, key, True, experiment._steps())["states"])
    assert np.allclose(recorded_segments[0][0, 0], states[2, :3])
    assert np.allclose(recorded_segments[0][49, 1], states[-1, :3])
    with pytest.raises(ValueError, match="slots"):
        plot_pca_trajectories(source, slots=[0], weights=weights)