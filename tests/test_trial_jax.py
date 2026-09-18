"""Tests for the JAX trial API; skipped when JAX is not installed."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

jax_available = importlib.util.find_spec("jax") is not None
pytestmark = pytest.mark.skipif(not jax_available, reason="JAX is not installed")


@pytest.mark.skipif(not jax_available, reason="JAX is not installed")
def test_trial_shapes_and_gradient():
    import jax
    import jax.numpy as jnp

    from vwm_scratch.trial import trial_loss, run_trial
    from vwm_scratch.weights import initialize_weights

    reference = initialize_weights(3, 4, 2, 50.0, 300.0, seed=3)
    weights = {name: jnp.asarray(getattr(reference, name)) for name in ("B", "W", "F", "tau", "dale_sign")}
    inputs = jnp.zeros((8, 3))
    initial_state = jnp.zeros(4)
    result = run_trial(weights, inputs, initial_state, 10.0, noise_type="none")
    assert result["states"].shape == (8, 4)
    assert result["readouts"].shape == (8, 2)
    assert result["final_state"].shape == (4,)

    target = jnp.zeros(1)
    presence = jnp.ones(1)
    value, gradient = jax.value_and_grad(
        lambda current: trial_loss(
            current, inputs, initial_state, target, presence, 0, 10.0, noise_type="none"
        )[0]
    )(weights)
    assert jnp.isfinite(value)
    assert all(jnp.all(jnp.isfinite(array)) for array in gradient.values())


@pytest.mark.skipif(not jax_available, reason="JAX is not installed")
def test_no_noise_dynamics_match_reference_matrix_equations():
    import jax
    import jax.numpy as jnp

    from vwm_scratch.trial import compiled_trial
    from vwm_scratch.weights import initialize_weights

    reference = initialize_weights(2, 3, 2, 50.0, 300.0, seed=11)
    weights = {name: jnp.asarray(getattr(reference, name)) for name in ("B", "W", "F", "tau", "dale_sign")}
    inputs = jnp.array([[0.1, 0.2], [0.0, 0.0], [0.3, -0.1]], dtype=jnp.float32)
    initial = jnp.array([0.2, 0.4, 0.1], dtype=jnp.float32)
    result = compiled_trial(weights, inputs, initial, 10.0, noise_type="none", key=jax.random.PRNGKey(0))

    state = np.asarray(initial)
    effective_w = np.asarray(reference.W) * np.asarray(reference.dale_sign)[None, :]
    expected_states = []
    expected_readouts = []
    for input_t in np.asarray(inputs):
        recurrent = effective_w @ state
        external = np.asarray(reference.B) @ input_t
        rate = 30.0 * (1.0 + np.tanh(0.14 * (recurrent + external) - 4.2))
        state = state + 10.0 * (-state + rate) / np.asarray(reference.tau)
        expected_states.append(state.copy())
        expected_readouts.append(np.asarray(reference.F) @ state)

    assert np.allclose(result["states"], np.asarray(expected_states), atol=1e-5)
    assert np.allclose(result["readouts"], np.asarray(expected_readouts), atol=1e-5)


@pytest.mark.skipif(not jax_available, reason="JAX is not installed")
def test_supported_noise_modes_return_finite_trials():
    import jax
    import jax.numpy as jnp

    from vwm_scratch.trial import compiled_trial
    from vwm_scratch.weights import initialize_weights

    reference = initialize_weights(2, 4, 2, 50.0, 300.0, seed=5)
    weights = {name: jnp.asarray(getattr(reference, name)) for name in ("B", "W", "F", "tau", "dale_sign")}
    inputs = jnp.zeros((3, 2))
    initial = jnp.ones(4)
    for mode in ("gamma", "gaussian", "puregauss", "csnr"):
        result = compiled_trial(weights, inputs, initial, 10.0, noise_type=mode, noise_factor=0.2, key=jax.random.PRNGKey(2))
        assert bool(jnp.isfinite(result["states"]).all())
        assert bool(jnp.isfinite(result["readouts"]).all())
