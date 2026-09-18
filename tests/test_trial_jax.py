"""Tests for the JAX trial API; skipped when JAX is not installed."""

import importlib.util
import sys
from pathlib import Path

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
