import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
jax_available = importlib.util.find_spec("jax") is not None
pytestmark = pytest.mark.skipif(not jax_available, reason="JAX is not installed")


def test_training_loss_matches_expected_l2_and_angular_metrics():
    import jax.numpy as jnp

    from vwm_scratch.losses import training_loss

    target_angles = jnp.array([[0.0, 0.0], [np.pi / 2, 0.0]])
    presence = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    target = jnp.stack((jnp.cos(target_angles), jnp.sin(target_angles)), axis=-1).reshape(2, 4)
    predictions = jnp.stack((target, target))
    train_mean, train_var, eval_mean, eval_var = training_loss(
        predictions, target, target_angles, presence, "l2"
    )
    assert float(train_mean) == pytest.approx(0.0, abs=1e-6)
    assert float(train_var) == pytest.approx(0.0, abs=1e-6)
    assert float(eval_mean) == pytest.approx(0.0, abs=1e-6)
    assert float(eval_var) == pytest.approx(0.0, abs=1e-6)


def test_training_losses_return_finite_values_for_report_shapes():
    import jax.numpy as jnp

    from vwm_scratch.losses import training_loss

    batch, items, time = 4, 3, 5
    angles = jnp.linspace(-2.0, 2.0, batch * items).reshape(batch, items)
    presence = jnp.ones((batch, items))
    target = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1).reshape(batch, -1)
    prediction = jnp.broadcast_to(target[None, ...], (time, batch, 2 * items)) * 0.8
    for error_type in ("l2", "sqrtl2", "norml2", "rad", "exp"):
        losses = training_loss(prediction, target, angles, presence, error_type)
        assert all(np.isfinite(float(value)) for value in losses)
