import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

jax_available = importlib.util.find_spec("jax") is not None
pytestmark = pytest.mark.skipif(not jax_available, reason="JAX is not installed")


@pytest.mark.skipif(not jax_available, reason="JAX is not installed")
def test_loss_registry_and_custom_loss():
    import jax.numpy as jnp

    from vwm_scratch.losses import get_loss

    predicted = jnp.array([1.0, 0.0, 0.0, 1.0])
    target = jnp.array([1.0, 0.0, 0.0, 1.0])
    presence = jnp.array([1.0, 1.0])
    assert float(get_loss("euclidean")(predicted, target, presence)) == pytest.approx(0.0)
    assert float(get_loss("rooted_euclidean")(predicted, target, presence)) == pytest.approx(0.0, abs=1e-6)
    assert float(get_loss("exponential")(predicted, target, presence)) == pytest.approx(-1.0)
    custom = lambda output, expected, mask: jnp.sum((output - expected) ** 2)
    assert float(get_loss("custom", custom)(predicted, target, presence)) == pytest.approx(0.0)
