"""Single-trial JAX dynamics and differentiable loss helpers."""

from __future__ import annotations

from typing import Any, Mapping


def _jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        raise ImportError(
            "JAX is required for vwm_scratch.trial. Install with "
            "pip install -e '.[jax]'"
        ) from error
    return jax, jnp


def _parameter(weights: Any, name: str):
    if isinstance(weights, Mapping):
        return weights[name]
    return getattr(weights, name)


def _activation(x, saturation_rate_hz: float):
    _, jnp = _jax()
    if saturation_rate_hz > 0:
        return saturation_rate_hz / 2.0 * (1.0 + jnp.tanh(0.14 * x - 4.2))
    return 0.18 * jnp.maximum(x - 10.0, 1e-10) ** 1.7


def _observed_state(key, rates, noise_type: str, noise_factor: float, dt_ms: float):
    jax, jnp = _jax()
    if noise_factor <= 0.0 or noise_type == "none":
        return rates
    if noise_type == "gamma":
        lam = (dt_ms / 1000.0) / noise_factor**2
        shape = jnp.maximum(rates * lam, 1e-10)
        return jax.random.gamma(key, shape) / lam
    if noise_type == "gaussian":
        noise = noise_factor * jnp.sqrt(rates * 1000.0 / dt_ms + 1e-10)
        return jnp.maximum(rates + noise * jax.random.normal(key, rates.shape), 0.0)
    raise ValueError("noise_type must be 'none', 'gamma', or 'gaussian'")


def run_trial(
    weights: Any,
    inputs,
    initial_state,
    dt_ms: float,
    saturation_rate_hz: float = 60.0,
    noise_type: str = "none",
    noise_factor: float = 0.0,
    key=None,
    store_states: bool = True,
):
    """Run one trial with differentiable JAX dynamics.

    Parameters
    ----------
    weights:
        A mapping or object with ``B``, ``W``, ``F``, ``tau``, and
        ``dale_sign``. Shapes are ``B=[N,I]``, ``W=[N,N]``, ``F=[O,N]``,
        ``tau=[N]``, and ``dale_sign=[N]``.
    inputs:
        Array with shape ``[time, input_dim]``.
    initial_state:
        Array with shape ``[neurons]``.
    store_states:
        If true, return the full state trajectory. Set false when only the
        final state/readout is needed; the scan still remains differentiable.

    Returns
    -------
    dict
        ``states`` is ``[time, neurons]`` when stored, otherwise ``None``;
        ``readouts`` is always ``[time, output_dim]``; ``final_state`` is
        ``[neurons]``. The dictionary is suitable for a loss function.
    """
    jax, jnp = _jax()
    B = _parameter(weights, "B")
    W = _parameter(weights, "W")
    F = _parameter(weights, "F")
    tau = _parameter(weights, "tau")
    dale_sign = _parameter(weights, "dale_sign")
    if key is None:
        key = jax.random.PRNGKey(0)
    effective_W = W * dale_sign[None, :]
    keys = jax.random.split(key, inputs.shape[0])

    def step(state, values):
        input_t, key_t = values
        recurrent_key, readout_key = jax.random.split(key_t)
        recurrent = effective_W @ _observed_state(
            recurrent_key, state, noise_type, noise_factor, dt_ms
        )
        external = B @ input_t
        rate = _activation(recurrent + external, saturation_rate_hz)
        next_state = state + dt_ms * (-state + rate) / tau
        readout = F @ _observed_state(
            readout_key, next_state, noise_type, noise_factor, dt_ms
        )
        return next_state, (next_state, readout) if store_states else readout

    final_state, scan_output = jax.lax.scan(step, initial_state, (inputs, keys))
    if store_states:
        states, readouts = scan_output
    else:
        states, readouts = None, scan_output
    return {
        "states": states if store_states else None,
        "readouts": readouts,
        "final_state": final_state,
    }


def decode_readouts(readouts, decode_start: int = 0):
    """Average a decode window and convert alternating cos/sin pairs to angles."""
    _, jnp = _jax()
    window = readouts[decode_start:]
    pairs = window.reshape(window.shape[0], -1, 2).mean(axis=0)
    return jnp.arctan2(pairs[:, 1], pairs[:, 0])


def angular_loss(decoded, target, presence):
    """Mean wrapped angular error over present items for one trial."""
    _, jnp = _jax()
    difference = (target - decoded + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    return jnp.sum(jnp.abs(difference) * presence) / jnp.sum(presence)


def trial_loss(
    weights: Any,
    inputs,
    initial_state,
    target,
    presence,
    decode_start: int,
    dt_ms: float,
    **run_options,
):
    """Run one trial and return a scalar loss plus prediction data."""
    result = run_trial(weights, inputs, initial_state, dt_ms, **run_options)
    decoded = decode_readouts(result["readouts"], decode_start)
    return angular_loss(decoded, target, presence), {**result, "decoded": decoded}


def value_and_grad_trial_loss(*args, **kwargs):
    """Return a JAX value-and-gradient function over the weight pytree."""
    jax, _ = _jax()
    return jax.value_and_grad(lambda weights: trial_loss(weights, *args, **kwargs)[0])
