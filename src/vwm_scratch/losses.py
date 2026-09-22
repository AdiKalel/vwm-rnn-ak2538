"""Loss functions and a registry shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _jax():
    try:
        import jax.numpy as jnp
    except ImportError as error:
        raise ImportError("Install JAX with: pip install -e '.[jax,test]'") from error
    return jnp


def circular_difference(predicted, target):
    jnp = _jax()
    return (target - predicted + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _present_mean(values, presence):
    jnp = _jax()
    return jnp.sum(values * presence) / jnp.sum(presence)


def angular(predicted, target, presence):
    """Mean absolute wrapped angular error in radians."""
    jnp = _jax()
    return _present_mean(jnp.abs(circular_difference(predicted, target)), presence)


def euclidean(predicted_output, target_output, presence):
    """Mean squared Euclidean error on cosine/sine output vectors."""
    jnp = _jax()
    predicted = predicted_output.reshape(-1, 2)
    target = target_output.reshape(-1, 2)
    mask = jnp.repeat(presence, 2).reshape(-1, 2)
    per_item = jnp.sum((predicted - target) ** 2, axis=-1)
    return jnp.sum(per_item * presence.reshape(-1)) / jnp.sum(presence)


def rooted_euclidean(predicted_output, target_output, presence):
    """Mean square-root Euclidean error on cosine/sine output vectors."""
    jnp = _jax()
    predicted = predicted_output.reshape(-1, 2)
    target = target_output.reshape(-1, 2)
    per_item = jnp.sqrt(jnp.sum((predicted - target) ** 2, axis=-1) + 1e-12)
    return jnp.sum(per_item * presence.reshape(-1)) / jnp.sum(presence)


def exponential(predicted_output, target_output, presence):
    """Psychophysical similarity loss; lower values are better."""
    jnp = _jax()
    predicted = predicted_output.reshape(-1, 2)
    target = target_output.reshape(-1, 2)
    predicted = predicted / (jnp.linalg.norm(predicted, axis=-1, keepdims=True) + 1e-8)
    target = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
    delta = jnp.arccos(jnp.clip(jnp.sum(predicted * target, axis=-1), -1.0, 1.0))
    similarity = jnp.exp(-3.0 * delta / jnp.pi)
    return -jnp.sum(similarity * presence.reshape(-1)) / jnp.sum(presence)


BUILTIN_LOSSES: dict[str, Callable[..., Any]] = {
    "angular": angular,
    "euclidean": euclidean,
    "rooted_euclidean": rooted_euclidean,
    "exponential": exponential,
}


def training_loss(readouts, target_output, target_theta, presence, error_type):
    """Match Derek's training losses for readouts [time, batch, output]."""
    jnp = _jax()
    steps = readouts.shape[0]
    predicted = readouts.reshape(steps, readouts.shape[1], -1, 2)
    target = target_output.reshape(target_output.shape[0], -1, 2)
    if error_type == "l2":
        per_item = jnp.sum((target - predicted.mean(axis=0)) ** 2, axis=-1)
    elif error_type == "sqrtl2":
        per_item = 10.0 * jnp.sqrt(jnp.linalg.norm(target - predicted.mean(axis=0), axis=-1))
    elif error_type == "norml2":
        pred_norm = predicted / (jnp.linalg.norm(predicted, axis=-1, keepdims=True) + 1e-12)
        target_norm = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + 1e-12)
        per_item = jnp.linalg.norm(target_norm - pred_norm.mean(axis=0), axis=-1)
    elif error_type in ("rad", "exp"):
        pred_norm = predicted / (jnp.linalg.norm(predicted, axis=-1, keepdims=True) + 1e-12)
        dot = jnp.sum(target * pred_norm.mean(axis=0), axis=-1)
        delta = jnp.arccos(jnp.clip(dot, -1.0, 1.0))
        per_item = delta if error_type == "rad" else -10.0 * jnp.exp(-3.0 * delta / jnp.pi)
    else:
        raise ValueError("error_type must be l2, sqrtl2, norml2, rad, or exp")
    per_trial = jnp.sum(per_item * presence, axis=-1) / jnp.sum(presence, axis=-1)
    train_mean = jnp.mean(per_trial)
    train_var = jnp.var(per_trial)
    decoded = jnp.arctan2(predicted.mean(axis=0)[..., 1], predicted.mean(axis=0)[..., 0])
    eval_error = circular_difference(decoded, target_theta)
    eval_per_trial = jnp.sum(jnp.abs(eval_error) * presence, axis=-1) / jnp.sum(presence, axis=-1)
    return train_mean, train_var, jnp.mean(eval_per_trial), jnp.var(eval_per_trial)


def get_loss(name: str, custom_loss: Callable[..., Any] | None = None):
    """Resolve a built-in loss or return the supplied custom callable."""
    if name == "custom":
        if custom_loss is None:
            raise ValueError("LOSS_TYPE='custom' requires CUSTOM_LOSS")
        return custom_loss
    try:
        return BUILTIN_LOSSES[name]
    except KeyError as error:
        valid = ", ".join((*BUILTIN_LOSSES, "custom"))
        raise ValueError(f"Unknown loss '{name}'. Choose: {valid}") from error
