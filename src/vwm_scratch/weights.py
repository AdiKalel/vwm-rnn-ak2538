"""Weight and node structure for the from-scratch VWM RNN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WeightShapes:
    """Dimensions of the three learned maps and recurrent state."""

    input_dim: int
    neurons: int
    output_dim: int

    @property
    def input_to_neuron(self) -> tuple[int, int]:
        return self.neurons, self.input_dim

    @property
    def recurrent(self) -> tuple[int, int]:
        return self.neurons, self.neurons

    @property
    def neuron_to_output(self) -> tuple[int, int]:
        return self.output_dim, self.neurons


@dataclass
class WeightMatrices:
    """All fixed and learned arrays needed by one RNN instance.

    Conventions follow the mathematical maps used by the model:

    - ``B``: input to neurons, shape ``[neurons, input_dim]``
    - ``W``: raw recurrent weights, shape ``[neurons, neurons]``
    - ``F``: neurons to output, shape ``[output_dim, neurons]``
    - ``tau``: one time constant per neuron, shape ``[neurons]``
    - ``dale_sign``: +1 or -1 per source neuron, shape ``[neurons]``

    The state is conventionally stored as ``[batch, neurons]``. With that
    convention, recurrent input is ``state @ effective_W.T`` and external
    input is ``inputs @ B.T``.
    """

    B: Any
    W: Any
    F: Any
    tau: Any
    dale_sign: Any

    @property
    def shapes(self) -> WeightShapes:
        return WeightShapes(
            input_dim=self.B.shape[1],
            neurons=self.B.shape[0],
            output_dim=self.F.shape[0],
        )

    @property
    def effective_W(self):
        """Apply Dale's sign to each source-neuron column of ``W``."""
        return self.W * self.dale_sign[None, :]

    def validate(self) -> None:
        """Raise a clear error if matrix/node dimensions are inconsistent."""
        shapes = self.shapes
        expected = {
            "W": shapes.recurrent,
            "F": shapes.neuron_to_output,
            "tau": (shapes.neurons,),
            "dale_sign": (shapes.neurons,),
        }
        actual = {
            "W": self.W.shape,
            "F": self.F.shape,
            "tau": self.tau.shape,
            "dale_sign": self.dale_sign.shape,
        }
        for name, expected_shape in expected.items():
            if actual[name] != expected_shape:
                raise ValueError(f"{name} has shape {actual[name]}, expected {expected_shape}")
        if not np.all(np.isin(np.asarray(self.dale_sign), (-1, 1))):
            raise ValueError("dale_sign must contain only -1 and +1")
        if np.any(np.asarray(self.tau) <= 0):
            raise ValueError("tau must be positive")

    def recurrent_input(self, state):
        """Compute recurrent input for state arrays shaped ``[batch, neurons]``."""
        return state @ self.effective_W.T

    def external_input(self, inputs):
        """Compute external input for arrays shaped ``[batch, input_dim]``."""
        return inputs @ self.B.T

    def readout(self, state):
        """Map neural state arrays shaped ``[batch, neurons]`` to outputs."""
        return state @ self.F.T

    def summary(self) -> dict[str, Any]:
        """Return serializable structure information for debugging and reports."""
        self.validate()
        return {
            "input_dim": self.shapes.input_dim,
            "neurons": self.shapes.neurons,
            "output_dim": self.shapes.output_dim,
            "B_shape": list(self.B.shape),
            "W_shape": list(self.W.shape),
            "F_shape": list(self.F.shape),
            "tau_shape": list(self.tau.shape),
            "dale_sign_shape": list(self.dale_sign.shape),
            "effective_W_shape": list(self.effective_W.shape),
        }


def initialize_weights(
    input_dim: int,
    neurons: int,
    output_dim: int,
    tau_min: float,
    tau_max: float,
    dale_law: bool = True,
    positive_input: bool = True,
    seed: int = 0,
) -> WeightMatrices:
    """Create reproducible reference weights using the report's initialization scales."""
    if input_dim < 1 or neurons < 1 or output_dim < 1:
        raise ValueError("input_dim, neurons, and output_dim must be positive")
    if tau_min <= 0 or tau_max < tau_min:
        raise ValueError("tau range must satisfy 0 < tau_min <= tau_max")

    rng = np.random.default_rng(seed)
    input_scale = 0.418
    recurrent_scale = 1.0 / np.sqrt(neurons * 0.682) if dale_law else 1.0 / np.sqrt(neurons)
    readout_scale = np.sqrt(2.0 / (neurons + output_dim))

    B = rng.normal(0.0, input_scale, size=(neurons, input_dim))
    if positive_input:
        B = np.abs(B)
    W = rng.normal(0.0, recurrent_scale, size=(neurons, neurons))
    if dale_law:
        W = np.abs(W)
    F = rng.normal(0.0, readout_scale, size=(output_dim, neurons))
    tau = np.exp(rng.uniform(np.log(tau_min), np.log(tau_max), size=neurons))
    dale_sign = np.ones(neurons) if not dale_law else np.where(np.arange(neurons) < neurons // 2, 1.0, -1.0)

    weights = WeightMatrices(B=B, W=W, F=F, tau=tau, dale_sign=dale_sign)
    weights.validate()
    return weights
