import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from vwm_scratch.weights import WeightMatrices, initialize_weights


def test_report_weight_shapes_and_reproducibility():
    first = initialize_weights(30, 64, 20, 50.0, 300.0, seed=7)
    second = initialize_weights(30, 64, 20, 50.0, 300.0, seed=7)
    assert first.summary()["B_shape"] == [64, 30]
    assert first.summary()["W_shape"] == [64, 64]
    assert first.summary()["F_shape"] == [20, 64]
    assert np.array_equal(first.B, second.B)
    assert np.array_equal(first.W, second.W)
    assert np.all(first.tau >= 50.0)
    assert np.all(first.tau <= 300.0)


def test_matrix_links_use_batch_by_node_convention():
    weights = WeightMatrices(
        B=np.array([[1.0, 2.0], [3.0, 4.0]]),
        W=np.array([[10.0, 20.0], [30.0, 40.0]]),
        F=np.array([[5.0, 6.0]]),
        tau=np.array([10.0, 20.0]),
        dale_sign=np.array([1.0, -1.0]),
    )
    weights.validate()
    state = np.array([[2.0, 3.0]])
    inputs = np.array([[7.0, 11.0]])
    assert np.array_equal(weights.external_input(inputs), [[29.0, 65.0]])
    assert np.array_equal(weights.recurrent_input(state), [[-40.0, -60.0]])
    assert np.array_equal(weights.readout(state), [[28.0]])


def test_dale_sign_changes_source_columns():
    weights = WeightMatrices(
        B=np.ones((2, 1)),
        W=np.ones((2, 2)),
        F=np.ones((1, 2)),
        tau=np.ones(2),
        dale_sign=np.array([1.0, -1.0]),
    )
    assert np.array_equal(weights.effective_W, [[1.0, -1.0], [1.0, -1.0]])
