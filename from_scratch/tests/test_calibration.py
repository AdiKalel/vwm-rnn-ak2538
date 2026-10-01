import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from vwm_scratch.calibration import derive_calibration, load_config
from vwm_scratch.equations import circular_difference, decode_orientation, positive_orientation_encoding


CONFIG = Path(__file__).parents[1] / "configs" / "report_optimal.yaml"


def test_report_dimensions_and_timing():
    derived = derive_calibration(load_config(CONFIG))
    assert derived["input_dim"] == 30
    assert derived["output_dim"] == 20
    assert derived["simulation_steps"] == 202
    assert derived["phase_indices"] == {
        "init": (0, 2),
        "stimulus": (2, 52),
        "delay": (52, 152),
        "decode": (152, 202),
    }


def test_positive_encoding_is_non_negative():
    theta = np.linspace(-np.pi, np.pi, 101)
    encoded = positive_orientation_encoding(theta)
    assert encoded.shape == (101, 3)
    assert encoded.min() >= -1e-12


def test_angle_round_trip_and_wrap():
    theta = np.array([-np.pi, -1.0, 0.0, 1.0, np.pi])
    cos_sin = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    assert np.allclose(circular_difference(decode_orientation(cos_sin), theta), 0.0, atol=1e-12)
