import numpy as np
import pytest

from src.ybus_builder import build_ybus


def test_ybus_symmetry_for_transmission_lines():
    line_data = [
        {"from_bus": 1, "to_bus": 2, "r": 0.02, "x": 0.06, "b": 0.06},
        {"from_bus": 1, "to_bus": 3, "r": 0.08, "x": 0.24, "b": 0.05},
        {"from_bus": 2, "to_bus": 3, "r": 0.06, "x": 0.18, "b": 0.04},
    ]

    Y_bus = build_ybus(3, line_data)

    assert Y_bus.shape == (3, 3)
    np.testing.assert_allclose(Y_bus, Y_bus.T)


def test_ybus_diagonal_positive_conductance():
    line_data = [{"from_bus": 1, "to_bus": 2, "r": 0.02, "x": 0.06, "b": 0.06}]

    Y_bus = build_ybus(2, line_data)

    assert np.real(Y_bus[0, 0]) > 0
    assert np.real(Y_bus[1, 1]) > 0


def test_ybus_includes_transformer_tap_and_bus_shunt():
    line_data = [{"from_bus": 1, "to_bus": 2, "r": 0.0, "x": 0.2, "b": 0.04, "tap": 0.95}]
    Y_bus = build_ybus(2, line_data, bus_shunts=[{"bus": 2, "b": 0.1}])

    series_admittance = 1.0 / complex(0.0, 0.2)
    charging_admittance = complex(0.0, 0.02)
    expected = np.array(
        [
            [(series_admittance + charging_admittance) / 0.95**2, -series_admittance / 0.95],
            [-series_admittance / 0.95, series_admittance + charging_admittance + 0.1j],
        ]
    )

    np.testing.assert_allclose(Y_bus, expected)


@pytest.mark.parametrize(
    "line_data",
    [
        [{"from_bus": 1, "to_bus": 2, "r": 0.0, "x": 0.0}],
        [{"from_bus": 1, "to_bus": 2, "r": 0.01, "x": 0.1, "tap": 0.0}],
        [{"from_bus": 1, "to_bus": 3, "r": 0.01, "x": 0.1}],
    ],
)
def test_ybus_rejects_invalid_branch_data(line_data):
    with pytest.raises(ValueError):
        build_ybus(2, line_data)
