import numpy as np

from src.bus_data import get_3bus_data
from src.jacobian_builder import build_jacobian
from src.mismatch_calculator import calculate_power_injections
from src.ybus_builder import build_ybus


def test_jacobian_matches_finite_difference_power_derivatives():
    bus_types, V, theta, _, _ = get_3bus_data()
    line_data = [
        {"from_bus": 1, "to_bus": 2, "r": 0.02, "x": 0.06, "b": 0.06},
        {"from_bus": 1, "to_bus": 3, "r": 0.08, "x": 0.24, "b": 0.05},
        {"from_bus": 2, "to_bus": 3, "r": 0.06, "x": 0.18, "b": 0.04},
    ]
    Y_bus = build_ybus(3, line_data)
    jacobian = build_jacobian(V, theta, Y_bus, bus_types)

    # State order: angle at buses 2 and 3, then voltage magnitude at bus 3.
    state = np.array([theta[1], theta[2], V[2]])

    def calculated_power(state_values):
        trial_V = V.copy()
        trial_theta = theta.copy()
        trial_theta[1:] = state_values[:2]
        trial_V[2] = state_values[2]
        P_calc, Q_calc = calculate_power_injections(trial_V, trial_theta, Y_bus)
        return np.array([P_calc[1], P_calc[2], Q_calc[2]])

    step = 1e-6
    numerical_jacobian = np.zeros_like(jacobian)
    for column in range(len(state)):
        state_plus = state.copy()
        state_minus = state.copy()
        state_plus[column] += step
        state_minus[column] -= step
        numerical_jacobian[:, column] = (
            calculated_power(state_plus) - calculated_power(state_minus)
        ) / (2.0 * step)

    np.testing.assert_allclose(jacobian, numerical_jacobian, rtol=1e-6, atol=1e-6)
