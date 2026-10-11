import numpy as np

from src.bus_data import get_3bus_data
from src.ieee14_data import get_ieee14_bus_shunts, get_ieee14_data
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


def test_jacobian_matches_finite_difference_on_ieee14_network():
    """Same check on the full benchmark network, which exercises the
    transformer tap model and the bus-9 shunt through ``Y_bus``."""
    num_buses, bus_types, V, theta, _, _, line_data = get_ieee14_data()
    Y_bus = build_ybus(num_buses, line_data, get_ieee14_bus_shunts())

    # Use a non-flat operating point so no derivative term degenerates.
    V = V.copy()
    theta = theta.copy()
    V[[3, 4, 8, 13]] = [1.012, 1.023, 1.051, 1.041]
    theta[1:] = np.radians([-4.9, -12.5, -10.2, -8.7, -14.1, -13.2, -13.2,
                            -14.8, -15.0, -14.7, -15.0, -15.1, -15.9])

    pv_pq = [i for i in range(num_buses) if bus_types[i] != "Slack"]
    pq = [i for i in range(num_buses) if bus_types[i] == "PQ"]
    jacobian = build_jacobian(V, theta, Y_bus, bus_types)
    assert jacobian.shape == (len(pv_pq) + len(pq), len(pv_pq) + len(pq))

    def residuals(state_values):
        trial_V = V.copy()
        trial_theta = theta.copy()
        trial_theta[pv_pq] = state_values[: len(pv_pq)]
        trial_V[pq] = state_values[len(pv_pq):]
        P_calc, Q_calc = calculate_power_injections(trial_V, trial_theta, Y_bus)
        return np.concatenate([P_calc[pv_pq], Q_calc[pq]])

    state = np.concatenate([theta[pv_pq], V[pq]])
    step = 1e-6
    numerical_jacobian = np.zeros_like(jacobian)
    for column in range(len(state)):
        state_plus = state.copy()
        state_minus = state.copy()
        state_plus[column] += step
        state_minus[column] -= step
        numerical_jacobian[:, column] = (
            residuals(state_plus) - residuals(state_minus)
        ) / (2.0 * step)

    np.testing.assert_allclose(jacobian, numerical_jacobian, rtol=1e-6, atol=1e-5)
