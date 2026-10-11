import numpy as np
import pytest

from src.bus_data import get_3bus_data
from src.ieee14_data import get_ieee14_bus_shunts, get_ieee14_data
from src.nr_solver import solve_newton_raphson
from src.ybus_builder import build_ybus

THREE_BUS_LINES = [
    {"from_bus": 1, "to_bus": 2, "r": 0.02, "x": 0.06, "b": 0.06},
    {"from_bus": 1, "to_bus": 3, "r": 0.08, "x": 0.24, "b": 0.05},
    {"from_bus": 2, "to_bus": 3, "r": 0.06, "x": 0.18, "b": 0.04},
]


def test_3bus_convergence_and_input_arrays_are_not_changed():
    bus_types, V_init, theta_init, P_spec, Q_spec = get_3bus_data()
    Y_bus = build_ybus(3, THREE_BUS_LINES)
    original_V = V_init.copy()
    original_theta = theta_init.copy()

    V_conv, theta_conv, history = solve_newton_raphson(
        V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types, max_iter=20, tol=1e-4
    )

    assert len(history) < 10
    assert history[-1] < 1e-4
    np.testing.assert_array_equal(V_init, original_V)
    np.testing.assert_array_equal(theta_init, original_theta)
    assert np.issubdtype(V_conv.dtype, np.floating)
    assert np.issubdtype(theta_conv.dtype, np.floating)


def test_ieee14_convergence_with_transformers_and_shunt():
    num_buses, bus_types, V_init, theta_init, P_spec, Q_spec, line_data = get_ieee14_data()
    Y_bus = build_ybus(num_buses, line_data, get_ieee14_bus_shunts())

    V_conv, _, history = solve_newton_raphson(
        V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types, max_iter=20, tol=1e-4
    )

    assert len(history) < 10
    assert history[-1] < 1e-4
    np.testing.assert_allclose(V_conv[8], 1.056, atol=0.002)


def test_solver_reports_nonconvergence_after_requested_iterations():
    bus_types, V_init, theta_init, P_spec, Q_spec = get_3bus_data()
    Y_bus = build_ybus(3, THREE_BUS_LINES)

    with pytest.raises(RuntimeError, match="did not converge"):
        solve_newton_raphson(
            V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types, max_iter=0, tol=1e-8
        )


def test_solver_rejects_invalid_bus_types():
    bus_types, V_init, theta_init, P_spec, Q_spec = get_3bus_data()
    Y_bus = build_ybus(3, THREE_BUS_LINES)
    bus_types[1] = "Generator"

    with pytest.raises(ValueError, match="bus types"):
        solve_newton_raphson(V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types)
