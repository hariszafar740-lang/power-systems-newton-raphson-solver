"""Regression tests for Newton-Raphson failure handling and diagnostics.

A failed solve must never be reported as converged: the solver raises
``RuntimeError`` for singular Jacobians and for runs that exhaust
``max_iter`` without reaching ``tol``, and ``ValueError`` for invalid
inputs.  The exception messages carry the actual final residual.
"""

import re

import numpy as np
import pytest

from src.bus_data import get_3bus_data
from src.ieee14_data import get_ieee14_bus_shunts, get_ieee14_data
from src.mismatch_calculator import calculate_power_mismatches
from src.nr_solver import solve_newton_raphson
from src.ybus_builder import build_ybus

THREE_BUS_LINES = [
    {"from_bus": 1, "to_bus": 2, "r": 0.02, "x": 0.06, "b": 0.06},
    {"from_bus": 1, "to_bus": 3, "r": 0.08, "x": 0.24, "b": 0.05},
    {"from_bus": 2, "to_bus": 3, "r": 0.06, "x": 0.18, "b": 0.04},
]


def _three_bus():
    bus_types, V_init, theta_init, P_spec, Q_spec = get_3bus_data()
    Y_bus = build_ybus(3, THREE_BUS_LINES)
    return bus_types, V_init, theta_init, P_spec, Q_spec, Y_bus


def test_convergence_history_describes_the_actual_computation():
    bus_types, V_init, theta_init, P_spec, Q_spec, Y_bus = _three_bus()
    V, theta, history = solve_newton_raphson(
        V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types, max_iter=20, tol=1e-8
    )

    # History starts with the mismatch of the initial state...
    _, _, delta_P0, delta_Q0 = calculate_power_mismatches(
        V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types
    )
    initial_mismatch = np.max(np.abs(np.concatenate([delta_P0, delta_Q0])))
    assert history[0] == pytest.approx(initial_mismatch)

    # ...ends with the mismatch of the returned state, and is consistent
    # with the returned solution (residual recomputed from scratch).
    _, _, delta_P, delta_Q = calculate_power_mismatches(
        V, theta, Y_bus, P_spec, Q_spec, bus_types
    )
    final_mismatch = np.max(np.abs(np.concatenate([delta_P, delta_Q])))
    assert history[-1] == pytest.approx(final_mismatch)
    assert history[-1] < 1e-8
    assert len(history) <= 21


def test_singular_jacobian_is_reported_as_a_failure():
    # Bus 3 is a PQ bus connected to nothing, so the Jacobian has an all
    # zero row and column block and is exactly singular.
    bus_types = ["Slack", "PV", "PQ"]
    V_init = np.array([1.05, 1.03, 1.0])
    theta_init = np.zeros(3)
    P_spec = np.array([0.0, 0.5, -0.5])
    Q_spec = np.array([0.0, 0.0, -0.25])
    Y_bus = build_ybus(3, THREE_BUS_LINES[:1])  # only the 1-2 branch

    with pytest.raises(RuntimeError, match="singular"):
        solve_newton_raphson(
            V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types, max_iter=20, tol=1e-8
        )


def test_nonconvergent_case_raises_with_truthful_residual():
    # A load far beyond the nose point of this two-bus link has no power
    # flow solution, so Newton-Raphson cannot converge.
    bus_types = ["Slack", "PQ"]
    V_init = np.array([1.0, 1.0])
    theta_init = np.zeros(2)
    P_spec = np.array([0.0, -10.0])
    Q_spec = np.array([0.0, -5.0])
    Y_bus = build_ybus(2, [{"from_bus": 1, "to_bus": 2, "r": 0.05, "x": 0.2, "b": 0.0}])

    with pytest.raises(RuntimeError) as excinfo:
        solve_newton_raphson(
            V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types, max_iter=15, tol=1e-6
        )
    message = str(excinfo.value)
    assert "did not converge" in message
    reported_residual = float(re.search(r"maximum mismatch is ([0-9.eE+-]+)", message).group(1))
    # The reported residual must be the honest one: at or above tolerance.
    assert reported_residual >= 1e-6


def test_failed_solve_never_returns_a_success_tuple():
    bus_types, V_init, theta_init, P_spec, Q_spec, Y_bus = _three_bus()
    with pytest.raises(RuntimeError):
        solve_newton_raphson(
            V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types, max_iter=0, tol=1e-12
        )


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda args: args[0].append("PV"), "one entry"),  # bus_types length
        (lambda args: args.__setitem__(1, args[1][:-1]), "equal length"),  # short theta
        (lambda args: args.__setitem__(3, args[3][:-1]), "equal length"),
    ],
    ids=["bus_types_length", "short_theta", "short_p_spec"],
)
def test_solver_rejects_inconsistent_inputs(mutate, match):
    bus_types, V_init, theta_init, P_spec, Q_spec, Y_bus = _three_bus()
    args = [bus_types, V_init.copy(), theta_init.copy(), P_spec.copy(), Q_spec.copy()]
    mutate(args)
    with pytest.raises(ValueError, match=match):
        solve_newton_raphson(args[1], args[2], Y_bus, args[3], args[4], args[0])


def test_solver_rejects_nan_specified_power():
    bus_types, V_init, theta_init, P_spec, Q_spec, Y_bus = _three_bus()
    P_spec = P_spec.copy()
    P_spec[2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solve_newton_raphson(V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types)


def test_solver_rejects_nonpositive_initial_voltage():
    bus_types, V_init, theta_init, P_spec, Q_spec, Y_bus = _three_bus()
    V_init = V_init.copy()
    V_init[2] = 0.0
    with pytest.raises(ValueError, match="positive"):
        solve_newton_raphson(V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types)


def test_solver_rejects_mismatched_ybus_shape():
    bus_types, V_init, theta_init, P_spec, Q_spec, _ = _three_bus()
    with pytest.raises(ValueError, match="Y_bus"):
        solve_newton_raphson(
            V_init, theta_init, np.eye(4, dtype=complex), P_spec, Q_spec, bus_types
        )


def test_solver_rejects_nonfinite_ybus():
    bus_types, V_init, theta_init, P_spec, Q_spec, Y_bus = _three_bus()
    Y_bus = Y_bus.copy()
    Y_bus[0, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        solve_newton_raphson(V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types)


@pytest.mark.parametrize("kwargs, match", [
    ({"max_iter": -1}, "max_iter"),
    ({"max_iter": 2.5}, "max_iter"),
    ({"tol": 0.0}, "tol"),
    ({"tol": -1e-4}, "tol"),
    ({"tol": np.nan}, "tol"),
])
def test_solver_rejects_invalid_iteration_controls(kwargs, match):
    bus_types, V_init, theta_init, P_spec, Q_spec, Y_bus = _three_bus()
    with pytest.raises(ValueError, match=match):
        solve_newton_raphson(V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types, **kwargs)


def test_ieee14_failure_status_is_never_masked_by_the_data_model():
    # Regression of the historical defect: with the wrong branch data the
    # solver "converged" to voltages ~0.028 p.u. away from the reference.
    # With the corrected data the tight solve finishes well inside 1e-10.
    num_buses, bus_types, V_init, theta_init, P_spec, Q_spec, line_data = get_ieee14_data()
    Y_bus = build_ybus(num_buses, line_data, get_ieee14_bus_shunts())
    _, _, history = solve_newton_raphson(
        V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types, max_iter=20, tol=1e-10
    )
    assert history[-1] < 1e-10
    assert len(history) <= 10
