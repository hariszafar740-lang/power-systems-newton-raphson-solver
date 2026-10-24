"""Regression and physics verification tests for PV-to-PQ reactive power
limit enforcement (Q_min / Q_max switching)."""

import numpy as np
import pytest

from src.bus_data import get_3bus_data
from src.ieee14_data import (
    get_ieee14_bus_shunts,
    get_ieee14_data,
    get_ieee14_q_limits,
    scale_ieee14_loads,
)
from src.ieee30_data import (
    get_ieee30_bus_shunts,
    get_ieee30_data,
    scale_ieee30_loads,
)
from src.nr_solver import solve_newton_raphson
from src.q_limit_solver import solve_nr_with_q_limits
from src.ybus_builder import build_ybus_sparse

CASE14_115X_VM_REF = np.array(
    [
        1.06, 1.04203667, 1.01, 1.01200458, 1.01409084, 1.07, 1.0564397,
        1.09, 1.04843671, 1.04363563, 1.05262674, 1.0524842, 1.04653573,
        1.0267189,
    ]
)

CASE14_125X_VM_REF = np.array(
    [
        1.06, 1.03274949, 0.99320008, 0.99754457, 1.00099805, 1.05515867,
        1.04185399, 1.08096354, 1.03085783, 1.02562622, 1.03575399,
        1.0357343, 1.02910737, 1.00691151,
    ]
)

CASE14_125X_VA_DEG_REF = np.array(
    [
        0.0, -6.55650726, -16.65495673, -13.39544575, -11.40424254,
        -18.5284833, -17.3315379, -17.3315379, -19.37914829, -19.59807,
        -19.22782697, -19.63246366, -19.72618784, -20.84441154,
    ]
)


def test_ieee14_base_case_has_no_q_limit_violations():
    num_buses, bus_types, V_init, theta_init, P_spec, Q_spec, lines = get_ieee14_data()
    y_sparse = build_ybus_sparse(num_buses, lines, get_ieee14_bus_shunts())

    v_unconstrained, th_unconstrained, _ = solve_newton_raphson(
        V_init, theta_init, y_sparse, P_spec, Q_spec, bus_types, tol=1e-10, sparse=True
    )
    res = solve_nr_with_q_limits(
        V_init,
        theta_init,
        y_sparse,
        P_spec,
        Q_spec,
        bus_types,
        get_ieee14_q_limits(1.0),
        tol=1e-10,
    )

    assert res.saturated_buses == {}
    assert res.outer_iterations == 1
    assert res.bus_types_final == bus_types
    np.testing.assert_allclose(res.V, v_unconstrained, atol=1e-12)
    np.testing.assert_allclose(res.theta, th_unconstrained, atol=1e-12)


def test_ieee14_partial_saturation_matches_pandapower_reference():
    num_buses, bus_types, V_init, theta_init, _, _, lines = get_ieee14_data()
    y_sparse = build_ybus_sparse(num_buses, lines, get_ieee14_bus_shunts())
    p_spec, q_spec, q_lims = scale_ieee14_loads(1.15)

    res = solve_nr_with_q_limits(
        V_init, theta_init, y_sparse, p_spec, q_spec, bus_types, q_lims, tol=1e-10
    )

    assert res.saturated_buses == {2: "Q_max"}
    assert res.bus_types_final[1] == "PQ"
    assert res.bus_types_final[2] == "PV"
    assert res.Q_gen[2] == pytest.approx(0.50, abs=1e-8)
    np.testing.assert_allclose(res.V, CASE14_115X_VM_REF, atol=1e-6)


@pytest.mark.parametrize("strategy", ["most_severe", "all"])
def test_ieee14_cascade_saturation_matches_pandapower_reference(strategy):
    num_buses, bus_types, V_init, theta_init, _, _, lines = get_ieee14_data()
    y_sparse = build_ybus_sparse(num_buses, lines, get_ieee14_bus_shunts())
    p_spec, q_spec, q_lims = scale_ieee14_loads(1.25)

    res = solve_nr_with_q_limits(
        V_init,
        theta_init,
        y_sparse,
        p_spec,
        q_spec,
        bus_types,
        q_lims,
        tol=1e-10,
        strategy=strategy,
    )

    assert res.saturated_buses == {2: "Q_max", 3: "Q_max", 6: "Q_max", 8: "Q_max"}
    np.testing.assert_allclose(res.V, CASE14_125X_VM_REF, atol=1e-6)
    np.testing.assert_allclose(np.degrees(res.theta), CASE14_125X_VA_DEG_REF, atol=1e-5)


def test_under_excitation_q_min_clamps_and_raises_bus_voltage():
    bus_types, V_init, theta_init, P_spec, _ = get_3bus_data()
    lines = [
        {"from_bus": 1, "to_bus": 2, "r": 0.02, "x": 0.06, "b": 0.06},
        {"from_bus": 1, "to_bus": 3, "r": 0.08, "x": 0.24, "b": 0.05},
        {"from_bus": 2, "to_bus": 3, "r": 0.06, "x": 0.18, "b": 0.04},
    ]
    y_sparse = build_ybus_sparse(3, lines)
    q_capacitive = np.array([0.0, 0.0, 0.85])
    q_limits = [{"bus": 2, "q_min": -0.05, "q_max": 0.50}]

    res = solve_nr_with_q_limits(
        V_init, theta_init, y_sparse, P_spec, q_capacitive, bus_types, q_limits, tol=1e-9
    )

    assert res.saturated_buses == {2: "Q_min"}
    assert res.Q_gen[2] == pytest.approx(-0.05, abs=1e-8)
    assert res.V[1] > V_init[1]


def test_ieee30_stressed_q_limits_enforces_generator_bounds():
    num_buses, bus_types, V_init, theta_init, _, _, lines = get_ieee30_data()
    y_sparse = build_ybus_sparse(num_buses, lines, get_ieee30_bus_shunts())
    p_spec, q_spec, q_lims = scale_ieee30_loads(1.30)

    res = solve_nr_with_q_limits(
        V_init, theta_init, y_sparse, p_spec, q_spec, bus_types, q_lims, tol=1e-8
    )

    assert len(res.saturated_buses) >= 1
    for bus_num, q_val in res.Q_gen.items():
        assert q_lims[bus_num]["q_min"] - 1e-6 <= q_val <= q_lims[bus_num]["q_max"] + 1e-6


@pytest.mark.parametrize(
    "bad_limits, kwargs, match",
    [
        ({1: {"q_min": -0.5, "q_max": 0.5}}, {}, "PV buses"),
        ({2: {"q_min": 0.6, "q_max": 0.2}}, {}, "q_min cannot exceed q_max"),
        ({99: {"q_min": -0.1, "q_max": 0.1}}, {}, "between 1 and num_buses"),
        ({2: {"q_min": -0.4, "q_max": 0.5}}, {"strategy": "invalid"}, "strategy"),
    ],
)
def test_q_limit_solver_rejects_invalid_configurations(bad_limits, kwargs, match):
    num_buses, bus_types, V_init, theta_init, P_spec, Q_spec, lines = get_ieee14_data()
    y_sparse = build_ybus_sparse(num_buses, lines, get_ieee14_bus_shunts())

    with pytest.raises(ValueError, match=match):
        solve_nr_with_q_limits(
            V_init, theta_init, y_sparse, P_spec, Q_spec, bus_types, bad_limits, **kwargs
        )
