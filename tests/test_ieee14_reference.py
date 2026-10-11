"""Regression tests pinning the IEEE 14-bus model to MATPOWER case14.

Data reference
--------------
The network data in ``src/ieee14_data.py`` is pinned to MATPOWER's
``case14.m`` (100 MVA base), file ``data/case14.m`` of
https://github.com/MATPOWER/matpower at commit
``2e0ef79a6c4526858ee923491d92781ad4507fb7`` (2026-08-04).  The identical
values appear in MATPOWER releases 7.1 and 8.0 and in PYPOWER 5.1.21's
``case14`` (all fetched and compared on 2026-10-11).  Beware that the raw
IEEE Common Data Format file and several textbook tables circulate a
row-shifted variant of the charging susceptances of branches 2-4, 2-5,
3-4, and 4-5; this repository deliberately follows MATPOWER case14.

Solution reference
------------------
The independent power-flow solution below was computed on 2026-10-11 with
pandapower 3.5.6 (``networks.case14``, ``runpp`` Newton-Raphson,
``tolerance_mva=1e-8``) and reproduces the solution columns stored in
MATPOWER's ``case14.m``.  The solver under test converges to a maximum
mismatch below 1e-10 here; observed agreement with the reference is
better than 5e-7 p.u. for every bus voltage.  Tolerances are set about
two orders of magnitude looser than the observed agreement so they catch
any data or modeling regression while remaining insensitive to floating
point differences across numpy versions:

* voltage magnitudes: atol = 1e-4 p.u.
* voltage angles:     atol = 1e-4 rad (about 0.006 degrees)
* losses and flows:   atol = 5e-4 p.u.
"""

import numpy as np
import pytest

from src.ieee14_data import get_ieee14_bus_shunts, get_ieee14_data
from src.line_flow_calculator import calculate_line_flows
from src.mismatch_calculator import calculate_power_injections
from src.nr_solver import solve_newton_raphson
from src.ybus_builder import build_ybus

# MATPOWER case14 branch table: (from, to, r, x, total b, tap ratio).
# A tap ratio of 1.0 marks a plain transmission line (MATPOWER stores 0
# for those, which means "no transformer"; it is normalized to 1.0 here).
MATPOWER_BRANCH_TABLE = [
    (1, 2, 0.01938, 0.05917, 0.0528, 1.0),
    (1, 5, 0.05403, 0.22304, 0.0492, 1.0),
    (2, 3, 0.04699, 0.19797, 0.0438, 1.0),
    (2, 4, 0.05811, 0.17632, 0.034, 1.0),
    (2, 5, 0.05695, 0.17388, 0.0346, 1.0),
    (3, 4, 0.06701, 0.17103, 0.0128, 1.0),
    (4, 5, 0.01335, 0.04211, 0.0, 1.0),
    (4, 7, 0.0, 0.20912, 0.0, 0.978),
    (4, 9, 0.0, 0.55618, 0.0, 0.969),
    (5, 6, 0.0, 0.25202, 0.0, 0.932),
    (6, 11, 0.09498, 0.19890, 0.0, 1.0),
    (6, 12, 0.12291, 0.25581, 0.0, 1.0),
    (6, 13, 0.06615, 0.13027, 0.0, 1.0),
    (7, 8, 0.0, 0.17615, 0.0, 1.0),
    (7, 9, 0.0, 0.11001, 0.0, 1.0),
    (9, 10, 0.03181, 0.08450, 0.0, 1.0),
    (9, 14, 0.12711, 0.27038, 0.0, 1.0),
    (10, 11, 0.08205, 0.19207, 0.0, 1.0),
    (12, 13, 0.22092, 0.19988, 0.0, 1.0),
    (13, 14, 0.17093, 0.34802, 0.0, 1.0),
]

# MATPOWER case14 bus loads Pd/Qd (MW/MVAr on a 100 MVA base) and generator
# active power Pg; PV/slack generator voltage setpoints Vg.
MATPOWER_LOAD_P = [0, 21.7, 94.2, 47.8, 7.6, 11.2, 0, 0, 29.5, 9, 3.5, 6.1, 13.5, 14.9]
MATPOWER_LOAD_Q = [0, 12.7, 19.0, -3.9, 1.6, 7.5, 0, 0, 16.6, 5.8, 1.8, 1.6, 5.8, 5.0]
MATPOWER_GEN_P = {1: 232.4, 2: 40.0, 3: 0.0, 6: 0.0, 8: 0.0}  # slack Pg floats
MATPOWER_GEN_V = {1: 1.06, 2: 1.045, 3: 1.01, 6: 1.07, 8: 1.09}

# Independent pandapower 3.5.6 Newton-Raphson solution of case14.
REFERENCE_VM = np.array(
    [1.06, 1.045, 1.01, 1.0176708537, 1.0195138598, 1.07, 1.0615195325,
     1.09, 1.0559317206, 1.050984625, 1.0569065185, 1.0551885632,
     1.0503817136, 1.0355299459]
)
REFERENCE_VA = np.array(
    [0.0, -0.0869625858, -0.2220948916, -0.1799940795, -0.1531326386,
     -0.2482023385, -0.2331694844, -0.2331694844, -0.260726382,
     -0.2634973918, -0.2581450529, -0.2631185865, -0.2645269244,
     -0.2798398881]
)
REFERENCE_P_LOSS = 0.1339327236
REFERENCE_Q_LOSS = 0.3012238804
REFERENCE_SLACK_P = 2.3239327235
REFERENCE_SLACK_Q = -0.1654930054
# From-end complex power flows for every branch, in MATPOWER branch order.
REFERENCE_BRANCH_P = np.array(
    [1.5688289053, 0.7551038182, 0.7323757924, 0.5613149594, 0.4151621502,
     -0.2328569010, -0.6115823044, 0.2807417592, 0.1607975758, 0.4408732086,
     0.0735327698, 0.0778606702, 0.1774797686, 0.0, 0.2807417592,
     0.0522755247, 0.0942638103, -0.0378532238, 0.0161425777, 0.0564385098]
)
REFERENCE_BRANCH_Q = np.array(
    [-0.2040429168, 0.0385499114, 0.0356020295, -0.0155035040, 0.0117099786,
     0.0447311562, 0.1582364194, -0.0968106571, -0.0042761118, 0.1247067981,
     0.0356047297, 0.0250341424, 0.0721657539, -0.1716297051, 0.0577869057,
     0.0421913780, 0.0361000624, -0.0161506292, 0.0075395917, 0.0174717378]
)

VOLTAGE_ATOL = 1e-4
ANGLE_ATOL = 1e-4
FLOW_ATOL = 5e-4


def _solve(tolerance=1e-10):
    num_buses, bus_types, V_init, theta_init, P_spec, Q_spec, line_data = get_ieee14_data()
    Y_bus = build_ybus(num_buses, line_data, get_ieee14_bus_shunts())
    V, theta, history = solve_newton_raphson(
        V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types, max_iter=50, tol=tolerance
    )
    return V, theta, history, line_data, Y_bus, bus_types


def test_branch_table_matches_matpower_case14():
    _, _, _, _, _, _, line_data = get_ieee14_data()
    assert len(line_data) == len(MATPOWER_BRANCH_TABLE)
    for line, expected in zip(line_data, MATPOWER_BRANCH_TABLE):
        from_bus, to_bus, r, x, b, tap = expected
        assert line["from_bus"] == from_bus
        assert line["to_bus"] == to_bus
        assert line["r"] == r
        assert line["x"] == x
        assert line["b"] == b
        assert line.get("tap", 1.0) == tap


def test_bus_types_and_generator_setpoints_match_matpower_case14():
    _, bus_types, V_init, _, _, _, _ = get_ieee14_data()
    expected_types = [
        "Slack", "PV", "PV", "PQ", "PQ", "PV", "PQ", "PV", "PQ", "PQ",
        "PQ", "PQ", "PQ", "PQ",
    ]
    assert bus_types == expected_types
    for bus, setpoint in MATPOWER_GEN_V.items():
        assert V_init[bus - 1] == setpoint


def test_net_injections_match_matpower_gen_minus_load():
    _, _, _, _, P_spec, Q_spec, _ = get_ieee14_data()
    for index in range(14):
        bus = index + 1
        # The slack injection is a solved quantity, so it is stored as 0.0;
        # every other bus is scheduled at generation minus load.
        if bus == 1:
            expected_p = 0.0
        else:
            expected_p = (MATPOWER_GEN_P.get(bus, 0.0) - MATPOWER_LOAD_P[index]) / 100.0
        expected_q = -MATPOWER_LOAD_Q[index] / 100.0
        # The case tables are exact to three decimals; allow only rounding
        # noise from the division above.
        assert P_spec[index] == pytest.approx(expected_p, abs=1e-9)
        # Q_spec only constrains PQ buses; slack/PV buses keep 0.0 here.
        if bus in MATPOWER_GEN_V:
            assert Q_spec[index] == 0.0
        else:
            assert Q_spec[index] == pytest.approx(expected_q, abs=1e-9)


def test_bus9_shunt_matches_matpower_case14():
    shunts = get_ieee14_bus_shunts()
    assert shunts == [{"bus": 9, "b": 0.19}]


def test_solution_voltages_match_independent_reference():
    V, theta, history, _, _, _ = _solve()
    assert history[-1] < 1e-10
    np.testing.assert_allclose(V, REFERENCE_VM, atol=VOLTAGE_ATOL)
    np.testing.assert_allclose(theta, REFERENCE_VA, atol=ANGLE_ATOL)


def test_solution_losses_and_flows_match_independent_reference():
    V, theta, _, line_data, _, _ = _solve()
    flows, total_p_loss, total_q_loss = calculate_line_flows(V, theta, line_data)
    np.testing.assert_allclose(total_p_loss, np.sum([flow["P_loss"] for flow in flows]))
    np.testing.assert_allclose(total_q_loss, np.sum([flow["Q_loss"] for flow in flows]))
    np.testing.assert_allclose(total_p_loss, REFERENCE_P_LOSS, atol=FLOW_ATOL)
    np.testing.assert_allclose(total_q_loss, REFERENCE_Q_LOSS, atol=FLOW_ATOL)
    p_flows = np.array([flow["P_ij"] for flow in flows])
    q_flows = np.array([flow["Q_ij"] for flow in flows])
    np.testing.assert_allclose(p_flows, REFERENCE_BRANCH_P, atol=FLOW_ATOL)
    np.testing.assert_allclose(q_flows, REFERENCE_BRANCH_Q, atol=FLOW_ATOL)


def test_slack_and_reactive_generation_match_independent_reference():
    V, theta, _, _, Y_bus, _ = _solve()
    P_calc, Q_calc = calculate_power_injections(V, theta, Y_bus)
    np.testing.assert_allclose(P_calc[0], REFERENCE_SLACK_P, atol=FLOW_ATOL)
    np.testing.assert_allclose(Q_calc[0], REFERENCE_SLACK_Q, atol=FLOW_ATOL)
    # Generator reactive output at PV buses equals the calculated injection
    # plus the local load; pandapower gives 0.435571, 0.250753, 0.127309,
    # and 0.176235 p.u. for buses 2, 3, 6, and 8.
    q_gen_reference = {2: 0.4355710013, 3: 0.2507534850, 6: 0.1273094440, 8: 0.1762345137}
    for bus, expected in q_gen_reference.items():
        q_gen = Q_calc[bus - 1] + MATPOWER_LOAD_Q[bus - 1] / 100.0
        np.testing.assert_allclose(q_gen, expected, atol=FLOW_ATOL)


def test_bus9_shunt_reactive_generation_is_consistent():
    # The 0.19 p.u. capacitive shunt generates V^2 * b reactive power.
    V, _, _, _, _, _ = _solve()
    np.testing.assert_allclose(V[8] ** 2 * 0.19, 0.2118484, atol=1e-5)
