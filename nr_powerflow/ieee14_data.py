import numpy as np


def get_ieee14_data():
    """Return the bus and branch data for the IEEE 14-bus benchmark.

    Bus 1 is the slack bus; buses 2, 3, 6, and 8 are PV buses.  The branch
    list uses total line-charging susceptance and includes off-nominal tap
    ratios for the three transformers.  Use :func:`get_ieee14_bus_shunts`
    with ``build_ybus`` to include the bus-9 shunt in the benchmark model.

    Reference
    ---------
    All network data are pinned to MATPOWER's ``case14.m`` (100 MVA base),
    file ``data/case14.m`` of https://github.com/MATPOWER/matpower at
    commit ``2e0ef79a6c4526858ee923491d92781ad4507fb7`` (2026-08-04); the
    same values appear in MATPOWER releases 7.1 and 8.0 and in PYPOWER
    5.1.21's ``case14``.  Note that four line-charging susceptances
    (branches 2-4, 2-5, 3-4, 4-5) differ from the raw IEEE Common Data
    Format file and some textbook tables, which circulate a row-shifted
    variant of these values.  This repository deliberately follows the
    MATPOWER case14 convention so results can be validated against the
    widely used MATPOWER solution; the discrepancy is regression-tested in
    ``tests/test_ieee14_reference.py``.
    """
    num_buses = 14
    bus_types = [
        "Slack", "PV", "PV", "PQ", "PQ", "PV", "PQ", "PV", "PQ", "PQ",
        "PQ", "PQ", "PQ", "PQ",
    ]

    V_init = np.array(
        [1.06, 1.045, 1.01, 1.0, 1.0, 1.07, 1.0, 1.09, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    theta_init = np.zeros(num_buses)

    P_spec = np.array(
        [0.0, 0.183, -0.942, -0.478, -0.076, -0.112, 0.0, 0.0,
         -0.295, -0.09, -0.035, -0.061, -0.135, -0.149]
    )
    Q_spec = np.array(
        [0.0, 0.0, 0.0, 0.039, -0.016, 0.0, 0.0, 0.0,
         -0.166, -0.058, -0.018, -0.016, -0.058, -0.05]
    )

    line_data = [
        {"from_bus": 1, "to_bus": 2, "r": 0.01938, "x": 0.05917, "b": 0.0528},
        {"from_bus": 1, "to_bus": 5, "r": 0.05403, "x": 0.22304, "b": 0.0492},
        {"from_bus": 2, "to_bus": 3, "r": 0.04699, "x": 0.19797, "b": 0.0438},
        {"from_bus": 2, "to_bus": 4, "r": 0.05811, "x": 0.17632, "b": 0.034},
        {"from_bus": 2, "to_bus": 5, "r": 0.05695, "x": 0.17388, "b": 0.0346},
        {"from_bus": 3, "to_bus": 4, "r": 0.06701, "x": 0.17103, "b": 0.0128},
        {"from_bus": 4, "to_bus": 5, "r": 0.01335, "x": 0.04211, "b": 0.0},
        {"from_bus": 4, "to_bus": 7, "r": 0.0, "x": 0.20912, "b": 0.0, "tap": 0.978},
        {"from_bus": 4, "to_bus": 9, "r": 0.0, "x": 0.55618, "b": 0.0, "tap": 0.969},
        {"from_bus": 5, "to_bus": 6, "r": 0.0, "x": 0.25202, "b": 0.0, "tap": 0.932},
        {"from_bus": 6, "to_bus": 11, "r": 0.09498, "x": 0.19890, "b": 0.0},
        {"from_bus": 6, "to_bus": 12, "r": 0.12291, "x": 0.25581, "b": 0.0},
        {"from_bus": 6, "to_bus": 13, "r": 0.06615, "x": 0.13027, "b": 0.0},
        {"from_bus": 7, "to_bus": 8, "r": 0.0, "x": 0.17615, "b": 0.0},
        {"from_bus": 7, "to_bus": 9, "r": 0.0, "x": 0.11001, "b": 0.0},
        {"from_bus": 9, "to_bus": 10, "r": 0.03181, "x": 0.08450, "b": 0.0},
        {"from_bus": 9, "to_bus": 14, "r": 0.12711, "x": 0.27038, "b": 0.0},
        {"from_bus": 10, "to_bus": 11, "r": 0.08205, "x": 0.19207, "b": 0.0},
        {"from_bus": 12, "to_bus": 13, "r": 0.22092, "x": 0.19988, "b": 0.0},
        {"from_bus": 13, "to_bus": 14, "r": 0.17093, "x": 0.34802, "b": 0.0},
    ]

    return num_buses, bus_types, V_init, theta_init, P_spec, Q_spec, line_data


def get_ieee14_bus_shunts():
    """Return IEEE 14-bus shunt admittances for ``build_ybus``."""
    return [{"bus": 9, "b": 0.19}]


def get_ieee14_q_limits(load_multiplier=1.0):
    """Return PV generator reactive capability limits from MATPOWER ``case14.m``.

    Values are in per unit on the 100 MVA base, including local reactive load
    ``q_load`` at PV buses 2, 3, and 6 so ``Q_gen = Q_inj + q_load``.
    """
    mult = float(load_multiplier)
    if not np.isfinite(mult) or mult < 0.0:
        raise ValueError("load_multiplier must be a finite non-negative number")
    return {
        2: {"q_min": -0.40, "q_max": 0.50, "q_load": 0.127 * mult},
        3: {"q_min": 0.00, "q_max": 0.40, "q_load": 0.190 * mult},
        6: {"q_min": -0.06, "q_max": 0.24, "q_load": 0.075 * mult},
        8: {"q_min": -0.06, "q_max": 0.24, "q_load": 0.0},
    }


def scale_ieee14_loads(load_multiplier=1.0):
    """Return ``(P_spec, Q_spec, q_limits)`` with all bus loads scaled by ``load_multiplier``."""
    mult = float(load_multiplier)
    if not np.isfinite(mult) or mult < 0.0:
        raise ValueError("load_multiplier must be a finite non-negative number")

    p_gen = np.array([0.0, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p_load = np.array(
        [0.0, 0.217, 0.942, 0.478, 0.076, 0.112, 0.0, 0.0,
         0.295, 0.09, 0.035, 0.061, 0.135, 0.149]
    )
    q_load_pq = np.array(
        [0.0, 0.0, 0.0, -0.039, 0.016, 0.0, 0.0, 0.0,
         0.166, 0.058, 0.018, 0.016, 0.058, 0.05]
    )
    p_spec = p_gen - mult * p_load
    q_spec = -mult * q_load_pq
    return p_spec, q_spec, get_ieee14_q_limits(mult)
