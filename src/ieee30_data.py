import numpy as np


def get_ieee30_data():
    """Return the bus and branch data for the IEEE 30-bus benchmark.

    Bus 1 is the slack bus; buses 2, 5, 8, 11, and 13 are PV buses.  Use
    :func:`get_ieee30_bus_shunts` with ``build_ybus`` / ``build_ybus_sparse``
    to include the capacitive shunts at buses 10 and 24.

    Reference
    ---------
    Pinned to MATPOWER's ``data/case_ieee30.m`` (100 MVA base, UW Power
    Systems Test Case Archive 1961 IEEE 30-bus system) at commit
    ``4d1215d0811e81d34384c159ef7fd420e0e00fa8`` of
    https://github.com/MATPOWER/matpower.
    """
    num_buses = 30
    bus_types = [
        "Slack", "PV", "PQ", "PQ", "PV", "PQ", "PQ", "PV", "PQ", "PQ",
        "PV", "PQ", "PV", "PQ", "PQ", "PQ", "PQ", "PQ", "PQ", "PQ",
        "PQ", "PQ", "PQ", "PQ", "PQ", "PQ", "PQ", "PQ", "PQ", "PQ",
    ]

    V_init = np.array(
        [
            1.06, 1.045, 1.0, 1.0, 1.01, 1.0, 1.0, 1.01, 1.0, 1.0,
            1.082, 1.0, 1.071, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ]
    )
    theta_init = np.zeros(num_buses)

    P_spec = np.array(
        [
            0.0, 0.183, -0.024, -0.076, -0.942, 0.0, -0.228, -0.300, 0.0, -0.058,
            0.0, -0.112, 0.0, -0.062, -0.082, -0.035, -0.090, -0.032, -0.095, -0.022,
            -0.175, 0.0, -0.032, -0.087, 0.0, -0.035, 0.0, 0.0, -0.024, -0.106,
        ]
    )
    Q_spec = np.array(
        [
            0.0, 0.0, -0.012, -0.016, 0.0, 0.0, -0.109, 0.0, 0.0, -0.020,
            0.0, -0.075, 0.0, -0.016, -0.025, -0.018, -0.058, -0.009, -0.034, -0.007,
            -0.112, 0.0, -0.016, -0.067, 0.0, -0.023, 0.0, 0.0, -0.009, -0.019,
        ]
    )

    line_data = [
        {"from_bus": 1, "to_bus": 2, "r": 0.0192, "x": 0.0575, "b": 0.0528},
        {"from_bus": 1, "to_bus": 3, "r": 0.0452, "x": 0.1652, "b": 0.0408},
        {"from_bus": 2, "to_bus": 4, "r": 0.0570, "x": 0.1737, "b": 0.0368},
        {"from_bus": 3, "to_bus": 4, "r": 0.0132, "x": 0.0379, "b": 0.0084},
        {"from_bus": 2, "to_bus": 5, "r": 0.0472, "x": 0.1983, "b": 0.0418},
        {"from_bus": 2, "to_bus": 6, "r": 0.0581, "x": 0.1763, "b": 0.0374},
        {"from_bus": 4, "to_bus": 6, "r": 0.0119, "x": 0.0414, "b": 0.0090},
        {"from_bus": 5, "to_bus": 7, "r": 0.0460, "x": 0.1160, "b": 0.0204},
        {"from_bus": 6, "to_bus": 7, "r": 0.0267, "x": 0.0820, "b": 0.0170},
        {"from_bus": 6, "to_bus": 8, "r": 0.0120, "x": 0.0420, "b": 0.0090},
        {"from_bus": 6, "to_bus": 9, "r": 0.0, "x": 0.2080, "b": 0.0, "tap": 0.978},
        {"from_bus": 6, "to_bus": 10, "r": 0.0, "x": 0.5560, "b": 0.0, "tap": 0.969},
        {"from_bus": 9, "to_bus": 11, "r": 0.0, "x": 0.2080, "b": 0.0, "tap": 1.0},
        {"from_bus": 9, "to_bus": 10, "r": 0.0, "x": 0.1100, "b": 0.0, "tap": 1.0},
        {"from_bus": 4, "to_bus": 12, "r": 0.0, "x": 0.2560, "b": 0.0, "tap": 0.932},
        {"from_bus": 12, "to_bus": 13, "r": 0.0, "x": 0.1400, "b": 0.0, "tap": 1.0},
        {"from_bus": 12, "to_bus": 14, "r": 0.1231, "x": 0.2559, "b": 0.0},
        {"from_bus": 12, "to_bus": 15, "r": 0.0662, "x": 0.1304, "b": 0.0},
        {"from_bus": 12, "to_bus": 16, "r": 0.0945, "x": 0.1987, "b": 0.0},
        {"from_bus": 14, "to_bus": 15, "r": 0.2210, "x": 0.1997, "b": 0.0},
        {"from_bus": 16, "to_bus": 17, "r": 0.0524, "x": 0.1923, "b": 0.0},
        {"from_bus": 15, "to_bus": 18, "r": 0.1073, "x": 0.2185, "b": 0.0},
        {"from_bus": 18, "to_bus": 19, "r": 0.0639, "x": 0.1292, "b": 0.0},
        {"from_bus": 19, "to_bus": 20, "r": 0.0340, "x": 0.0680, "b": 0.0},
        {"from_bus": 10, "to_bus": 20, "r": 0.0936, "x": 0.2090, "b": 0.0},
        {"from_bus": 10, "to_bus": 17, "r": 0.0324, "x": 0.0845, "b": 0.0},
        {"from_bus": 10, "to_bus": 21, "r": 0.0348, "x": 0.0749, "b": 0.0},
        {"from_bus": 10, "to_bus": 22, "r": 0.0727, "x": 0.1499, "b": 0.0},
        {"from_bus": 21, "to_bus": 22, "r": 0.0116, "x": 0.0236, "b": 0.0},
        {"from_bus": 15, "to_bus": 23, "r": 0.1000, "x": 0.2020, "b": 0.0},
        {"from_bus": 22, "to_bus": 24, "r": 0.1150, "x": 0.1790, "b": 0.0},
        {"from_bus": 23, "to_bus": 24, "r": 0.1320, "x": 0.2700, "b": 0.0},
        {"from_bus": 24, "to_bus": 25, "r": 0.1885, "x": 0.3292, "b": 0.0},
        {"from_bus": 25, "to_bus": 26, "r": 0.2544, "x": 0.3800, "b": 0.0},
        {"from_bus": 25, "to_bus": 27, "r": 0.1093, "x": 0.2087, "b": 0.0},
        {"from_bus": 28, "to_bus": 27, "r": 0.0, "x": 0.3960, "b": 0.0, "tap": 0.968},
        {"from_bus": 27, "to_bus": 29, "r": 0.2198, "x": 0.4153, "b": 0.0},
        {"from_bus": 27, "to_bus": 30, "r": 0.3202, "x": 0.6027, "b": 0.0},
        {"from_bus": 29, "to_bus": 30, "r": 0.2399, "x": 0.4533, "b": 0.0},
        {"from_bus": 8, "to_bus": 28, "r": 0.0636, "x": 0.2000, "b": 0.0428},
        {"from_bus": 6, "to_bus": 28, "r": 0.0169, "x": 0.0599, "b": 0.0130},
    ]

    return num_buses, bus_types, V_init, theta_init, P_spec, Q_spec, line_data


def get_ieee30_bus_shunts():
    """Return IEEE 30-bus shunt admittances for ``build_ybus``.

    Matches MATPOWER ``case_ieee30.m`` (19 MVAr at bus 10 and 4.3 MVAr at
    bus 24 on the 100 MVA system base).
    """
    return [{"bus": 10, "b": 0.19}, {"bus": 24, "b": 0.043}]
