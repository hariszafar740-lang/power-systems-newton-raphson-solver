import numpy as np


def calculate_power_injections(V, theta, Y_bus):
    """Calculate active and reactive power injected at every bus.

    Positive values represent power injected into the network.  The voltage
    angles are expressed in radians.
    """
    num_buses = len(V)
    conductance = Y_bus.real
    susceptance = Y_bus.imag

    P_calc = np.zeros(num_buses)
    Q_calc = np.zeros(num_buses)

    for i in range(num_buses):
        for j in range(num_buses):
            theta_ij = theta[i] - theta[j]
            P_calc[i] += V[i] * V[j] * (
                conductance[i, j] * np.cos(theta_ij)
                + susceptance[i, j] * np.sin(theta_ij)
            )
            Q_calc[i] += V[i] * V[j] * (
                conductance[i, j] * np.sin(theta_ij)
                - susceptance[i, j] * np.cos(theta_ij)
            )

    return P_calc, Q_calc


def calculate_power_mismatches(V, theta, Y_bus, P_spec, Q_spec, bus_types):
    """Return calculated injections and the Newton-Raphson mismatch vectors.

    The active-power mismatch contains PV and PQ buses in bus order.  The
    reactive-power mismatch contains PQ buses only, also in bus order.
    """
    P_calc, Q_calc = calculate_power_injections(V, theta, Y_bus)

    delta_P = []
    delta_Q = []
    for i, bus_type in enumerate(bus_types):
        if bus_type != "Slack":
            delta_P.append(P_spec[i] - P_calc[i])
        if bus_type == "PQ":
            delta_Q.append(Q_spec[i] - Q_calc[i])

    return P_calc, Q_calc, np.array(delta_P), np.array(delta_Q)
