import numpy as np

from src.mismatch_calculator import calculate_power_injections


def build_jacobian(V, theta, Y_bus, bus_types):
    """Construct the reduced Newton-Raphson power-flow Jacobian.

    The row order is active-power equations for PV and PQ buses followed by
    reactive-power equations for PQ buses.  The column order is the matching
    voltage angles followed by PQ-bus voltage magnitudes.
    """
    num_buses = len(V)
    conductance = Y_bus.real
    susceptance = Y_bus.imag

    pv_pq_buses = [i for i in range(num_buses) if bus_types[i] != "Slack"]
    pq_buses = [i for i in range(num_buses) if bus_types[i] == "PQ"]

    n_p = len(pv_pq_buses)
    n_q = len(pq_buses)
    J11 = np.zeros((n_p, n_p))
    J12 = np.zeros((n_p, n_q))
    J21 = np.zeros((n_q, n_p))
    J22 = np.zeros((n_q, n_q))

    P_calc, Q_calc = calculate_power_injections(V, theta, Y_bus)

    for row, i in enumerate(pv_pq_buses):
        for column, j in enumerate(pv_pq_buses):
            theta_ij = theta[i] - theta[j]
            if i == j:
                J11[row, column] = -Q_calc[i] - V[i] ** 2 * susceptance[i, i]
            else:
                J11[row, column] = V[i] * V[j] * (
                    conductance[i, j] * np.sin(theta_ij)
                    - susceptance[i, j] * np.cos(theta_ij)
                )

        for column, j in enumerate(pq_buses):
            theta_ij = theta[i] - theta[j]
            if i == j:
                J12[row, column] = P_calc[i] / V[i] + V[i] * conductance[i, i]
            else:
                J12[row, column] = V[i] * (
                    conductance[i, j] * np.cos(theta_ij)
                    + susceptance[i, j] * np.sin(theta_ij)
                )

    for row, i in enumerate(pq_buses):
        for column, j in enumerate(pv_pq_buses):
            theta_ij = theta[i] - theta[j]
            if i == j:
                J21[row, column] = P_calc[i] - V[i] ** 2 * conductance[i, i]
            else:
                J21[row, column] = -V[i] * V[j] * (
                    conductance[i, j] * np.cos(theta_ij)
                    + susceptance[i, j] * np.sin(theta_ij)
                )

        for column, j in enumerate(pq_buses):
            theta_ij = theta[i] - theta[j]
            if i == j:
                J22[row, column] = Q_calc[i] / V[i] - V[i] * susceptance[i, i]
            else:
                J22[row, column] = V[i] * (
                    conductance[i, j] * np.sin(theta_ij)
                    - susceptance[i, j] * np.cos(theta_ij)
                )

    return np.block([[J11, J12], [J21, J22]])
