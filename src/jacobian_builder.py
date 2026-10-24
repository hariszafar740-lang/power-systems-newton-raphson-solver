import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, issparse

from src.mismatch_calculator import calculate_power_injections


def build_jacobian_sparse(V, theta, Y_bus, bus_types):
    """Construct the reduced Newton-Raphson Jacobian directly as a CSC matrix.

    Complexity scales as ``O(nnz(Y_bus))`` in both runtime and memory,
    iterating solely over non-zero admittance couplings and bus diagonals.
    """
    num_buses = len(V)
    pv_pq_buses = [i for i in range(num_buses) if bus_types[i] != "Slack"]
    pq_buses = [i for i in range(num_buses) if bus_types[i] == "PQ"]

    n_p = len(pv_pq_buses)
    n_q = len(pq_buses)
    dim = n_p + n_q

    p_pos = np.full(num_buses, -1, dtype=int)
    q_pos = np.full(num_buses, -1, dtype=int)
    for idx, bus in enumerate(pv_pq_buses):
        p_pos[bus] = idx
    for idx, bus in enumerate(pq_buses):
        q_pos[bus] = idx

    P_calc, Q_calc = calculate_power_injections(V, theta, Y_bus)

    y_csc = Y_bus.tocsc() if issparse(Y_bus) else csc_matrix(Y_bus)
    diag = y_csc.diagonal()
    g_diag = diag.real
    b_diag = diag.imag

    rows = []
    cols = []
    data = []

    for row_p, i in enumerate(pv_pq_buses):
        rows.append(row_p)
        cols.append(row_p)
        data.append(float(-Q_calc[i] - V[i] ** 2 * b_diag[i]))

        col_q = int(q_pos[i])
        if col_q >= 0:
            rows.append(row_p)
            cols.append(n_p + col_q)
            data.append(float(P_calc[i] / V[i] + V[i] * g_diag[i]))

    for row_q, i in enumerate(pq_buses):
        col_p = int(p_pos[i])
        rows.append(n_p + row_q)
        cols.append(col_p)
        data.append(float(P_calc[i] - V[i] ** 2 * g_diag[i]))

        rows.append(n_p + row_q)
        cols.append(n_p + row_q)
        data.append(float(Q_calc[i] / V[i] - V[i] * b_diag[i]))

    y_coo = y_csc.tocoo()
    for i, j, y_ij in zip(y_coo.row, y_coo.col, y_coo.data, strict=True):
        if i == j:
            continue
        row_p = int(p_pos[i])
        if row_p < 0:
            continue

        g_ij = float(y_ij.real)
        b_ij = float(y_ij.imag)
        theta_ij = float(theta[i] - theta[j])
        sin_t = float(np.sin(theta_ij))
        cos_t = float(np.cos(theta_ij))

        col_p = int(p_pos[j])
        if col_p >= 0:
            rows.append(row_p)
            cols.append(col_p)
            data.append(float(V[i] * V[j] * (g_ij * sin_t - b_ij * cos_t)))

        col_q = int(q_pos[j])
        if col_q >= 0:
            rows.append(row_p)
            cols.append(n_p + col_q)
            data.append(float(V[i] * (g_ij * cos_t + b_ij * sin_t)))

        row_q = int(q_pos[i])
        if row_q >= 0:
            if col_p >= 0:
                rows.append(n_p + row_q)
                cols.append(col_p)
                data.append(float(-V[i] * V[j] * (g_ij * cos_t + b_ij * sin_t)))
            if col_q >= 0:
                rows.append(n_p + row_q)
                cols.append(n_p + col_q)
                data.append(float(V[i] * (g_ij * sin_t - b_ij * cos_t)))

    return coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=float).tocsc()


def build_jacobian(V, theta, Y_bus, bus_types, *, sparse=False):
    """Construct the reduced Newton-Raphson power-flow Jacobian.

    The row order is active-power equations for PV and PQ buses followed by
    reactive-power equations for PQ buses.  The column order is the matching
    voltage angles followed by PQ-bus voltage magnitudes.
    """
    if sparse:
        return build_jacobian_sparse(V, theta, Y_bus, bus_types)

    y_dense = Y_bus.toarray() if issparse(Y_bus) else np.asarray(Y_bus, dtype=complex)
    num_buses = len(V)
    conductance = y_dense.real
    susceptance = y_dense.imag

    pv_pq_buses = [i for i in range(num_buses) if bus_types[i] != "Slack"]
    pq_buses = [i for i in range(num_buses) if bus_types[i] == "PQ"]

    n_p = len(pv_pq_buses)
    n_q = len(pq_buses)
    J11 = np.zeros((n_p, n_p))
    J12 = np.zeros((n_p, n_q))
    J21 = np.zeros((n_q, n_p))
    J22 = np.zeros((n_q, n_q))

    P_calc, Q_calc = calculate_power_injections(V, theta, y_dense)

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
