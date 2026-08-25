import numpy as np

def build_jacobian(V, theta, Y_bus, bus_types):
    """
    Constructs the Jacobian Matrix (J) for Newton-Raphson load flow.
    Submatrices:
      J11 = dP / dtheta
      J12 = dP / d|V|
      J21 = dQ / dtheta
      J22 = dQ / d|V|
    """
    num_buses = len(V)
    G = Y_bus.real
    B = Y_bus.imag
    
    pv_pq_buses = [i for i in range(num_buses) if bus_types[i] != 'Slack']
    pq_buses = [i for i in range(num_buses) if bus_types[i] == 'PQ']
    
    n_p = len(pv_pq_buses)
    n_q = len(pq_buses)
    
    J11 = np.zeros((n_p, n_p))
    J12 = np.zeros((n_p, n_q))
    J21 = np.zeros((n_q, n_p))
    J22 = np.zeros((n_q, n_q))
    
    P_calc = np.zeros(num_buses)
    Q_calc = np.zeros(num_buses)
    for i in range(num_buses):
        for j in range(num_buses):
            th_ij = theta[i] - theta[j]
            P_calc[i] += V[i] * V[j] * (G[i, j] * np.cos(th_ij) + B[i, j] * np.sin(th_ij))
            Q_calc[i] += V[i] * V[j] * (G[i, j] * np.sin(th_ij) - B[i, j] * np.cos(th_ij))

    for idx_i, i in enumerate(pv_pq_buses):
        for idx_j, j in enumerate(pv_pq_buses):
            th_ij = theta[i] - theta[j]
            if i == j:
                J11[idx_i, idx_j] = -Q_calc[i] - (V[i]**2) * B[i, i]
            else:
                J11[idx_i, idx_j] = V[i] * V[j] * (G[i, j] * np.sin(th_ij) - B[i, j] * np.cos(th_ij))
        
        for idx_j, j in enumerate(pq_buses):
            th_ij = theta[i] - theta[j]
            if i == j:
                J12[idx_i, idx_j] = (P_calc[i] / V[i]) + V[i] * G[i, i]
            else:
                J12[idx_i, idx_j] = V[i] * (G[i, j] * np.cos(th_ij) + B[i, j] * np.sin(th_ij))

    for idx_i, i in enumerate(pq_buses):
        for idx_j, j in enumerate(pv_pq_buses):
            th_ij = theta[i] - theta[j]
            if i == j:
                J21[idx_i, idx_j] = P_calc[i] - (V[i]**2) * G[i, i]
            else:
                J21[idx_i, idx_j] = -V[i] * V[j] * (G[i, j] * np.cos(th_ij) + B[i, j] * np.sin(th_ij))
        
        for idx_j, j in enumerate(pq_buses):
            th_ij = theta[i] - theta[j]
            if i == j:
                J22[idx_i, idx_j] = (Q_calc[i] / V[i]) - V[i] * B[i, i]
            else:
                J22[idx_i, idx_j] = V[i] * (G[i, j] * np.sin(th_ij) - B[i, j] * np.cos(th_ij))

    J = np.block([[J11, J12],
                  [J21, J22]])
    return J
