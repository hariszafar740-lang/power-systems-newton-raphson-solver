import numpy as np

def calculate_power_mismatches(V, theta, Y_bus, P_spec, Q_spec, bus_types):
    """
    Calculates power injected into each bus and computes mismatch vectors (Delta P, Delta Q).
    """
    num_buses = len(V)
    G = Y_bus.real
    B = Y_bus.imag
    
    P_calc = np.zeros(num_buses)
    Q_calc = np.zeros(num_buses)
    
    for i in range(num_buses):
        for j in range(num_buses):
            theta_ij = theta[i] - theta[j]
            P_calc[i] += V[i] * V[j] * (G[i, j] * np.cos(theta_ij) + B[i, j] * np.sin(theta_ij))
            Q_calc[i] += V[i] * V[j] * (G[i, j] * np.sin(theta_ij) - B[i, j] * np.cos(theta_ij))
            
    delta_P = []
    delta_Q = []
    
    for i in range(num_buses):
        if bus_types[i] != 'Slack':
            delta_P.append(P_spec[i] - P_calc[i])
        if bus_types[i] == 'PQ':
            delta_Q.append(Q_spec[i] - Q_calc[i])
            
    return P_calc, Q_calc, np.array(delta_P), np.array(delta_Q)
