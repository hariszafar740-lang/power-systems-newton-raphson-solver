import numpy as np
from src.mismatch_calculator import calculate_power_mismatches
from src.jacobian_builder import build_jacobian

def solve_newton_raphson(V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types, max_iter=20, tol=1e-4):
    """
    Executes full Newton-Raphson load flow iterative loop until max mismatch < tol.
    """
    V = V_init.copy()
    theta = theta_init.copy()
    num_buses = len(V)
    
    pv_pq_buses = [i for i in range(num_buses) if bus_types[i] != 'Slack']
    pq_buses = [i for i in range(num_buses) if bus_types[i] == 'PQ']
    
    history = []
    
    for iteration in range(max_iter):
        P_calc, Q_calc, delta_P, delta_Q = calculate_power_mismatches(
            V, theta, Y_bus, P_spec, Q_spec, bus_types
        )
        
        mismatch_vector = np.concatenate([delta_P, delta_Q])
        max_err = np.max(np.abs(mismatch_vector))
        history.append(max_err)
        
        if max_err < tol:
            print(f"Convergence achieved in {iteration} iterations! Max Mismatch: {max_err:.6e}")
            break
            
        J = build_jacobian(V, theta, Y_bus, bus_types)
        delta_x = np.linalg.solve(J, mismatch_vector)
        
        # Update angles (theta) for PV & PQ buses
        n_p = len(pv_pq_buses)
        for idx, bus_idx in enumerate(pv_pq_buses):
            theta[bus_idx] += delta_x[idx]
            
        # Update voltage magnitudes (|V|) for PQ buses
        for idx, bus_idx in enumerate(pq_buses):
            V[bus_idx] += delta_x[n_p + idx]
    else:
        print("Warning: Maximum iterations reached without full convergence.")
        
    return V, theta, history
