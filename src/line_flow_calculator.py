import numpy as np

def calculate_line_flows(V, theta, line_data):
    """
    Calculates complex power flows (P_ij, Q_ij) and transmission losses across all lines.
    """
    flows = []
    total_P_loss = 0.0
    total_Q_loss = 0.0
    
    for line in line_data:
        from_b = line['from_bus'] - 1
        to_b = line['to_bus'] - 1
        r = line['r']
        x = line['x']
        b_charging = line.get('b', 0.0)
        
        z = complex(r, x)
        y = 1.0 / z
        y_c = complex(0.0, b_charging / 2.0)
        
        V_i = V[from_b] * np.exp(1j * theta[from_b])
        V_j = V[to_b] * np.exp(1j * theta[to_b])
        
        # Branch Currents
        I_ij = (V_i - V_j) * y + V_i * y_c
        I_ji = (V_j - V_i) * y + V_j * y_c
        
        # Complex Power Flows (S = V * I*)
        S_ij = V_i * np.conj(I_ij)
        S_ji = V_j * np.conj(I_ji)
        
        # Transmission Losses
        S_loss = S_ij + S_ji
        total_P_loss += S_loss.real
        total_Q_loss += S_loss.imag
        
        flows.append({
            'from': line['from_bus'],
            'to': line['to_bus'],
            'P_ij': S_ij.real,
            'Q_ij': S_ij.imag,
            'P_ji': S_ji.real,
            'Q_ji': S_ji.imag,
            'P_loss': S_loss.real,
            'Q_loss': S_loss.imag
        })
        
    return flows, total_P_loss, total_Q_loss
