import numpy as np

def build_ybus(num_buses, line_data):
    """
    Constructs the complex Bus Admittance Matrix (Y_bus) for a power system.
    
    Parameters:
        num_buses (int): Total number of system buses.
        line_data (list of dict): List containing branch parameters:
            - 'from_bus': int (1-indexed)
            - 'to_bus': int (1-indexed)
            - 'r': float (resistance p.u.)
            - 'x': float (reactance p.u.)
            - 'b': float (total line charging susceptance p.u., default 0.0)
            
    Returns:
        np.ndarray: (num_buses x num_buses) complex Y_bus matrix.
    """
    Y_bus = np.zeros((num_buses, num_buses), dtype=complex)
    
    for line in line_data:
        i = line['from_bus'] - 1  # Convert to 0-indexed array
        j = line['to_bus'] - 1
        r = line['r']
        x = line['x']
        b_total = line.get('b', 0.0)
        
        # Branch impedance and admittance
        z = complex(r, x)
        y = 1.0 / z
        y_charging = complex(0.0, b_total / 2.0)
        
        # Off-diagonal elements (Symmetric for passive lines)
        Y_bus[i, j] -= y
        Y_bus[j, i] -= y
        
        # Diagonal elements (Self-admittance + line charging)
        Y_bus[i, i] += y + y_charging
        Y_bus[j, j] += y + y_charging
        
    return Y_bus
