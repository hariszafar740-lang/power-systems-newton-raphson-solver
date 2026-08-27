import numpy as np

def get_ieee14_data():
    """
    IEEE 14-Bus Benchmark System Specification:
    - Bus 1: Slack Bus (V = 1.06 p.u., theta = 0 rad)
    - Buses 2, 3, 6, 8: PV Buses
    - Buses 4, 5, 7, 9, 10, 11, 12, 13, 14: PQ Buses
    """
    num_buses = 14
    bus_types = [
        'Slack', 'PV', 'PV', 'PQ', 'PQ', 
        'PV', 'PQ', 'PV', 'PQ', 'PQ', 
        'PQ', 'PQ', 'PQ', 'PQ'
    ]
    
    # Voltage targets (p.u.) & initial flat-start angles (rad)
    V_init = np.array([1.06, 1.045, 1.01, 1.0, 1.0, 1.07, 1.0, 1.09, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    theta_init = np.zeros(num_buses)
    
    # Net scheduled power injections (P_gen - P_load, Q_gen - Q_load in p.u.)
    P_spec = np.array([0.0, 0.183, -0.942, -0.478, -0.076, -0.112, 0.0, 0.0, -0.295, -0.09, -0.035, -0.061, -0.135, -0.149])
    Q_spec = np.array([0.0, 0.0, 0.0, 0.039, -0.016, 0.0, 0.0, 0.0, -0.166, -0.058, -0.018, -0.016, -0.058, -0.05])
    
    # Transmission branch data [From, To, R, X, Half-line B charging]
    line_data = [
        {'from_bus': 1, 'to_bus': 2, 'r': 0.01938, 'x': 0.05917, 'b': 0.0528},
        {'from_bus': 1, 'to_bus': 5, 'r': 0.05403, 'x': 0.22304, 'b': 0.0492},
        {'from_bus': 2, 'to_bus': 3, 'r': 0.04699, 'x': 0.19797, 'b': 0.0438},
        {'from_bus': 2, 'to_bus': 4, 'r': 0.05811, 'x': 0.17632, 'b': 0.0374},
        {'from_bus': 2, 'to_bus': 5, 'r': 0.05695, 'x': 0.17388, 'b': 0.0340},
        {'from_bus': 3, 'to_bus': 4, 'r': 0.06701, 'x': 0.17103, 'b': 0.0346},
        {'from_bus': 4, 'to_bus': 5, 'r': 0.01335, 'x': 0.04211, 'b': 0.0128},
        {'from_bus': 4, 'to_bus': 7, 'r': 0.0,     'x': 0.20912, 'b': 0.0},
        {'from_bus': 4, 'to_bus': 9, 'r': 0.0,     'x': 0.55618, 'b': 0.0},
        {'from_bus': 5, 'to_bus': 6, 'r': 0.0,     'x': 0.25202, 'b': 0.0},
        {'from_bus': 6, 'to_bus': 11, 'r': 0.09498, 'x': 0.19890, 'b': 0.0},
        {'from_bus': 6, 'to_bus': 12, 'r': 0.12291, 'x': 0.25581, 'b': 0.0},
        {'from_bus': 6, 'to_bus': 13, 'r': 0.06615, 'x': 0.13027, 'b': 0.0},
        {'from_bus': 7, 'to_bus': 8, 'r': 0.0,     'x': 0.17615, 'b': 0.0},
        {'from_bus': 7, 'to_bus': 9, 'r': 0.0,     'x': 0.11001, 'b': 0.0},
        {'from_bus': 9, 'to_bus': 10, 'r': 0.03181, 'x': 0.08450, 'b': 0.0},
        {'from_bus': 9, 'to_bus': 14, 'r': 0.12711, 'x': 0.27038, 'b': 0.0},
        {'from_bus': 10, 'to_bus': 11, 'r': 0.08205, 'x': 0.19207, 'b': 0.0},
        {'from_bus': 12, 'to_bus': 13, 'r': 0.22092, 'x': 0.19988, 'b': 0.0},
        {'from_bus': 13, 'to_bus': 14, 'r': 0.17093, 'x': 0.34802, 'b': 0.0}
    ]
    
    return num_buses, bus_types, V_init, theta_init, P_spec, Q_spec, line_data
