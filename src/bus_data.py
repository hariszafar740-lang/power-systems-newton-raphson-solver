import numpy as np

def get_3bus_data():
    """
    Standard 3-bus test system specifications:
    - Bus 1: Slack Bus (V = 1.05 p.u., theta = 0 rad)
    - Bus 2: PV Bus (V = 1.03 p.u., P_net = +0.5 p.u.)
    - Bus 3: PQ Bus (P_net = -1.0 p.u., Q_net = -0.5 p.u.)
    """
    bus_types = ['Slack', 'PV', 'PQ']
    V = np.array([1.05, 1.03, 1.00])
    theta = np.array([0.0, 0.0, 0.0])
    
    # Specified net power injections (p.u.)
    P_spec = np.array([0.0, 0.5, -1.0])
    Q_spec = np.array([0.0, 0.0, -0.5])
    
    return bus_types, V, theta, P_spec, Q_spec
