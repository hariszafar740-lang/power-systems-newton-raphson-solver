import pytest
import numpy as np
from src.ybus_builder import build_ybus
from src.bus_data import get_3bus_data
from src.ieee14_data import get_ieee14_data
from src.nr_solver import solve_newton_raphson

def test_3bus_convergence():
    bus_types, V_init, theta_init, P_spec, Q_spec = get_3bus_data()
    line_data = [
        {'from_bus': 1, 'to_bus': 2, 'r': 0.02, 'x': 0.06, 'b': 0.06},
        {'from_bus': 1, 'to_bus': 3, 'r': 0.08, 'x': 0.24, 'b': 0.05},
        {'from_bus': 2, 'to_bus': 3, 'r': 0.06, 'x': 0.18, 'b': 0.04}
    ]
    Y_bus = build_ybus(3, line_data)
    V_conv, theta_conv, history = solve_newton_raphson(
        V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types, max_iter=20, tol=1e-4
    )
    assert len(history) < 10
    assert history[-1] < 1e-4

def test_ieee14_convergence():
    num_buses, bus_types, V_init, theta_init, P_spec, Q_spec, line_data = get_ieee14_data()
    Y_bus = build_ybus(num_buses, line_data)
    V_conv, theta_conv, history = solve_newton_raphson(
        V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types, max_iter=20, tol=1e-4
    )
    assert len(history) < 10
    assert history[-1] < 1e-4
