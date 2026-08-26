import pytest
import numpy as np
from src.ybus_builder import build_ybus

def test_ybus_symmetry():
    line_data = [
        {'from_bus': 1, 'to_bus': 2, 'r': 0.02, 'x': 0.06, 'b': 0.06},
        {'from_bus': 1, 'to_bus': 3, 'r': 0.08, 'x': 0.24, 'b': 0.05},
        {'from_bus': 2, 'to_bus': 3, 'r': 0.06, 'x': 0.18, 'b': 0.04}
    ]
    Y_bus = build_ybus(3, line_data)
    
    # Assert matrix is square
    assert Y_bus.shape == (3, 3)
    
    # Assert matrix symmetry Y_ij == Y_ji
    np.testing.assert_almost_equal(Y_bus, Y_bus.T)

def test_ybus_diagonal_positive_conductance():
    line_data = [
        {'from_bus': 1, 'to_bus': 2, 'r': 0.02, 'x': 0.06, 'b': 0.06}
    ]
    Y_bus = build_ybus(2, line_data)
    
    # Diagonal real parts (conductances) must be positive
    assert np.real(Y_bus[0, 0]) > 0
    assert np.real(Y_bus[1, 1]) > 0
