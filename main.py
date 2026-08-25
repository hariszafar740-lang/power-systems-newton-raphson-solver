import numpy as np
from src.ybus_builder import build_ybus
from src.bus_data import get_3bus_data
from src.mismatch_calculator import calculate_power_mismatches
from src.jacobian_builder import build_jacobian

def main():
    num_buses = 3
    line_data = [
        {'from_bus': 1, 'to_bus': 2, 'r': 0.02, 'x': 0.06, 'b': 0.06},
        {'from_bus': 1, 'to_bus': 3, 'r': 0.08, 'x': 0.24, 'b': 0.05},
        {'from_bus': 2, 'to_bus': 3, 'r': 0.06, 'x': 0.18, 'b': 0.04}
    ]
    
    Y_bus = build_ybus(num_buses, line_data)
    bus_types, V, theta, P_spec, Q_spec = get_3bus_data()
    
    _, _, delta_P, delta_Q = calculate_power_mismatches(
        V, theta, Y_bus, P_spec, Q_spec, bus_types
    )
    
    mismatch_vector = np.concatenate([delta_P, delta_Q])
    J = build_jacobian(V, theta, Y_bus, bus_types)
    
    # Solve linear system J * delta_x = mismatch_vector
    delta_x = np.linalg.solve(J, mismatch_vector)
    
    print("=" * 60)
    print("DAY 5: JACOBIAN MATRIX & CORRECTION VECTOR (ITERATION 0)")
    print("=" * 60)
    print("3x3 Jacobian Matrix J:\n")
    for row in J:
        print("  ".join([f"{val:+8.4f}" for val in row]))
    print("-" * 60)
    print(f"Mismatch Vector [Delta P, Delta Q]^T (p.u.): {np.round(mismatch_vector, 4)}")
    print(f"Correction Vector [Delta Theta_2, Delta Theta_3, Delta |V_3|]^T: {np.round(delta_x, 4)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
