import numpy as np
from src.ybus_builder import build_ybus
from src.bus_data import get_3bus_data
from src.mismatch_calculator import calculate_power_mismatches

def main():
    num_buses = 3
    line_data = [
        {'from_bus': 1, 'to_bus': 2, 'r': 0.02, 'x': 0.06, 'b': 0.06},
        {'from_bus': 1, 'to_bus': 3, 'r': 0.08, 'x': 0.24, 'b': 0.05},
        {'from_bus': 2, 'to_bus': 3, 'r': 0.06, 'x': 0.18, 'b': 0.04}
    ]
    
    Y_bus = build_ybus(num_buses, line_data)
    bus_types, V, theta, P_spec, Q_spec = get_3bus_data()
    
    P_calc, Q_calc, delta_P, delta_Q = calculate_power_mismatches(
        V, theta, Y_bus, P_spec, Q_spec, bus_types
    )
    
    print("=" * 60)
    print("DAY 4: INITIAL POWER MISMATCH EVALUATION (ITERATION 0)")
    print("=" * 60)
    print(f"Bus Types: {bus_types}")
    print(f"Calculated Real Power P_calc (p.u.): {np.round(P_calc, 4)}")
    print(f"Calculated Reactive Power Q_calc (p.u.): {np.round(Q_calc, 4)}")
    print("-" * 60)
    print(f"Active Power Mismatch Vector Delta P (p.u.): {np.round(delta_P, 4)}")
    print(f"Reactive Power Mismatch Vector Delta Q (p.u.): {np.round(delta_Q, 4)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
