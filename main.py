import numpy as np
from src.ybus_builder import build_ybus

def main():
    # Standard 3-Bus Test System Data (p.u. impedance)
    num_buses = 3
    line_data = [
        {'from_bus': 1, 'to_bus': 2, 'r': 0.02, 'x': 0.06, 'b': 0.06},
        {'from_bus': 1, 'to_bus': 3, 'r': 0.08, 'x': 0.24, 'b': 0.05},
        {'from_bus': 2, 'to_bus': 3, 'r': 0.06, 'x': 0.18, 'b': 0.04}
    ]
    
    Y_bus = build_ybus(num_buses, line_data)
    
    print("=" * 60)
    print("POWER SYSTEMS ANALYSIS: BUS ADMITTANCE MATRIX (Y_bus)")
    print("=" * 60)
    print(f"Buses: {num_buses} | Transmission Lines: {len(line_data)}\n")
    
    print("Complex Y_bus Matrix (p.u.):\n")
    for row in Y_bus:
        formatted_row = "  ".join([f"{val.real:+.3f}{val.imag:+.3f}j" for val in row])
        print(f"[{formatted_row}]")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
