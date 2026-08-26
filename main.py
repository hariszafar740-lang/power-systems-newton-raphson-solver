import numpy as np
from src.ybus_builder import build_ybus
from src.bus_data import get_3bus_data
from src.nr_solver import solve_newton_raphson
from src.line_flow_calculator import calculate_line_flows
from src.visualizer import plot_convergence, plot_voltage_profile

def main():
    num_buses = 3
    line_data = [
        {'from_bus': 1, 'to_bus': 2, 'r': 0.02, 'x': 0.06, 'b': 0.06},
        {'from_bus': 1, 'to_bus': 3, 'r': 0.08, 'x': 0.24, 'b': 0.05},
        {'from_bus': 2, 'to_bus': 3, 'r': 0.06, 'x': 0.18, 'b': 0.04}
    ]
    
    Y_bus = build_ybus(num_buses, line_data)
    bus_types, V_init, theta_init, P_spec, Q_spec = get_3bus_data()
    
    V_conv, theta_conv, history = solve_newton_raphson(
        V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types, max_iter=20, tol=1e-4
    )
    
    flows, total_p_loss, total_q_loss = calculate_line_flows(V_conv, theta_conv, line_data)
    
    # Save Visual Artifacts
    plot_convergence(history)
    plot_voltage_profile(V_conv, bus_types)
    
    print("=" * 65)
    print("NEWTON-RAPHSON LOAD FLOW SOLVER (ENHANCED PIPELINE)")
    print("=" * 65)
    print("Bus Voltage Profiles:")
    for i in range(num_buses):
        deg = np.degrees(theta_conv[i])
        print(f"  Bus {i+1} ({bus_types[i]:5s}): |V| = {V_conv[i]:.4f} p.u. | Theta = {deg:+.4f} deg")
        
    print("-" * 65)
    print(f"Total Active Losses: {total_p_loss:.4f} p.u. | Total Reactive Losses: {total_q_loss:.4f} p.u.")
    print("Visual artifacts saved to: /results/convergence_curve.png & /results/voltage_profile.png")
    print("=" * 65)

if __name__ == "__main__":
    main()
