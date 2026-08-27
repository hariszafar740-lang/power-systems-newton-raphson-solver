import numpy as np
from src.ybus_builder import build_ybus
from src.ieee14_data import get_ieee14_data
from src.nr_solver import solve_newton_raphson
from src.line_flow_calculator import calculate_line_flows
from src.visualizer import plot_convergence, plot_voltage_profile

def main():
    num_buses, bus_types, V_init, theta_init, P_spec, Q_spec, line_data = get_ieee14_data()
    
    Y_bus = build_ybus(num_buses, line_data)
    
    V_conv, theta_conv, history = solve_newton_raphson(
        V_init, theta_init, Y_bus, P_spec, Q_spec, bus_types, max_iter=20, tol=1e-4
    )
    
    flows, total_p_loss, total_q_loss = calculate_line_flows(V_conv, theta_conv, line_data)
    
    # Save Visual Artifacts
    plot_convergence(history, save_path='results/convergence_curve.png')
    plot_voltage_profile(V_conv, bus_types, save_path='results/voltage_profile.png')
    
    print("=" * 70)
    print("NEWTON-RAPHSON LOAD FLOW SOLVER (IEEE 14-BUS BENCHMARK SYSTEM)")
    print("=" * 70)
    print(f"Convergence achieved in {len(history)-1} iterations.")
    print("Bus Voltage Profiles:")
    for i in range(num_buses):
        deg = np.degrees(theta_conv[i])
        print(f"  Bus {i+1:2d} ({bus_types[i]:5s}): |V| = {V_conv[i]:.4f} p.u. | Theta = {deg:+.4f} deg")
        
    print("-" * 70)
    print(f"Total IEEE 14 Active Losses (P_loss):   {total_p_loss:.4f} p.u.")
    print(f"Total IEEE 14 Reactive Losses (Q_loss): {total_q_loss:.4f} p.u.")
    print("Artifacts saved: /results/convergence_curve.png & /results/voltage_profile.png")
    print("=" * 70)

if __name__ == "__main__":
    main()
