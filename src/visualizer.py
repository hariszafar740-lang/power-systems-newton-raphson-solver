import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(history, save_path='results/convergence_curve.png'):
    """Generates semi-log convergence curve of maximum power mismatch."""
    plt.figure(figsize=(8, 5))
    plt.semilogy(range(len(history)), history, 'o-', color='#1f77b4', linewidth=2, markersize=6)
    plt.axhline(y=1e-4, color='r', linestyle='--', label='Tolerance Limit (1e-4 p.u.)')
    plt.title('Newton-Raphson Solver Convergence Trajectory', fontsize=12, fontweight='bold')
    plt.xlabel('Iteration Number', fontsize=10)
    plt.ylabel('Max Power Mismatch |ΔP, ΔQ| (p.u.)', fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_voltage_profile(V, bus_types, save_path='results/voltage_profile.png'):
    """Generates bar plot comparing bus voltage magnitudes against standard limits."""
    buses = [f"Bus {i+1}\n({bus_types[i]})" for i in range(len(V))]
    colors = ['#2ca02c' if t == 'Slack' else '#ff7f0e' if t == 'PV' else '#1f77b4' for t in bus_types]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(buses, V, color=colors, width=0.4, edgecolor='black', alpha=0.85)
    plt.axhline(y=1.05, color='r', linestyle=':', label='Upper Voltage Limit (1.05 p.u.)')
    plt.axhline(y=0.95, color='r', linestyle=':', label='Lower Voltage Limit (0.95 p.u.)')
    plt.ylim(0.90, 1.10)
    plt.title('Bus Voltage Magnitude Profile', fontsize=12, fontweight='bold')
    plt.ylabel('Voltage Magnitude |V| (p.u.)', fontsize=10)
    plt.grid(axis='y', ls='--', alpha=0.5)
    plt.legend()
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
