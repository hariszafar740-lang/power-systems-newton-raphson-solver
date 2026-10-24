from pathlib import Path

import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, issparse


def plot_convergence(history, save_path="results/convergence_curve.png"):
    """Generate a semi-log convergence plot of the maximum power mismatch."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.semilogy(range(len(history)), history, "o-", color="#1f77b4", linewidth=2, markersize=6)
    plt.axhline(y=1e-4, color="r", linestyle="--", label="Tolerance Limit (1e-4 p.u.)")
    plt.title("Newton-Raphson Solver Convergence Trajectory", fontsize=12, fontweight="bold")
    plt.xlabel("Iteration Number", fontsize=10)
    plt.ylabel("Max Power Mismatch |ΔP, ΔQ| (p.u.)", fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_voltage_profile(V, bus_types, save_path="results/voltage_profile.png"):
    """Generate a bus-voltage magnitude bar chart."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    buses = [f"Bus {i + 1}\n({bus_types[i]})" for i in range(len(V))]
    colors = [
        "#2ca02c" if bus_type == "Slack" else "#ff7f0e" if bus_type == "PV" else "#1f77b4"
        for bus_type in bus_types
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(buses, V, color=colors, width=0.4, edgecolor="black", alpha=0.85)
    plt.axhline(y=1.05, color="r", linestyle=":", label="Upper Voltage Limit (1.05 p.u.)")
    plt.axhline(y=0.95, color="r", linestyle=":", label="Lower Voltage Limit (0.95 p.u.)")
    plt.ylim(0.90, 1.10)
    plt.title("Bus Voltage Magnitude Profile", fontsize=12, fontweight="bold")
    plt.ylabel("Voltage Magnitude |V| (p.u.)", fontsize=10)
    plt.grid(axis="y", ls="--", alpha=0.5)
    plt.legend()

    for bar in bars:
        voltage = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            voltage + 0.005,
            f"{voltage:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_jacobian_sparsity(jacobian, save_path="results/jacobian_sparsity.png"):
    """Render the non-zero sparsity pattern of a dense or CSC Jacobian matrix."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    j_csc = jacobian.tocsc() if issparse(jacobian) else csc_matrix(jacobian)
    dim = j_csc.shape[0]
    sparsity_pct = 100.0 * (1.0 - j_csc.nnz / (dim * dim)) if dim > 0 else 0.0

    plt.figure(figsize=(6, 6))
    plt.spy(j_csc, markersize=4, color="#1f77b4")
    plt.title(
        f"Jacobian Sparsity Pattern ({dim}x{dim}, nnz={j_csc.nnz}, {sparsity_pct:.1f}% zeros)",
        fontsize=11,
        fontweight="bold",
    )
    plt.xlabel("State Variable Column Index", fontsize=10)
    plt.ylabel("Mismatch Equation Row Index", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_contingency_ranking(report, save_path="results/contingency_ranking.png", top_n=10):
    """Plot post-contingency severity index ranking for converged N-1 outages."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    converged_cases = [c for c in report.ranked_cases if c.converged][: int(top_n)]
    labels = [f"{c.from_bus}-{c.to_bus}" for c in converged_cases]
    severities = [c.severity_index for c in converged_cases]
    colors = ["#d62728" if c.status == "VIOLATION" else "#1f77b4" for c in converged_cases]

    plt.figure(figsize=(8, 4.8))
    if converged_cases:
        plt.barh(labels[::-1], severities[::-1], color=colors[::-1], edgecolor="black", alpha=0.85)
    plt.title(
        "Top N-1 Contingency Severity Ranking (Converged Outages)",
        fontsize=11,
        fontweight="bold",
    )
    plt.xlabel("Composite Security Severity Index (PI)", fontsize=10)
    plt.ylabel("Tripped Branch Corridor", fontsize=10)
    plt.grid(axis="x", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
