from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.sparse import csc_matrix, issparse


def plot_convergence(history, save_path="results/convergence_curve.png"):
    """Generate a semi-log convergence plot of the maximum power mismatch."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8.2, 4.8))
    steps = list(range(len(history)))
    plt.semilogy(steps, history, "o-", color="#1f77b4", linewidth=2.2, markersize=6.5)
    plt.axhline(y=1e-4, color="#d62728", linestyle="--", label="Tolerance Limit (1e-4 p.u.)")
    plt.xticks(steps)
    plt.title("Newton-Raphson Solver Convergence Trajectory", fontsize=12, fontweight="bold")
    plt.xlabel("Iteration Number", fontsize=10)
    plt.ylabel("Max Power Mismatch |ΔP, ΔQ| (p.u.)", fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_voltage_profile(V, bus_types, save_path="results/voltage_profile.png"):
    """Generate a bus-voltage magnitude bar chart with adaptive width and headroom."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    n = len(V)
    prefix = "B" if n > 16 else "Bus "
    buses = [f"{prefix}{i + 1}\n({bus_types[i]})" for i in range(n)]
    colors = [
        "#2ca02c" if bus_type == "Slack" else "#ff7f0e" if bus_type == "PV" else "#1f77b4"
        for bus_type in bus_types
    ]

    fig_width = max(8.5, min(16.0, 0.76 * n))
    plt.figure(figsize=(fig_width, 5.2))
    bars = plt.bar(buses, V, color=colors, width=0.52, edgecolor="black", alpha=0.88)
    plt.axhline(
        y=1.05, color="#d62728", linestyle=":", linewidth=1.4, label="Upper Limit (1.05 p.u.)"
    )
    plt.axhline(
        y=0.95, color="#d62728", linestyle="--", linewidth=1.3, label="Lower Limit (0.95 p.u.)"
    )

    v_min = min(0.90, float(min(V)) - 0.03)
    v_max = max(1.118, float(max(V)) + 0.028)
    plt.ylim(v_min, v_max)
    plt.xticks(fontsize=8.5 if n > 10 else 9.5)
    plt.title("Bus Voltage Magnitude Profile", fontsize=12, fontweight="bold")
    plt.ylabel("Voltage Magnitude |V| (p.u.)", fontsize=10)
    plt.grid(axis="y", ls="--", alpha=0.5)
    plt.legend(loc="upper right")

    label_font = 8.5 if n <= 16 else 7.5
    for bar in bars:
        voltage = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            voltage + 0.004,
            f"{voltage:.4f}",
            ha="center",
            va="bottom",
            fontsize=label_font,
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

    plt.figure(figsize=(6.2, 6.2))
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

    converged_cases = sorted(
        [c for c in report.ranked_cases if c.converged],
        key=lambda c: c.severity_index,
        reverse=True,
    )[: int(top_n)]
    labels = [f"{c.from_bus}-{c.to_bus}" for c in converged_cases]
    severities = [c.severity_index for c in converged_cases]
    colors = ["#d62728" if c.status == "VIOLATION" else "#1f77b4" for c in converged_cases]

    plt.figure(figsize=(8.4, 4.9))
    if converged_cases:
        plt.barh(
            labels[::-1], severities[::-1], color=colors[::-1], edgecolor="black", alpha=0.88
        )
        legend_handles = [
            Patch(facecolor="#d62728", edgecolor="black", label="Limit Violation"),
            Patch(facecolor="#1f77b4", edgecolor="black", label="Secure Post-Contingency"),
        ]
        plt.legend(handles=legend_handles, loc="lower right", fontsize=9)
    plt.title(
        "Top N-1 Contingency Severity Ranking (Converged Outages)",
        fontsize=11.5,
        fontweight="bold",
    )
    plt.xlabel("Composite Security Severity Index (PI)", fontsize=10)
    plt.ylabel("Tripped Branch Corridor", fontsize=10)
    plt.grid(axis="x", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pv_nose_curve(
    unconstrained_res,
    qlim_res=None,
    bus=14,
    save_path="results/pv_nose_curve.png",
):
    """Plot P-V voltage stability nose curves comparing unconstrained vs Q-limited margins."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    bus_idx = int(bus) - 1

    plt.figure(figsize=(8.4, 5.0))
    plt.plot(
        unconstrained_res.total_load_mw,
        unconstrained_res.voltages[:, bus_idx],
        "-",
        color="#1f77b4",
        linewidth=2.2,
        label=f"Unconstrained PV (λ_max = {unconstrained_res.lambda_max:.2f}x)",
    )
    plt.scatter(
        [unconstrained_res.max_load_mw],
        [unconstrained_res.voltages[-1, bus_idx]],
        color="#1f77b4",
        s=55,
        zorder=5,
    )

    if qlim_res is not None:
        plt.plot(
            qlim_res.total_load_mw,
            qlim_res.voltages[:, bus_idx],
            "-",
            color="#d62728",
            linewidth=2.2,
            label=f"Enforced Q-Limits (λ_max = {qlim_res.lambda_max:.2f}x)",
        )
        plt.scatter(
            [qlim_res.max_load_mw],
            [qlim_res.voltages[-1, bus_idx]],
            color="#d62728",
            s=55,
            zorder=5,
        )

    plt.axhline(y=0.95, color="gray", linestyle=":", label="0.95 p.u. Statutory Floor")
    plt.title(f"Voltage Stability P-V Nose Curve at Bus {bus}", fontsize=12, fontweight="bold")
    plt.xlabel("Total System Active Load (MW)", fontsize=10)
    plt.ylabel(f"Bus {bus} Voltage Magnitude |V| (p.u.)", fontsize=10)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
