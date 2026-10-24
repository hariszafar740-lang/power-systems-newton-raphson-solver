import numpy as np

from src.contingency_analyzer import run_n1_contingency_analysis
from src.ieee14_data import (
    get_ieee14_bus_shunts,
    get_ieee14_data,
    get_ieee14_q_limits,
    scale_ieee14_loads,
)
from src.ieee30_data import get_ieee30_bus_shunts, get_ieee30_data
from src.jacobian_builder import build_jacobian_sparse
from src.line_flow_calculator import calculate_line_flows
from src.nr_solver import solve_newton_raphson
from src.q_limit_solver import solve_nr_with_q_limits
from src.visualizer import plot_convergence, plot_voltage_profile
from src.ybus_builder import build_ybus_sparse


def main():
    num_buses, bus_types, V_init, theta_init, P_spec, Q_spec, line_data = get_ieee14_data()
    shunts14 = get_ieee14_bus_shunts()
    Y_bus = build_ybus_sparse(num_buses, line_data, shunts14)

    V_conv, theta_conv, history = solve_newton_raphson(
        V_init,
        theta_init,
        Y_bus,
        P_spec,
        Q_spec,
        bus_types,
        max_iter=20,
        tol=1e-4,
        sparse=True,
    )
    _flows, total_p_loss, total_q_loss = calculate_line_flows(V_conv, theta_conv, line_data)

    plot_convergence(history, save_path="results/convergence_curve.png")
    plot_voltage_profile(V_conv, bus_types, save_path="results/voltage_profile.png")

    p_stressed, q_stressed, q_lims_stressed = scale_ieee14_loads(1.25)
    q_res = solve_nr_with_q_limits(
        V_init,
        theta_init,
        Y_bus,
        p_stressed,
        q_stressed,
        bus_types,
        q_lims_stressed,
        tol=1e-6,
        sparse=True,
    )

    n1_rep = run_n1_contingency_analysis(
        num_buses,
        bus_types,
        V_init,
        theta_init,
        P_spec,
        Q_spec,
        line_data,
        shunts14,
        q_limits=get_ieee14_q_limits(1.0),
        v_min=0.94,
        v_max=1.06,
    )

    n30, btypes30, v30_init, th30_init, p30, q30, lines30 = get_ieee30_data()
    y30_sparse = build_ybus_sparse(n30, lines30, get_ieee30_bus_shunts())
    v30_conv, th30_conv, hist30 = solve_newton_raphson(
        v30_init,
        th30_init,
        y30_sparse,
        p30,
        q30,
        btypes30,
        max_iter=20,
        tol=1e-6,
        sparse=True,
    )
    _, p30_loss, q30_loss = calculate_line_flows(v30_conv, th30_conv, lines30)
    j30_sparse = build_jacobian_sparse(v30_conv, th30_conv, y30_sparse, btypes30)

    print("=" * 70)
    print("NEWTON-RAPHSON SPARSE CSC LOAD FLOW SOLVER (IEEE 14 & IEEE 30)")
    print("=" * 70)
    print(f"IEEE 14-Bus converged in {len(history) - 1} iterations.")
    for i in range(num_buses):
        deg = np.degrees(theta_conv[i])
        print(
            f"  Bus {i + 1:2d} ({bus_types[i]:5s}): "
            f"|V| = {V_conv[i]:.4f} p.u. | Theta = {deg:+.4f} deg"
        )
    print("-" * 70)
    print(f"IEEE 14 Losses: P = {total_p_loss:.4f} p.u. | Q = {total_q_loss:.4f} p.u.")
    print(
        f"IEEE 14 Q-Limits (1.25x load): {len(q_res.saturated_buses)} PV buses saturated "
        f"{q_res.saturated_buses} | min |V| = {np.min(q_res.V):.4f} p.u."
    )
    print(
        f"IEEE 14 N-1 Security: {n1_rep.total_contingencies} contingencies | "
        f"{n1_rep.secure_count} secure, {n1_rep.violated_count} violated, "
        f"{n1_rep.islanded_count} islanded, {n1_rep.diverged_count} collapse"
    )
    sparsity30 = 100.0 * (1.0 - j30_sparse.nnz / (j30_sparse.shape[0] ** 2))
    print(
        f"IEEE 30-Bus Sparse CSC: converged in {len(hist30) - 1} iterations | "
        f"P_loss = {p30_loss:.4f} p.u., Q_loss = {q30_loss:.4f} p.u. | "
        f"Jacobian nnz = {j30_sparse.nnz}/{j30_sparse.shape[0] ** 2} ({sparsity30:.1f}% sparse)"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
