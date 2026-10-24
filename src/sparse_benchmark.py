import time

import numpy as np

from src.jacobian_builder import build_jacobian_sparse
from src.nr_solver import solve_newton_raphson
from src.ybus_builder import build_ybus, build_ybus_sparse


def build_synthetic_grid(num_buses):
    """Build a well-conditioned synthetic meshed transmission network of size N.

    Topology combines a backbone ring with deterministic chord interconnections
    so average bus degree stays bounded (~3.5), mirroring realistic high-voltage
    transmission sparsity (>90% zeros for N >= 50).
    """
    if not isinstance(num_buses, (int, np.integer)) or num_buses < 4:
        raise ValueError("num_buses must be an integer >= 4")

    bus_types = ["Slack"]
    V_init = np.ones(num_buses, dtype=float)
    V_init[0] = 1.04
    theta_init = np.zeros(num_buses, dtype=float)
    P_spec = np.zeros(num_buses, dtype=float)
    Q_spec = np.zeros(num_buses, dtype=float)

    for bus_idx in range(1, num_buses):
        if bus_idx % 8 == 0:
            bus_types.append("PV")
            V_init[bus_idx] = 1.02
            P_spec[bus_idx] = 0.12
        else:
            bus_types.append("PQ")
            P_spec[bus_idx] = -0.025
            Q_spec[bus_idx] = -0.010

    line_data = []
    seen_pairs = set()

    def _add_branch(u, v, r_val, x_val, b_val):
        pair = (min(u, v), max(u, v))
        if u != v and pair not in seen_pairs:
            seen_pairs.add(pair)
            line_data.append(
                {"from_bus": u, "to_bus": v, "r": r_val, "x": x_val, "b": b_val}
            )

    for i in range(1, num_buses + 1):
        nxt = 1 if i == num_buses else i + 1
        _add_branch(i, nxt, 0.012, 0.045, 0.015)
        if i + 4 <= num_buses and i % 2 == 1:
            _add_branch(i, i + 4, 0.018, 0.065, 0.010)
        if i == 1 and num_buses >= 12:
            for anchor in range(9, num_buses + 1, 8):
                _add_branch(1, anchor, 0.010, 0.038, 0.020)

    return num_buses, bus_types, V_init, theta_init, P_spec, Q_spec, line_data


def benchmark_dense_vs_sparse(bus_sizes=(14, 30, 80, 160), tol=1e-8):
    """Compare dense NumPy vs CSC SciPy sparse Newton-Raphson scaling."""
    results = []
    for n_bus in bus_sizes:
        n, btypes, v0, th0, ps, qs, lines = build_synthetic_grid(int(n_bus))

        t0 = time.perf_counter()
        y_dense = build_ybus(n, lines, sparse=False)
        v_dense, th_dense, hist_dense = solve_newton_raphson(
            v0, th0, y_dense, ps, qs, btypes, max_iter=25, tol=tol, sparse=False
        )
        dense_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        y_sparse = build_ybus_sparse(n, lines)
        v_sparse, th_sparse, hist_sparse = solve_newton_raphson(
            v0, th0, y_sparse, ps, qs, btypes, max_iter=25, tol=tol, sparse=True
        )
        sparse_ms = (time.perf_counter() - t1) * 1000.0

        j_sp = build_jacobian_sparse(v_sparse, th_sparse, y_sparse, btypes)
        dim = j_sp.shape[0]
        total_entries = dim * dim
        sparsity_pct = 100.0 * (1.0 - (j_sp.nnz / total_entries)) if total_entries else 0.0

        results.append(
            {
                "num_buses": n,
                "jacobian_dim": dim,
                "jacobian_nnz": int(j_sp.nnz),
                "sparsity_percent": sparsity_pct,
                "iterations": len(hist_sparse) - 1,
                "dense_ms": dense_ms,
                "sparse_ms": sparse_ms,
                "speedup": dense_ms / max(sparse_ms, 1e-9),
                "max_v_diff": float(np.max(np.abs(v_dense - v_sparse))),
                "max_theta_diff": float(np.max(np.abs(th_dense - th_sparse))),
                "dense_iterations": len(hist_dense) - 1,
            }
        )
    return results
