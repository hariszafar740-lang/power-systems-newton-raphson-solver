"""Command-line interface (``nr-powerflow``) for AC load flow, N-1 security
screening, P-V voltage stability continuation, and sparse scaling benchmarks."""

import argparse
import json
from pathlib import Path

import numpy as np

from src import __version__
from src.bus_data import get_3bus_data
from src.contingency_analyzer import (
    format_contingency_table,
    run_n1_contingency_analysis,
)
from src.ieee14_data import (
    get_ieee14_bus_shunts,
    get_ieee14_data,
    scale_ieee14_loads,
)
from src.ieee30_data import (
    get_ieee30_bus_shunts,
    get_ieee30_data,
    scale_ieee30_loads,
)
from src.jacobian_builder import build_jacobian_sparse
from src.line_flow_calculator import calculate_line_flows
from src.nr_solver import solve_newton_raphson
from src.q_limit_solver import solve_nr_with_q_limits
from src.sparse_benchmark import benchmark_dense_vs_sparse
from src.visualizer import (
    plot_convergence,
    plot_jacobian_sparsity,
    plot_pv_nose_curve,
    plot_voltage_profile,
)
from src.voltage_stability import trace_ieee14_pv_curves
from src.ybus_builder import build_ybus


def _load_case(case_name, load_mult=1.0):
    if case_name == "3bus":
        btypes, v0, th0, ps, qs = get_3bus_data()
        lines = [
            {"from_bus": 1, "to_bus": 2, "r": 0.02, "x": 0.06, "b": 0.06},
            {"from_bus": 1, "to_bus": 3, "r": 0.08, "x": 0.24, "b": 0.05},
            {"from_bus": 2, "to_bus": 3, "r": 0.06, "x": 0.18, "b": 0.04},
        ]
        q_lims = {2: {"q_min": -0.40, "q_max": 0.60, "q_load": 0.0}}
        return 3, btypes, v0, th0, ps * load_mult, qs * load_mult, lines, None, q_lims
    if case_name == "ieee14":
        n, btypes, v0, th0, _, _, lines = get_ieee14_data()
        ps, qs, q_lims = scale_ieee14_loads(load_mult)
        return n, btypes, v0, th0, ps, qs, lines, get_ieee14_bus_shunts(), q_lims
    if case_name == "ieee30":
        n, btypes, v0, th0, _, _, lines = get_ieee30_data()
        ps, qs, q_lims = scale_ieee30_loads(load_mult)
        return n, btypes, v0, th0, ps, qs, lines, get_ieee30_bus_shunts(), q_lims
    raise ValueError(f"Unsupported case: {case_name}")


def _cmd_solve(args):
    n, btypes, v0, th0, ps, qs, lines, shunts, q_lims = _load_case(
        args.case, args.load_mult
    )
    use_sparse = not args.dense
    y_mat = build_ybus(n, lines, shunts, sparse=use_sparse)

    if args.enforce_q_limits:
        q_res = solve_nr_with_q_limits(
            v0, th0, y_mat, ps, qs, btypes, q_lims,
            max_iter=args.max_iter, tol=args.tol, sparse=use_sparse,
        )
        v_sol, th_sol, history = q_res.V, q_res.theta, q_res.history
        final_types = q_res.bus_types_final
        saturated = q_res.saturated_buses
    else:
        v_sol, th_sol, history = solve_newton_raphson(
            v0, th0, y_mat, ps, qs, btypes,
            max_iter=args.max_iter, tol=args.tol, sparse=use_sparse,
        )
        final_types = list(btypes)
        saturated = {}

    _, p_loss, q_loss = calculate_line_flows(v_sol, th_sol, lines)

    if args.plots_dir:
        out_dir = Path(args.plots_dir)
        plot_convergence(history, save_path=str(out_dir / "convergence_curve.png"))
        plot_voltage_profile(v_sol, final_types, save_path=str(out_dir / "voltage_profile.png"))
        j_sp = build_jacobian_sparse(v_sol, th_sol, y_mat, final_types)
        plot_jacobian_sparsity(j_sp, save_path=str(out_dir / "jacobian_sparsity.png"))

    payload = {
        "case": args.case,
        "sparse": use_sparse,
        "load_multiplier": args.load_mult,
        "enforce_q_limits": args.enforce_q_limits,
        "iterations": len(history) - 1,
        "final_mismatch_pu": float(history[-1]),
        "p_loss_pu": float(p_loss),
        "q_loss_pu": float(q_loss),
        "saturated_pv_buses": {str(k): v for k, v in saturated.items()},
        "buses": [
            {
                "bus": i + 1,
                "type": final_types[i],
                "vm_pu": round(float(v_sol[i]), 6),
                "va_deg": round(float(np.degrees(th_sol[i])), 6),
            }
            for i in range(n)
        ],
    }

    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        mode_str = "Sparse CSC" if use_sparse else "Dense NumPy"
        print(f"Case: {args.case.upper()} | Engine: {mode_str} | Load: {args.load_mult:.2f}x")
        print(
            f"Converged Steps: {payload['iterations']} | "
            f"P_loss: {p_loss:.4f} p.u. | Q_loss: {q_loss:.4f} p.u."
        )
        if saturated:
            print(f"Saturated PV Buses: {saturated}")
        for row in payload["buses"]:
            print(
                f"  Bus {row['bus']:2d} ({row['type']:5s}): "
                f"|V| = {row['vm_pu']:.4f} p.u. | Theta = {row['va_deg']:+.4f} deg"
            )
    return 0


def _cmd_contingency(args):
    n, btypes, v0, th0, ps, qs, lines, shunts, q_lims = _load_case(args.case, 1.0)
    rep = run_n1_contingency_analysis(
        n,
        btypes,
        v0,
        th0,
        ps,
        qs,
        lines,
        shunts,
        q_limits=q_lims if args.enforce_q_limits else None,
        v_min=args.v_min,
        v_max=args.v_max,
    )
    if args.format == "json":
        data = {
            "case": args.case,
            "total_contingencies": rep.total_contingencies,
            "secure_count": rep.secure_count,
            "violated_count": rep.violated_count,
            "islanded_count": rep.islanded_count,
            "diverged_count": rep.diverged_count,
            "top_cases": [
                {
                    "outage": f"{c.from_bus}-{c.to_bus}",
                    "status": c.status,
                    "min_v_pu": c.min_v_pu,
                    "max_branch_mva": c.max_branch_mva,
                    "severity_index": round(c.severity_index, 4),
                }
                for c in rep.ranked_cases[: args.top]
            ],
        }
        print(json.dumps(data, indent=2))
    else:
        print(format_contingency_table(rep, top_n=args.top))
    return 0


def _cmd_pv_curve(args):
    unc, qlim = trace_ieee14_pv_curves(min_step=args.min_step)
    if args.plot:
        plot_pv_nose_curve(unc, qlim_res=qlim, bus=args.bus, save_path=args.plot)

    payload = {
        "case": "ieee14",
        "monitored_bus": args.bus,
        "unconstrained_lambda_max": round(unc.lambda_max, 4),
        "unconstrained_max_load_mw": round(unc.max_load_mw, 2),
        "q_limited_lambda_max": round(qlim.lambda_max, 4),
        "q_limited_max_load_mw": round(qlim.max_load_mw, 2),
        "q_limited_margin_pct": round(qlim.loadability_margin_pct, 2),
        "weakest_bus": qlim.weakest_bus,
        "critical_voltage_pu": round(qlim.critical_voltage_pu, 4),
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"IEEE 14 P-V Stability: Q-limited λ_max = {qlim.lambda_max:.4f}x "
            f"({qlim.max_load_mw:.1f} MW, +{qlim.loadability_margin_pct:.1f}% margin) | "
            f"Unconstrained λ_max = {unc.lambda_max:.4f}x ({unc.max_load_mw:.1f} MW)"
        )
        print(
            f"Weakest PQ Bus at Collapse: Bus {qlim.weakest_bus} "
            f"(|V|_crit = {qlim.critical_voltage_pu:.4f} p.u.)"
        )
    return 0


def _cmd_benchmark(args):
    rows = benchmark_dense_vs_sparse(bus_sizes=tuple(args.sizes))
    if args.format == "json":
        print(json.dumps(rows, indent=2))
    else:
        print("| Buses | Jacobian Dim | NNZ | Sparsity (%) | Dense (ms) | Sparse (ms) | Speedup |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            print(
                f"| {r['num_buses']} | {r['jacobian_dim']} | {r['jacobian_nnz']} | "
                f"{r['sparsity_percent']:.1f}% | {r['dense_ms']:.2f} | "
                f"{r['sparse_ms']:.2f} | {r['speedup']:.2f}x |"
            )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="nr-powerflow",
        description="Sparse CSC Newton-Raphson AC Power-Flow & Grid Security CLI",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_solve = sub.add_parser("solve", help="Solve balanced AC power flow")
    p_solve.add_argument("--case", choices=["3bus", "ieee14", "ieee30"], default="ieee14")
    p_solve.add_argument("--load-mult", type=float, default=1.0)
    p_solve.add_argument("--enforce-q-limits", action="store_true")
    p_solve.add_argument("--dense", action="store_true", help="Use dense NumPy solver")
    p_solve.add_argument("--tol", type=float, default=1e-6)
    p_solve.add_argument("--max-iter", type=int, default=25)
    p_solve.add_argument("--format", choices=["text", "json"], default="text")
    p_solve.add_argument("--plots-dir", type=str, default=None)
    p_solve.set_defaults(func=_cmd_solve)

    p_cont = sub.add_parser("contingency", help="Run N-1 branch security screening")
    p_cont.add_argument("--case", choices=["ieee14", "ieee30"], default="ieee14")
    p_cont.add_argument("--enforce-q-limits", action="store_true")
    p_cont.add_argument("--v-min", type=float, default=0.94)
    p_cont.add_argument("--v-max", type=float, default=1.06)
    p_cont.add_argument("--top", type=int, default=10)
    p_cont.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p_cont.set_defaults(func=_cmd_contingency)

    p_pv = sub.add_parser("pv-curve", help="Trace P-V nose curve and collapse margin")
    p_pv.add_argument("--bus", type=int, default=14)
    p_pv.add_argument("--min-step", type=float, default=5e-3)
    p_pv.add_argument("--plot", type=str, default=None)
    p_pv.add_argument("--format", choices=["text", "json"], default="text")
    p_pv.set_defaults(func=_cmd_pv_curve)

    p_bench = sub.add_parser("benchmark", help="Run dense vs sparse CSC scaling benchmark")
    p_bench.add_argument("--sizes", type=int, nargs="+", default=[14, 30, 80, 160])
    p_bench.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p_bench.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
