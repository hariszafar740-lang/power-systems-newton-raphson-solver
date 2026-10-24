from collections import deque
from dataclasses import dataclass

import numpy as np

from src.line_flow_calculator import calculate_line_flows
from src.nr_solver import solve_newton_raphson
from src.q_limit_solver import solve_nr_with_q_limits
from src.ybus_builder import build_ybus_sparse


@dataclass
class ContingencyCase:
    """Post-contingency telemetry for a single N-1 branch outage."""

    branch_index: int
    from_bus: int
    to_bus: int
    status: str
    converged: bool
    islanded_buses: list[int]
    min_v_pu: float | None
    min_v_bus: int | None
    max_v_pu: float | None
    max_branch_mva: float | None
    max_mva_corridor: tuple[int, int] | None
    p_loss_pu: float | None
    voltage_violations: list[dict]
    thermal_violations: list[dict]
    saturated_pv_buses: dict[int, str]
    severity_index: float


@dataclass
class ContingencyReport:
    """Complete N-1 security assessment summary and ranked outage list."""

    base_p_loss_pu: float
    base_max_branch_mva: float
    total_contingencies: int
    converged_count: int
    islanded_count: int
    diverged_count: int
    violated_count: int
    secure_count: int
    ranked_cases: list[ContingencyCase]


def find_islanded_buses(num_buses, line_data, slack_bus=1):
    """Return sorted 1-based bus numbers disconnected from ``slack_bus``."""
    if not isinstance(num_buses, (int, np.integer)) or num_buses < 1:
        raise ValueError("num_buses must be a positive integer")
    if not isinstance(slack_bus, (int, np.integer)) or not 1 <= int(slack_bus) <= num_buses:
        raise ValueError("slack_bus must be an integer between 1 and num_buses")

    adj = {bus: [] for bus in range(1, num_buses + 1)}
    for line in line_data:
        u = int(line["from_bus"])
        v = int(line["to_bus"])
        adj[u].append(v)
        adj[v].append(u)

    visited = {int(slack_bus)}
    queue = deque([int(slack_bus)])
    while queue:
        curr = queue.popleft()
        for nxt in adj[curr]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)

    return sorted(bus for bus in range(1, num_buses + 1) if bus not in visited)


def _branch_mva_metrics(flows, line_data, base_mva):
    """Compute maximum apparent power flow (MVA) and any thermal overloads."""
    max_mva = 0.0
    max_corridor = None
    thermal_violations = []
    pi_thermal = 0.0

    for flow, line in zip(flows, line_data, strict=True):
        s_ij = float(np.hypot(flow["P_ij"], flow["Q_ij"]) * base_mva)
        s_ji = float(np.hypot(flow["P_ji"], flow["Q_ji"]) * base_mva)
        s_peak = max(s_ij, s_ji)
        corridor = (int(flow["from"]), int(flow["to"]))
        if s_peak > max_mva or max_corridor is None:
            max_mva = s_peak
            max_corridor = corridor

        rate_mva = float(line.get("rate_mva", 0.0))
        if rate_mva > 0.0 and s_peak > rate_mva:
            loading_pct = 100.0 * s_peak / rate_mva
            thermal_violations.append(
                {
                    "from_bus": corridor[0],
                    "to_bus": corridor[1],
                    "flow_mva": s_peak,
                    "rate_mva": rate_mva,
                    "loading_percent": loading_pct,
                }
            )
            pi_thermal += ((s_peak - rate_mva) / rate_mva) ** 2

    return max_mva, max_corridor, thermal_violations, pi_thermal


def run_n1_contingency_analysis(
    num_buses,
    bus_types,
    V_init,
    theta_init,
    P_spec,
    Q_spec,
    line_data,
    bus_shunts=None,
    *,
    q_limits=None,
    v_min=0.95,
    v_max=1.05,
    base_mva=100.0,
    tol=1e-5,
    max_iter=25,
):
    """Execute automated N-1 single-branch contingency security screening.

    Screens every branch outage for topological islanding, AC load-flow
    convergence (with optional PV Q-limit enforcement), PQ/saturated-bus
    voltage limit violations, branch MVA loading, and composite severity PI.
    """
    if not np.isfinite([v_min, v_max, base_mva]).all() or v_min <= 0.0 or v_min >= v_max:
        raise ValueError("voltage bounds must satisfy 0 < v_min < v_max")
    if base_mva <= 0.0:
        raise ValueError("base_mva must be positive")
    if sum(bt == "Slack" for bt in bus_types) != 1:
        raise ValueError("exactly one Slack bus is required")

    slack_bus = bus_types.index("Slack") + 1
    y_base = build_ybus_sparse(num_buses, line_data, bus_shunts)

    if q_limits is not None:
        base_res = solve_nr_with_q_limits(
            V_init, theta_init, y_base, P_spec, Q_spec, bus_types, q_limits,
            max_iter=max_iter, tol=tol, sparse=True,
        )
        v_base, th_base = base_res.V, base_res.theta
    else:
        v_base, th_base, _ = solve_newton_raphson(
            V_init, theta_init, y_base, P_spec, Q_spec, bus_types,
            max_iter=max_iter, tol=tol, sparse=True,
        )

    base_flows, base_p_loss, _ = calculate_line_flows(v_base, th_base, line_data)
    base_max_mva, _, _, _ = _branch_mva_metrics(base_flows, line_data, base_mva)

    cases = []
    for idx, tripped in enumerate(line_data):
        u = int(tripped["from_bus"])
        v = int(tripped["to_bus"])
        remaining_lines = [line for k, line in enumerate(line_data) if k != idx]

        islands = find_islanded_buses(num_buses, remaining_lines, slack_bus=slack_bus)
        if islands:
            cases.append(
                ContingencyCase(
                    branch_index=idx,
                    from_bus=u,
                    to_bus=v,
                    status="ISLANDED",
                    converged=False,
                    islanded_buses=islands,
                    min_v_pu=None,
                    min_v_bus=None,
                    max_v_pu=None,
                    max_branch_mva=None,
                    max_mva_corridor=None,
                    p_loss_pu=None,
                    voltage_violations=[],
                    thermal_violations=[],
                    saturated_pv_buses={},
                    severity_index=1000.0 + float(len(islands)),
                )
            )
            continue

        y_out = build_ybus_sparse(num_buses, remaining_lines, bus_shunts)
        v_start = V_init.copy()
        pq_mask = np.array([bt == "PQ" for bt in bus_types])
        v_start[pq_mask] = v_base[pq_mask]

        try:
            if q_limits is not None:
                out_res = solve_nr_with_q_limits(
                    v_start, th_base, y_out, P_spec, Q_spec, bus_types, q_limits,
                    max_iter=max_iter, tol=tol, sparse=True,
                )
                v_post = out_res.V
                th_post = out_res.theta
                final_types = out_res.bus_types_final
                sat_buses = out_res.saturated_buses
            else:
                v_post, th_post, _ = solve_newton_raphson(
                    v_start, th_base, y_out, P_spec, Q_spec, bus_types,
                    max_iter=max_iter, tol=tol, sparse=True,
                )
                final_types = list(bus_types)
                sat_buses = {}
        except RuntimeError:
            cases.append(
                ContingencyCase(
                    branch_index=idx,
                    from_bus=u,
                    to_bus=v,
                    status="NON_CONVERGED",
                    converged=False,
                    islanded_buses=[],
                    min_v_pu=None,
                    min_v_bus=None,
                    max_v_pu=None,
                    max_branch_mva=None,
                    max_mva_corridor=None,
                    p_loss_pu=None,
                    voltage_violations=[],
                    thermal_violations=[],
                    saturated_pv_buses={},
                    severity_index=900.0,
                )
            )
            continue

        flows_post, p_loss_post, _ = calculate_line_flows(v_post, th_post, remaining_lines)
        max_mva, max_corr, thermal_viols, pi_s = _branch_mva_metrics(
            flows_post, remaining_lines, base_mva
        )

        v_viols = []
        pi_v = 0.0
        span = 0.5 * (v_max - v_min)
        monitored_indices = [i for i, bt in enumerate(final_types) if bt == "PQ"]
        for i in monitored_indices:
            vm = float(v_post[i])
            if vm < v_min:
                v_viols.append({"bus": i + 1, "vm_pu": vm, "type": "UNDERVOLTAGE", "limit": v_min})
                pi_v += ((v_min - vm) / span) ** 2
            elif vm > v_max:
                v_viols.append({"bus": i + 1, "vm_pu": vm, "type": "OVERVOLTAGE", "limit": v_max})
                pi_v += ((vm - v_max) / span) ** 2

        loss_ratio = max(0.0, (p_loss_post - base_p_loss) / max(base_p_loss, 1e-6))
        mva_ratio = max_mva / max(base_max_mva, 1e-6)
        severity = 10.0 * (pi_v + pi_s) + loss_ratio + 0.25 * mva_ratio

        monitored_v = v_post[monitored_indices] if monitored_indices else v_post
        min_idx = (
            monitored_indices[int(np.argmin(monitored_v))]
            if monitored_indices
            else int(np.argmin(v_post))
        )
        status = "VIOLATION" if (v_viols or thermal_viols) else "SECURE"

        cases.append(
            ContingencyCase(
                branch_index=idx,
                from_bus=u,
                to_bus=v,
                status=status,
                converged=True,
                islanded_buses=[],
                min_v_pu=float(v_post[min_idx]),
                min_v_bus=min_idx + 1,
                max_v_pu=float(np.max(monitored_v)),
                max_branch_mva=max_mva,
                max_mva_corridor=max_corr,
                p_loss_pu=float(p_loss_post),
                voltage_violations=v_viols,
                thermal_violations=thermal_viols,
                saturated_pv_buses=sat_buses,
                severity_index=float(severity),
            )
        )

    status_priority = {"ISLANDED": 0, "NON_CONVERGED": 1, "VIOLATION": 2, "SECURE": 3}
    ranked = sorted(
        cases,
        key=lambda c: (status_priority[c.status], -c.severity_index, c.branch_index),
    )

    return ContingencyReport(
        base_p_loss_pu=float(base_p_loss),
        base_max_branch_mva=float(base_max_mva),
        total_contingencies=len(cases),
        converged_count=sum(1 for c in cases if c.converged),
        islanded_count=sum(1 for c in cases if c.status == "ISLANDED"),
        diverged_count=sum(1 for c in cases if c.status == "NON_CONVERGED"),
        violated_count=sum(1 for c in cases if c.status == "VIOLATION"),
        secure_count=sum(1 for c in cases if c.status == "SECURE"),
        ranked_cases=ranked,
    )


def format_contingency_table(report, top_n=10):
    """Render a concise Markdown table of top-ranked N-1 contingencies."""
    lines = [
        "| Rank | Outage | Status | Min V (p.u.) | Max Flow (MVA) | P_loss (p.u.) | Severity PI |",
        "|---:|:---:|:---|---:|---:|---:|---:|",
    ]
    for rank, case in enumerate(report.ranked_cases[: int(top_n)], start=1):
        outage = f"{case.from_bus}-{case.to_bus} (#{case.branch_index + 1})"
        min_v = f"{case.min_v_pu:.4f} (B{case.min_v_bus})" if case.min_v_pu is not None else "N/A"
        max_s = f"{case.max_branch_mva:.2f}" if case.max_branch_mva is not None else "N/A"
        ploss = f"{case.p_loss_pu:.4f}" if case.p_loss_pu is not None else "N/A"
        lines.append(
            f"| {rank} | {outage} | {case.status} | {min_v} | {max_s} | "
            f"{ploss} | {case.severity_index:.3f} |"
        )
    return "\n".join(lines)
