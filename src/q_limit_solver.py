from dataclasses import dataclass

import numpy as np

from src.mismatch_calculator import calculate_power_injections
from src.nr_solver import solve_newton_raphson
from src.ybus_builder import build_ybus_sparse


@dataclass
class QLimitResult:
    """Structured solution and switching diagnostics for Q-limit load flow."""

    V: np.ndarray
    theta: np.ndarray
    bus_types_final: list[str]
    Q_gen: dict[int, float]
    saturated_buses: dict[int, str]
    switching_events: list[dict]
    history: list[float]
    outer_iterations: int


def _normalize_q_limits(q_limits, num_buses, bus_types):
    """Validate and normalize generator reactive capability limits."""
    normalized = {}
    items = q_limits.items() if isinstance(q_limits, dict) else [
        (entry["bus"], entry) for entry in q_limits
    ]

    for raw_bus, spec in items:
        if not isinstance(raw_bus, (int, np.integer)) or not 1 <= int(raw_bus) <= num_buses:
            raise ValueError("q_limits bus numbers must be integers between 1 and num_buses")
        bus_num = int(raw_bus)
        idx = bus_num - 1
        if bus_types[idx] != "PV":
            raise ValueError(f"q_limits can only be assigned to PV buses (got bus {bus_num})")

        q_min = float(spec["q_min"])
        q_max = float(spec["q_max"])
        q_load = float(spec.get("q_load", 0.0))
        if not np.isfinite([q_min, q_max, q_load]).all():
            raise ValueError("q_limits parameters must be finite numbers")
        if q_min > q_max:
            raise ValueError(f"q_min cannot exceed q_max at bus {bus_num}")

        normalized[bus_num] = {"q_min": q_min, "q_max": q_max, "q_load": q_load}

    return normalized


def solve_nr_with_q_limits(
    V_init,
    theta_init,
    Y_bus,
    P_spec,
    Q_spec,
    bus_types,
    q_limits,
    *,
    max_iter=20,
    tol=1e-4,
    sparse=True,
    strategy="most_severe",
    allow_reversion=True,
    max_q_switches=30,
    q_tol=1e-7,
):
    """Solve AC power flow enforcing PV generator reactive limits (Q_min/Q_max).

    Matches industrial ``pf.enforce_q_lims`` formulations (MATPOWER / PSS/E):
    at each outer stage a converged Newton-Raphson solution is checked for PV
    reactive violations ``Q_gen = Q_inj + Q_load`` outside ``[q_min, q_max]``.
    Violated PV buses are converted to PQ buses with net reactive injection
    clamped to ``q_max - q_load`` or ``q_min - q_load``.
    """
    if strategy not in ("most_severe", "all"):
        raise ValueError("strategy must be 'most_severe' or 'all'")
    if not isinstance(max_q_switches, (int, np.integer)) or max_q_switches < 1:
        raise ValueError("max_q_switches must be a positive integer")

    v_setpoints = np.asarray(V_init, dtype=float).copy()
    v_curr = v_setpoints.copy()
    theta_curr = np.asarray(theta_init, dtype=float).copy()
    p_curr = np.asarray(P_spec, dtype=float).copy()
    q_curr = np.asarray(Q_spec, dtype=float).copy()
    active_types = list(bus_types)
    num_buses = len(v_curr)

    norm_limits = _normalize_q_limits(q_limits, num_buses, active_types)
    saturated = {}
    events = []
    combined_history = []

    for outer in range(1, int(max_q_switches) + 1):
        v_sol, theta_sol, inner_hist = solve_newton_raphson(
            v_curr,
            theta_curr,
            Y_bus,
            p_curr,
            q_curr,
            active_types,
            max_iter=max_iter,
            tol=tol,
            sparse=sparse,
        )
        if combined_history:
            combined_history.extend(inner_hist[1:])
        else:
            combined_history.extend(inner_hist)

        _, q_inj = calculate_power_injections(v_sol, theta_sol, Y_bus)
        switched_this_round = False

        if allow_reversion and saturated:
            for bus_num in sorted(list(saturated.keys())):
                idx = bus_num - 1
                bound_tag = saturated[bus_num]
                v_set = v_setpoints[idx]
                if bound_tag == "Q_max" and v_sol[idx] > v_set + q_tol:
                    active_types[idx] = "PV"
                    v_sol[idx] = v_set
                    del saturated[bus_num]
                    events.append(
                        {
                            "outer_step": outer,
                            "bus": bus_num,
                            "action": "PQ->PV (backoff)",
                            "q_gen_pu": float(q_inj[idx] + norm_limits[bus_num]["q_load"]),
                            "limit_pu": float(norm_limits[bus_num]["q_max"]),
                        }
                    )
                    switched_this_round = True
                elif bound_tag == "Q_min" and v_sol[idx] < v_set - q_tol:
                    active_types[idx] = "PV"
                    v_sol[idx] = v_set
                    del saturated[bus_num]
                    events.append(
                        {
                            "outer_step": outer,
                            "bus": bus_num,
                            "action": "PQ->PV (backoff)",
                            "q_gen_pu": float(q_inj[idx] + norm_limits[bus_num]["q_load"]),
                            "limit_pu": float(norm_limits[bus_num]["q_min"]),
                        }
                    )
                    switched_this_round = True

        violations = []
        for bus_num, lim in norm_limits.items():
            idx = bus_num - 1
            if active_types[idx] != "PV":
                continue
            q_gen = float(q_inj[idx] + lim["q_load"])
            if q_gen > lim["q_max"] + q_tol:
                violations.append((q_gen - lim["q_max"], bus_num, "Q_max", q_gen, lim["q_max"]))
            elif q_gen < lim["q_min"] - q_tol:
                violations.append((lim["q_min"] - q_gen, bus_num, "Q_min", q_gen, lim["q_min"]))

        if violations:
            violations.sort(key=lambda item: (-item[0], item[1]))
            selected = violations[:1] if strategy == "most_severe" else violations
            for _, bus_num, tag, q_gen_val, bound_val in selected:
                idx = bus_num - 1
                active_types[idx] = "PQ"
                q_curr[idx] = bound_val - norm_limits[bus_num]["q_load"]
                saturated[bus_num] = tag
                events.append(
                    {
                        "outer_step": outer,
                        "bus": bus_num,
                        "action": f"PV->PQ ({tag})",
                        "q_gen_pu": q_gen_val,
                        "limit_pu": bound_val,
                    }
                )
            switched_this_round = True

        v_curr = v_sol
        theta_curr = theta_sol

        if not switched_this_round:
            q_gen_final = {
                bus_num: float(q_inj[bus_num - 1] + lim["q_load"])
                for bus_num, lim in norm_limits.items()
            }
            return QLimitResult(
                V=v_sol,
                theta=theta_sol,
                bus_types_final=active_types,
                Q_gen=q_gen_final,
                saturated_buses=dict(saturated),
                switching_events=events,
                history=combined_history,
                outer_iterations=outer,
            )

    raise RuntimeError(
        f"Q-limit enforcement did not stabilize within {max_q_switches} outer switching stages"
    )


__all__ = ["QLimitResult", "build_ybus_sparse", "solve_nr_with_q_limits"]
