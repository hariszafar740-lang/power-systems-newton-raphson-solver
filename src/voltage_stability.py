from dataclasses import dataclass

import numpy as np

from src.jacobian_builder import build_jacobian_sparse
from src.nr_solver import solve_newton_raphson
from src.q_limit_solver import solve_nr_with_q_limits
from src.ybus_builder import build_ybus_sparse


@dataclass
class PVCurveResult:
    """Voltage stability P-V nose curve trajectory and bifurcation margin."""

    lambdas: np.ndarray
    voltages: np.ndarray
    total_load_mw: np.ndarray
    min_singular_values: np.ndarray
    q_saturation_lambdas: dict[int, float]
    lambda_max: float
    base_load_mw: float
    max_load_mw: float
    loadability_margin_mw: float
    loadability_margin_pct: float
    weakest_bus: int
    critical_voltage_pu: float


def trace_pv_nose_curve(
    num_buses,
    bus_types,
    V_init,
    theta_init,
    line_data,
    p_load_base,
    q_load_base,
    *,
    p_gen_base=None,
    bus_shunts=None,
    q_limits_fn=None,
    lambda_start=0.0,
    initial_step=0.10,
    min_step=1e-3,
    lambda_cap=15.0,
    base_mva=100.0,
    tol=1e-6,
    max_iter=20,
):
    """Sweep loading factor ``lambda`` up to the saddle-node bifurcation nose point.

    Uses warm-started sparse Newton-Raphson solves with adaptive binary step
    halving near the voltage collapse boundary.  When ``q_limits_fn(lam)`` is
    provided, PV generator reactive saturation kinks are tracked automatically.
    """
    p_load = np.asarray(p_load_base, dtype=float)
    q_load = np.asarray(q_load_base, dtype=float)
    p_gen = (
        np.zeros(num_buses, dtype=float)
        if p_gen_base is None
        else np.asarray(p_gen_base, dtype=float)
    )

    if p_load.shape != (num_buses,) or q_load.shape != (num_buses,) or p_gen.shape != (num_buses,):
        raise ValueError("load and generation base arrays must have length num_buses")
    if not np.isfinite([lambda_start, initial_step, min_step, lambda_cap, base_mva]).all():
        raise ValueError("sweep parameters must be finite numbers")
    if lambda_start < 0.0 or initial_step <= 0.0 or min_step <= 0.0 or min_step > initial_step:
        raise ValueError("require lambda_start >= 0 and 0 < min_step <= initial_step")
    if base_mva <= 0.0:
        raise ValueError("base_mva must be positive")
    if np.sum(p_load) <= 0.0 and np.sum(np.abs(q_load)) <= 0.0:
        raise ValueError("at least one positive load component is required to trace a P-V curve")

    y_sparse = build_ybus_sparse(num_buses, line_data, bus_shunts)
    v_prev = np.asarray(V_init, dtype=float).copy()
    th_prev = np.asarray(theta_init, dtype=float).copy()
    pq_mask = np.array([bt == "PQ" for bt in bus_types])

    lam = float(lambda_start)
    step = float(initial_step)

    lambdas = []
    voltages = []
    min_sigmas = []
    sat_first_lambda = {}

    while lam <= lambda_cap:
        p_spec = p_gen - lam * p_load
        q_spec = -lam * q_load

        v_guess = np.asarray(V_init, dtype=float).copy()
        v_guess[pq_mask] = v_prev[pq_mask]

        try:
            if q_limits_fn is not None:
                q_lims = q_limits_fn(lam)
                res = solve_nr_with_q_limits(
                    v_guess,
                    th_prev,
                    y_sparse,
                    p_spec,
                    q_spec,
                    bus_types,
                    q_lims,
                    max_iter=max_iter,
                    tol=tol,
                    sparse=True,
                )
                v_sol, th_sol = res.V, res.theta
                eff_types = res.bus_types_final
                for b_num in res.saturated_buses:
                    if b_num not in sat_first_lambda:
                        sat_first_lambda[b_num] = lam
            else:
                v_sol, th_sol, _ = solve_newton_raphson(
                    v_guess,
                    th_prev,
                    y_sparse,
                    p_spec,
                    q_spec,
                    bus_types,
                    max_iter=max_iter,
                    tol=tol,
                    sparse=True,
                )
                eff_types = list(bus_types)

            j_sp = build_jacobian_sparse(v_sol, th_sol, y_sparse, eff_types)
            sigma_min = float(np.linalg.svd(j_sp.toarray(), compute_uv=False)[-1])

            lambdas.append(lam)
            voltages.append(v_sol.copy())
            min_sigmas.append(sigma_min)

            v_prev = v_sol
            th_prev = th_sol
            lam += step
        except RuntimeError:
            if not lambdas:
                raise RuntimeError("base starting loading point failed to converge") from None
            lam -= step
            step *= 0.5
            if step < min_step:
                break
            lam += step

    lam_arr = np.asarray(lambdas, dtype=float)
    v_arr = np.vstack(voltages)
    sigma_arr = np.asarray(min_sigmas, dtype=float)
    base_mw = float(np.sum(p_load) * base_mva)
    load_mw_arr = lam_arr * base_mw

    lam_max = float(lam_arr[-1])
    max_mw = float(load_mw_arr[-1])
    v_crit_vec = v_arr[-1]

    pq_indices = [i for i, bt in enumerate(bus_types) if bt == "PQ"]
    if pq_indices:
        weak_local = int(np.argmin(v_crit_vec[pq_indices]))
        weak_idx = pq_indices[weak_local]
    else:
        weak_idx = int(np.argmin(v_crit_vec))

    return PVCurveResult(
        lambdas=lam_arr,
        voltages=v_arr,
        total_load_mw=load_mw_arr,
        min_singular_values=sigma_arr,
        q_saturation_lambdas=sat_first_lambda,
        lambda_max=lam_max,
        base_load_mw=base_mw,
        max_load_mw=max_mw,
        loadability_margin_mw=max_mw - base_mw,
        loadability_margin_pct=100.0 * (lam_max - 1.0),
        weakest_bus=weak_idx + 1,
        critical_voltage_pu=float(v_crit_vec[weak_idx]),
    )


def trace_ieee14_pv_curves(min_step=5e-3):
    """Convenience runner returning ``(unconstrained, q_limited)`` IEEE 14 P-V curves."""
    from src.ieee14_data import (
        get_ieee14_bus_shunts,
        get_ieee14_data,
        get_ieee14_q_limits,
    )

    num_buses, bus_types, V_init, theta_init, _, _, lines = get_ieee14_data()
    p_gen = np.array([0.0, 0.40, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    p_load = np.array(
        [0.0, 0.217, 0.942, 0.478, 0.076, 0.112, 0.0, 0.0,
         0.295, 0.09, 0.035, 0.061, 0.135, 0.149]
    )
    q_load_pq = np.array(
        [0.0, 0.0, 0.0, -0.039, 0.016, 0.0, 0.0, 0.0,
         0.166, 0.058, 0.018, 0.016, 0.058, 0.05]
    )
    shunts = get_ieee14_bus_shunts()

    unconstrained = trace_pv_nose_curve(
        num_buses,
        bus_types,
        V_init,
        theta_init,
        lines,
        p_load,
        q_load_pq,
        p_gen_base=p_gen,
        bus_shunts=shunts,
        q_limits_fn=None,
        lambda_start=1.0,
        initial_step=0.20,
        min_step=min_step,
    )
    q_limited = trace_pv_nose_curve(
        num_buses,
        bus_types,
        V_init,
        theta_init,
        lines,
        p_load,
        q_load_pq,
        p_gen_base=p_gen,
        bus_shunts=shunts,
        q_limits_fn=get_ieee14_q_limits,
        lambda_start=1.0,
        initial_step=0.05,
        min_step=min_step,
    )
    return unconstrained, q_limited
