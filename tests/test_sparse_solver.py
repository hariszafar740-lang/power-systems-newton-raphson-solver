"""Verification tests for the SciPy CSC sparse solver, IEEE 30-bus benchmark,
and dense-vs-sparse scaling diagnostics."""

import numpy as np
import pytest
from scipy.sparse import csc_matrix, issparse

from src.bus_data import get_3bus_data
from src.ieee14_data import get_ieee14_bus_shunts, get_ieee14_data
from src.ieee30_data import get_ieee30_bus_shunts, get_ieee30_data
from src.jacobian_builder import build_jacobian, build_jacobian_sparse
from src.line_flow_calculator import calculate_line_flows
from src.mismatch_calculator import calculate_power_injections
from src.nr_solver import solve_newton_raphson
from src.sparse_benchmark import benchmark_dense_vs_sparse, build_synthetic_grid
from src.visualizer import plot_jacobian_sparsity
from src.ybus_builder import build_ybus, build_ybus_sparse

IEEE30_REF_VM_PU = np.array(
    [
        1.06, 1.045, 1.021178, 1.0123, 1.01, 1.010626, 1.002597, 1.01,
        1.051132, 1.045379, 1.082, 1.057339, 1.071, 1.042508, 1.037916,
        1.044626, 1.04015, 1.028396, 1.0259, 1.029987, 1.032982, 1.033514,
        1.027429, 1.021846, 1.017619, 0.999946, 1.023539, 1.007101, 1.003706,
        0.992235,
    ]
)

IEEE30_REF_VA_DEG = np.array(
    [
        0.0, -5.378243, -7.52866, -9.279432, -14.148767, -11.055023,
        -12.852319, -11.797385, -14.097969, -15.688173, -14.097969, -14.932908,
        -14.932908, -15.824522, -15.916363, -15.515424, -15.849948, -16.530189,
        -16.703722, -16.507193, -16.130667, -16.116437, -16.306626, -16.482787,
        -16.054559, -16.473981, -15.53008, -11.677297, -16.759313, -17.641613,
    ]
)


def test_sparse_ybus_matches_dense_across_benchmarks():
    for getter, shunt_getter in (
        (get_ieee14_data, get_ieee14_bus_shunts),
        (get_ieee30_data, get_ieee30_bus_shunts),
    ):
        num_buses, _, _, _, _, _, lines = getter()
        shunts = shunt_getter()
        y_dense = build_ybus(num_buses, lines, shunts, sparse=False)
        y_sparse = build_ybus_sparse(num_buses, lines, shunts)
        assert issparse(y_sparse)
        assert isinstance(y_sparse, csc_matrix)
        np.testing.assert_allclose(y_sparse.toarray(), y_dense, rtol=1e-14, atol=1e-14)


def test_sparse_jacobian_matches_dense_and_finite_differences_on_ieee30():
    num_buses, bus_types, V, theta, _, _, lines = get_ieee30_data()
    y_sparse = build_ybus_sparse(num_buses, lines, get_ieee30_bus_shunts())
    y_dense = y_sparse.toarray()

    rng = np.random.default_rng(42)
    trial_v = V.copy()
    pq_mask = np.array([bt == "PQ" for bt in bus_types])
    trial_v[pq_mask] += rng.uniform(-0.02, 0.02, size=int(np.sum(pq_mask)))
    trial_theta = theta.copy()
    trial_theta[1:] = rng.uniform(-0.25, -0.02, size=num_buses - 1)

    j_dense = build_jacobian(trial_v, trial_theta, y_dense, bus_types, sparse=False)
    j_sparse = build_jacobian_sparse(trial_v, trial_theta, y_sparse, bus_types)

    assert issparse(j_sparse)
    np.testing.assert_allclose(j_sparse.toarray(), j_dense, rtol=1e-13, atol=1e-13)

    pv_pq = [i for i in range(num_buses) if bus_types[i] != "Slack"]
    pq = [i for i in range(num_buses) if bus_types[i] == "PQ"]
    state = np.concatenate([trial_theta[pv_pq], trial_v[pq]])
    step = 1e-6
    j_fd = np.zeros_like(j_dense)

    for col in range(len(state)):
        s_plus = state.copy()
        s_minus = state.copy()
        s_plus[col] += step
        s_minus[col] -= step

        v_p, th_p = trial_v.copy(), trial_theta.copy()
        th_p[pv_pq] = s_plus[: len(pv_pq)]
        v_p[pq] = s_plus[len(pv_pq) :]
        p_p, q_p = calculate_power_injections(v_p, th_p, y_sparse)

        v_m, th_m = trial_v.copy(), trial_theta.copy()
        th_m[pv_pq] = s_minus[: len(pv_pq)]
        v_m[pq] = s_minus[len(pv_pq) :]
        p_m, q_m = calculate_power_injections(v_m, th_m, y_sparse)

        j_fd[:, col] = (
            np.concatenate([p_p[pv_pq], q_p[pq]]) - np.concatenate([p_m[pv_pq], q_m[pq]])
        ) / (2.0 * step)

    np.testing.assert_allclose(j_sparse.toarray(), j_fd, rtol=1e-6, atol=1e-5)


def test_sparse_and_dense_ieee14_solutions_agree_to_machine_precision():
    num_buses, bus_types, V_init, theta_init, P_spec, Q_spec, lines = get_ieee14_data()
    shunts = get_ieee14_bus_shunts()

    y_dense = build_ybus(num_buses, lines, shunts)
    y_sparse = build_ybus_sparse(num_buses, lines, shunts)

    v_d, th_d, hist_d = solve_newton_raphson(
        V_init, theta_init, y_dense, P_spec, Q_spec, bus_types, tol=1e-10, sparse=False
    )
    v_s, th_s, hist_s = solve_newton_raphson(
        V_init, theta_init, y_sparse, P_spec, Q_spec, bus_types, tol=1e-10, sparse=True
    )

    assert len(hist_s) == len(hist_d)
    np.testing.assert_allclose(v_s, v_d, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(th_s, th_d, rtol=1e-13, atol=1e-13)


def test_ieee30_sparse_solution_matches_matpower_reference():
    num_buses, bus_types, V_init, theta_init, P_spec, Q_spec, lines = get_ieee30_data()
    y_sparse = build_ybus_sparse(num_buses, lines, get_ieee30_bus_shunts())

    V_sol, theta_sol, history = solve_newton_raphson(
        V_init, theta_init, y_sparse, P_spec, Q_spec, bus_types, max_iter=20, tol=1e-9, sparse=True
    )
    _, p_loss, q_loss = calculate_line_flows(V_sol, theta_sol, lines)

    assert len(history) - 1 == 4
    np.testing.assert_allclose(V_sol, IEEE30_REF_VM_PU, atol=5e-6)
    np.testing.assert_allclose(np.degrees(theta_sol), IEEE30_REF_VA_DEG, atol=5e-5)
    assert p_loss == pytest.approx(0.175569, abs=1e-5)
    assert q_loss == pytest.approx(0.329833, abs=1e-5)


def test_sparse_solver_reports_singular_jacobian_and_nonfinite_ybus():
    bus_types, V_init, theta_init, P_spec, Q_spec = get_3bus_data()
    disconnected_y = build_ybus_sparse(
        3, [{"from_bus": 1, "to_bus": 2, "r": 0.02, "x": 0.06, "b": 0.06}]
    )

    with pytest.raises(RuntimeError, match="singular"):
        solve_newton_raphson(
            V_init, theta_init, disconnected_y, P_spec, Q_spec, bus_types, sparse=True
        )

    bad_y = disconnected_y.copy()
    bad_y.data[0] = complex(np.inf, 0.0)
    with pytest.raises(ValueError, match="finite"):
        solve_newton_raphson(V_init, theta_init, bad_y, P_spec, Q_spec, bus_types, sparse=True)


def test_scaling_benchmark_demonstrates_high_sparsity_and_parity():
    rows = benchmark_dense_vs_sparse(bus_sizes=(16, 80), tol=1e-8)
    assert len(rows) == 2
    large = rows[1]
    assert large["num_buses"] == 80
    assert large["sparsity_percent"] > 90.0
    assert large["max_v_diff"] < 1e-12
    assert large["max_theta_diff"] < 1e-12


def test_synthetic_grid_rejects_too_small_bus_count():
    with pytest.raises(ValueError, match=">= 4"):
        build_synthetic_grid(3)


def test_plot_jacobian_sparsity_writes_png(tmp_path):
    num_buses, bus_types, V_init, theta_init, _, _, lines = get_ieee30_data()
    y_sparse = build_ybus_sparse(num_buses, lines, get_ieee30_bus_shunts())
    j_sparse = build_jacobian_sparse(V_init, theta_init, y_sparse, bus_types)
    out = tmp_path / "sparsity.png"
    plot_jacobian_sparsity(j_sparse, save_path=str(out))
    assert out.exists() and out.stat().st_size > 1000
