import warnings

import numpy as np
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from src.jacobian_builder import build_jacobian, build_jacobian_sparse
from src.mismatch_calculator import calculate_power_mismatches


def solve_newton_raphson(
    V_init,
    theta_init,
    Y_bus,
    P_spec,
    Q_spec,
    bus_types,
    max_iter=20,
    tol=1e-4,
    *,
    sparse=False,
):
    """Solve a balanced AC power flow with the Newton-Raphson method.

    Parameters use per-unit voltage magnitudes and powers, with voltage angles
    in radians.  When ``sparse=True`` (or when ``Y_bus`` is supplied as a
    ``scipy.sparse`` matrix), the solver assembles a CSC sparse Jacobian and
    solves each Newton step via sparse LU factorization (``spsolve``).

    Raises
    ------
    ValueError
        If the network input is inconsistent or physically invalid.
    RuntimeError
        If the Jacobian is singular or the requested tolerance is not reached.
    """
    V = np.asarray(V_init, dtype=float).copy()
    theta = np.asarray(theta_init, dtype=float).copy()
    P_spec = np.asarray(P_spec, dtype=float)
    Q_spec = np.asarray(Q_spec, dtype=float)
    use_sparse = bool(sparse) or issparse(Y_bus)

    if any(values.ndim != 1 for values in (V, theta, P_spec, Q_spec)):
        raise ValueError("voltage, angle, and specified-power arrays must be one-dimensional")
    num_buses = len(V)
    if num_buses == 0:
        raise ValueError("the network must contain at least one bus")
    if len(theta) != num_buses or len(P_spec) != num_buses or len(Q_spec) != num_buses:
        raise ValueError("voltage, angle, and specified-power arrays must have equal length")

    if issparse(Y_bus):
        y_matrix = Y_bus.tocsc().astype(complex)
        if y_matrix.shape != (num_buses, num_buses):
            raise ValueError("Y_bus must be a square matrix matching the number of buses")
        if not np.isfinite(y_matrix.data.real).all() or not np.isfinite(y_matrix.data.imag).all():
            raise ValueError("Y_bus entries must be finite")
    else:
        y_dense = np.asarray(Y_bus, dtype=complex)
        if y_dense.shape != (num_buses, num_buses):
            raise ValueError("Y_bus must be a square matrix matching the number of buses")
        if not np.isfinite(y_dense.real).all() or not np.isfinite(y_dense.imag).all():
            raise ValueError("Y_bus entries must be finite")
        y_matrix = csc_matrix(y_dense) if use_sparse else y_dense

    if len(bus_types) != num_buses:
        raise ValueError("bus_types must contain one entry for each bus")
    if any(bus_type not in ("Slack", "PV", "PQ") for bus_type in bus_types):
        raise ValueError("bus types must be 'Slack', 'PV', or 'PQ'")
    if sum(bus_type == "Slack" for bus_type in bus_types) != 1:
        raise ValueError("exactly one Slack bus is required")
    if not np.isfinite(V).all() or not np.isfinite(theta).all():
        raise ValueError("initial voltages and angles must be finite")
    if not np.isfinite(P_spec).all() or not np.isfinite(Q_spec).all():
        raise ValueError("specified powers must be finite")
    if np.any(V <= 0.0):
        raise ValueError("initial voltage magnitudes must be positive")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter must be a non-negative integer")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be a positive finite number")

    pv_pq_buses = [i for i in range(num_buses) if bus_types[i] != "Slack"]
    pq_buses = [i for i in range(num_buses) if bus_types[i] == "PQ"]
    n_p = len(pv_pq_buses)
    history = []

    for iteration in range(max_iter + 1):
        _, _, delta_P, delta_Q = calculate_power_mismatches(
            V, theta, y_matrix, P_spec, Q_spec, bus_types
        )
        mismatch_vector = np.concatenate([delta_P, delta_Q])
        max_error = (
            float(np.max(np.abs(mismatch_vector))) if mismatch_vector.size else 0.0
        )
        history.append(max_error)

        if max_error < tol:
            return V, theta, history
        if iteration == max_iter:
            break

        if use_sparse:
            jacobian = build_jacobian_sparse(V, theta, y_matrix, bus_types)
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                try:
                    state_change = np.atleast_1d(
                        np.asarray(spsolve(jacobian, mismatch_vector), dtype=float)
                    )
                except (RuntimeError, np.linalg.LinAlgError, MatrixRankWarning) as error:
                    raise RuntimeError("Newton-Raphson Jacobian is singular") from error
        else:
            jacobian = build_jacobian(V, theta, y_matrix, bus_types)
            try:
                state_change = np.linalg.solve(jacobian, mismatch_vector)
            except np.linalg.LinAlgError as error:
                raise RuntimeError("Newton-Raphson Jacobian is singular") from error

        if not np.isfinite(state_change).all():
            raise RuntimeError("Newton-Raphson Jacobian is singular or ill-conditioned")

        step_size = 1.0
        voltage_change = state_change[n_p:]
        while pq_buses and np.any(V[pq_buses] + step_size * voltage_change <= 0.0):
            step_size *= 0.5
            if step_size < 1e-6:
                raise RuntimeError("Newton-Raphson step would create a non-positive voltage")

        theta[pv_pq_buses] += step_size * state_change[:n_p]
        V[pq_buses] += step_size * voltage_change

    raise RuntimeError(
        "Newton-Raphson did not converge within "
        f"{max_iter} iterations; maximum mismatch is {history[-1]:.6e}"
    )
