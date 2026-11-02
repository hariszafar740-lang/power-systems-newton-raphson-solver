import numpy as np


def calculate_power_injections(V, theta, Y_bus):
    """Calculate active and reactive power injected at every bus.

    Works seamlessly with both dense ``numpy.ndarray`` and ``scipy.sparse``
    admittance matrices using complex phasor matrix-vector multiplication
    ``S = V_phasor * conj(Y_bus @ V_phasor)``.
    """
    v_phasor = np.asarray(V, dtype=float) * np.exp(1j * np.asarray(theta, dtype=float))
    i_injected = Y_bus @ v_phasor
    s_injected = v_phasor * np.conj(i_injected)
    return np.asarray(s_injected.real, dtype=float), np.asarray(s_injected.imag, dtype=float)


def calculate_power_mismatches(V, theta, Y_bus, P_spec, Q_spec, bus_types):
    """Return calculated injections and the Newton-Raphson mismatch vectors.

    The active-power mismatch contains PV and PQ buses in bus order.  The
    reactive-power mismatch contains PQ buses only, also in bus order.
    """
    P_calc, Q_calc = calculate_power_injections(V, theta, Y_bus)

    pv_pq_indices = [i for i, btype in enumerate(bus_types) if btype != "Slack"]
    pq_indices = [i for i, btype in enumerate(bus_types) if btype == "PQ"]

    delta_P = np.asarray(P_spec, dtype=float)[pv_pq_indices] - P_calc[pv_pq_indices]
    delta_Q = np.asarray(Q_spec, dtype=float)[pq_indices] - Q_calc[pq_indices]

    return P_calc, Q_calc, delta_P, delta_Q
