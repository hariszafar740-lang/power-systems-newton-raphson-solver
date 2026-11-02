import numpy as np


def calculate_line_flows(V, theta, line_data):
    """Calculate branch complex-power flows and total network losses.

    ``b`` in each branch is total line charging susceptance.  An optional
    positive, real ``tap`` ratio is placed on the from-bus side, matching the
    convention used by :func:`src.ybus_builder.build_ybus`.
    """
    V = np.asarray(V, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if V.ndim != 1 or theta.ndim != 1 or len(V) != len(theta):
        raise ValueError("V and theta must be one-dimensional arrays of equal length")
    if not np.isfinite(V).all() or not np.isfinite(theta).all() or np.any(V <= 0.0):
        raise ValueError("voltage magnitudes must be positive finite values")

    flows = []
    total_P_loss = 0.0
    total_Q_loss = 0.0

    for line in line_data:
        from_bus = line["from_bus"]
        to_bus = line["to_bus"]
        if not (
            isinstance(from_bus, (int, np.integer))
            and isinstance(to_bus, (int, np.integer))
            and 1 <= from_bus <= len(V)
            and 1 <= to_bus <= len(V)
            and from_bus != to_bus
        ):
            raise ValueError("branch bus numbers must identify different buses in V")

        r = float(line["r"])
        x = float(line["x"])
        b_total = float(line.get("b", 0.0))
        tap = float(line.get("tap", 1.0))
        if not np.isfinite([r, x, b_total, tap]).all():
            raise ValueError("branch parameters must be finite numbers")
        if r == 0.0 and x == 0.0:
            raise ValueError("branch impedance cannot be zero")
        if tap <= 0.0:
            raise ValueError("transformer tap ratio must be positive")

        from_index = from_bus - 1
        to_index = to_bus - 1
        series_admittance = 1.0 / complex(r, x)
        charging_admittance = complex(0.0, b_total / 2.0)
        V_from = V[from_index] * np.exp(1j * theta[from_index])
        V_to = V[to_index] * np.exp(1j * theta[to_index])

        # Currents injected into the branch at the from and to ends.  The tap
        # is on the from side, so it affects its self-admittance by tap**2.
        I_from_to = (
            V_from * (series_admittance + charging_admittance) / tap**2
            - V_to * series_admittance / tap
        )
        I_to_from = (
            V_to * (series_admittance + charging_admittance)
            - V_from * series_admittance / tap
        )

        S_from_to = V_from * np.conj(I_from_to)
        S_to_from = V_to * np.conj(I_to_from)
        S_loss = S_from_to + S_to_from

        total_P_loss += S_loss.real
        total_Q_loss += S_loss.imag
        flows.append(
            {
                "from": from_bus,
                "to": to_bus,
                "P_ij": S_from_to.real,
                "Q_ij": S_from_to.imag,
                "P_ji": S_to_from.real,
                "Q_ji": S_to_from.imag,
                "P_loss": S_loss.real,
                "Q_loss": S_loss.imag,
            }
        )

    return flows, total_P_loss, total_Q_loss
