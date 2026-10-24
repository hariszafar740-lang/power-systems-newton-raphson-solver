import numpy as np
from scipy.sparse import csc_matrix


def _assemble_ybus_triplets(num_buses, line_data, bus_shunts=None):
    """Validate network topology and assemble COO triplets for ``Y_bus``."""
    if not isinstance(num_buses, (int, np.integer)) or num_buses <= 0:
        raise ValueError("num_buses must be a positive integer")

    rows = []
    cols = []
    data = []

    for line in line_data:
        from_bus = line["from_bus"]
        to_bus = line["to_bus"]
        if not (
            isinstance(from_bus, (int, np.integer))
            and isinstance(to_bus, (int, np.integer))
            and 1 <= from_bus <= num_buses
            and 1 <= to_bus <= num_buses
        ):
            raise ValueError("branch bus numbers must be integers between 1 and num_buses")
        if from_bus == to_bus:
            raise ValueError("a branch must connect two different buses")

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

        i = int(from_bus) - 1
        j = int(to_bus) - 1
        series_admittance = 1.0 / complex(r, x)
        charging_admittance = complex(0.0, b_total / 2.0)

        rows.extend([i, j, i, j])
        cols.extend([i, j, j, i])
        data.extend(
            [
                (series_admittance + charging_admittance) / tap**2,
                series_admittance + charging_admittance,
                -series_admittance / tap,
                -series_admittance / tap,
            ]
        )

    if bus_shunts is not None:
        for shunt in bus_shunts:
            bus = shunt["bus"]
            if not isinstance(bus, (int, np.integer)) or not 1 <= bus <= num_buses:
                raise ValueError("shunt bus numbers must be integers between 1 and num_buses")

            conductance = float(shunt.get("g", 0.0))
            susceptance = float(shunt.get("b", 0.0))
            if not np.isfinite([conductance, susceptance]).all():
                raise ValueError("shunt parameters must be finite numbers")

            idx = int(bus) - 1
            rows.append(idx)
            cols.append(idx)
            data.append(complex(conductance, susceptance))

    return rows, cols, data


def build_ybus_sparse(num_buses, line_data, bus_shunts=None):
    """Construct the complex bus-admittance matrix directly in CSC sparse format."""
    rows, cols, data = _assemble_ybus_triplets(num_buses, line_data, bus_shunts)
    return csc_matrix((data, (rows, cols)), shape=(num_buses, num_buses), dtype=complex)


def build_ybus(num_buses, line_data, bus_shunts=None, *, sparse=False):
    """Construct the complex bus-admittance matrix for a power system.

    Parameters
    ----------
    num_buses : int
        Number of buses in the network.
    line_data : list of dict
        Branch data using one-based bus numbers.  Each branch requires
        ``from_bus``, ``to_bus``, ``r``, and ``x``.  ``b`` is the total line
        charging susceptance and is split equally between both ends.  ``tap``
        is an optional positive, real off-nominal transformer ratio on the
        from-bus side; it defaults to 1.0 for a transmission line.
    bus_shunts : list of dict, optional
        Shunt admittances connected to buses.  Each entry contains ``bus``
        and optional conductance ``g`` and susceptance ``b`` in per unit.
    sparse : bool, optional
        When ``True``, return a ``scipy.sparse.csc_matrix`` assembled directly
        from branch triplets without allocating an ``N x N`` dense array.

    Returns
    -------
    numpy.ndarray or scipy.sparse.csc_matrix
        A ``num_buses`` by ``num_buses`` complex admittance matrix.
    """
    y_sparse = build_ybus_sparse(num_buses, line_data, bus_shunts)
    if sparse:
        return y_sparse
    return y_sparse.toarray()
