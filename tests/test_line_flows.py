import numpy as np

from src.line_flow_calculator import calculate_line_flows
from src.mismatch_calculator import calculate_power_injections
from src.ybus_builder import build_ybus


def test_line_flows_match_ybus_power_injections_with_transformer_tap():
    line_data = [
        {"from_bus": 1, "to_bus": 2, "r": 0.02, "x": 0.16, "b": 0.04, "tap": 0.95}
    ]
    V = np.array([1.04, 0.98])
    theta = np.radians(np.array([4.0, -2.0]))
    Y_bus = build_ybus(2, line_data)

    flows, total_P_loss, total_Q_loss = calculate_line_flows(V, theta, line_data)
    P_calc, Q_calc = calculate_power_injections(V, theta, Y_bus)
    flow = flows[0]

    np.testing.assert_allclose(P_calc, [flow["P_ij"], flow["P_ji"]])
    np.testing.assert_allclose(Q_calc, [flow["Q_ij"], flow["Q_ji"]])
    np.testing.assert_allclose(total_P_loss, flow["P_loss"])
    np.testing.assert_allclose(total_Q_loss, flow["Q_loss"])
    assert total_P_loss > 0.0
