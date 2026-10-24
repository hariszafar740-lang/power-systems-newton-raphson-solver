"""Analytical and benchmark verification tests for P-V nose-curve tracing and
saddle-node bifurcation loadability margins."""

import numpy as np
import pytest

from src.visualizer import plot_pv_nose_curve
from src.voltage_stability import trace_ieee14_pv_curves, trace_pv_nose_curve


def test_two_bus_lossless_line_matches_exact_analytical_nose_point():
    v1 = 1.04
    x_line = 0.20
    expected_p_max_pu = (v1**2) / (2.0 * x_line)
    expected_v_crit_pu = v1 / np.sqrt(2.0)

    res = trace_pv_nose_curve(
        num_buses=2,
        bus_types=["Slack", "PQ"],
        V_init=np.array([v1, 1.0]),
        theta_init=np.zeros(2),
        line_data=[{"from_bus": 1, "to_bus": 2, "r": 0.0, "x": x_line, "b": 0.0}],
        p_load_base=np.array([0.0, 1.0]),
        q_load_base=np.zeros(2),
        lambda_start=0.1,
        initial_step=0.25,
        min_step=1e-4,
        tol=1e-9,
    )

    assert res.lambda_max == pytest.approx(expected_p_max_pu, rel=5e-4)
    assert res.critical_voltage_pu == pytest.approx(expected_v_crit_pu, rel=5e-3)
    assert res.weakest_bus == 2
    assert res.min_singular_values[-1] < 0.08 * res.min_singular_values[0]


def test_ieee14_q_limits_significantly_reduce_collapse_margin_and_plot(tmp_path):
    unc, qlim = trace_ieee14_pv_curves(min_step=5e-3)

    assert qlim.lambda_max == pytest.approx(1.76, abs=0.02)
    assert unc.lambda_max == pytest.approx(4.00, abs=0.03)
    assert qlim.lambda_max < 0.5 * unc.lambda_max
    assert qlim.weakest_bus == 14
    assert set(qlim.q_saturation_lambdas.keys()) == {2, 3, 6, 8}
    assert qlim.q_saturation_lambdas[2] < qlim.q_saturation_lambdas[8]

    out_png = tmp_path / "pv_nose.png"
    plot_pv_nose_curve(unc, qlim_res=qlim, bus=14, save_path=str(out_png))
    assert out_png.exists() and out_png.stat().st_size > 1000


@pytest.mark.parametrize(
    "p_load, kwargs, match",
    [
        (np.zeros(2), {}, "positive load"),
        (np.array([0.0, 1.0]), {"min_step": 0.5, "initial_step": 0.1}, "min_step <="),
        (np.array([0.0, 1.0]), {"base_mva": 0.0}, "base_mva"),
        (np.array([0.0, 1.0, 0.5]), {}, "num_buses"),
    ],
)
def test_pv_curve_tracer_rejects_invalid_inputs(p_load, kwargs, match):
    with pytest.raises(ValueError, match=match):
        trace_pv_nose_curve(
            num_buses=2,
            bus_types=["Slack", "PQ"],
            V_init=np.array([1.0, 1.0]),
            theta_init=np.zeros(2),
            line_data=[{"from_bus": 1, "to_bus": 2, "r": 0.01, "x": 0.1, "b": 0.0}],
            p_load_base=p_load,
            q_load_base=np.zeros_like(p_load),
            **kwargs,
        )
