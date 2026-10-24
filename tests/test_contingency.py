"""Verification tests for automated N-1 single-branch contingency screening."""

import pytest

from src.contingency_analyzer import (
    find_islanded_buses,
    format_contingency_table,
    run_n1_contingency_analysis,
)
from src.ieee14_data import (
    get_ieee14_bus_shunts,
    get_ieee14_data,
    get_ieee14_q_limits,
)
from src.ieee30_data import get_ieee30_data
from src.visualizer import plot_contingency_ranking


def test_topological_islanding_detection_on_ieee14_and_ieee30():
    n14, _, _, _, _, _, lines14 = get_ieee14_data()
    without_7_8 = [line for idx, line in enumerate(lines14) if idx != 13]
    assert find_islanded_buses(n14, without_7_8, slack_bus=1) == [8]
    assert find_islanded_buses(n14, lines14, slack_bus=1) == []

    n30, _, _, _, _, _, lines30 = get_ieee30_data()
    without_25_26 = [
        line for line in lines30 if not (line["from_bus"] == 25 and line["to_bus"] == 26)
    ]
    assert find_islanded_buses(n30, without_25_26, slack_bus=1) == [26]


def test_ieee14_contingency_reveals_line_1_2_collapse_under_q_limits():
    n14, bt14, v14, th14, p14, q14, lines14 = get_ieee14_data()
    shunts14 = get_ieee14_bus_shunts()

    rep_unconstrained = run_n1_contingency_analysis(
        n14, bt14, v14, th14, p14, q14, lines14, shunts14, q_limits=None, v_min=0.94, v_max=1.06
    )
    case_12_unconstrained = next(
        c for c in rep_unconstrained.ranked_cases if (c.from_bus, c.to_bus) == (1, 2)
    )
    assert case_12_unconstrained.converged is True

    rep_qlim = run_n1_contingency_analysis(
        n14,
        bt14,
        v14,
        th14,
        p14,
        q14,
        lines14,
        shunts14,
        q_limits=get_ieee14_q_limits(1.0),
        v_min=0.94,
        v_max=1.06,
    )
    assert rep_qlim.total_contingencies == 20
    assert rep_qlim.islanded_count == 1
    case_78 = rep_qlim.ranked_cases[0]
    assert (case_78.from_bus, case_78.to_bus) == (7, 8)
    assert case_78.status == "ISLANDED"
    assert case_78.islanded_buses == [8]

    case_12_qlim = next(c for c in rep_qlim.ranked_cases if (c.from_bus, c.to_bus) == (1, 2))
    assert case_12_qlim.status == "NON_CONVERGED"
    assert case_12_qlim.converged is False


def test_thermal_branch_overload_detection_and_ranking():
    n14, bt14, v14, th14, p14, q14, lines14 = get_ieee14_data()
    rated_lines = [{**line, "rate_mva": 120.0} for line in lines14]

    rep = run_n1_contingency_analysis(
        n14, bt14, v14, th14, p14, q14, rated_lines, get_ieee14_bus_shunts(), v_min=0.94, v_max=1.07
    )
    overloaded = [c for c in rep.ranked_cases if c.thermal_violations]
    assert len(overloaded) >= 1
    assert overloaded[0].status == "VIOLATION"
    assert overloaded[0].thermal_violations[0]["loading_percent"] > 100.0


def test_contingency_markdown_table_and_plot(tmp_path):
    n14, bt14, v14, th14, p14, q14, lines14 = get_ieee14_data()
    rep = run_n1_contingency_analysis(
        n14, bt14, v14, th14, p14, q14, lines14, get_ieee14_bus_shunts(), v_min=0.94, v_max=1.06
    )
    md = format_contingency_table(rep, top_n=5)
    assert "| Rank | Outage | Status |" in md
    assert "7-8 (#14)" in md

    out_png = tmp_path / "contingency.png"
    plot_contingency_ranking(rep, save_path=str(out_png), top_n=8)
    assert out_png.exists() and out_png.stat().st_size > 1000


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"v_min": 1.05, "v_max": 0.95}, "v_min < v_max"),
        ({"base_mva": -100.0}, "base_mva"),
    ],
)
def test_contingency_analyzer_rejects_invalid_bounds(kwargs, match):
    n14, bt14, v14, th14, p14, q14, lines14 = get_ieee14_data()
    with pytest.raises(ValueError, match=match):
        run_n1_contingency_analysis(
            n14, bt14, v14, th14, p14, q14, lines14, get_ieee14_bus_shunts(), **kwargs
        )
