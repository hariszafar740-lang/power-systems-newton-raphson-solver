"""Static interactive HTML5 dashboard generator for GitHub Pages deployment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src import __version__
from src.contingency_analyzer import run_n1_contingency_analysis
from src.ieee14_data import (
    get_ieee14_bus_shunts,
    get_ieee14_data,
    scale_ieee14_loads,
)
from src.ieee30_data import (
    get_ieee30_bus_shunts,
    get_ieee30_data,
    scale_ieee30_loads,
)
from src.jacobian_builder import build_jacobian_sparse
from src.line_flow_calculator import calculate_line_flows
from src.nr_solver import solve_newton_raphson
from src.q_limit_solver import solve_nr_with_q_limits
from src.sparse_benchmark import benchmark_dense_vs_sparse
from src.visualizer import (
    plot_contingency_ranking,
    plot_convergence,
    plot_jacobian_sparsity,
    plot_pv_nose_curve,
    plot_voltage_profile,
)
from src.voltage_stability import trace_ieee14_pv_curves
from src.ybus_builder import build_ybus_sparse


def _bus_rows(
    v: np.ndarray,
    theta: np.ndarray,
    bus_types: Sequence[str],
    saturated: dict[int, str] | None = None,
) -> list[dict[str, object]]:
    sat = saturated or {}
    rows: list[dict[str, object]] = []
    for idx in range(len(v)):
        bus_num = idx + 1
        status = f"PV->PQ ({sat[bus_num]})" if bus_num in sat else str(bus_types[idx])
        rows.append(
            {
                "bus": bus_num,
                "type": status,
                "v_pu": round(float(v[idx]), 5),
                "angle_deg": round(float(np.degrees(theta[idx])), 4),
            }
        )
    return rows


def build_dashboard_site(out_dir: str | Path = "site") -> Path:
    """Solve benchmark grids and build a self-contained static interactive site."""
    target = Path(out_dir)
    assets = target / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (target / ".nojekyll").write_text("", encoding="utf-8")

    # 1. IEEE 14 Base & Stressed (1.25x with Q-limits)
    n14, bt14, v0_14, th0_14, p14, q14, lines14 = get_ieee14_data()
    shunts14 = get_ieee14_bus_shunts()
    _, _, q_lims14_base = scale_ieee14_loads(1.0)
    y14 = build_ybus_sparse(n14, lines14, shunts14)
    v14_base, th14_base, hist14 = solve_newton_raphson(
        v0_14, th0_14, y14, p14, q14, bt14, sparse=True
    )
    _, loss_p14, loss_q14 = calculate_line_flows(v14_base, th14_base, lines14)

    p14_s, q14_s, q_lims14_s = scale_ieee14_loads(1.25)
    q14_res = solve_nr_with_q_limits(
        v0_14, th0_14, y14, p14_s, q14_s, bt14, q_lims14_s, sparse=True
    )
    _, loss_p14_s, loss_q14_s = calculate_line_flows(q14_res.V, q14_res.theta, lines14)

    # 2. IEEE 30 Base & Stressed (1.35x with Q-limits)
    n30, bt30, v0_30, th0_30, p30, q30, lines30 = get_ieee30_data()
    shunts30 = get_ieee30_bus_shunts()
    y30 = build_ybus_sparse(n30, lines30, shunts30)
    v30_base, th30_base, hist30 = solve_newton_raphson(
        v0_30, th0_30, y30, p30, q30, bt30, sparse=True
    )
    _, loss_p30, loss_q30 = calculate_line_flows(v30_base, th30_base, lines30)
    j30 = build_jacobian_sparse(v30_base, th30_base, y30, bt30)

    p30_s, q30_s, q_lims30_s = scale_ieee30_loads(1.35)
    q30_res = solve_nr_with_q_limits(
        v0_30, th0_30, y30, p30_s, q30_s, bt30, q_lims30_s, sparse=True
    )
    _, loss_p30_s, loss_q30_s = calculate_line_flows(q30_res.V, q30_res.theta, lines30)

    # 3. N-1 Contingency Security Screening (IEEE 14 with Q-limits & 0.94-1.06 bounds)
    n1_rep = run_n1_contingency_analysis(
        n14,
        bt14,
        v0_14,
        th0_14,
        p14,
        q14,
        lines14,
        shunts14,
        q_limits=q_lims14_base,
        v_min=0.94,
        v_max=1.06,
    )

    # 4. P-V Voltage Stability Nose Curves (IEEE 14)
    pv_unc, pv_qlim = trace_ieee14_pv_curves(min_step=5e-3)

    # 5. Sparse vs Dense Scaling Benchmark
    bench_rows = benchmark_dense_vs_sparse(bus_sizes=(14, 30, 64, 100))

    # Render diagnostic visual assets
    plot_voltage_profile(v14_base, bt14, save_path=str(assets / "ieee14_voltage.png"))
    plot_convergence(hist14, save_path=str(assets / "ieee14_convergence.png"))
    plot_jacobian_sparsity(j30, save_path=str(assets / "ieee30_jacobian_spy.png"))
    plot_contingency_ranking(
        n1_rep, save_path=str(assets / "ieee14_contingency.png"), top_n=10
    )
    plot_pv_nose_curve(
        pv_unc,
        qlim_res=pv_qlim,
        bus=14,
        save_path=str(assets / "ieee14_pv_nose.png"),
    )

    payload = {
        "version": __version__,
        "cases": {
            "ieee14_base": {
                "title": "IEEE 14-Bus, Nominal Base Case (1.00x Load)",
                "buses": n14,
                "branches": len(lines14),
                "iterations": len(hist14) - 1,
                "loss_mw": round(float(loss_p14) * 100.0, 3),
                "loss_mvar": round(float(loss_q14) * 100.0, 3),
                "min_v": round(float(np.min(v14_base)), 4),
                "min_v_bus": int(np.argmin(v14_base)) + 1,
                "saturated_count": 0,
                "rows": _bus_rows(v14_base, th14_base, bt14),
            },
            "ieee14_stressed": {
                "title": "IEEE 14-Bus, Heavy Load (1.25x) + Q-Limits Enforced",
                "buses": n14,
                "branches": len(lines14),
                "iterations": len(q14_res.history) - 1,
                "loss_mw": round(float(loss_p14_s) * 100.0, 3),
                "loss_mvar": round(float(loss_q14_s) * 100.0, 3),
                "min_v": round(float(np.min(q14_res.V)), 4),
                "min_v_bus": int(np.argmin(q14_res.V)) + 1,
                "saturated_count": len(q14_res.saturated_buses),
                "rows": _bus_rows(
                    q14_res.V, q14_res.theta, bt14, q14_res.saturated_buses
                ),
            },
            "ieee30_base": {
                "title": "IEEE 30-Bus, Nominal Base Case (1.00x Load)",
                "buses": n30,
                "branches": len(lines30),
                "iterations": len(hist30) - 1,
                "loss_mw": round(float(loss_p30) * 100.0, 3),
                "loss_mvar": round(float(loss_q30) * 100.0, 3),
                "min_v": round(float(np.min(v30_base)), 4),
                "min_v_bus": int(np.argmin(v30_base)) + 1,
                "saturated_count": 0,
                "rows": _bus_rows(v30_base, th30_base, bt30),
            },
            "ieee30_stressed": {
                "title": "IEEE 30-Bus, Stressed Load (1.35x) + Q-Limits Enforced",
                "buses": n30,
                "branches": len(lines30),
                "iterations": len(q30_res.history) - 1,
                "loss_mw": round(float(loss_p30_s) * 100.0, 3),
                "loss_mvar": round(float(loss_q30_s) * 100.0, 3),
                "min_v": round(float(np.min(q30_res.V)), 4),
                "min_v_bus": int(np.argmin(q30_res.V)) + 1,
                "saturated_count": len(q30_res.saturated_buses),
                "rows": _bus_rows(
                    q30_res.V, q30_res.theta, bt30, q30_res.saturated_buses
                ),
            },
        },
        "contingency_summary": {
            "total": int(n1_rep.total_contingencies),
            "secure": int(n1_rep.secure_count),
            "violated": int(n1_rep.violated_count),
            "islanded": int(n1_rep.islanded_count),
            "diverged": int(n1_rep.diverged_count),
            "v_bounds": "0.94 - 1.06 p.u.",
        },
        "contingencies": [
            {
                "rank": idx + 1,
                "branch": f"L{item.branch_index + 1}",
                "corridor": f"{item.from_bus}-{item.to_bus}",
                "status": item.status,
                "min_v": (
                    f"{item.min_v_pu:.4f} (B{item.min_v_bus})"
                    if item.converged and item.min_v_pu is not None
                    else "COLLAPSE"
                ),
                "q_sat": len(item.saturated_pv_buses),
                "severity": round(float(item.severity_index), 2),
            }
            for idx, item in enumerate(n1_rep.ranked_cases[:12])
        ],
        "pv_curve": {
            "unconstrained_lambda": round(float(pv_unc.lambda_max), 4),
            "unconstrained_mw": round(float(pv_unc.max_load_mw), 2),
            "qlimited_lambda": round(float(pv_qlim.lambda_max), 4),
            "qlimited_mw": round(float(pv_qlim.max_load_mw), 2),
            "weakest_bus": int(pv_qlim.weakest_bus),
            "q_events": {
                str(k): round(float(v), 3)
                for k, v in pv_qlim.q_saturation_lambdas.items()
            },
            "points_qlim": [
                {
                    "lam": round(float(pv_qlim.lambdas[i]), 4),
                    "mw": round(float(pv_qlim.total_load_mw[i]), 2),
                    "sigma_min": round(float(pv_qlim.min_singular_values[i]), 4),
                    "v_buses": [
                        round(float(val), 4) for val in pv_qlim.voltages[i]
                    ],
                }
                for i in range(len(pv_qlim.lambdas))
            ],
        },
        "benchmarks": [
            {
                "buses": int(row["num_buses"]),
                "jac_shape": f"{row['jacobian_dim']}x{row['jacobian_dim']}",
                "nnz": int(row["jacobian_nnz"]),
                "sparsity_pct": round(float(row["sparsity_percent"]), 2),
                "dense_ms": round(float(row["dense_ms"]), 2),
                "sparse_ms": round(float(row["sparse_ms"]), 2),
                "speedup": round(float(row["speedup"]), 2),
            }
            for row in bench_rows
        ],
    }

    html = _render_html(payload)
    index_path = target / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def _render_html(payload: dict[str, object]) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NR Power-Flow & Grid Security Explorer | v__VERSION__</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: #131b33;
      --panel-alt: #192342;
      --border: #283663;
      --text: #e8eefc;
      --muted: #9fb0d9;
      --accent: #4cc9f0;
      --ok: #2ec4b6;
      --warn: #ff9f1c;
      --crit: #ef476f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: radial-gradient(circle at top right, #18254a 0%, var(--bg) 60%);
      color: var(--text);
      line-height: 1.55;
    }
    header {
      max-width: 1200px;
      margin: 0 auto;
      padding: 2.2rem 1.4rem 1.2rem;
      border-bottom: 1px solid var(--border);
    }
    .badge-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.7rem; }
    .pill {
      font-size: 0.78rem;
      font-weight: 600;
      padding: 0.22rem 0.68rem;
      border-radius: 999px;
      background: var(--panel-alt);
      border: 1px solid var(--border);
      color: var(--accent);
    }
    h1 { margin: 0 0 0.45rem; font-size: 1.85rem; letter-spacing: -0.02em; }
    header p { margin: 0; color: var(--muted); max-width: 860px; }
    header a { color: var(--accent); text-decoration: none; font-weight: 600; }
    main { max-width: 1200px; margin: 0 auto; padding: 1.4rem; }
    .tabs { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-bottom: 1.3rem; }
    .tab-btn {
      background: var(--panel);
      color: var(--muted);
      border: 1px solid var(--border);
      padding: 0.62rem 1.05rem;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.92rem;
    }
    .tab-btn.active {
      background: var(--accent);
      color: #081021;
      border-color: var(--accent);
    }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
      gap: 0.85rem;
      margin-bottom: 1.2rem;
    }
    .kpi {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.9rem 1rem;
    }
    .kpi span { display: block; font-size: 0.78rem; color: var(--muted); }
    .kpi strong { font-size: 1.35rem; color: var(--text); }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 0.8rem;
      align-items: center;
      margin-bottom: 1rem;
      background: var(--panel);
      padding: 0.85rem 1rem;
      border-radius: 10px;
      border: 1px solid var(--border);
    }
    select, input {
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 0.48rem 0.75rem;
      border-radius: 6px;
      font-size: 0.9rem;
    }
    .split {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 1.1rem;
      align-items: start;
    }
    @media (max-width: 920px) { .split { grid-template-columns: 1fr; } }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem;
      overflow-x: auto;
    }
    .card h3 { margin-top: 0; font-size: 1.05rem; color: var(--accent); }
    .card img { width: 100%; border-radius: 8px; background: #fff; padding: 4px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th, td {
      padding: 0.52rem 0.65rem;
      border-bottom: 1px solid var(--border);
      text-align: left;
    }
    th { color: var(--muted); font-weight: 600; }
    .tag-ok { color: var(--ok); font-weight: 600; }
    .tag-warn { color: var(--warn); font-weight: 600; }
    .tag-crit { color: var(--crit); font-weight: 700; }
    footer {
      max-width: 1200px;
      margin: 2rem auto 1.5rem;
      padding: 1rem 1.4rem 0;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.84rem;
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.5rem;
    }
  </style>
</head>
<body>
  <header>
    <div class="badge-row">
      <span class="pill">nr-powerflow-solver v__VERSION__</span>
      <span class="pill">SciPy CSC Sparse Engine</span>
      <span class="pill">MATPOWER case14 &amp; case_ieee30 Verified</span>
      <span class="pill">72 Automated Pytest Checks</span>
    </div>
    <h1>Power Systems Sparse Newton-Raphson &amp; Grid Security Explorer</h1>
    <p>
      Interactive benchmark explorer for balanced AC load flow, PV generator
      Q<sub>min</sub>/Q<sub>max</sub> capability enforcement, automated N-1 contingency
      screening, and saddle-node bifurcation P-V voltage stability margins.
      <a href="https://github.com/hariszafar740-lang/power-systems-newton-raphson-solver" target="_blank" rel="noopener">View Source &amp; CLI Docs on GitHub &rarr;</a>
    </p>
  </header>

  <main>
    <div class="tabs">
      <button class="tab-btn active" data-tab="powerflow">1. AC Load Flow &amp; Q-Limits</button>
      <button class="tab-btn" data-tab="contingency">2. N-1 Contingency Security</button>
      <button class="tab-btn" data-tab="stability">3. P-V Voltage Collapse Margin</button>
      <button class="tab-btn" data-tab="sparsity">4. Sparse CSC Scaling Benchmark</button>
    </div>

    <section id="tab-powerflow" class="tab-panel active">
      <div class="controls">
        <label for="caseSelect"><strong>Operating Scenario:</strong></label>
        <select id="caseSelect">
          <option value="ieee14_base">IEEE 14-Bus (Nominal 1.00x)</option>
          <option value="ieee14_stressed">IEEE 14-Bus (1.25x Stressed + Q-Limits)</option>
          <option value="ieee30_base">IEEE 30-Bus (Nominal 1.00x)</option>
          <option value="ieee30_stressed">IEEE 30-Bus (1.35x Stressed + Q-Limits)</option>
        </select>
        <label for="busFilter"><strong>Filter Bus / Type:</strong></label>
        <input id="busFilter" type="text" placeholder="e.g. PV->PQ or 14" />
      </div>
      <div id="pfKpis" class="kpi-grid"></div>
      <div class="split">
        <div class="card">
          <h3 id="pfTableTitle">Bus Phasor Solution</h3>
          <table>
            <thead>
              <tr><th>Bus</th><th>Regime / Status</th><th>|V| (p.u.)</th><th>Angle (deg)</th></tr>
            </thead>
            <tbody id="pfTableBody"></tbody>
          </table>
        </div>
        <div class="card">
          <h3>IEEE 14-Bus Convergence &amp; Voltage Diagnostics</h3>
          <img src="assets/ieee14_voltage.png" alt="IEEE 14 Voltage Profile" />
          <div style="height:0.8rem"></div>
          <img src="assets/ieee14_convergence.png" alt="IEEE 14 Quadratic Convergence" />
        </div>
      </div>
    </section>

    <section id="tab-contingency" class="tab-panel">
      <div id="n1Kpis" class="kpi-grid"></div>
      <div class="split">
        <div class="card">
          <h3>Top Ranked N-1 Contingencies (IEEE 14-Bus, 0.94&ndash;1.06 p.u., MATPOWER Q-Limits)</h3>
          <table>
            <thead>
              <tr><th>Rank</th><th>Branch</th><th>Corridor</th><th>Status</th><th>Min |V|</th><th>Q-Sat</th><th>Severity</th></tr>
            </thead>
            <tbody id="n1TableBody"></tbody>
          </table>
        </div>
        <div class="card">
          <h3>Severity Index Screening Chart</h3>
          <img src="assets/ieee14_contingency.png" alt="N-1 Contingency Ranking" />
        </div>
      </div>
    </section>

    <section id="tab-stability" class="tab-panel">
      <div id="pvKpis" class="kpi-grid"></div>
      <div class="controls">
        <label for="pvBusSelect"><strong>Inspect Bus Trajectory (Q-Limited Continuation):</strong></label>
        <select id="pvBusSelect"></select>
      </div>
      <div class="split">
        <div class="card">
          <h3>Continuation Operating Points &amp; Jacobian Conditioning</h3>
          <table>
            <thead>
              <tr><th>Lambda (x)</th><th>Total Load (MW)</th><th>Selected Bus |V| (p.u.)</th><th>Min Singular Value &sigma;<sub>min</sub>(J)</th></tr>
            </thead>
            <tbody id="pvTableBody"></tbody>
          </table>
        </div>
        <div class="card">
          <h3>P-V Nose Curve: Unconstrained vs. Generator Q-Limited</h3>
          <img src="assets/ieee14_pv_nose.png" alt="IEEE 14 P-V Nose Curve" />
        </div>
      </div>
    </section>

    <section id="tab-sparsity" class="tab-panel">
      <div class="split">
        <div class="card">
          <h3>Dense NumPy LU vs. Direct SciPy CSC spsolve</h3>
          <table>
            <thead>
              <tr><th>Buses</th><th>Jacobian</th><th>nnz(J)</th><th>Zeros (%)</th><th>Dense (ms)</th><th>Sparse (ms)</th><th>Speedup</th></tr>
            </thead>
            <tbody id="benchTableBody"></tbody>
          </table>
        </div>
        <div class="card">
          <h3>IEEE 30-Bus Reduced Jacobian Non-Zero Pattern</h3>
          <img src="assets/ieee30_jacobian_spy.png" alt="IEEE 30 Sparse Jacobian Spy Plot" />
        </div>
      </div>
    </section>
  </main>

  <footer>
    <span>Built automatically from Python solver benchmarks via GitHub Actions.</span>
    <span>MIT License &bull; Haris Zafar (@hariszafar740-lang)</span>
  </footer>

  <script>
    const DATA = __DASHBOARD_JSON__;

    document.querySelectorAll(".tab-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        document.querySelectorAll(".tab-btn").forEach(function (b) { b.classList.remove("active"); });
        document.querySelectorAll(".tab-panel").forEach(function (p) { p.classList.remove("active"); });
        btn.classList.add("active");
        document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
      });
    });

    function renderPowerFlow() {
      const key = document.getElementById("caseSelect").value;
      const filter = document.getElementById("busFilter").value.trim().toLowerCase();
      const info = DATA.cases[key];
      document.getElementById("pfTableTitle").textContent = info.title;
      document.getElementById("pfKpis").innerHTML =
        "<div class='kpi'><span>Network Size</span><strong>" + info.buses + " Buses / " + info.branches + " Lines</strong></div>" +
        "<div class='kpi'><span>NR Steps</span><strong>" + info.iterations + "</strong></div>" +
        "<div class='kpi'><span>Active Losses</span><strong>" + info.loss_mw + " MW</strong></div>" +
        "<div class='kpi'><span>Minimum Voltage</span><strong>" + info.min_v + " p.u. (B" + info.min_v_bus + ")</strong></div>" +
        "<div class='kpi'><span>Saturated PV Buses</span><strong>" + info.saturated_count + "</strong></div>";

      const rowsHtml = info.rows.filter(function (r) {
        if (!filter) return true;
        return String(r.bus).includes(filter) || r.type.toLowerCase().includes(filter);
      }).map(function (r) {
        const cls = r.v_pu < 0.95 ? "tag-crit" : (r.type.indexOf("PV->PQ") >= 0 ? "tag-warn" : "");
        return "<tr><td>" + r.bus + "</td><td class='" + cls + "'>" + r.type + "</td><td>" +
          r.v_pu.toFixed(5) + "</td><td>" + r.angle_deg.toFixed(4) + "</td></tr>";
      }).join("");
      document.getElementById("pfTableBody").innerHTML = rowsHtml;
    }

    function renderContingencies() {
      const cs = DATA.contingency_summary;
      document.getElementById("n1Kpis").innerHTML =
        "<div class='kpi'><span>Evaluated Outages (" + cs.v_bounds + ")</span><strong>" + cs.total + " Branches</strong></div>" +
        "<div class='kpi'><span>Secure Outages</span><strong class='tag-ok'>" + cs.secure + "</strong></div>" +
        "<div class='kpi'><span>Limit Violations</span><strong class='tag-warn'>" + cs.violated + "</strong></div>" +
        "<div class='kpi'><span>Islanded / Collapsed</span><strong class='tag-crit'>" + cs.islanded + " Islanded / " + cs.diverged + " Collapse</strong></div>";

      document.getElementById("n1TableBody").innerHTML = DATA.contingencies.map(function (c) {
        const cls = c.status === "SECURE" ? "tag-ok" : (c.status === "VIOLATION" ? "tag-warn" : "tag-crit");
        return "<tr><td>" + c.rank + "</td><td>" + c.branch + "</td><td>" + c.corridor +
          "</td><td class='" + cls + "'>" + c.status + "</td><td>" + c.min_v +
          "</td><td>" + c.q_sat + "</td><td><strong>" + c.severity.toFixed(2) + "</strong></td></tr>";
      }).join("");
    }

    function renderStability() {
      const pv = DATA.pv_curve;
      document.getElementById("pvKpis").innerHTML =
        "<div class='kpi'><span>Q-Limited Nose (&lambda;<sub>max</sub>)</span><strong class='tag-warn'>" + pv.qlimited_lambda + "x (" + pv.qlimited_mw + " MW)</strong></div>" +
        "<div class='kpi'><span>Unconstrained Nose</span><strong>" + pv.unconstrained_lambda + "x (" + pv.unconstrained_mw + " MW)</strong></div>" +
        "<div class='kpi'><span>Margin Overestimation</span><strong class='tag-crit'>" + (pv.unconstrained_lambda / pv.qlimited_lambda).toFixed(2) + "x Without Q-Limits</strong></div>" +
        "<div class='kpi'><span>Critical Weakest Bus</span><strong>Bus " + pv.weakest_bus + "</strong></div>";

      const busSel = document.getElementById("pvBusSelect");
      if (!busSel.options.length) {
        for (let b = 1; b <= 14; b++) {
          const opt = document.createElement("option");
          opt.value = String(b - 1);
          opt.textContent = "Bus " + b + (b === pv.weakest_bus ? " (Weakest Load Bus)" : "");
          if (b === pv.weakest_bus) opt.selected = true;
          busSel.appendChild(opt);
        }
      }
      const bIdx = parseInt(busSel.value, 10);
      const pts = pv.points_qlim;
      const stride = Math.max(1, Math.floor(pts.length / 14));
      const sampled = pts.filter(function (_, idx) { return idx % stride === 0 || idx === pts.length - 1; });
      document.getElementById("pvTableBody").innerHTML = sampled.map(function (pt) {
        return "<tr><td>" + pt.lam.toFixed(4) + "</td><td>" + pt.mw.toFixed(2) +
          "</td><td>" + pt.v_buses[bIdx].toFixed(4) + "</td><td>" + pt.sigma_min.toFixed(4) + "</td></tr>";
      }).join("");
    }

    function renderBenchmarks() {
      document.getElementById("benchTableBody").innerHTML = DATA.benchmarks.map(function (b) {
        return "<tr><td>" + b.buses + "</td><td>" + b.jac_shape +
          "</td><td>" + b.nnz + "</td><td>" + b.sparsity_pct.toFixed(1) + "%</td><td>" +
          b.dense_ms.toFixed(2) + "</td><td>" + b.sparse_ms.toFixed(2) +
          "</td><td class='tag-ok'><strong>" + b.speedup.toFixed(2) + "x</strong></td></tr>";
      }).join("");
    }

    document.getElementById("caseSelect").addEventListener("change", renderPowerFlow);
    document.getElementById("busFilter").addEventListener("input", renderPowerFlow);
    document.getElementById("pvBusSelect").addEventListener("change", renderStability);

    renderPowerFlow();
    renderContingencies();
    renderStability();
    renderBenchmarks();
  </script>
</body>
</html>
"""
    return template.replace("__VERSION__", str(payload["version"])).replace(
        "__DASHBOARD_JSON__", data_json
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build static interactive HTML5 power-flow explorer."
    )
    parser.add_argument(
        "--out-dir",
        default="site",
        help="Target output directory for static HTML/assets (default: site)",
    )
    args = parser.parse_args(argv)
    index_path = build_dashboard_site(args.out_dir)
    print(f"Built interactive web dashboard at {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
