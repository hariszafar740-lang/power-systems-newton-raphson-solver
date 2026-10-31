"""Tests for the static GitHub Pages interactive web explorer builder."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.web_dashboard import build_dashboard_site, main


def test_build_dashboard_site_creates_html_and_all_visual_assets(tmp_path: Path):
    index_path = build_dashboard_site(tmp_path / "site")
    assert index_path.exists()
    assert (tmp_path / "site" / ".nojekyll").exists()

    expected_assets = [
        "ieee14_voltage.png",
        "ieee14_convergence.png",
        "ieee30_jacobian_spy.png",
        "ieee14_contingency.png",
        "ieee14_pv_nose.png",
    ]
    for asset_name in expected_assets:
        asset_file = tmp_path / "site" / "assets" / asset_name
        assert asset_file.exists(), f"Missing dashboard asset: {asset_name}"
        assert asset_file.stat().st_size > 1024

    html = index_path.read_text(encoding="utf-8")
    assert "Power Systems Sparse Newton-Raphson" in html
    match = re.search(r"const DATA = ({.*});", html)
    assert match is not None
    payload = json.loads(match.group(1))
    summary = payload["contingency_summary"]
    assert summary["total"] == 20
    assert summary["secure"] == 10
    assert summary["violated"] == 8
    assert summary["islanded"] == 1
    assert summary["diverged"] == 1
    statuses = {row["status"] for row in payload["contingencies"]}
    assert {"ISLANDED", "NON_CONVERGED", "VIOLATION", "SECURE"} <= statuses

    rc = main(["--out-dir", str(tmp_path / "cli_site")])
    assert rc == 0
    assert (tmp_path / "cli_site" / "index.html").exists()
