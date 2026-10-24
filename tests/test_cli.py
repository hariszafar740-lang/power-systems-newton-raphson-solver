"""End-to-end verification tests for the ``nr-powerflow`` CLI and package API."""

import json

import src
from src.cli import main as cli_main


def test_package_exposes_version_and_public_api():
    assert src.__version__ == "1.0.0"
    assert callable(src.solve_newton_raphson)
    assert callable(src.solve_nr_with_q_limits)
    assert callable(src.run_n1_contingency_analysis)
    assert callable(src.trace_pv_nose_curve)


def test_cli_solve_json_and_plot_export(capsys, tmp_path):
    rc = cli_main(
        [
            "solve",
            "--case",
            "ieee14",
            "--load-mult",
            "1.25",
            "--enforce-q-limits",
            "--format",
            "json",
            "--plots-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["case"] == "ieee14"
    assert payload["enforce_q_limits"] is True
    assert set(payload["saturated_pv_buses"].keys()) == {"2", "3", "6", "8"}
    assert (tmp_path / "convergence_curve.png").exists()
    assert (tmp_path / "voltage_profile.png").exists()
    assert (tmp_path / "jacobian_sparsity.png").exists()


def test_cli_contingency_and_pv_curve_and_benchmark(capsys, tmp_path):
    rc_cont = cli_main(
        ["contingency", "--case", "ieee14", "--enforce-q-limits", "--top", "4", "--format", "json"]
    )
    assert rc_cont == 0
    cont_data = json.loads(capsys.readouterr().out)
    assert cont_data["total_contingencies"] == 20
    assert cont_data["islanded_count"] == 1

    pv_png = tmp_path / "nose.png"
    rc_pv = cli_main(["pv-curve", "--bus", "14", "--format", "json", "--plot", str(pv_png)])
    assert rc_pv == 0
    pv_data = json.loads(capsys.readouterr().out)
    assert pv_data["q_limited_lambda_max"] < pv_data["unconstrained_lambda_max"]
    assert pv_png.exists()

    rc_bench = cli_main(["benchmark", "--sizes", "16", "48", "--format", "markdown"])
    assert rc_bench == 0
    bench_out = capsys.readouterr().out
    assert "| Buses | Jacobian Dim |" in bench_out
