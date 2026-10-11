# Power Systems Newton-Raphson Load Flow Solver

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Build Status](https://img.shields.io/badge/pytest-passing-brightgreen)
![Domain](https://img.shields.io/badge/Focus-Power%20Systems%20Analysis-orange)

An object-oriented Python implementation of the **Newton-Raphson Load Flow Algorithm** engineered for power grid dynamic stability and power flow calculation. Validated against standard **3-bus** and **IEEE 14-bus benchmark networks**.

## Key Features
* **Admittance Matrix Formulation:** Automated $Y_{bus}$ construction for line resistance ($R$), reactance ($X$), total line charging ($B$), bus shunts, and real off-nominal transformer taps.
* **Full Jacobian Engine:** Complete derivative evaluation across four dynamic sub-matrices ($J_{11}, J_{12}, J_{21}, J_{22}$) supporting Slack, PV, and PQ buses, verified against central finite differences.
* **Transmission Loss Analysis:** Complex power flow ($S_{ij}$) and real/reactive transmission loss calculations ($S_{loss}$).
* **Honest Convergence Reporting:** A solve that does not reach the tolerance raises `RuntimeError` (never a silent "success"), and invalid network data raises `ValueError`.
* **Automated Test Suite:** `pytest` verification covering $Y_{bus}$ construction, finite-difference Jacobian checks, failure handling, and regression of the full IEEE 14-bus solution against an independent MATPOWER-grade reference.

## Mathematical Formulation
The non-linear active and reactive power balance equations solved per bus $i$:

$$P_i = \sum_{j=1}^{N} \vert{}V_i\vert{}\vert{}V_j\vert{}(G_{ij}\cos\theta_{ij} + B_{ij}\sin\theta_{ij})$$

$$Q_i = \sum_{j=1}^{N} \vert{}V_i\vert{}\vert{}V_j\vert{}(G_{ij}\sin\theta_{ij} - B_{ij}\cos\theta_{ij})$$

State updates $\Delta x = [\Delta \theta, \Delta \vert{}V\vert{}]^T$ are calculated iteratively via the inverse Jacobian system:

$$\begin{bmatrix} \Delta P \\ \Delta Q \end{bmatrix} = \begin{bmatrix} J_{11} & J_{12} \\ J_{21} & J_{22} \end{bmatrix} \begin{bmatrix} \Delta \theta \\ \Delta \vert{}V\vert{} \end{bmatrix}$$

## Simulation Artifacts & Convergence
Convergence is achieved quadratically within **4 iterations** on the IEEE 14-bus benchmark ($tol = 10^{-4}\text{ p.u.}$).

### Convergence Trajectory
![Convergence Curve](results/convergence_curve.png)

### Bus Voltage Profile
![Voltage Profile](results/voltage_profile.png)

## Validation Against MATPOWER case14
The IEEE 14-bus model is pinned to **MATPOWER `case14.m`** (100 MVA base):
`data/case14.m` of https://github.com/MATPOWER/matpower at commit
`2e0ef79a6c4526858ee923491d92781ad4507fb7` (2026-08-04). The same values were
cross-checked in MATPOWER releases 7.1 and 8.0 and in PYPOWER 5.1.21.

Known convention note: the line-charging susceptances of branches **2-4, 2-5,
3-4, and 4-5** differ from the raw IEEE Common Data Format file and some
textbook tables, which circulate a row-shifted variant of these four values.
This repository follows the MATPOWER convention deliberately so its solution
can be compared with the widely used MATPOWER result; the choice is locked
down by `tests/test_ieee14_reference.py`.

The independent reference solution was computed with pandapower 3.5.6
(`networks.case14`, Newton-Raphson, `tolerance_mva=1e-8`) and reproduces the
solution columns stored in MATPOWER's case file. With the corrected data this
solver agrees with that reference to better than **5e-7 p.u.** on every bus
voltage magnitude (angle error < 1e-8 rad); the regression tests assert
tolerances of `1e-4 p.u.` / `1e-4 rad` for voltages/angles and `5e-4 p.u.`
for losses and branch flows. Total losses: **P = 0.1339 p.u., Q = 0.3012
p.u.**; slack generation **2.3239 − j0.1655 p.u.**

Modeling assumptions (matching the default MATPOWER power-flow formulation):
PV-bus voltage magnitudes are held at their setpoints supplied in `V_init`,
generator reactive-power limits are not enforced, and the bus-9 shunt
(+j0.19 p.u.) is included via `get_ieee14_bus_shunts()`.

## Running the Solver and Tests
```bash
pip install -r requirements.txt
python main.py              # solve IEEE 14-bus, write results/*.png
python -m pytest tests/ -v  # 38 tests
ruff check .                # lint
```
CI (`.github/workflows/ci.yml`) runs the test suite, the Ruff lint pass, and a
`main.py` smoke run on Python 3.10–3.13.

## Repository Structure
```text
├── main.py                   # Primary solver entry point
├── requirements.txt          # Package dependencies
├── pyproject.toml            # Ruff & pytest configuration
├── src/                      # Source modules
│   ├── bus_data.py           # 3-bus test network specification
│   ├── ieee14_data.py        # IEEE 14-bus benchmark parameters (MATPOWER case14)
│   ├── ybus_builder.py       # Admittance matrix builder (taps, shunts)
│   ├── mismatch_calculator.py# Delta P & Delta Q mismatch engine
│   ├── jacobian_builder.py   # Jacobian matrix constructor
│   ├── nr_solver.py          # Iterative Newton-Raphson loop
│   ├── line_flow_calculator.py # Line flow & loss calculation
│   └── visualizer.py         # Matplotlib rendering module
├── tests/                    # Unit testing suite
│   ├── test_ybus.py          # Admittance matrix test cases
│   ├── test_jacobian.py      # Finite-difference Jacobian verification
│   ├── test_nr_solver.py     # Convergence test cases
│   ├── test_nr_failure_handling.py # Failure/diagnostic regression tests
│   ├── test_line_flows.py    # Branch flow consistency tests
│   └── test_ieee14_reference.py    # MATPOWER case14 data & solution regression
└── results/                  # Exported plots and simulation output
```
