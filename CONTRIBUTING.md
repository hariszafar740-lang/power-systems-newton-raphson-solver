# Contributing to Power Systems Newton-Raphson Solver

Thank you for your interest in contributing to this power-systems numerical analysis project.

## Engineering & Numerical Verification Standards

1. Physical & Mathematical Integrity:
 - Every change to admittance matrices (Y_bus), power mismatch vectors, or Jacobian sub-matrices (J_11, J_12, J_21, J_22) must be backed by analytical or central finite-difference verification tests.
 - Benchmark reference tests (`tests/test_ieee14_reference.py`) must remain reproducible against MATPOWER `case14` within documented per-unit tolerances.
2. Fail-Closed Error Handling:
 - Non-convergent solves must raise `RuntimeError` rather than returning silent unconverged states.
 - Invalid bus/branch topologies or non-finite inputs must raise `ValueError`.

## Local Development Workflow (Modular Git + Safe Rebase)

~~~bash
git checkout main
git pull --rebase origin main
git checkout -b feat/your-feature-name

python -m pip install -r requirements.txt ruff
ruff check.
python -m pytest tests/ -v
python main.py
~~~

All pull requests must pass the Python 3.10–3.13 CI matrix before merge.
