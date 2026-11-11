# Fairness-Driven Downlink Optimization in PD-NOMA-MEC with UAV-IRS for 5G/6G Networks — Simulation Code

Simulation code for the paper *"Fairness-Driven Downlink Optimization in PD-NOMA-MEC with UAV-IRS for 5G/6G Networks"*.

## Overview

This repository implements the numerical experiments for a user-fairness optimization problem in an Airborne Intelligent Reflecting Surface (AIRS)-assisted, power-domain Non-Orthogonal Multiple Access (PD-NOMA) mobile edge computing (MEC) network.

The problem maximizes the minimum (max-min) downlink transmission rate of ground users by jointly optimizing:

1. the UAV's 3D position,
2. the IRS reflection matrix,
3. the per-user power allocation,

under line-of-sight and non-line-of-sight propagation.

The high-dimensional non-convex problem is solved with an iterative **Block Coordinate Descent (BCD)** framework that decomposes it into three sub-problems:

- **IRS reflection matrix** — penalty method + Successive Convex Approximation (SCA),
- **UAV 3D position** — variable substitution + SCA,
- **Power allocation** — convex log-power subproblem (Lagrange multiplier / dual decomposition).

## Repository structure

```
.
├── 1.py                                # Main simulation entry (AR_IRS + BCD framework)
├── lunwen2.py                          # Main simulation entry (alternate variant)
├── baseline_plot.py                    # Baseline comparison plotting
├── runtime_scaling.py                  # Runtime scaling analysis
├── ci_sweep.py                         # Configuration/parameter sweep (CI)
├── mc_jensen_validation.py             # Monte Carlo validation of Jensen bound
├── regen_*.py                          # Figure/result regeneration scripts
├── revised_*.py                        # Revision-response experiments (reviewers 1-5)
├── run_*.py / run_*.sh                 # Experiment runners and pipelines
├── func/                               # Helper modules (earlier version)
│   ├── algorithms.py                   # AR_IRS class + BCD optimization
│   ├── alldata.py                      # Data generation and parameter definitions
│   ├── myfunc.py                       # Utility functions
│   └── ploter.py                       # Plotting helpers
├── new_func/                           # Helper modules (current version)
│   ├── algorithms.py
│   ├── alldata.py
│   ├── myfunc.py
│   └── ploter.py
├── new_data/                           # Raw numerical results (CSV)
├── revised_results/                    # Raw numerical results (JSON) for the revision experiments
├── irs_rank_results/plot.py            # IRS rank/eigenvalue analysis plotting
└── environment.txt                     # Conda environment export (dependency list)
```

The naming convention for the parameter-sweep scripts refers to swept quantities, e.g. `difL` (IRS elements), `difP` (power), `difM` (number of users), `difW` (bandwidth), `difBeta`, `difPHAP` (HAP power), and `difQ` (UAV position).

## Dependencies

The full environment is listed in `environment.txt` (a `conda env export`). Key packages:

- `numpy`, `scipy`, `matplotlib`, `pandas`
- `cvxpy` (1.5.1) — convex optimization modeling
- `networkx`
- `mosek` — commercial solver (requires a license)
- `osqp`, `ecos`, `clarabel` — open-source conic solvers (fallbacks)

> **Note:** some scripts `import mosek`. MOSEK is a commercial solver and requires a separate license. The open-source solvers (`osqp`, `ecos`, `clarabel`) can be used by changing the solver argument in the `cvxpy` calls if a MOSEK license is not available.

## Running the experiments

Set up the environment, then run the main simulation:

```bash
conda create -n aris python=3.12
conda activate aris
# install dependencies (see environment.txt), then:
python 1.py
# or
python lunwen2.py
```

Individual parameter sweeps and revision experiments are launched via their own `revised_*.py` / `run_*.py` scripts.

## Data and reproducibility

- `new_data/*.csv` and `revised_results/*.json` hold the raw numerical results consumed by the plotting/regeneration scripts.
- Random seeds are set inside the simulation scripts (e.g. `np.random.seed(...)`) for reproducibility.
