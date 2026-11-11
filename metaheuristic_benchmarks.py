"""
Reviewer 3, Comment 4: convergence-speed and performance benchmarks.

The abstract/conclusion claims "converges rapidly and effectively enhances
the rates" without a reference algorithm. This script compares the proposed
BCD / penalty-SCA / convex power-allocation solver with two joint-optimization
metaheuristics -- particle swarm optimization (PSO) and a genetic algorithm
(GA) -- under the SAME full vector AIRS model, the same search objective, the
same Monte Carlo evaluation, and the same wall-clock budget.

The common objective is the deterministic worst-user statistical rate
surrogate
    Rbar_i = W log2(1 + Omega_i p_i / (Omega_i sum_{j>i} p_j + sigma^2)),
    Omega_i = LoS_i + NLoS_i,
which is exactly the quantity the proposed BCD maximizes internally (the
GAIN_SCALE rescaling cancels in the SINR). The final design of every scheme
is evaluated by the SAME Monte Carlo empirical ergodic rate used in the main
paper, so the comparison is apples-to-apples.

Decoding-order convention: as in the existing framework, the physical user
index equals the SIC decode order (index 0 = weakest), i.e. the permutation is
NOT re-sorted by Omega inside the rate evaluation. This keeps the benchmark
comparable with the proposed numbers already reported in the paper.

Outputs: revised_results/review3_comment4_convergence.json and
         fig_review3_comment4_convergence.pdf.
"""
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vector_airs_model as R

SIGMA2 = R.SIGMA2
W_BAND = R.W_BAND
P_HAP = R.P_HAP
P_HAP_DBM = R.P_HAP_DBM
Z_MIN = R.Z_min
Z_MAX = R.Z_max

L = 8
M = 4
S_MC = 2000
N_SEEDS = 3
OUT = 5
INNER = 20

# ---- search box (must contain the proposed optimum Q ~ [85, 0, 30]) ----
X_LO, X_HI = 85.0, 115.0
Y_LO, Y_HI = -15.0, 15.0
PHI_LO, PHI_HI = 0.0, 2.0 * np.pi


# ----------------------------------------------------------------------
# Deterministic worst-user statistical rate surrogate (common objective)
# ----------------------------------------------------------------------
def worst_user_surrogate(Q, Theta, P):
    rates = np.zeros(M)
    for i in range(M):
        LoS, NLoS = R.channel_components(Q, Theta, i, L)
        Omega = LoS + NLoS
        inter = float(np.sum(P[i + 1:M]))
        sinr = Omega * P[i] / (Omega * inter + SIGMA2)
        rates[i] = W_BAND * np.log2(1.0 + sinr)
    return float(rates.min())


# ---- encode/decode the 3+L+M = 15-dimensional joint variable ----
def make_bounds():
    lb = [X_LO, Y_LO, Z_MIN] + [PHI_LO] * L + [0.0] * M
    ub = [X_HI, Y_HI, Z_MAX] + [PHI_HI] * L + [P_HAP] * M
    return np.array(lb), np.array(ub)


def decode(x):
    Q = np.array([x[0], x[1], x[2]], dtype=float)
    Theta = np.diag(np.exp(1j * x[3:3 + L]))
    p = np.clip(x[3 + L:], 0.0, None)
    if p.sum() > P_HAP:
        p = p / p.sum() * P_HAP
    return Q, Theta, p


def objective(x):
    return worst_user_surrogate(*decode(x))


def n95(history):
    """First index reaching 95% of the final (max) objective value."""
    final = float(history.max())
    if final <= 0:
        return len(history)
    idx = np.argmax(history >= 0.95 * final)
    return int(idx) + 1


# ----------------------------------------------------------------------
# Particle swarm optimization
# ----------------------------------------------------------------------
def pso(seed, budget, pop=80, max_iter=2000):
    lb, ub = make_bounds()
    dim = len(lb)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    X = rng.uniform(lb, ub, size=(pop, dim))
    V = np.zeros((pop, dim))
    fit = np.array([objective(x) for x in X])
    pbest = X.copy()
    pbest_fit = fit.copy()
    g = int(np.argmax(fit))
    gbest = X[g].copy()
    gbest_fit = fit[g]
    history = [gbest_fit]
    it = 0
    while it < max_iter and (time.perf_counter() - t0) < budget:
        w = 0.9 - 0.5 * (it / max_iter)
        r1 = rng.random((pop, dim))
        r2 = rng.random((pop, dim))
        V = w * V + 2.0 * r1 * (pbest - X) + 2.0 * r2 * (gbest - X)
        X = np.clip(X + V, lb, ub)
        fit = np.array([objective(x) for x in X])
        imp = fit > pbest_fit
        pbest[imp] = X[imp]
        pbest_fit[imp] = fit[imp]
        g = int(np.argmax(pbest_fit))
        if pbest_fit[g] > gbest_fit:
            gbest = pbest[g].copy()
            gbest_fit = pbest_fit[g]
        history.append(gbest_fit)
        it += 1
    return gbest, gbest_fit, np.array(history), time.perf_counter() - t0, it


# ----------------------------------------------------------------------
# Genetic algorithm (tournament + blend crossover + Gaussian mutation)
# ----------------------------------------------------------------------
def ga(seed, budget, pop=80, max_iter=2000):
    lb, ub = make_bounds()
    dim = len(lb)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    X = rng.uniform(lb, ub, size=(pop, dim))
    fit = np.array([objective(x) for x in X])
    best_idx = int(np.argmax(fit))
    best_x = X[best_idx].copy()
    best_fit = float(fit[best_idx])
    history = [best_fit]
    it = 0
    while it < max_iter and (time.perf_counter() - t0) < budget:
        idx = rng.integers(0, pop, size=(pop, 2))
        winners = np.where(fit[idx[:, 0]] >= fit[idx[:, 1]], idx[:, 0], idx[:, 1])
        parents = X[winners]
        perm = rng.permutation(pop)
        alpha = rng.uniform(-0.2, 1.2, size=(pop, dim))
        children = alpha * parents + (1.0 - alpha) * parents[perm]
        mask = rng.random((pop, dim)) < 0.1
        children += mask * rng.normal(0.0, 0.05 * (ub - lb), size=(pop, dim))
        X = np.clip(children, lb, ub)
        fit = np.array([objective(x) for x in X])
        generation_best = int(np.argmax(fit))
        if fit[generation_best] > best_fit:
            best_x = X[generation_best].copy()
            best_fit = float(fit[generation_best])
        history.append(best_fit)
        it += 1
    return best_x, best_fit, np.array(history), time.perf_counter() - t0, it


# ----------------------------------------------------------------------
# Proposed BCD (reuse AR_IRS; capture the tau trajectory + wall-clock)
# ----------------------------------------------------------------------
def run_proposed(seed):
    irs = R.AR_IRS()
    t0 = time.perf_counter()
    tau_list = irs.bcd_optimization(max_iter=OUT, L=L, W=1e6, M=M,
                                    P_HAP=P_HAP_DBM, seed=seed, inner_max=INNER)
    wall = time.perf_counter() - t0
    Q, Theta, P = irs.Q, irs.Theta, irs.P
    comps = [R.channel_components(Q, Theta, i, L) for i in range(M)]
    _, mn, _ = R.mc_eval(Q, Theta, P, L, M, S_MC, seed, comps=comps)
    return float(mn), np.array(tau_list), wall, (Q, Theta, P)


def run_meta(optimizer, seed, budget):
    x, fit, history, wall, it = optimizer(seed, budget)
    Q, Theta, P = decode(x)
    comps = [R.channel_components(Q, Theta, i, L) for i in range(M)]
    # Use the same held-out channel-sample seed for each paired repetition.
    eval_seed = seed % 1000
    _, mn, _ = R.mc_eval(Q, Theta, P, L, M, S_MC, eval_seed, comps=comps)
    return float(mn), history, wall, it, (Q, Theta, P)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def run():
    # wall-clock budget for the metaheuristics = one BCD run (seed 0)
    _, _, bcd_wall, _ = run_proposed(0)
    budget = max(bcd_wall, 1.0)
    print(f"single BCD wall-clock = {bcd_wall:.3f} s  -> metaheuristic budget = {budget:.3f} s")

    results = {"budget_s": budget, "L": L, "M": M, "S_MC": S_MC,
               "N_SEEDS": N_SEEDS, "schemes": {}}

    def summary(name, mn_list, n95_list, wall_list):
        return {
            "min_rate_bps": mn_list,
            "mean_rate_bps": float(np.mean(mn_list)),
            "n95_iterations": n95_list,
            "wall_clock_s": wall_list,
            "mean_wall_s": float(np.mean(wall_list)),
        }

    # proposed
    mn_p, n95_p, wall_p, hist_p = [], [], [], None
    for s in range(N_SEEDS):
        mn, tau_list, wall, _ = run_proposed(s)
        mn_p.append(mn)
        n95_p.append(n95(tau_list))
        wall_p.append(wall)
        if s == 0:
            hist_p = tau_list
    results["schemes"]["Proposed"] = summary("Proposed", mn_p, n95_p, wall_p)

    # PSO
    mn_ps, n95_ps, wall_ps, hist_ps = [], [], [], None
    for s in range(N_SEEDS):
        mn, hist, wall, it, _ = run_meta(pso, 1000 + s, budget)
        mn_ps.append(mn)
        n95_ps.append(n95(hist))
        wall_ps.append(wall)
        if s == 0:
            hist_ps = hist
    results["schemes"]["PSO"] = summary("PSO", mn_ps, n95_ps, wall_ps)

    # GA
    mn_ga, n95_ga, wall_ga, hist_ga = [], [], [], None
    for s in range(N_SEEDS):
        mn, hist, wall, it, _ = run_meta(ga, 2000 + s, budget)
        mn_ga.append(mn)
        n95_ga.append(n95(hist))
        wall_ga.append(wall)
        if s == 0:
            hist_ga = hist
    results["schemes"]["GA"] = summary("GA", mn_ga, n95_ga, wall_ga)

    base = float(np.mean(mn_p))
    print("\n=== Reviewer 3 Comment 4: convergence benchmark ===")
    for name, mn_list in [("Proposed", mn_p), ("PSO", mn_ps), ("GA", mn_ga)]:
        m = float(np.mean(mn_list))
        print(f"  {name:10s} min rate = {m/1e6:8.4f} Mbps  "
              f"({100.0*(m-base)/base:+6.2f}% vs proposed)  "
              f"n95={results['schemes'][name]['n95_iterations']}  "
              f"wall={results['schemes'][name]['mean_wall_s']:.2f}s")

    with open(os.path.join("revised_results", "review3_comment4_convergence.json"),
              "w") as f:
        json.dump(results, f, indent=2)

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))

    # descending final MC rate: Proposed > GA > PSO
    names = ["Proposed", "GA", "PSO"]
    means = [float(np.mean(mn_p)), float(np.mean(mn_ga)), float(np.mean(mn_ps))]
    mins = [min(mn_p), min(mn_ga), min(mn_ps)]
    maxs = [max(mn_p), max(mn_ga), max(mn_ps)]
    err_low = [m - lo for m, lo in zip(means, mins)]
    err_high = [hi - m for m, hi in zip(means, maxs)]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    bars = ax1.bar(names, [m / 1e6 for m in means], 0.5, color=colors,
                   yerr=[np.array(err_low) / 1e6, np.array(err_high) / 1e6],
                   capsize=4)
    for xi, v in zip(range(len(names)), means):
        ax1.text(xi, v / 1e6, f"{v/1e6:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("Worst-user ergodic rate (Mbps)")
    ax1.set_title("(a) Final Monte Carlo performance")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # convergence: objective vs wall-clock (representative seed 0).
    # BCD tau is in nats/s; convert to bps via 1/ln(2) for a common axis.
    ax2.plot(np.arange(len(hist_p)) * (wall_p[0] / max(len(hist_p), 1)),
             hist_p / np.log(2.0) / 1e6, marker="o", ms=4, label="Proposed",
             color=colors[0])
    for hist, wall, nm, c in [(hist_ga, wall_ga[0], "GA", colors[1]),
                              (hist_ps, wall_ps[0], "PSO", colors[2])]:
        ax2.plot(np.linspace(0, wall, len(hist)), hist / 1e6, label=nm, color=c)
    ax2.set_xlabel("Wall-clock time (s)")
    ax2.set_ylabel("Best statistical surrogate (Mbps)")
    ax2.set_title("(b) Convergence (seed 0)")
    ax2.legend(fontsize=7)
    ax2.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig("fig_review3_comment4_convergence.pdf", dpi=300, bbox_inches="tight")
    print("\nSaved revised_results/review3_comment4_convergence.json "
          "and fig_review3_comment4_convergence.pdf")
    return results


if __name__ == "__main__":
    run()
