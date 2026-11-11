"""
Experiments for Reviewer 1, Comments 2--4.

All experiments reuse the clean scalar Rician channel model and the closed-form
max-min power allocation from mc_jensen_validation.py, and additionally use the
SIC-feasible achievable rate (minimum over all required SIC decoders) so that
random order reversals are handled consistently.

  Comment 2 (decoding order): enumerate all M! orders (M=3,4), compare the
      proposed statistical-gain order against fixed/distance/random orders.
  Comment 3 (fairness): Jain index, rate variance, average rate, 10th-percentile
      rate; compare max-min vs proportional fairness vs weighted max-min.
  Comment 4 (solution quality): small-scale grid-search reference (M=2,L=2) and
      multi-random-initialization statistics.
"""

import os
import json
import numpy as np
from itertools import permutations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ----------------------------------------------------------------------
# System parameters (identical to mc_jensen_validation.py)
# ----------------------------------------------------------------------
rho0 = 0.001
SIGMA2 = 1e-8
W_BAND = 1e6
Z_MIN, Z_MAX = 30.0, 60.0

K_DEF_DB = 10.0      # Rician factor (dB) of the AIRS-GMD LoS link (K2); the
                     # direct HAP-GMD link is blocked -> pure NLoS (kappa1 = 0)
L_DEF = 16
M_DEF = 5
PHAP_DEF_DB = 30.0
BETA1, BETA2 = 3.5, 2.0   # BETA1: blocked/NLoS ground link; BETA2: aerial LoS link

W_S = np.array([0.0, 0.0, 20.0])

# GMDs distributed over a ~350 m^2 region located ~100 m from the HAP, matching
# the "HAP at a considerable distance / obstacles blocking the direct path"
# scenario in Section II (the direct HAP-GMD link is blocked, so the AIRS
# reflected path dominates).
USER_POSITIONS = np.array([
    [100.0,   0.00, 0.0],
    [106.0,   1.00, 0.0],
    [104.0,  -1.25, 0.0],
    [ 97.0,   1.50, 0.0],
    [109.0,  -0.50, 0.0],
    [ 93.0,   1.25, 0.0],
    [101.0,  -2.00, 0.0],
    [107.0,  -1.50, 0.0],
    [ 95.0,  -1.00, 0.0],
    [103.0,   2.00, 0.0],
    [ 99.0,   0.50, 0.0],
    [108.0,   0.75, 0.0],
    [ 92.0,  -1.75, 0.0],
    [105.0,   1.75, 0.0],
])

# Heterogeneous topology used for the fairness comparison (Comment 3): four
# near users and one distant cell-edge user that acts as the rate bottleneck.
FAIRNESS_POSITIONS = np.array([
    [ 96.0,   0.00, 0.0],
    [102.0,   1.00, 0.0],
    [ 99.0,  -1.25, 0.0],
    [105.0,   0.25, 0.0],
    [160.0,   0.00, 0.0],
])

S = 20000
SEED = 2026


def kappa_from_db(K_db):
    K_lin = 10.0 ** (K_db / 10.0)
    return K_lin * rho0 / (K_lin + 1.0)


def channel_stats(Q, w_i, K_db, L):
    # Direct HAP-GMD link is blocked (obstacle) -> pure NLoS: kappa1 = 0.
    # K_db is the Rician factor (dB) of the AIRS-GMD LoS link (kappa2).
    kappa1 = 0.0
    kappa2 = kappa_from_db(K_db)
    d_si = np.linalg.norm(W_S - w_i)
    d_ui = np.linalg.norm(Q - w_i)
    d_us = np.linalg.norm(Q - W_S)
    a_i = np.sqrt(kappa1) / (d_si ** (BETA1 / 2.0))                     # direct LoS (= 0, blocked)
    b_i = np.sqrt(kappa2 * rho0 * L * L) / ((d_ui ** (BETA2 / 2.0)) * d_us)  # reflected LoS
    L_i = (a_i + b_i) ** 2
    gamma_i = (rho0 - kappa1) / (d_si ** BETA1)                        # direct NLoS
    tau_i = L * rho0 * (rho0 - kappa2) / ((d_ui ** BETA2) * (d_us ** 2))    # reflected NLoS
    s2_i = gamma_i + tau_i
    return L_i + s2_i, L_i, s2_i


def channel_stats_for_users(Q, W_u, K_db, L):
    M = len(W_u)
    Omega = np.zeros(M)
    L_vec = np.zeros(M)
    s2_vec = np.zeros(M)
    for i in range(M):
        Omega[i], L_vec[i], s2_vec[i] = channel_stats(Q, W_u[i], K_db, L)
    return Omega, L_vec, s2_vec


def uav_position(M):
    W_u = USER_POSITIONS[:M]
    cen = np.mean(W_u, axis=0)
    cen[2] = 0.5 * (Z_MIN + Z_MAX)
    return cen


def max_min_power(Omega, P_HAP, tol=1e-9, iters=200):
    """Closed-form max-min power for a given decoding order (Omega in that order)."""
    M = len(Omega)
    Omega = np.asarray(Omega, dtype=float)
    lo, hi = 0.0, 200.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        g = 2.0 ** mid - 1.0
        P = np.zeros(M)
        P[M - 1] = g * SIGMA2 / Omega[M - 1]
        for r in range(M - 2, -1, -1):
            P[r] = g * (np.sum(P[r + 1:M]) + SIGMA2 / Omega[r])
        if np.sum(P) > P_HAP:
            hi = mid
        else:
            lo = mid
    tau = 0.5 * (lo + hi)
    g = 2.0 ** tau - 1.0
    P = np.zeros(M)
    P[M - 1] = g * SIGMA2 / Omega[M - 1]
    for r in range(M - 2, -1, -1):
        P[r] = g * (np.sum(P[r + 1:M]) + SIGMA2 / Omega[r])
    return tau, P


def jensen_rates(Omega, P):
    M = len(Omega)
    Rbar = np.zeros(M)
    for i in range(M):
        inter = np.sum(P[i + 1:M])
        Rbar[i] = W_BAND * np.log2(1.0 + Omega[i] * P[i] / (Omega[i] * inter + SIGMA2))
    return Rbar


def sample_gains(L_vec, s2_vec, S, rng):
    """Instantaneous channel power gains |H_i|^2 for each physical user (M, S)."""
    L_vec = np.asarray(L_vec, dtype=float)
    s2_vec = np.asarray(s2_vec, dtype=float)
    M = len(L_vec)
    amp = np.sqrt(np.maximum(L_vec, 0.0))
    std = np.sqrt(np.maximum(s2_vec, 0.0))
    gains = np.zeros((M, S))
    for i in range(M):
        wr = rng.normal(0.0, std[i] / np.sqrt(2.0), S)
        wi = rng.normal(0.0, std[i] / np.sqrt(2.0), S)
        gains[i] = (amp[i] + wr) ** 2 + wi ** 2
    return gains


def sic_rate(gains_in_order, P):
    """SIC-feasible empirical rate (bps) for messages in decoding order.

    For message r, the supported rate is the minimum over all decoders k >= r.
    """
    M = len(P)
    P = np.asarray(P, dtype=float)
    R = np.zeros(M)
    for r in range(M):
        inter = np.sum(P[r + 1:M])
        sinr = (P[r] * gains_in_order[r:]) / (gains_in_order[r:] * inter + SIGMA2)
        R[r] = np.mean(np.min(W_BAND * np.log2(1.0 + sinr), axis=0))
    return R


def mc_min_rate_for_order(Omega, gains, P_HAP, order=None):
    """Return (min SIC-feasible MC rate, power) for a given decoding order."""
    M = len(Omega)
    if order is None:
        order = np.argsort(Omega)  # ascending statistical gain
    order = np.asarray(order)
    Om_s = Omega[order]
    tau, P = max_min_power(Om_s, P_HAP)
    R = sic_rate(gains[order], P)
    return float(np.min(R)), P, order


def jensen_min_rate_for_order(Omega, P_HAP, order=None):
    M = len(Omega)
    if order is None:
        order = np.argsort(Omega)
    order = np.asarray(order)
    Om_s = Omega[order]
    tau, P = max_min_power(Om_s, P_HAP)
    Rbar = jensen_rates(Om_s, P)
    return float(np.min(Rbar)), P, order


# ======================================================================
# Comment 2 : decoding-order enumeration and strategy comparison
# ======================================================================
def comment2(M, K_db, L, PHAP_db, S, seed):
    rng = np.random.default_rng(seed)
    Q = uav_position(M)
    W_u = USER_POSITIONS[:M]
    P_HAP = 10.0 ** (PHAP_db / 10.0)
    Omega, L_vec, s2_vec = channel_stats_for_users(Q, W_u, K_db, L)
    gains = sample_gains(L_vec, s2_vec, S, rng)

    # --- best enumerated order (over all M! permutations) ---
    best = -1.0
    best_pi = None
    for pi in permutations(range(M)):
        pi = list(pi)
        Om_pi = Omega[pi]
        tau, P = max_min_power(Om_pi, P_HAP)
        R = sic_rate(gains[pi], P)
        mn = float(np.min(R))
        if mn > best:
            best = mn
            best_pi = pi
    enum_best = best

    # --- proposed statistical-gain (ascending Omega) order ---
    stat_min, _, _ = mc_min_rate_for_order(Omega, gains, P_HAP, order=np.argsort(Omega))

    # --- distance-based order (ascending distance to HAP) ---
    d_si = np.array([np.linalg.norm(W_S - w) for w in W_u])
    dist_order = np.argsort(d_si)  # closest first
    dist_min, _, _ = mc_min_rate_for_order(Omega, gains, P_HAP, order=dist_order)

    # --- random fixed orders (average over several) ---
    rand_vals = []
    n_rand = 20
    for _ in range(n_rand):
        pi = rng.permutation(M)
        mn, _, _ = mc_min_rate_for_order(Omega, gains, P_HAP, order=pi)
        rand_vals.append(mn)
    rand_mean = float(np.mean(rand_vals))

    return {
        "M": M,
        "enum_best_bps": enum_best,
        "statistical_bps": stat_min,
        "distance_bps": dist_min,
        "random_bps": rand_mean,
        "eta_stat_vs_enum_pct": 100.0 * stat_min / enum_best,
        "gap_stat_vs_enum_pct": 100.0 * (enum_best - stat_min) / enum_best,
        "gain_stat_vs_fixed_dist_pct": 100.0 * (stat_min - dist_min) / dist_min,
    }


# ======================================================================
# Comment 3 : fairness metrics and alternative fairness objectives
# ======================================================================
def pf_power(Omega, P_HAP, eps=1e-3, n_start=12, seed=0):
    """Power allocation maximizing proportional fairness sum log(Rbar + eps)."""
    M = len(Omega)
    Omega = np.asarray(Omega, dtype=float)
    rng = np.random.default_rng(seed)

    def neg_pf(p):
        Rbar = np.zeros(M)
        for i in range(M):
            inter = np.sum(p[i + 1:M])
            Rbar[i] = W_BAND * np.log2(1.0 + Omega[i] * p[i] / (Omega[i] * inter + SIGMA2))
        return -np.sum(np.log(Rbar + eps))

    bounds = [(0.0, P_HAP)] * M
    cons = [{"type": "eq", "fun": lambda p: P_HAP - np.sum(p)}]
    best = None
    for _ in range(n_start):
        p0 = rng.dirichlet(np.ones(M)) * P_HAP
        res = minimize(neg_pf, p0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 2000, "ftol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def weighted_max_min_power(Omega, d, P_HAP, iters=200):
    """Max-min power for weighted objective Rbar_r >= tau * d_r (d_r = weights)."""
    M = len(Omega)
    Omega = np.asarray(Omega, dtype=float)
    d = np.asarray(d, dtype=float)
    lo, hi = 0.0, 200.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        P = np.zeros(M)
        ok = True
        for r in range(M - 1, -1, -1):
            g = 2.0 ** (mid * d[r]) - 1.0
            P[r] = g * (np.sum(P[r + 1:M]) + SIGMA2 / Omega[r])
            if P[r] < 0:
                ok = False
        if ok and np.sum(P) <= P_HAP:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    P = np.zeros(M)
    for r in range(M - 1, -1, -1):
        g = 2.0 ** (mid * d[r]) - 1.0
        P[r] = g * (np.sum(P[r + 1:M]) + SIGMA2 / Omega[r])
    return P


def fairness_metrics(R):
    R = np.asarray(R, dtype=float)
    M = len(R)
    jain = np.sum(R) ** 2 / (M * np.sum(R ** 2))
    avg = np.mean(R)
    var = np.mean((R - avg) ** 2)
    return {"jain": float(jain), "avg_bps": float(avg), "var": float(var),
            "min_bps": float(np.min(R))}


def comment3(M, K_db, L, PHAP_db, S, seed):
    """Fairness comparison on the Jensen/ergodic upper-bound rates Rbar_i.

    Three fairness objectives are compared under a heterogeneous topology with
    one cell-edge user acting as the rate bottleneck:
      - max-min   : max tau s.t. Rbar_i >= tau for all i (proposed)
      - propfair  : max sum_i log(Rbar_i)
      - weighted  : max tau s.t. Rbar_i >= tau * d_i (edge user d=1.5, else 1)
    """
    W_u = FAIRNESS_POSITIONS[:M]
    cen = np.mean(W_u, axis=0)
    cen[2] = 0.5 * (Z_MIN + Z_MAX)
    Q = cen
    P_HAP = 10.0 ** (PHAP_db / 10.0)
    Omega, L_vec, s2_vec = channel_stats_for_users(Q, W_u, K_db, L)
    order = np.argsort(Omega)
    Om_s = Omega[order]
    d_si = np.array([np.linalg.norm(W_S - w) for w in W_u])

    def metrics_jensen(P):
        Rb = jensen_rates(Om_s, P)
        jain = np.sum(Rb) ** 2 / (M * np.sum(Rb ** 2))
        return {
            "min_bps": float(np.min(Rb)),
            "avg_bps": float(np.mean(Rb)),
            "sum_bps": float(np.sum(Rb)),
            "jain": float(jain),
            "rates_bps": Rb.tolist(),
        }

    results = {}

    tau_mm, P_mm = max_min_power(Om_s, P_HAP)
    results["maxmin"] = metrics_jensen(P_mm)
    results["maxmin"]["tau_bps"] = float(W_BAND * np.log2(1.0 + (2.0 ** tau_mm - 1.0)))

    P_pf = pf_power(Om_s, P_HAP, seed=seed)
    results["propfair"] = metrics_jensen(P_pf)

    d = np.ones(M)
    d[0] = 1.5  # cell-edge (weakest) user given higher priority weight
    P_wm = weighted_max_min_power(Om_s, d, P_HAP)
    results["weighted"] = metrics_jensen(P_wm)

    edge_user = int(np.argmax(d_si))
    edge_sorted_idx = int(np.where(order == edge_user)[0][0])
    results["edge_user_idx"] = edge_user
    results["edge_sorted_idx"] = edge_sorted_idx
    results["d_si"] = d_si.tolist()
    results["order"] = order.tolist()

    return results


# ======================================================================
# Comment 4 : small-scale grid-search reference and multi-init statistics
# ======================================================================
def comment4_grid(M, L, K_db, PHAP_db, z_grid, step_grid, seed):
    """Grid search over UAV (x,y) at fixed heights; enumerate orders; closed-form power.

    Returns the best Jensen min rate over the grid and the best (x,y,z,order).
    """
    P_HAP = 10.0 ** (PHAP_db / 10.0)
    W_u = USER_POSITIONS[:M]
    xs = np.arange(85.0, 115.0 + 1e-9, step_grid)
    ys = np.arange(-15.0, 15.0 + 1e-9, step_grid)
    best = -1.0
    best_cfg = None
    for x in xs:
        for y in ys:
            for z in z_grid:
                Q = np.array([x, y, z])
                Omega, _, _ = channel_stats_for_users(Q, W_u, K_db, L)
                for pi in permutations(range(M)):
                    pi = list(pi)
                    Om_pi = Omega[pi]
                    tau, P = max_min_power(Om_pi, P_HAP)
                    Rbar = jensen_rates(Om_pi, P)
                    mn = float(np.min(Rbar))
                    if mn > best:
                        best = mn
                        best_cfg = (x, y, z, pi)
    return best, best_cfg


def comment4_multiinit(M, L, K_db, PHAP_db, S, n_init, seed):
    """Random UAV positions (30--50) -> distribution of min MC rate (SIC-feasible)."""
    rng = np.random.default_rng(seed)
    P_HAP = 10.0 ** (PHAP_db / 10.0)
    W_u = USER_POSITIONS[:M]
    vals = []
    for _ in range(n_init):
        Q = np.array([rng.uniform(85, 115), rng.uniform(-15, 15),
                      rng.uniform(Z_MIN, Z_MAX)])
        Omega, L_vec, s2_vec = channel_stats_for_users(Q, W_u, K_db, L)
        order = np.argsort(Omega)
        Om_s = Omega[order]
        L_s = L_vec[order]
        s2_s = s2_vec[order]
        tau, P = max_min_power(Om_s, P_HAP)
        # fresh gains per run (different positions -> different statistics)
        rng2 = np.random.default_rng(seed + len(vals) + 1)
        gains = sample_gains(L_s, s2_s, S, rng2)
        R = sic_rate(gains, P)
        vals.append(float(np.min(R)))
    vals = np.array(vals)
    return {
        "n_init": n_init,
        "mean_bps": float(np.mean(vals)),
        "std_bps": float(np.std(vals)),
        "best_bps": float(np.max(vals)),
        "worst_bps": float(np.min(vals)),
        "cv_pct": 100.0 * np.std(vals) / np.mean(vals),
        "vals_bps": vals.tolist(),
    }


# ======================================================================
# Run everything and save results + figures
# ======================================================================
if __name__ == "__main__":
    out_dir = "revised_results"
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    # ---- Comment 2 ----
    c2 = {}
    for M in [3, 4]:
        r = comment2(M, K_DEF_DB, L_DEF, PHAP_DEF_DB, S, SEED)
        c2[str(M)] = r
        print(f"[Comment2 M={M}] enum={r['enum_best_bps']/1e6:.3f} Mbps, "
              f"stat={r['statistical_bps']/1e6:.3f}, dist={r['distance_bps']/1e6:.3f}, "
              f"rand={r['random_bps']/1e6:.3f} | eta={r['eta_stat_vs_enum_pct']:.2f}%, "
              f"gap={r['gap_stat_vs_enum_pct']:.2f}%, gain_vs_dist={r['gain_stat_vs_fixed_dist_pct']:.2f}%")
    all_results["comment2"] = c2

    # ---- Comment 3 ----
    c3 = comment3(M_DEF, K_DEF_DB, L_DEF, PHAP_DEF_DB, S, SEED)
    print("\n[Comment3 fairness metrics (Jensen rates)]")
    for k in ["maxmin", "propfair", "weighted"]:
        v = c3[k]
        print(f"  {k:12s} min={v['min_bps']/1e6:.4f} avg={v['avg_bps']/1e6:.4f} "
              f"sum={v['sum_bps']/1e6:.4f} Mbps  Jain={v['jain']:.5f}")
    print(f"  edge_user_idx={c3['edge_user_idx']} (sorted idx {c3['edge_sorted_idx']}), "
          f"d_si={np.round(np.array(c3['d_si']),2).tolist()}")
    print(f"  edge-user rate: maxmin={c3['maxmin']['rates_bps'][c3['edge_sorted_idx']]/1e6:.4f} "
          f"propfair={c3['propfair']['rates_bps'][c3['edge_sorted_idx']]/1e6:.4f} "
          f"weighted={c3['weighted']['rates_bps'][c3['edge_sorted_idx']]/1e6:.4f} Mbps")
    all_results["comment3"] = c3

    # ---- Comment 4 ----
    # grid refinement (M=2, L=2)
    grid_refs = {}
    for step in [5.0, 2.5, 1.0]:
        ref, cfg = comment4_grid(2, 2, K_DEF_DB, PHAP_DEF_DB, [30.0, 45.0, 60.0], step, SEED)
        grid_refs[str(step)] = ref
        print(f"[Comment4 grid step={step}] ref={ref/1e6:.3f} Mbps  cfg={cfg}")
    all_results["comment4_grid"] = grid_refs

    # proposed (centroid + ascending order) Jensen min rate for comparison
    Qc = uav_position(2)
    Omc, _, _ = channel_stats_for_users(Qc, USER_POSITIONS[:2], K_DEF_DB, 2)
    afoa_val, _, _ = jensen_min_rate_for_order(Omc, 10.0 ** (PHAP_DEF_DB / 10.0))
    fine_ref = grid_refs["1.0"]
    all_results["comment4_afoa_bps"] = afoa_val
    all_results["comment4_eta_afoa_pct"] = 100.0 * afoa_val / fine_ref
    all_results["comment4_gap_afoa_pct"] = 100.0 * (fine_ref - afoa_val) / fine_ref
    print(f"\n[Comment4] AFOA(centroid)={afoa_val/1e6:.3f} Mbps, fine-grid ref={fine_ref/1e6:.3f} "
          f"-> eta={100*afoa_val/fine_ref:.2f}%, gap={100*(fine_ref-afoa_val)/fine_ref:.2f}%")

    # grid stability: % change fine vs medium
    med_ref = grid_refs["2.5"]
    grid_change = 100.0 * abs(fine_ref - med_ref) / med_ref
    all_results["comment4_grid_change_pct"] = grid_change
    print(f"[Comment4] grid ref change medium->fine = {grid_change:.3f}%")

    # multi-init (M=2, L=2, 30 runs)
    mi = comment4_multiinit(2, 2, K_DEF_DB, PHAP_DEF_DB, S, 30, SEED)
    all_results["comment4_multiinit"] = mi
    print(f"[Comment4 multi-init] mean={mi['mean_bps']/1e6:.3f} std={mi['std_bps']/1e6:.3f} "
          f"best={mi['best_bps']/1e6:.3f} worst={mi['worst_bps']/1e6:.3f} CV={mi['cv_pct']:.2f}%")

    with open(os.path.join(out_dir, "comments_234_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ---- Figures ----
    # Fig B: decoding-order strategy comparison (bar chart, two panels M=3, M=4)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    labels = ["Best\nenumerated", "Statistical\ngain", "Distance\nbased", "Random"]
    for ax, M in zip(axes, [3, 4]):
        r = c2[str(M)]
        vals = [r["enum_best_bps"], r["statistical_bps"], r["distance_bps"], r["random_bps"]]
        vals = [v / 1e6 for v in vals]
        bars = ax.bar(labels, vals, color=["#7f7f7f", "#2ca02c", "#1f77b4", "#ff7f0e"], width=0.6)
        ax.set_title(f"(a) M={M}" if M == 3 else f"(b) M={M}", fontsize=10)
        ax.set_ylabel("Minimum rate (Mbps)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_order.pdf", dpi=300, bbox_inches="tight")
    print("\nSaved fig_order.pdf")

    # Fig C: multi-init boxplot + grid reference line
    fig2, ax2 = plt.subplots(figsize=(5, 3.6))
    vals_mbps = np.array(mi["vals_bps"]) / 1e6
    ax2.boxplot(vals_mbps, vert=True, widths=0.4)
    ax2.axhline(fine_ref / 1e6, color="r", linestyle="--", label="Grid reference")
    ax2.axhline(afoa_val / 1e6, color="g", linestyle="-.", label="Proposed (centroid)")
    ax2.set_ylabel("Minimum rate (Mbps)")
    ax2.set_xticks([])
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    fig2.tight_layout()
    fig2.savefig("fig_init.pdf", dpi=300, bbox_inches="tight")
    print("Saved fig_init.pdf")

    # Fig B2: per-user rates under the three fairness objectives (Comment 3)
    M = M_DEF
    edge_s = c3["edge_sorted_idx"]
    users = [f"U{i+1}\n({'edge' if i == edge_s else 'near'})" for i in range(M)]
    schemes = ["maxmin", "propfair", "weighted"]
    scheme_labels = ["Max-min", "Prop. fair", "Weighted\nmax-min"]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    x = np.arange(M)
    width = 0.26
    fig3, ax3 = plt.subplots(figsize=(7.5, 3.6))
    for s, lab, col in zip(schemes, scheme_labels, colors):
        vals = np.array(c3[s]["rates_bps"]) / 1e6
        ax3.bar(x + (schemes.index(s) - 1) * width, vals, width,
                label=lab, color=col)
    ax3.set_xticks(x)
    ax3.set_xticklabels(users, fontsize=8)
    ax3.set_ylabel("Ergodic rate $\\bar{R}_i$ (Mbps)")
    ax3.legend(fontsize=8, ncol=3)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    fig3.tight_layout()
    fig3.savefig("fig_fairness.pdf", dpi=300, bbox_inches="tight")
    print("Saved fig_fairness.pdf")
    print("All done.")
