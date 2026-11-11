"""
Reviewer 1 Comment 2 follow-up (review item 3.1): direct comparison between a
dynamically-updated statistical-gain decoding order and a fixed initial
statistical-gain order, together with the number of order switches N_switch.

The scalar Rician model (scalar_rician_model) is embedded in a lightweight
outer BCD loop over the UAV position Q: each outer cycle re-optimizes Q (local,
Nelder-Mead multi-start) with the current decoding order held fixed, then the
statistical channel gains are recomputed at the new Q. The DYNAMIC rule re-sorts
the users after every outer cycle; the FIXED rule freezes the order obtained at
the centroid. N_switch counts the outer cycles (after the first) in which the
dynamic order actually changed.

The five order strategies requested by the reviewer (best enumerated / dynamic
statistical-gain / fixed initial statistical-gain / distance / random) are then
evaluated at the SAME optimized position Q* with the SAME Monte Carlo channel
realizations, so the bars are directly comparable.

Outputs: revised_results/comment2_order.json and fig_order.pdf.
"""
import os
import json
import numpy as np
from itertools import permutations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import scalar_rician_model as R


# ----------------------------------------------------------------------
# Scalar Q optimization (maximize min Jensen-surrogate rate, order fixed)
# ----------------------------------------------------------------------
def optimize_Q_scalar(Q0, W_u, order, P_HAP, K_db, L, seed=0, n_restarts=4):
    rng = np.random.default_rng(seed)
    bounds = np.array([[85.0, 115.0], [-15.0, 15.0], [R.Z_MIN, R.Z_MAX]])
    order = np.asarray(order)

    def neg_min(q):
        Omega, _, _ = R.channel_stats_for_users(np.asarray(q, float), W_u, K_db, L)
        Om = Omega[order]
        tau, P = R.max_min_power(Om, P_HAP)
        return -float(np.min(R.jensen_rates(Om, P)))

    best = None
    for s in range(n_restarts):
        q0 = np.array(Q0, dtype=float).copy()
        if s > 0:
            q0 = np.array([rng.uniform(*b) for b in bounds])
        res = minimize(neg_min, q0, method="Nelder-Mead",
                       options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-9})
        q = np.clip(res.x, bounds[:, 0], bounds[:, 1])
        val = neg_min(q)
        if best is None or val < best[1]:
            best = (q, val)
    return best[0]


def run_bcd(M, K_db, L, PHAP_db, S, seed, n_outer=8, dynamic=True):
    """Scalar BCD over Q; returns (Q_final, order_final, n_switch, min_mc_rate)."""
    W_u = R.USER_POSITIONS[:M]
    P_HAP = 10.0 ** (PHAP_db / 10.0)
    Q = R.uav_position(M).copy()

    Omega, _, _ = R.channel_stats_for_users(Q, W_u, K_db, L)
    order = np.argsort(Omega)
    n_switch = 0

    for t in range(n_outer):
        Q = optimize_Q_scalar(Q, W_u, order, P_HAP, K_db, L, seed=seed + t)
        Omega, L_vec, s2_vec = R.channel_stats_for_users(Q, W_u, K_db, L)
        order_new = np.argsort(Omega)
        if dynamic and not np.array_equal(order_new, order):
            n_switch += 1
        if dynamic:
            order = order_new
        # else: keep the frozen centroid order

    rng = np.random.default_rng(seed + 1000)
    gains = R.sample_gains(L_vec, s2_vec, S, rng)
    min_mc = R.mc_min_rate_for_order(Omega, gains, P_HAP, order=order)[0]
    return Q, order, n_switch, min_mc


def comment2_dynamic_fixed(M, K_db, L, PHAP_db, S, seed, n_outer=8):
    W_u = R.USER_POSITIONS[:M]
    P_HAP = 10.0 ** (PHAP_db / 10.0)
    centroid = R.uav_position(M)

    # ---- dynamic BCD ----
    Q_dyn, order_dyn, n_switch, R_dyn = run_bcd(
        M, K_db, L, PHAP_db, S, seed, n_outer=n_outer, dynamic=True)

    # ---- fixed BCD (order frozen at centroid) ----
    Q_fix, order_fix, _, R_fix = run_bcd(
        M, K_db, L, PHAP_db, S, seed, n_outer=n_outer, dynamic=False)

    # ---- five strategies, all at the optimized position Q_dyn, same gains ----
    Omega, L_vec, s2_vec = R.channel_stats_for_users(Q_dyn, W_u, K_db, L)
    rng = np.random.default_rng(seed + 2000)
    gains = R.sample_gains(L_vec, s2_vec, S, rng)

    # 1. best enumerated
    best = -1.0
    for pi in permutations(range(M)):
        pi = list(pi)
        v = R.mc_min_rate_for_order(Omega, gains, P_HAP, order=pi)[0]
        if v > best:
            best = v

    # 2. dynamic (= current sort at Q_dyn)
    dyn_at_q = R.mc_min_rate_for_order(Omega, gains, P_HAP,
                                       order=np.argsort(Omega))[0]

    # 3. fixed initial (= centroid sort, evaluated at Q_dyn)
    Omega0, _, _ = R.channel_stats_for_users(centroid, W_u, K_db, L)
    fix_at_q = R.mc_min_rate_for_order(Omega, gains, P_HAP,
                                       order=np.argsort(Omega0))[0]

    # 4. distance-based
    d_si = np.array([np.linalg.norm(R.W_S - w) for w in W_u])
    dist_at_q = R.mc_min_rate_for_order(Omega, gains, P_HAP,
                                        order=np.argsort(d_si))[0]

    # 5. random fixed (average over permutations)
    rng2 = np.random.default_rng(seed + 3000)
    rand_vals = []
    for _ in range(20):
        pi = rng2.permutation(M)
        rand_vals.append(R.mc_min_rate_for_order(Omega, gains, P_HAP,
                                                 order=pi)[0])
    rand_at_q = float(np.mean(rand_vals))

    G_dyn = 100.0 * (R_dyn - R_fix) / R_fix

    return {
        "M": M,
        "n_outer": n_outer,
        "n_switch": int(n_switch),
        "order_dynamic_final": [int(x) for x in order_dyn],
        "order_fixed_final": [int(x) for x in order_fix],
        "R_dyn_bps": float(R_dyn),
        "R_fix_bps": float(R_fix),
        "G_dyn_pct": float(G_dyn),
        "strategies_bps": {
            "best_enumerated": float(best),
            "dynamic": float(dyn_at_q),
            "fixed_initial": float(fix_at_q),
            "distance": float(dist_at_q),
            "random": float(rand_at_q),
        },
    }


if __name__ == "__main__":
    out_dir = "revised_results"
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for M in [3, 4]:
        r = comment2_dynamic_fixed(M, R.K_DEF_DB, R.L_DEF, R.PHAP_DEF_DB,
                                   R.S, R.SEED)
        results[str(M)] = r
        s = r["strategies_bps"]
        print(f"[Comment2 dynamic/fixed M={M}] N_switch={r['n_switch']} "
              f"R_dyn={r['R_dyn_bps']/1e6:.4f} R_fix={r['R_fix_bps']/1e6:.4f} "
              f"G_dyn={r['G_dyn_pct']:+.2f}%")
        print(f"  best={s['best_enumerated']/1e6:.4f} "
              f"dyn={s['dynamic']/1e6:.4f} fix={s['fixed_initial']/1e6:.4f} "
              f"dist={s['distance']/1e6:.4f} rand={s['random']/1e6:.4f} Mbps")

    with open(os.path.join(out_dir, "comment2_order.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- regenerated fig_order.pdf with five strategies ----
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    labels = ["Best\nenumerated", "Dynamic\nstat-gain", "Fixed init\nstat-gain",
              "Distance\nbased", "Random"]
    colors = ["#7f7f7f", "#2ca02c", "#98df8a", "#1f77b4", "#ff7f0e"]
    for ax, M in zip(axes, [3, 4]):
        s = results[str(M)]["strategies_bps"]
        vals = [s["best_enumerated"], s["dynamic"], s["fixed_initial"],
                s["distance"], s["random"]]
        vals = [v / 1e6 for v in vals]
        bars = ax.bar(labels, vals, color=colors, width=0.62)
        ax.set_title(f"(a) M={M}" if M == 3 else f"(b) M={M}", fontsize=10)
        ax.set_ylabel("Minimum rate (Mbps)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", labelsize=7)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig("fig_order.pdf", dpi=300, bbox_inches="tight")
    print("\nSaved revised_results/comment2_order.json and fig_order.pdf")
