"""
Focused fixes for Reviewer 1 Comments 3 and 4 (scalar Rician model, fast).

  Comment 3: 5th-percentile pooled user rate across T random topologies
             (max-min / proportional-fairness / weighted max-min), so the
             "lower-percentile user rate" explicitly requested by the reviewer
             is reported rather than only min/avg/Jain/variance.
  Comment 4: evaluate BOTH the Jensen surrogate and the Monte Carlo SIC-feasible
             minimum rate at the grid-optimal configuration and at the centroid
             (AFOA) configuration, so the grid reference and the AFOA solution
             are compared on a single metric. Regenerates fig_init.pdf with
             consistent Monte Carlo reference lines.

Outputs: revised_results/comments_3_4_fix.json and fig_init.pdf.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scalar_rician_model as R


def mc_min_at(Q, order, M, L, K_db, PHAP_db, S, seed):
    """Monte Carlo SIC-feasible min rate at a fixed (Q, order) design."""
    P_HAP = 10.0 ** (PHAP_db / 10.0)
    W_u = R.USER_POSITIONS[:M]
    Omega, L_vec, s2_vec = R.channel_stats_for_users(Q, W_u, K_db, L)
    Om = Omega[list(order)]
    tau, P = R.max_min_power(Om, P_HAP)
    rng = np.random.default_rng(seed)
    gains = R.sample_gains(L_vec[list(order)], s2_vec[list(order)], S, rng)
    Rmc = R.sic_rate(gains, P)
    Rsur = R.jensen_rates(Om, P)
    return float(np.min(Rsur)), float(np.min(Rmc))


# ============================ Comment 3 ============================
def lower_percentile(T=50, M=5, K_db=10.0, L=16, PHAP_db=30.0, S=20000,
                     seed=2026):
    rng = np.random.default_rng(seed)
    P_HAP = 10.0 ** (PHAP_db / 10.0)
    pooled = {"maxmin": [], "propfair": [], "weighted": []}
    for t in range(T):
        # 4 near users around (100, 0), one edge user at 150--170 m
        near = np.array([[100 + rng.uniform(-6, 6), rng.uniform(-6, 6), 0.0]
                         for _ in range(M - 1)])
        edge = np.array([rng.uniform(150, 170), rng.uniform(-8, 8), 0.0])
        W_u = np.vstack([near, edge])
        cen = np.mean(W_u, axis=0)
        cen[2] = 0.5 * (R.Z_MIN + R.Z_MAX)
        Q = cen
        Omega, L_vec, s2_vec = R.channel_stats_for_users(Q, W_u, K_db, L)
        order = np.argsort(Omega)
        Om_s = Omega[order]

        tau_mm, P_mm = R.max_min_power(Om_s, P_HAP)
        pooled["maxmin"].extend(R.jensen_rates(Om_s, P_mm).tolist())

        P_pf = R.pf_power(Om_s, P_HAP, seed=seed + t)
        pooled["propfair"].extend(R.jensen_rates(Om_s, P_pf).tolist())

        d = np.ones(M)
        d[np.argmin(Om_s)] = 1.5  # weakest (edge) user weighted higher
        P_wm = R.weighted_max_min_power(Om_s, d, P_HAP)
        pooled["weighted"].extend(R.jensen_rates(Om_s, P_wm).tolist())

    out = {}
    for k, vals in pooled.items():
        v = np.array(vals) / 1e6  # Mbps
        out[k] = {
            "n_topologies": T,
            "n_samples": len(v),
            "p5_mbps": float(np.percentile(v, 5)),
            "p10_mbps": float(np.percentile(v, 10)),
            "min_mbps": float(v.min()),
            "mean_mbps": float(v.mean()),
        }
    return out


# ============================ Comment 4 ============================
def comment4_reconcile(M=2, L=2, K_db=10.0, PHAP_db=30.0, S=20000, seed=2026,
                       n_init=30):
    # grid reference (step 1.0 m)
    best, cfg = R.comment4_grid(M, L, K_db, PHAP_db, [30.0, 45.0, 60.0], 1.0, seed)
    x, y, z, pi = cfg
    Q_grid = np.array([float(x), float(y), float(z)])
    sur_grid, mc_grid = mc_min_at(Q_grid, pi, M, L, K_db, PHAP_db, S, seed + 1)

    # AFOA (centroid + ascending order)
    Qc = R.uav_position(M)
    W_u = R.USER_POSITIONS[:M]
    Omc, _, _ = R.channel_stats_for_users(Qc, W_u, K_db, L)
    order_c = list(np.argsort(Omc))
    sur_afoa, mc_afoa = mc_min_at(Qc, order_c, M, L, K_db, PHAP_db, S, seed + 2)

    # multi-init (MC)
    mi = R.comment4_multiinit(M, L, K_db, PHAP_db, S, n_init, seed)

    # regenerate fig_init.pdf with consistent MC reference lines
    fig, ax = plt.subplots(figsize=(5, 3.6))
    vals_mbps = np.array(mi["vals_bps"]) / 1e6
    ax.boxplot(vals_mbps, vert=True, widths=0.4)
    ax.axhline(mc_grid / 1e6, color="r", linestyle="--",
               label="Grid reference (MC)")
    ax.axhline(mc_afoa / 1e6, color="g", linestyle="-.",
               label="Proposed centroid (MC)")
    ax.set_ylabel("Min. Monte Carlo rate (Mbps)")
    ax.set_xticks([])
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("fig_init.pdf", dpi=300, bbox_inches="tight")

    return {
        "grid_cfg": cfg,
        "sur_grid_bps": sur_grid,
        "mc_grid_bps": mc_grid,
        "sur_afoa_bps": sur_afoa,
        "mc_afoa_bps": mc_afoa,
        "sur_afoa_over_grid_pct": 100.0 * sur_afoa / sur_grid,
        "mc_afoa_over_grid_pct": 100.0 * mc_afoa / mc_grid,
        "jensen_gap_grid_pct": 100.0 * (sur_grid - mc_grid) / mc_grid,
        "jensen_gap_afoa_pct": 100.0 * (sur_afoa - mc_afoa) / mc_afoa,
        "multiinit": {k: v for k, v in mi.items() if k != "vals_bps"},
    }


if __name__ == "__main__":
    out_dir = "revised_results"
    os.makedirs(out_dir, exist_ok=True)

    pct = lower_percentile()
    print("=== Comment 3 lower-percentile (Jensen rates, pooled over 50 topologies) ===")
    for k, v in pct.items():
        print(f"  {k:10s} p5={v['p5_mbps']:.4f} p10={v['p10_mbps']:.4f} "
              f"min={v['min_mbps']:.4f} mean={v['mean_mbps']:.4f} Mbps "
              f"(n={v['n_samples']})")

    rec = comment4_reconcile()
    print("\n=== Comment 4 reconciliation ===")
    print(f"  grid cfg={rec['grid_cfg']}")
    print(f"  surrogate: grid={rec['sur_grid_bps']/1e6:.4f} afoa={rec['sur_afoa_bps']/1e6:.4f} "
          f"-> {rec['sur_afoa_over_grid_pct']:.2f}%")
    print(f"  MC:        grid={rec['mc_grid_bps']/1e6:.4f} afoa={rec['mc_afoa_bps']/1e6:.4f} "
          f"-> {rec['mc_afoa_over_grid_pct']:.2f}%")
    print(f"  Jensen gap: grid={rec['jensen_gap_grid_pct']:.2f}% afoa={rec['jensen_gap_afoa_pct']:.2f}%")

    with open(os.path.join(out_dir, "comments_3_4_fix.json"), "w") as f:
        json.dump({"comment3_percentile": pct, "comment4_reconcile": rec}, f, indent=2)
    print("\nSaved revised_results/comments_3_4_fix.json and fig_init.pdf")
