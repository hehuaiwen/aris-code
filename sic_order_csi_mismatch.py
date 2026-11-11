"""
Reviewer 4, Comment 4 follow-up: quantify the RATE IMPACT of a noisy
statistical-CSI SIC decoding order.

The existing Section 5.12 experiment reports only the permutation-mismatch
probability P_ord (up to 92.34% at NMSE = -5 dB) but does not answer whether a
wrong order actually reduces the SIC-feasible minimum rate. This script adds the
missing paired experiment.

Protocol (isolate the ordering effect only):
  - Fix the DESIGN (UAV position, IRS phase, and per-user power allocation);
    the scalar Rician model in scalar_rician_model assumes coherent phase
    alignment, so the IRS phase is already at its best-case value.
  - True statistical gains   Omega_i^true = channel_stats_for_users(...)
  - Noisy estimated gains    Omega_i^hat  = max{ Omega_i^true (1 + delta_i), 0 }
                             delta_i ~ N(0, nu),  nu = 10^{NMSE_dB / 10}
    (identical multiplicative model used in Section 5.12).
  - Two orders:
        oracle order   = argsort(Omega_i^true)   (reference only, not deployable)
        estimated order = argsort(Omega_i^hat)   (deployable, based on noisy CSI)
  - Power is computed once from the TRUE gains in the oracle order and is then
    FIXED (mapped to physical users). Only the decoding order changes.
  - The SIC-feasible rate (Eq. (12)) is evaluated with PAIRED channel
    realizations shared across the two orders.

Reported per NMSE level:
  - permutation-mismatch probability P_ord (across-topology mean),
  - SIC-feasible worst-user rate for the oracle and estimated orders,
  - relative rate loss (oracle - estimated) / oracle,
  all with across-topology Student-t 95% confidence intervals.

Outputs:
  revised_results/review4_comment4_sic_order.json
  revised_results/fig_sic_order_nmse.pdf
"""

import os
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scalar_rician_model as R

# ----------------------------------------------------------------------
# Configuration (matches Section 5.12: L = 8, M = 4, P_HAP = 30 dBm)
# ----------------------------------------------------------------------
M = 4
L = 8
K_DB = R.K_DEF_DB          # 10 dB (AIRS--GMD Rician factor; direct link blocked)
PHAP_DB = R.PHAP_DEF_DB    # 30 dBm
P_HAP = 10.0 ** (PHAP_DB / 10.0)
NMSE_DBS = [-30, -20, -15, -10, -5]

N_TOPOLOGIES = 30          # independent user topologies
S = 2000                   # paired channel realizations per topology
N_NOISE = 200              # noisy-gain draws per topology per NMSE (order distribution)

BASE_SEED = 2026

# user-region sampling (Section 5.12 / ci_sweep region ~100 m from the HAP)
X_RANGE = [92.0, 109.0]
Y_RANGE = [-2.0, 2.0]

OUT_DIR = "revised_results"


def mean_t_ci(x, conf=0.95):
    """Student-t mean and half-width CI across topologies."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    tcrit = float(stats.t.ppf((1.0 + conf) / 2.0, n - 1))
    return m, tcrit * s / np.sqrt(n)


def centroid_q(W_u):
    q = np.mean(W_u, axis=0)
    q[2] = 0.5 * (R.Z_MIN + R.Z_MAX)
    return q


def rate_for_order(gains, P_physical, order):
    """SIC-feasible per-user rate (bps) for decoding order `order` (physical-user
    indices in decode order) under the FIXED per-user power P_physical."""
    order = np.asarray(order)
    return R.sic_rate(gains[order], P_physical[order])


def run_topology(W_u, topo_seed):
    rng = np.random.default_rng(topo_seed)

    # --- fixed design ---
    Q = centroid_q(W_u)
    Omega_true, L_vec, s2_vec = R.channel_stats_for_users(Q, W_u, K_DB, L)
    oracle_order = np.argsort(Omega_true)            # ascending statistical gain
    Om_oracle = Omega_true[oracle_order]
    _, P_oracle = R.max_min_power(Om_oracle, P_HAP)  # power in oracle order
    P_physical = np.empty(M)
    P_physical[oracle_order] = P_oracle              # fixed per-physical-user power

    # paired channel realizations shared by both orders and all NMSE levels
    gains = R.sample_gains(L_vec, s2_vec, S, rng)

    R_oracle = float(np.min(rate_for_order(gains, P_physical, oracle_order)))

    # base standard-normal draws reused (scaled) across NMSE levels -> monotone P_ord
    base = rng.normal(0.0, 1.0, (N_NOISE, M))

    out = {"oracle_rate_bps": R_oracle, "nmse_db": [], "p_ord": [],
           "est_rate_bps": [], "rate_loss_pct": []}
    for nmse_db in NMSE_DBS:
        nu = 10.0 ** (nmse_db / 10.0)
        delta = np.sqrt(nu) * base                  # (N_NOISE, M)
        Omega_hat = np.maximum(Omega_true[None, :] * (1.0 + delta), 0.0)
        est_orders = np.argsort(Omega_hat, axis=1)  # (N_NOISE, M)

        mismatch = np.any(est_orders != oracle_order[None, :], axis=1)
        p_ord = float(np.mean(mismatch))

        # expected SIC-feasible min rate under the estimated order (Monte Carlo
        # over the noisy-gain draws, sharing the S channel realizations)
        rates = np.empty(N_NOISE)
        for k in range(N_NOISE):
            rates[k] = np.min(rate_for_order(gains, P_physical, est_orders[k]))
        R_est = float(np.mean(rates))
        loss_pct = 100.0 * (R_oracle - R_est) / R_oracle

        out["nmse_db"].append(nmse_db)
        out["p_ord"].append(p_ord)
        out["est_rate_bps"].append(R_est)
        out["rate_loss_pct"].append(loss_pct)

    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    topo_rng = np.random.default_rng(BASE_SEED)

    per_topo = []
    for t in range(N_TOPOLOGIES):
        # sample 4 users uniformly in the service region
        xs = topo_rng.uniform(*X_RANGE, M)
        ys = topo_rng.uniform(*Y_RANGE, M)
        W_u = np.column_stack([xs, ys, np.zeros(M)])
        res = run_topology(W_u, BASE_SEED + 1000 + t)
        per_topo.append(res)
        print(f"[topo {t+1:2d}/{N_TOPOLOGIES}] oracle={res['oracle_rate_bps']/1e6:.4f} Mbps "
              f"| P_ord(-5dB)={res['p_ord'][-1]*100:.1f}% "
              f"loss(-5dB)={res['rate_loss_pct'][-1]:+.2f}%")

    # aggregate across topologies
    summary = {"nmse_db": NMSE_DBS,
               "p_ord_mean": [], "p_ord_ci": [],
               "oracle_rate_mean": [], "oracle_rate_ci": [],
               "est_rate_mean": [], "est_rate_ci": [],
               "rate_loss_mean_pct": [], "rate_loss_ci_pct": []}
    for j in range(len(NMSE_DBS)):
        p_ord = [res["p_ord"][j] for res in per_topo]
        R_o = [res["oracle_rate_bps"] for res in per_topo]
        R_e = [res["est_rate_bps"][j] for res in per_topo]
        loss = [res["rate_loss_pct"][j] for res in per_topo]

        m_po, c_po = mean_t_ci(p_ord)
        m_o, c_o = mean_t_ci(R_o)
        m_e, c_e = mean_t_ci(R_e)
        m_l, c_l = mean_t_ci(loss)

        summary["p_ord_mean"].append(m_po); summary["p_ord_ci"].append(c_po)
        summary["oracle_rate_mean"].append(m_o); summary["oracle_rate_ci"].append(c_o)
        summary["est_rate_mean"].append(m_e); summary["est_rate_ci"].append(c_e)
        summary["rate_loss_mean_pct"].append(m_l); summary["rate_loss_ci_pct"].append(c_l)

    summary["n_topologies"] = N_TOPOLOGIES
    summary["S"] = S
    summary["N_noise"] = N_NOISE
    summary["M"] = M
    summary["L"] = L
    summary["per_topology"] = per_topo

    with open(os.path.join(OUT_DIR, "review4_comment4_sic_order.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- figure: (a) SIC-feasible worst-user rate (oracle vs estimated),
    #               (b) relative rate loss ----
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    ax = axes[0]
    ax.errorbar(NMSE_DBS, [v / 1e6 for v in summary["oracle_rate_mean"]],
                yerr=[v / 1e6 for v in summary["oracle_rate_ci"]],
                fmt="o-", color="#2ca02c", capsize=4, label="True-gain (oracle) order")
    ax.errorbar(NMSE_DBS, [v / 1e6 for v in summary["est_rate_mean"]],
                yerr=[v / 1e6 for v in summary["est_rate_ci"]],
                fmt="s--", color="#d62728", capsize=4, label="Noisy-gain (estimated) order")
    ax.set_xlabel("CSI NMSE (dB)")
    ax.set_ylabel("Worst-user SIC-feasible rate (Mbps)")
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_title("(a)", fontsize=10)

    ax = axes[1]
    ax.errorbar(NMSE_DBS, summary["rate_loss_mean_pct"], yerr=summary["rate_loss_ci_pct"],
                fmt="s-", color="#d62728", capsize=4)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("CSI NMSE (dB)")
    ax.set_ylabel("Worst-user rate loss vs. oracle (%)")
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_title("(b)", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_sic_order_nmse.pdf"), dpi=300, bbox_inches="tight")
    print("\nSaved", os.path.join(OUT_DIR, "fig_sic_order_nmse.pdf"))
    print(json.dumps({k: summary[k] for k in
                      ["nmse_db", "p_ord_mean", "p_ord_ci", "rate_loss_mean_pct", "rate_loss_ci_pct"]},
                     indent=2))


if __name__ == "__main__":
    main()
