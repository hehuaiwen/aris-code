"""
Reviewer 4, Comment 2: residual SIC and SIC-ordering-error experiments.

Two lightweight, self-contained additions (reusing the proposed design, no
re-optimization) that quantify the two effects the reviewer raised:

  1. Residual (imperfect) SIC. In ideal NOMA the interference from already
     decoded weaker users is fully cancelled. With imperfect cancellation a
     fraction xi of that power remains. We sweep xi in {0, 0.01, 0.05, 0.1}
     and report the worst-user empirical ergodic rate, together with a
     fixed-resource OMA-AIRS reference (which has no SIC and is therefore
     xi-independent), to show how much the NOMA--OMA gap shrinks.

  2. SIC-ordering error under CSI mismatch. The statistical channel gains
     Omega_i = LoS_i + NLoS_i are perturbed by independent multiplicative
     Gaussian errors (the same model used in the robustness section):
       Omega_hat_i = LoS_i (1 + delta_L) + NLoS_i (1 + delta_N),
       delta ~ N(0, NMSE).
     We sweep NMSE in {-30,...,-5} dB and report the probability that sorting
     the noisy gains produces a permutation different from the true one.

The decoding-order convention (physical index = decode order) matches the
existing framework, and no existing result is changed.

Outputs: revised_results/review4_comment2.json and
         fig_review4_comment2.pdf.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vector_airs_model as R

SIGMA2 = R.SIGMA2
W_BAND = R.W_BAND
P_HAP = R.P_HAP

L = 8
M = 4
S_MC = 2000
OUT = 5
INNER = 20


# ----------------------------------------------------------------------
# Residual (imperfect) SIC
# ----------------------------------------------------------------------
def sic_rates_residual(g, P, xi):
    """SIC-feasible rates with residual interference xi on cancelled users."""
    rates = np.zeros(M)
    for i in range(M):
        inter_unc = float(np.sum(P[i + 1:M]))   # stronger users, not cancelled
        inter_res = float(np.sum(P[0:i]))       # weaker users, cancelled with residual xi
        dec = []
        for k in range(i, M):
            sinr = g[k] * P[i] / (g[k] * (inter_unc + xi * inter_res) + SIGMA2)
            dec.append(W_BAND * np.log2(1.0 + sinr))
        rates[i] = min(dec)
    return rates


def mc_eval_residual(Q, Theta, P, xi, comps, seed):
    rng = np.random.default_rng(seed)
    rate_sums = np.zeros(M)
    for _ in range(S_MC):
        g = np.array([R.sample_gain(c[0], c[1], rng) for c in comps])
        rate_sums += sic_rates_residual(g, P, xi)
    return float(rate_sums.min() / S_MC)


# ----------------------------------------------------------------------
# SIC-ordering error probability under CSI mismatch
# ----------------------------------------------------------------------
def ordering_error_prob(comps, nmse_db, n_samples, seed):
    nmse = 10.0 ** (nmse_db / 10.0)
    Omega = np.array([l + n for (l, n) in comps])
    true_order = np.argsort(Omega)
    rng = np.random.default_rng(seed)
    err = 0
    for _ in range(n_samples):
        Oh = np.maximum(Omega * (1.0 + rng.normal(0.0, np.sqrt(nmse), size=Omega.shape)), 0.0)
        if not np.array_equal(np.argsort(Oh), true_order):
            err += 1
    return err / n_samples


def mc_eval_oma(comps, p_oma, seed):
    """Monte Carlo worst-user ergodic rate for fixed-resource OMA-AIRS."""
    rng = np.random.default_rng(seed)
    rate_sums = np.zeros(M)
    for _ in range(S_MC):
        g = np.array([R.sample_gain(c[0], c[1], rng) for c in comps])
        for i in range(M):
            rate_sums[i] += W_BAND / M * np.log2(1.0 + g[i] * p_oma / SIGMA2)
    return float(rate_sums.min() / S_MC)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def run(seed=0, n_ord=20000):
    Q, Theta, P, _ = R.optimize_design(L, M, seed=seed, out=OUT, inner=INNER)
    comps = [R.channel_components(Q, Theta, i, L) for i in range(M)]
    Omega = np.array([l + n for (l, n) in comps])

    # --- residual SIC sweep ---
    xi_list = [0.0, 0.01, 0.05, 0.1]
    rates_xi = [mc_eval_residual(Q, Theta, P, xi, comps, seed + 1) for xi in xi_list]

    # --- fixed-resource OMA-AIRS reference (no SIC, xi-independent) ---
    p_oma = P_HAP / M
    oma_ref = mc_eval_oma(comps, p_oma, seed + 1)

    # --- SIC-ordering error vs NMSE ---
    nmse_list = [-30, -20, -15, -10, -5]
    p_ord = [ordering_error_prob(comps, nmse, n_ord, seed + 100) for nmse in nmse_list]

    res = {
        "L": L, "M": M, "S_MC": S_MC, "n_ord": n_ord,
        "Omega": Omega.tolist(),
        "true_order": np.argsort(Omega).tolist(),
        "xi_list": xi_list,
        "worst_user_rate_vs_xi_bps": rates_xi,
        "oma_ref_bps": oma_ref,
        "nmse_db_list": nmse_list,
        "ordering_error_prob": p_ord,
    }

    print("=== Reviewer 4 Comment 2 ===")
    print(f"Omega = {np.array2string(Omega, precision=3, floatmode='maxprec_equal')}  "
          f"true order = {np.argsort(Omega)}")
    print("\nResidual SIC sweep (worst-user ergodic rate):")
    for xi, r in zip(xi_list, rates_xi):
        print(f"  xi = {xi:4.2f}  ->  {r/1e6:.4f} Mbps  "
              f"({100.0*(r-rates_xi[0])/rates_xi[0]:+6.2f}% vs ideal)")
    print(f"  OMA-AIRS reference = {oma_ref/1e6:.4f} Mbps")
    print("\nSIC-ordering error probability vs NMSE:")
    for nmse, p in zip(nmse_list, p_ord):
        print(f"  NMSE = {nmse:4d} dB  ->  P_ord = {100.0*p:.3f}%")

    with open(os.path.join("revised_results", "review4_comment2.json"), "w") as f:
        json.dump(res, f, indent=2)

    # --- figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))

    # (a) residual SIC
    ax1.plot(xi_list, [r / 1e6 for r in rates_xi], marker="o", color="#1f77b4",
             label="Proposed (AFOA)")
    ax1.axhline(oma_ref / 1e6, color="#d62728", linestyle="--",
                label="Fixed-resource OMA-AIRS")
    ax1.set_xlabel("Residual SIC coefficient $\\xi$")
    ax1.set_ylabel("Worst-user ergodic rate (Mbps)")
    ax1.set_title("(a) Imperfect SIC")
    ax1.grid(linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8)

    # (b) ordering error
    ax2.plot(nmse_list, [100.0 * p for p in p_ord], marker="o", color="#2ca02c")
    ax2.set_xlabel("Normalized MSE (NMSE) of CSI estimate (dB)")
    ax2.set_ylabel("Ordering-error probability $P_{\\rm ord}$ (%)")
    ax2.set_title("(b) SIC ordering under CSI error")
    ax2.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig("fig_review4_comment2.pdf", dpi=300, bbox_inches="tight")
    print("\nSaved revised_results/review4_comment2.json and fig_review4_comment2.pdf")
    return res


if __name__ == "__main__":
    run()
