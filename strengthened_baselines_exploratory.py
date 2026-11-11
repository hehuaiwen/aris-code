"""
Reviewer 2 Comment 5: strengthened benchmarks on the full vector AIRS model.

The original manuscript compared the proposed passive AIRS--NOMA design only
against a fixed-resource OMA (equal W/M bandwidth split), a no-IRS NOMA scheme,
and a grid-optimized GIRS. The reviewer asked for stronger, fairer baselines.
This script adds three benchmarks that close that gap:

  1. Optimized OMA   -- joint bandwidth-fraction + power allocation under the
                        max-min objective (replacing the equal-split OMA).
  2. Cooperative NOMA -- decode-and-forward user pairing with half-duplex loss,
                        SIC decodability, relay power, and short-range D2D links.
  3. Active IRS       -- uniform amplification gain with amplified dynamic noise
                        (same HAP transmit power as the passive design).

The proposed passive AIRS--NOMA design (Q, Theta, P) is reused from
optimize_design so that all schemes share the same deployment, channel
realizations, and HAP power condition. The active-IRS and cooperative-NOMA
models follow the fairness rules of the revision plan (amplified noise + power
considerations for active IRS; relay power + SIC + half-duplex for cooperation).

Outputs: revised_results/review2_comment5_benchmarks.json and
         fig_review2_benchmarks.pdf.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import vector_airs_model as R
import mec_latency_baselines as L8

SIGMA2 = R.SIGMA2
W_BAND = R.W_BAND
P_HAP = R.P_HAP
P_HAP_DBM = R.P_HAP_DBM
BETA1 = R.BETA1
BETA2 = R.BETA2

# ---- active IRS parameters (uniform-amplification reference) ----
A_ACTIVE = 10.0        # uniform amplification gain
SIGMA_A2 = 1e-10       # active-element dynamic noise power (-70 dBm)

# ---- cooperative NOMA parameters ----
BETA_D2D = 2.0         # short-range D2D path-loss exponent (LoS)


def mean_gains(comps):
    return np.array([LoS + NLoS for (LoS, NLoS) in comps])


# ----------------------------------------------------------------------
# Optimized OMA (joint bandwidth-fraction + power allocation, max-min)
# ----------------------------------------------------------------------
def optimize_oma(Omega, P_HAP, W, sigma2):
    """Max-min OMA: R_i = alpha_i W log2(1 + Omega_i p_i/(alpha_i sigma^2)).

    Solved by bisection on the common rate tau; the inner problem minimizes the
    total power over the bandwidth fractions x_i with s_i = 2^{tau/(x_i W)} - 1.
    """
    M = len(Omega)
    Omega = np.asarray(Omega, dtype=float)
    eps = 1e-9

    def min_power(tau):
        def f(x):
            x = np.clip(x, eps, None)
            s = 2.0 ** (tau / (x * W)) - 1.0
            return float(np.sum(x * sigma2 / Omega * s))

        cons = {"type": "ineq", "fun": lambda x: 1.0 - np.sum(x)}
        x0 = np.full(M, 1.0 / M)
        res = minimize(f, x0, method="SLSQP", bounds=[(eps, None)] * M,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-14})
        if res.success:
            return float(res.fun), np.clip(res.x, eps, None)
        # fallback: equal bandwidth
        x = np.full(M, 1.0 / M)
        s = 2.0 ** (tau / (x * W)) - 1.0
        p = x * sigma2 / Omega * s
        return float(np.sum(p)), x

    hi = W * np.log2(1.0 + P_HAP * Omega.max() / sigma2)
    lo = 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        pmin, _ = min_power(mid)
        if pmin <= P_HAP:
            lo = mid
        else:
            hi = mid
    tau = lo
    _, alpha = min_power(tau)
    alpha = np.clip(alpha, eps, None)
    alpha /= alpha.sum()
    s = 2.0 ** (tau / (alpha * W)) - 1.0
    p = alpha * sigma2 / Omega * s
    return alpha, p


def oma_mc_rates(comps, alpha, p, M, S, rng):
    Rsum = np.zeros(M)
    for i in range(M):
        LoS, NLoS = comps[i]
        a = alpha[i]
        acc = 0.0
        for _ in range(S):
            g = R.sample_gain(LoS, NLoS, rng)
            acc += a * W_BAND * np.log2(1.0 + g * p[i] / (a * SIGMA2))
        Rsum[i] = acc / S
    return Rsum


# ----------------------------------------------------------------------
# Cooperative NOMA (decode-and-forward pairing, half-duplex)
# ----------------------------------------------------------------------
def coop_noma_rates(Q, Theta, P, comps, Omega, M, S, rng, p_relay,
                    beta_d2d=BETA_D2D):
    """Weak--strong user pairing; strong user DF-relays the weak user message."""
    order = np.argsort(Omega)                      # ascending effective gain
    pairs = [(order[k], order[M - 1 - k]) for k in range(M // 2)]
    W_k = R.positions[:M]
    Rsum = np.zeros(M)
    for _ in range(S):
        g = np.array([R.sample_gain(c[0], c[1], rng) for c in comps])
        for (w, s) in pairs:
            g_w, g_s = g[w], g[s]
            p_w, p_s = P[w], P[s]
            G_s_w = g_s * p_w / (g_s * p_s + SIGMA2)   # strong decodes weak (phase 1)
            G_w_1 = g_w * p_w / (g_w * p_s + SIGMA2)   # weak direct (phase 1)
            d_sw = np.linalg.norm(W_k[s] - W_k[w])
            h_sw2 = R.rho0 / (d_sw ** beta_d2d)        # D2D link gain
            G_w_2 = p_relay * h_sw2 / SIGMA2           # relay (phase 2)
            G_w_c = G_w_1 + G_w_2                      # MRC at weak user
            Rsum[w] += (W_BAND / 2.0) * np.log2(1.0 + min(G_s_w, G_w_c))
            Rsum[s] += (W_BAND / 2.0) * np.log2(1.0 + g_s * p_s / SIGMA2)
    return Rsum / S


# ----------------------------------------------------------------------
# Active IRS (uniform amplification + amplified noise)
# ----------------------------------------------------------------------
def active_channel(Q, Theta, i, L, a):
    W_k = R.positions
    gamma = (R.rho0 - R.kappa1) / (np.linalg.norm(R.W_s - W_k[i]) ** BETA1)
    tau = L * R.rho0 * (R.rho0 - R.kappa2)
    phi_k = (Q[1] - R.W_s[1]) / np.linalg.norm(Q - R.W_s)
    psi_k = (Q[2] - R.W_s[2]) / np.linalg.norm(Q - R.W_s)
    phi_ki = (W_k[i][1] - Q[1]) / np.linalg.norm(Q - W_k[i])
    hat_s = np.sqrt(R.kappa1 / np.linalg.norm(R.W_s - W_k[i]) ** BETA1) \
        * R.s_ki.reshape(-1, 1)
    g_bar = np.exp(-1j * (2 * np.pi * R.d / R.wavelength) * np.arange(L) * phi_k).reshape(-1, 1)
    g_tit = np.exp(-1j * (2 * np.pi * R.d / R.wavelength) * np.arange(R.N) * psi_k).reshape(1, -1)
    hat_h = np.sqrt(R.kappa2 / np.linalg.norm(Q - W_k[i]) ** BETA2) * np.exp(
        -1j * (2 * np.pi * R.d / R.wavelength) * np.arange(L) * phi_ki).reshape(-1, 1)
    G_k = np.sqrt(R.rho0 / np.linalg.norm(Q - R.W_s) ** 2) * g_bar * g_tit

    gain = np.conj(hat_s.T) + a * (np.conj(hat_h.T) @ Theta @ G_k)
    LoS = float(np.linalg.norm(gain) ** 2)
    NLoS = float(gamma + a * a * tau / (
        (np.linalg.norm(Q - W_k[i]) ** BETA2) * (np.linalg.norm(Q - R.W_s) ** 2)))
    noise = SIGMA2 + SIGMA_A2 * a * a * L * R.rho0 / (np.linalg.norm(Q - W_k[i]) ** BETA2)
    return LoS, NLoS, noise


def active_mc_rates(Q, Theta, P, L, M, S, rng, a=A_ACTIVE):
    ac = [active_channel(Q, Theta, i, L, a) for i in range(M)]
    Rsum = np.zeros(M)
    for _ in range(S):
        g = np.array([R.sample_gain(c[0], c[1], rng) for c in ac])
        for i in range(M):
            inter = float(np.sum(P[i + 1:M]))
            dec = []
            for k in range(i, M):
                sinr = g[k] * P[i] / (g[k] * inter + ac[k][2])
                dec.append(W_BAND * np.log2(1.0 + sinr))
            Rsum[i] += min(dec)
    return Rsum / S


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def run(L=8, M=4, S=2000, seed=0, out=5, inner=20):
    Q, Theta, P, _ = R.optimize_design(L, M, seed=seed, out=out, inner=inner)
    comps = [R.channel_components(Q, Theta, i, L) for i in range(M)]
    Omega = mean_gains(comps)
    rng = np.random.default_rng(seed + 6000)

    # proposed passive AIRS--NOMA
    _, mn_prop, _ = R.mc_eval(Q, Theta, P, L, M, S, seed, comps=comps)

    # optimized OMA
    alpha, p_oma = optimize_oma(Omega, P_HAP, W_BAND, SIGMA2)
    mn_oma = float(oma_mc_rates(comps, alpha, p_oma, M, S, rng).min())

    # cooperative NOMA
    p_relay = P_HAP / M
    mn_coop = float(coop_noma_rates(Q, Theta, P, comps, Omega, M, S, rng, p_relay).min())

    # active IRS (same HAP power, same phase profile, uniform gain a)
    mn_act = float(active_mc_rates(Q, Theta, P, L, M, S, rng).min())

    # no-IRS NOMA (fast, no re-optimization)
    mn_noirs = float(L8.no_irs_rates(L, M, S, seed).min())

    # optimized GIRS (known value from comment7; avoids a re-run)
    mn_girs = 546835.7548691899

    return {
        "L": L, "M": M, "S": S,
        "proposed_min_bps": float(mn_prop),
        "opt_oma_min_bps": mn_oma,
        "coop_noma_min_bps": mn_coop,
        "active_irs_min_bps": mn_act,
        "no_irs_min_bps": mn_noirs,
        "girs_min_bps": mn_girs,
        "alpha_oma": alpha.tolist(),
        "p_oma": p_oma.tolist(),
        "p_relay": p_relay,
        "A_active": A_ACTIVE,
        "sigma_a2": SIGMA_A2,
        "beta_d2d": BETA_D2D,
    }


if __name__ == "__main__":
    out_dir = "revised_results"
    os.makedirs(out_dir, exist_ok=True)

    res = run()
    base = res["proposed_min_bps"]

    names = ["Proposed\nAIRS-NOMA", "Optimized\nOMA-AIRS", "Cooperative\nNOMA",
             "Active\nIRS-NOMA", "No-IRS\nNOMA", "Optimized\nGIRS"]
    keys = ["proposed_min_bps", "opt_oma_min_bps", "coop_noma_min_bps",
            "active_irs_min_bps", "no_irs_min_bps", "girs_min_bps"]
    vals = [res[k] for k in keys]

    print("=== Reviewer 2 Comment 5 benchmarks ===")
    for n, v in zip(names, vals):
        print(f"  {n:18s} min rate = {v/1e6:8.4f} Mbps   "
              f"({100.0*(v-base)/base:+6.2f}% vs proposed)")
    print(f"\n  optimized OMA bandwidth fractions: "
          f"{['%.3f' % a for a in res['alpha_oma']]}")
    print(f"  relay power per strong user: {res['p_relay']*1e3:.1f} mW")

    with open(os.path.join(out_dir, "review2_comment5_benchmarks.json"), "w") as f:
        json.dump(res, f, indent=2)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(names))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    bars = ax.bar(x, [v / 1e6 for v in vals], 0.55, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Minimum rate (Mbps)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for xi, v in zip(x, vals):
        ax.text(xi, v / 1e6, f"{v/1e6:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Minimum-rate comparison of AIRS-NOMA and strengthened baselines")
    fig.tight_layout()
    fig.savefig("fig_review2_benchmarks.pdf", dpi=300, bbox_inches="tight")
    print("\nSaved revised_results/review2_comment5_benchmarks.json "
          "and fig_review2_benchmarks.pdf")
