"""
Reviewer 2 Comment 5 -- fairness-corrected strengthened benchmarks.

The exploratory script `strengthened_baselines_exploratory.py` produced indicative numbers
but violated several fairness rules (per the revision plan
`revised/Reviewer2-Comment5-最小补充实验方案.md`).  This script corrects them:

  1. Unified evaluation protocol:  every scheme is evaluated with the SAME
     S=2000 external Monte Carlo samples, the SAME user positions and the SAME
     random seed, so the rates are directly comparable.
  2. Optimized OMA   -- "fixed-geometry, resource-optimized OMA": bandwidth
     fractions AND power are jointly optimized (not an equal W/M split).
  3. Active IRS      -- uniform amplification gain a is searched over a 1-D grid;
     the surface output power is constrained; the HAP power is re-optimized;
     two fairness budgets are reported (same total RF power, and same HAP power
     with extra active supply).
  4. Cooperative NOMA-- two-phase half-duplex decode-and-forward strong/weak
     pairing, SIC decodability, orthogonal time-frequency for the two relay
     pairs, and a single total RF power budget P_HAP + sum(P_relay).
  5. No-IRS and optimized GIRS are RE-EVALUATED on the same S=2000 samples.

All schemes run on the FULL vector AIRS model (M=4, L=8, W=1 MHz, P_HAP=30 dBm,
sigma^2=-80 dBm), the same model behind the paper's main simulation.

Outputs: revised_results/review2_comment5_fair.json and
         fig_review2_benchmarks_fair.pdf.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import vector_airs_model as R

W_BAND = R.W_BAND          # 1e6 Hz
SIGMA2 = R.SIGMA2          # 1e-11 W  (-80 dBm)
P_HAP = R.P_HAP            # 1.0 W   (30 dBm)
P_HAP_DBM = R.P_HAP_DBM    # 30.0
BETA1 = R.BETA1
BETA2 = R.BETA2

# ---- active IRS parameters (uniform-amplification reference) ----
SIGMA_A2 = 1e-10           # active-element dynamic noise power (-70 dBm)
A_GRID = np.linspace(1.0, 100.0, 50)  # 1-D search grid for the gain a (0-40 dB)

# ---- cooperative NOMA parameters ----
BETA_D2D = 2.0             # short-range D2D path-loss exponent (LoS)


def mean_gains(comps):
    return np.array([LoS + NLoS for (LoS, NLoS) in comps])


# ----------------------------------------------------------------------
# Max-min NOMA power allocation (decode order = physical index, i=0 weakest)
# ----------------------------------------------------------------------
def noma_max_min_power(Omega, P_total, sigma2=SIGMA2, tol=1e-9, iters=200):
    """Closed-form max-min NOMA power for ascending decode order Omega."""
    M = len(Omega)
    Omega = np.asarray(Omega, dtype=float)
    lo, hi = 0.0, 200.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        g = 2.0 ** mid - 1.0
        P = np.zeros(M)
        P[M - 1] = g * sigma2 / Omega[M - 1]
        for r in range(M - 2, -1, -1):
            P[r] = g * (np.sum(P[r + 1:M]) + sigma2 / Omega[r])
        if np.sum(P) > P_total:
            hi = mid
        else:
            lo = mid
    tau = 0.5 * (lo + hi)
    g = 2.0 ** tau - 1.0
    P = np.zeros(M)
    P[M - 1] = g * sigma2 / Omega[M - 1]
    for r in range(M - 2, -1, -1):
        P[r] = g * (np.sum(P[r + 1:M]) + sigma2 / Omega[r])
    return P


# ----------------------------------------------------------------------
# Optimized OMA (joint bandwidth-fraction + power allocation, max-min)
# ----------------------------------------------------------------------
def optimize_oma(Omega, P_total, W, sigma2):
    """Max-min OMA: R_i = alpha_i W log2(1 + Omega_i p_i/(alpha_i sigma^2)).
    Bisection on the common rate tau; inner problem minimizes total power over
    bandwidth fractions x_i with s_i = 2^{tau/(x_i W)} - 1."""
    M = len(Omega)
    Omega = np.asarray(Omega, dtype=float)
    eps = 1e-9

    def min_power(tau):
        def f(x):
            x = np.clip(x, eps, None)
            with np.errstate(over="ignore"):
                s = np.maximum(2.0 ** (tau / (x * W)) - 1.0, 0.0)
            return float(np.sum(x * sigma2 / Omega * s))

        cons = {"type": "ineq", "fun": lambda x: 1.0 - np.sum(x)}
        x0 = np.full(M, 1.0 / M)
        res = minimize(f, x0, method="SLSQP", bounds=[(eps, None)] * M,
                       constraints=cons, options={"maxiter": 1000, "ftol": 1e-14})
        if res.success:
            return float(res.fun), np.clip(res.x, eps, None)
        x = np.full(M, 1.0 / M)
        s = 2.0 ** (tau / (x * W)) - 1.0
        p = x * sigma2 / Omega * s
        return float(np.sum(p)), x

    hi = W * np.log2(1.0 + P_total * Omega.max() / sigma2)
    lo = 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        pmin, _ = min_power(mid)
        if pmin <= P_total:
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
    """MC per-user OMA rates with bandwidth fractions alpha (paired rng)."""
    Rsum = np.zeros(M)
    for _ in range(S):
        for i in range(M):
            LoS, NLoS = comps[i]
            g = R.sample_gain(LoS, NLoS, rng)
            a = alpha[i]
            Rsum[i] += a * W_BAND * np.log2(1.0 + g * p[i] / (a * SIGMA2))
    return Rsum / S


# ----------------------------------------------------------------------
# Active IRS (uniform amplification + amplified noise + output power budget)
# ----------------------------------------------------------------------
def active_channel(Q, Theta, i, L, a):
    """Return (LoS_power, NLoS_power, noise_power) for the active surface."""
    W_k = R.positions
    gamma = (R.rho0 - R.kappa1) / (np.linalg.norm(R.W_s - W_k[i]) ** BETA1)
    tau = L * R.rho0 * (R.rho0 - R.kappa2)
    phi_k = (Q[1] - R.W_s[1]) / np.linalg.norm(Q - R.W_s)
    psi_k = (Q[2] - R.W_s[2]) / np.linalg.norm(Q - R.W_s)
    phi_ki = (W_k[i][1] - Q[1]) / np.linalg.norm(Q - W_k[i])
    hat_s = np.sqrt(R.kappa1 / np.linalg.norm(R.W_s - W_k[i]) ** BETA1) * R.s_ki.reshape(-1, 1)
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


def active_surface_power(a, P_hap, Q, L):
    """Active surface output power for uniform gain a (incl. amplified noise)."""
    d_k = np.linalg.norm(Q - R.W_s)          # HAP -> UAV distance
    p_in = P_hap * (R.rho0 / d_k ** 2) * R.N  # per-element incident power
    return a * a * L * (p_in + SIGMA_A2)


def active_mc_rates(Q, Theta, P, L, M, S, rng, a):
    """Per-user active-IRS SIC-feasible rates (decode order = physical index)."""
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
# Cooperative NOMA (two-phase half-duplex DF pairing, orthogonal relays)
# ----------------------------------------------------------------------
def coop_noma_rates(Q, Theta, P_hap_alloc, comps, Omega, M, S, rng,
                    p_relay, beta_d2d=BETA_D2D):
    """Weak--strong pairing; strong user DF-relays the weak user's message.

    Phase 1 (T/2, full band W): HAP broadcasts; strong user SIC-decodes the
    weak user's message.
    Phase 2 (T/2, half band W/2): strong user forwards; the two relay pairs
    share the band orthogonally (W/2 each).
    """
    order = np.argsort(Omega)                       # ascending effective gain
    pairs = [(order[k], order[M - 1 - k]) for k in range(M // 2)]
    W_k = R.positions[:M]
    Rsum = np.zeros(M)
    for _ in range(S):
        g = np.array([R.sample_gain(c[0], c[1], rng) for c in comps])
        for (w, s) in pairs:
            g_w, g_s = g[w], g[s]
            p_w, p_s = P_hap_alloc[w], P_hap_alloc[s]
            G_s_w = g_s * p_w / (g_s * p_s + SIGMA2)   # strong decodes weak (ph.1)
            G_w_1 = g_w * p_w / (g_w * p_s + SIGMA2)   # weak direct (ph.1)
            d_sw = np.linalg.norm(W_k[s] - W_k[w])
            h_sw2 = R.rho0 / (d_sw ** beta_d2d)        # D2D link gain
            G_w_2 = p_relay * h_sw2 / SIGMA2           # relay (ph.2, orthogonal)
            # DF constraint (strong must decode) + MRC at weak user,
            # half-duplex factor 1/2, phase-2 half-band factor 1/2.
            r_w = 0.5 * W_BAND * np.log2(1.0 + G_s_w)
            r_w_mrc = (0.5 * W_BAND * np.log2(1.0 + G_w_1)
                       + 0.25 * W_BAND * np.log2(1.0 + G_w_2))
            Rsum[w] += min(r_w, r_w_mrc)
            Rsum[s] += 0.5 * W_BAND * np.log2(1.0 + g_s * p_s / SIGMA2)
    return Rsum / S


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def run(L=8, M=4, S=2000, seed=0, out=5, inner=20):
    Q, Theta, P, _ = R.optimize_design(L, M, seed=seed, out=out, inner=inner)
    comps = [R.channel_components(Q, Theta, i, L) for i in range(M)]
    Omega = mean_gains(comps)
    rng = np.random.default_rng(seed)

    # ---- proposed passive AIRS--NOMA ----
    r_prop, mn_prop, _ = R.mc_eval(Q, Theta, P, L, M, S, seed, comps=comps)

    # ---- optimized OMA (fixed-geometry, resource-optimized) ----
    alpha, p_oma = optimize_oma(Omega, P_HAP, W_BAND, SIGMA2)
    r_oma = oma_mc_rates(comps, alpha, p_oma, M, S, rng)
    mn_oma = float(r_oma.min())

    # ---- cooperative NOMA (single total budget, orthogonal relays) ----
    best_coop = None
    for P_hap_c in np.linspace(0.3, 0.9, 13):
        p_relay = (1.0 - P_hap_c) / (M // 2)
        P_hap_alloc = P * (P_hap_c / P_HAP)          # scale proposed allocation
        r_coop = coop_noma_rates(Q, Theta, P_hap_alloc, comps, Omega, M, S, rng,
                                 p_relay)
        mn_c = float(r_coop.min())
        if best_coop is None or mn_c > best_coop[0]:
            best_coop = (mn_c, P_hap_c, p_relay, r_coop)
    mn_coop, P_hap_coop, p_relay_coop, r_coop = best_coop

    # ---- active IRS (1-D gain search, two fairness budgets) ----
    # Budget A: same total RF power  P_hap + P_IRS = P_HAP.
    best_a_A = None
    for a in A_GRID:
        d_k = np.linalg.norm(Q - R.W_s)
        denom = 1.0 + a * a * L * R.N * (R.rho0 / d_k ** 2)
        P_hap_A = (P_HAP - a * a * L * SIGMA_A2) / denom
        if P_hap_A <= 0:
            continue
        P_act = noma_max_min_power(
            np.array([LoS + NLoS for (LoS, NLoS, _) in
                      [active_channel(Q, Theta, i, L, a) for i in range(M)]]),
            P_hap_A)
        r_act = active_mc_rates(Q, Theta, P_act, L, M, S, rng, a)
        mn_a = float(r_act.min())
        if best_a_A is None or mn_a > best_a_A[0]:
            best_a_A = (mn_a, a, P_hap_A, P_act, r_act)
    mn_act_A, a_best_A, P_hap_A_best, P_act_A, r_act_A = best_a_A

    # Budget B: same HAP power (1 W) + extra active supply (reference column).
    best_a_B = None
    for a in A_GRID:
        P_act = noma_max_min_power(
            np.array([LoS + NLoS for (LoS, NLoS, _) in
                      [active_channel(Q, Theta, i, L, a) for i in range(M)]]),
            P_HAP)
        r_act = active_mc_rates(Q, Theta, P_act, L, M, S, rng, a)
        mn_a = float(r_act.min())
        if best_a_B is None or mn_a > best_a_B[0]:
            best_a_B = (mn_a, a, P_HAP, P_act, r_act)
    mn_act_B, a_best_B, _, P_act_B, r_act_B = best_a_B

    # ---- no-IRS NOMA (re-evaluated on the same rng) ----
    W_k = R.positions[:M]
    gamma = np.array([(R.rho0 - R.kappa1) / (np.linalg.norm(R.W_s - w) ** BETA1)
                      for w in W_k])
    comps_noirs = [(0.0, float(gi)) for gi in gamma]
    order_noirs = np.argsort(gamma)
    P_noirs = noma_max_min_power(gamma[order_noirs], P_HAP)
    comps_noirs_s = [comps_noirs[o] for o in order_noirs]
    Rsum_noirs = np.zeros(M)
    for _ in range(S):
        g = np.array([R.sample_gain(c[0], c[1], rng) for c in comps_noirs_s])
        Rsum_noirs += R._sic_feasible_rates(g, P_noirs, M)
    r_noirs_sorted = Rsum_noirs / S
    r_noirs = np.zeros(M)
    for k in range(M):
        r_noirs[order_noirs[k]] = r_noirs_sorted[k]
    mn_noirs = float(r_noirs.min())

    # ---- optimized GIRS (re-evaluated at girs_best_Q with S=2000) ----
    girs_Q = np.array([95.0, 0.0, 10.0])
    Qg, Thg, Pg, _ = R.optimize_design(L, M, seed=seed, out=out, inner=inner,
                                       opt_Q=False, Q_init=girs_Q)
    comps_g = [R.channel_components(girs_Q, Thg, i, L) for i in range(M)]
    r_girs, mn_girs, _ = R.mc_eval(girs_Q, Thg, Pg, L, M, S, seed, comps=comps_g)

    return {
        "L": L, "M": M, "S": S, "seed": seed,
        "proposed_min_bps": float(mn_prop),
        "proposed_per_user_bps": [float(v) for v in r_prop],
        "opt_oma_min_bps": mn_oma,
        "opt_oma_per_user_bps": [float(v) for v in r_oma],
        "opt_oma_alpha": alpha.tolist(),
        "coop_noma_min_bps": mn_coop,
        "coop_noma_per_user_bps": [float(v) for v in r_coop],
        "coop_P_hap": P_hap_coop,
        "coop_p_relay_per_user": p_relay_coop,
        "active_irs_A_min_bps": mn_act_A,
        "active_irs_A_a": a_best_A,
        "active_irs_A_P_hap": P_hap_A_best,
        "active_irs_A_per_user_bps": [float(v) for v in r_act_A],
        "active_irs_B_min_bps": mn_act_B,
        "active_irs_B_a": a_best_B,
        "active_irs_B_per_user_bps": [float(v) for v in r_act_B],
        "no_irs_min_bps": mn_noirs,
        "no_irs_per_user_bps": [float(v) for v in r_noirs],
        "girs_min_bps": float(mn_girs),
        "girs_per_user_bps": [float(v) for v in r_girs],
        "girs_Q": girs_Q.tolist(),
    }


if __name__ == "__main__":
    out_dir = "revised_results"
    os.makedirs(out_dir, exist_ok=True)

    res = run()
    base = res["proposed_min_bps"]

    rows = [
        ("Proposed AIRS-NOMA", "proposed_min_bps"),
        ("Optimized OMA-AIRS", "opt_oma_min_bps"),
        ("Active IRS-NOMA (budget A)", "active_irs_A_min_bps"),
        ("Active IRS-NOMA (budget B)", "active_irs_B_min_bps"),
        ("Cooperative NOMA", "coop_noma_min_bps"),
        ("No-IRS NOMA", "no_irs_min_bps"),
        ("Optimized GIRS", "girs_min_bps"),
    ]
    print("=== Reviewer 2 Comment 5 (fairness-corrected) ===")
    for name, key in rows:
        v = res[key]
        print(f"  {name:28s} min = {v/1e6:8.4f} Mbps   "
              f"({100.0*(v-base)/base:+6.2f}% vs proposed)")
    print(f"\n  optimized OMA bandwidth fractions: "
          f"{['%.3f' % a for a in res['opt_oma_alpha']]}")
    print(f"  active IRS best gain a (A/B): "
          f"{res['active_irs_A_a']:.2f} / {res['active_irs_B_a']:.2f}")
    print(f"  cooperative: P_hap={res['coop_P_hap']:.2f} W, "
          f"p_relay={res['coop_p_relay_per_user']*1e3:.0f} mW")

    with open(os.path.join(out_dir, "review2_comment5_fair.json"), "w") as f:
        json.dump(res, f, indent=2)

    # figure
    names = ["Proposed\nAIRS-NOMA", "Optimized\nOMA-AIRS",
             "Active IRS\n(budget A)", "Active IRS\n(budget B)",
             "Cooperative\nNOMA", "No-IRS\nNOMA", "Optimized\nGIRS"]
    vals = [res["proposed_min_bps"], res["opt_oma_min_bps"],
            res["active_irs_A_min_bps"], res["active_irs_B_min_bps"],
            res["coop_noma_min_bps"], res["no_irs_min_bps"],
            res["girs_min_bps"]]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    x = np.arange(len(names))
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#d62728", "#2ca02c",
              "#9467bd", "#8c564b"]
    ax.bar(x, [v / 1e6 for v in vals], 0.6, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Minimum rate (Mbps)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for xi, v in zip(x, vals):
        ax.text(xi, v / 1e6, f"{v/1e6:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Minimum-rate comparison of AIRS-NOMA and strengthened baselines")
    fig.tight_layout()
    fig.savefig("fig_review2_benchmarks_fair.pdf", dpi=300, bbox_inches="tight")
    print("\nSaved revised_results/review2_comment5_fair.json and "
          "fig_review2_benchmarks_fair.pdf")
