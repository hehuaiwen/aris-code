"""
Reviewer 1 Comment 8 follow-up (review item 5): strengthened latency baselines.

The earlier latency experiment compared the proposed design only against a
fixed-resource baseline (equal power + fixed UAV + random phase), which shuts
off all three communication blocks at once and makes the 33.9% improvement look
larger than its decomposition warrants. This script adds the three missing
communication baselines so that the NOMA gain, the AIRS gain, and the
communication-optimization gain can be separated:

    proposed NOMA-AIRS + equal CPU      (full communication optimization)
    OMA-AIRS          + equal CPU       (isolates the NOMA gain)
    no-IRS PD-NOMA    + equal CPU       (isolates the AIRS gain)
    optimized GIRS    + equal CPU       (airborne-vs-ground deployment)
    fixed-resource    + equal CPU       (all communication blocks off)

All downlink rates are evaluated on the FULL vector AIRS model with the same
Monte Carlo realizations and the same task/CPU profile, so the latencies are
directly comparable. The proposed design also keeps the delay-aware CPU row.

Outputs: revised_results/comment8_latency.json and fig_latency_baselines.pdf.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vector_airs_model as R

W_BAND = R.W_BAND
SIGMA2 = R.SIGMA2
P_HAP = R.P_HAP
P_HAP_DBM = R.P_HAP_DBM
BETA1 = R.BETA1
BETA2 = R.BETA2


# ----------------------------------------------------------------------
# Power allocations (full-model noise SIGMA2)
# ----------------------------------------------------------------------
def noma_max_min_power(Omega, P_HAP, tol=1e-9, iters=200):
    """Max-min NOMA power for a decoding order Omega (ascending gains)."""
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
    return P


def oma_max_min_power(Omega, P_HAP):
    """OMA max-min power: equalize Omega_i p_i across users."""
    Omega = np.asarray(Omega, dtype=float)
    c = P_HAP / np.sum(1.0 / Omega)
    return c / Omega


# ----------------------------------------------------------------------
# Monte Carlo downlink rate evaluations
# ----------------------------------------------------------------------
def sic_mc_rates(comps, P, M, S, rng):
    """Per-user SIC-feasible NOMA rates (comps already in decode order)."""
    Rsum = np.zeros(M)
    for _ in range(S):
        g = np.array([R.sample_gain(c[0], c[1], rng) for c in comps])
        Rsum += R._sic_feasible_rates(g, P, M)
    return Rsum / S


def oma_mc_rates(comps, P, M, S, rng):
    """Per-user OMA rates: (W/M) log2(1 + g_i p_i / sigma^2)."""
    Rsum = np.zeros(M)
    for i in range(M):
        LoS, NLoS = comps[i]
        acc = 0.0
        for _ in range(S):
            g = R.sample_gain(LoS, NLoS, rng)
            acc += (W_BAND / M) * np.log2(1.0 + g * P[i] / SIGMA2)
        Rsum[i] = acc / S
    return Rsum


def mean_gains(comps):
    return np.array([LoS + NLoS for (LoS, NLoS) in comps])


# ----------------------------------------------------------------------
# Per-scheme downlink rate (per PHYSICAL user, index order preserved)
# ----------------------------------------------------------------------
def proposed_rates(L, M, S, seed, out, inner):
    Q, Theta, P, _ = R.optimize_design(L, M, seed=seed, out=out, inner=inner)
    comps = [R.channel_components(Q, Theta, i, L) for i in range(M)]
    R_dl, _, _ = R.mc_eval(Q, Theta, P, L, M, S, seed, comps=comps)
    return R_dl


def oma_airs_rates(L, M, S, seed, out, inner):
    Q, Theta, _, _ = R.optimize_design(L, M, seed=seed, out=out, inner=inner)
    comps = [R.channel_components(Q, Theta, i, L) for i in range(M)]
    Omega = mean_gains(comps)
    P_oma = oma_max_min_power(Omega, P_HAP)
    rng = np.random.default_rng(seed + 4000)
    return oma_mc_rates(comps, P_oma, M, S, rng)


def no_irs_rates(L, M, S, seed):
    # direct HAP-GMD link only (blocked LoS -> pure NLoS scattered power)
    W_k = R.positions[:M]
    gamma = np.array([(R.rho0 - R.kappa1) / (np.linalg.norm(R.W_s - w) ** BETA1)
                      for w in W_k])
    comps = [(0.0, float(gi)) for gi in gamma]
    order = np.argsort(gamma)                       # ascending gain decode order
    P = noma_max_min_power(gamma[order], P_HAP)
    comps_s = [comps[o] for o in order]
    rng = np.random.default_rng(seed + 5000)
    R_sorted = sic_mc_rates(comps_s, P, M, S, rng)
    R_dl = np.zeros(M)                              # map back to physical order
    for k in range(M):
        R_dl[order[k]] = R_sorted[k]
    return R_dl


def girs_rates(L, M, S, seed, out, inner):
    candidates = []
    for x in (95.0, 100.0, 105.0):
        for y in (-5.0, 0.0, 5.0):
            candidates.append(np.array([x, y, 10.0]))
    best = -np.inf
    best_Rdl = None
    for gq in candidates:
        Qg, Thg, Pg, _ = R.optimize_design(L, M, seed=seed, out=out, inner=inner,
                                           opt_Q=False, Q_init=gq)
        comps = [R.channel_components(gq, Thg, i, L) for i in range(M)]
        R_dl, mn, _ = R.mc_eval(gq, Thg, Pg, L, M, S // 4, seed, comps=comps)
        if mn > best:
            best = mn
            best_Rdl = R_dl
            best_Q = gq
    # refine best candidate with full S for a stable per-user rate
    Qg, Thg, Pg, _ = R.optimize_design(L, M, seed=seed, out=out, inner=inner,
                                       opt_Q=False, Q_init=best_Q)
    comps = [R.channel_components(best_Q, Thg, i, L) for i in range(M)]
    R_dl, _, _ = R.mc_eval(best_Q, Thg, Pg, L, M, S, seed, comps=comps)
    return R_dl, best_Q


# ----------------------------------------------------------------------
# End-to-end latency
# ----------------------------------------------------------------------
def e2e_latency(D_in, c, D_out, R_ul, R_dl, f, T_queue=0.0):
    C = c * D_in
    T_ul = D_in / R_ul
    T_comp = C / f
    T_dl = D_out / R_dl
    T = T_queue + T_ul + T_comp + T_dl
    return T, {"uplink": float(T_ul.mean()), "compute": float(T_comp.mean()),
               "downlink": float(T_dl.mean())}


def build_schemes(L=8, M=4, S=1000, seed=0, out=5, inner=30):
    D_in = np.array([1.0e6] * (M - 1) + [1.5e6])     # bits; heavy user index M-1
    c = np.full(M, 200.0)                            # cycles/bit
    D_out = D_in * 0.5                               # bits
    R_ul = np.full(M, 2.0e6)                         # uplink bits/s (fixed)
    F_max = 10.0e9                                   # 10 GHz total CPU
    f_eq = np.full(M, F_max / M)

    schemes = []

    # 1. proposed NOMA-AIRS + equal CPU
    R_dl = proposed_rates(L, M, S, seed, out, inner)
    T, _ = e2e_latency(D_in, c, D_out, R_ul, R_dl, f_eq)
    schemes.append(("Proposed\n+ equal CPU", float(T.max()), float(T.mean())))

    # 2. proposed + delay-aware CPU
    b = D_in / R_ul + D_out / R_dl
    f_da, T_da = R.delay_aware_cpu(c * D_in, b, F_max)
    T, _ = e2e_latency(D_in, c, D_out, R_ul, R_dl, f_da)
    schemes.append(("Proposed\n+ delay-aware", float(T.max()), float(T.mean())))

    # 3. OMA-AIRS + equal CPU
    R_dl = oma_airs_rates(L, M, S, seed, out, inner)
    T, _ = e2e_latency(D_in, c, D_out, R_ul, R_dl, f_eq)
    schemes.append(("OMA-AIRS\n+ equal CPU", float(T.max()), float(T.mean())))

    # 4. no-IRS PD-NOMA + equal CPU
    R_dl = no_irs_rates(L, M, S, seed)
    T, _ = e2e_latency(D_in, c, D_out, R_ul, R_dl, f_eq)
    schemes.append(("No-IRS PD-NOMA\n+ equal CPU", float(T.max()), float(T.mean())))

    # 5. optimized GIRS + equal CPU
    R_dl, gq = girs_rates(L, M, S, seed, out, inner)
    T, _ = e2e_latency(D_in, c, D_out, R_ul, R_dl, f_eq)
    schemes.append(("Opt. GIRS\n+ equal CPU", float(T.max()), float(T.mean())))

    # 6. fixed-resource + equal CPU
    centroid = np.array([R.positions[:M, 0].mean(), R.positions[:M, 1].mean(),
                         R.Z_min])
    P_equal = np.full(M, P_HAP / M)
    rng = np.random.default_rng(seed + 2)
    Theta_r = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, L)))
    comps_r = [R.channel_components(centroid, Theta_r, i, L) for i in range(M)]
    R_dl, _, _ = R.mc_eval(centroid, Theta_r, P_equal, L, M, S, seed, comps=comps_r)
    T, _ = e2e_latency(D_in, c, D_out, R_ul, R_dl, f_eq)
    schemes.append(("Fixed-resource\n+ equal CPU", float(T.max()), float(T.mean())))

    names = [s[0] for s in schemes]
    tmax = [s[1] for s in schemes]
    tavg = [s[2] for s in schemes]

    # decomposition (relative to fixed-resource baseline)
    base = tmax[-1]
    gains = {names[i]: 100.0 * (base - tmax[i]) / base for i in range(len(names))}

    return {"names": names, "T_max": tmax, "T_avg": tavg,
            "reduction_pct": gains, "base_T_max": base,
            "girs_best_Q": gq.tolist()}


if __name__ == "__main__":
    out_dir = "revised_results"
    os.makedirs(out_dir, exist_ok=True)

    res = build_schemes()
    for n, t, g in zip(res["names"], res["T_max"], res["reduction_pct"].values()):
        print(f"  {n:24s} T_max={t:.4f} s   reduction={g:+.2f}%")

    with open(os.path.join(out_dir, "comment8_latency.json"), "w") as f:
        json.dump(res, f, indent=2)

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(res["names"]))
    w = 0.38
    ax.bar(x - w / 2, res["T_max"], w, label="Max E2E latency", color="#1f77b4")
    ax.bar(x + w / 2, res["T_avg"], w, label="Avg E2E latency", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(res["names"], fontsize=8)
    ax.set_ylabel("End-to-end latency (s)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    for i, v in enumerate(res["T_max"]):
        ax.text(i - w / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig("fig_latency_baselines.pdf", dpi=300, bbox_inches="tight")
    print("\nSaved revised_results/comment8_latency.json and fig_latency_baselines.pdf")
