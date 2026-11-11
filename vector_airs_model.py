"""
Reviewer 1, Comments 6-8 experiments on the FULL vector AIRS model (AR_IRS).

  Comment 6 -- robustness: CSI estimation error (NMSE), UAV position error
               (horizontal/vertical), and discrete (b-bit) IRS phase control.
  Comment 7 -- ablation study (remove Q / Theta / P optimization) and an
               optimized ground-IRS (GIRS) benchmark.
  Comment 8 -- end-to-end MEC task-latency evaluation with several CPU
               allocation strategies.

The design (UAV position Q, IRS phase Theta, power P) is obtained with the
same AR_IRS penalty-SCA / SDP solver used in the main simulation. The final
empirical ergodic rate is evaluated by Monte Carlo: for every realization the
instantaneous effective channel gain is sampled as a Rician variable whose
line-of-sight and non-line-of-sight powers are given by the vector-array
channel model, and the SIC-feasible NOMA rate is computed.

All numerical results are written to
`revised_results/comments_678_full_results.json` and the figures
`fig_robustness_full.pdf`, `fig_ablation_full.pdf`, `fig_latency_full.pdf`.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from runtime_scaling import (
    AR_IRS, positions, W_s, N, rho0, K1, K2, sigmma,
    d, wavelength, kappa1, kappa2, Z_min, Z_max, s_ki,
)

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
W_BAND = 1.0e6        # bandwidth (Hz)
SIGMA2 = 1.0e-11      # physical noise power (-80 dBm), used by the Monte Carlo
                      # evaluation. Note: runtime_scaling.sigmma is the rescaled
                      # (=1.0) value fed to the SDP/SCA solvers.
BETA1 = 3.5           # blocked/NLoS ground link
BETA2 = 2.0           # aerial LoS link
P_HAP_DBM = 30.0
P_HAP = 10 ** (P_HAP_DBM / 10 - 3)   # total HAP transmit power (W)


# ----------------------------------------------------------------------
# Channel model: line-of-sight power and non-line-of-sight power
# ----------------------------------------------------------------------
def channel_components(Q, Theta, i, L, beta1=BETA1, beta2=BETA2, W_k=None):
    """Return (LoS_power, NLoS_power) for user i, matching compute_channel_gain."""
    if W_k is None:
        W_k = positions
    gamma_k_i = (rho0 - kappa1) / (np.linalg.norm(W_s - W_k[i]) ** beta1)
    tau_k_i = L * rho0 * (rho0 - kappa2)

    phi_k = (Q[1] - W_s[1]) / np.linalg.norm(Q - W_s)
    psi_k = (Q[2] - W_s[2]) / np.linalg.norm(Q - W_s)
    phi_ki = (W_k[i][1] - Q[1]) / np.linalg.norm(Q - W_k[i])
    hat_s_ki = np.sqrt(kappa1 / np.linalg.norm(W_s - W_k[i]) ** beta1) * s_ki.reshape(-1, 1)
    g_bar_k = np.exp(-1j * (2 * np.pi * d / wavelength) * np.arange(L) * phi_k).reshape(-1, 1)
    g_title_k = np.exp(-1j * (2 * np.pi * d / wavelength) * np.arange(N) * psi_k).reshape(1, -1)
    hat_h_k_i = np.sqrt(kappa2 / np.linalg.norm(Q - W_k[i]) ** beta2) * np.exp(
        -1j * (2 * np.pi * d / wavelength) * np.arange(L) * phi_ki).reshape(-1, 1)
    G_k = np.sqrt(rho0 / np.linalg.norm(Q - W_s) ** 2) * g_bar_k * g_title_k

    gain = np.conj(hat_s_ki.T) + np.conj(hat_h_k_i.T) @ Theta @ G_k
    LoS = float(np.linalg.norm(gain) ** 2)
    NLoS = float(gamma_k_i + tau_k_i / (
        (np.linalg.norm(Q - W_k[i]) ** beta2) * (np.linalg.norm(Q - W_s) ** 2)))
    return LoS, NLoS


def sample_gain(LoS, NLoS, rng):
    """Instantaneous effective channel gain |sqrt(LoS) + CN(0, NLoS)|^2."""
    n = rng.normal(0.0, np.sqrt(NLoS / 2.0), 2)
    return np.abs(np.sqrt(LoS) + (n[0] + 1j * n[1])) ** 2


def _sic_feasible_rates(g, P, M):
    """SIC-feasible NOMA rates for instantaneous gains g (index 0 = weakest).

    R_i = min_{k>=i} W log2(1 + g_k p_i / (g_k * sum_{j>i} p_j + sigma^2)).
    """
    rates = np.zeros(M)
    for i in range(M):
        inter_coeff = float(np.sum(P[i + 1:M]))
        decodable = []
        for k in range(i, M):
            sinr = g[k] * P[i] / (g[k] * inter_coeff + SIGMA2)
            decodable.append(W_BAND * np.log2(1.0 + sinr))
        rates[i] = min(decodable)
    return rates


def mc_eval(Q, Theta, P, L, M, S, seed, beta1=BETA1, beta2=BETA2,
            comps=None, rng=None):
    """Monte Carlo empirical ergodic rates.

    Returns (mean_per_user_rates, mean_min_rate, per_realization_min_rates).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if comps is None:
        comps = [channel_components(Q, Theta, i, L, beta1, beta2) for i in range(M)]
    rate_sums = np.zeros(M)
    min_rates = np.empty(S)
    for s in range(S):
        g = np.array([sample_gain(c[0], c[1], rng) for c in comps])
        r = _sic_feasible_rates(g, P, M)
        rate_sums += r
        min_rates[s] = r.min()
    return rate_sums / S, float(rate_sums.min() / S), min_rates


def optimize_design(L, M, P_HAP_dbm=P_HAP_DBM, seed=0, out=5, inner=20,
                    opt_theta=True, opt_Q=True, opt_P=True, Q_init=None,
                    P_init=None):
    """Run AR_IRS and return (Q, Theta, P, irs_obj)."""
    irs = AR_IRS()
    irs.bcd_optimization(max_iter=out, L=L, W=1e6, M=M, P_HAP=P_HAP_dbm,
                         seed=seed, inner_max=inner,
                         opt_theta=opt_theta, opt_Q=opt_Q, opt_P=opt_P,
                         Q_init=Q_init, P_init=P_init)
    return irs.Q, irs.Theta, irs.P, irs


# ======================================================================
# Comment 6: robustness
# ======================================================================
def comment6(L=8, M=4, S=1000, seed=0, out=5, inner=20,
             nmse_dbs=(-30, -20, -15, -10, -5),
             sig_h=(0.0, 0.5, 1.0, 2.0, 5.0),
             sig_v=(0.0, 0.2, 0.5, 1.0, 2.0)):
    Q, Theta, P, _ = optimize_design(L, M, seed=seed, out=out, inner=inner)
    rng = np.random.default_rng(seed + 999)

    # ---- (a) CSI estimation error ----
    comps_nom = [channel_components(Q, Theta, i, L) for i in range(M)]
    base = mc_eval(Q, Theta, P, L, M, S, seed, comps=comps_nom)[1]
    csi = {"nmse_db": list(nmse_dbs), "min_rate": [], "degradation_pct": [],
           "base_min_rate": base}
    for nmse_db in nmse_dbs:
        nmse = 10 ** (nmse_db / 10.0)
        min_rates = []
        for _ in range(5):  # average over perturbation realizations
            comps_p = []
            for (LoS, NLoS) in comps_nom:
                delta = rng.normal(0.0, np.sqrt(nmse))
                comps_p.append((max(LoS * (1 + delta), 0.0),
                                max(NLoS * (1 + delta), 0.0)))
            min_rates.append(mc_eval(Q, Theta, P, L, M, S, seed, comps=comps_p)[1])
        mn = float(np.mean(min_rates))
        csi["min_rate"].append(mn)
        csi["degradation_pct"].append((base - mn) / base * 100.0)

    # ---- (b) UAV position error ----
    pos = {"sig_h": list(sig_h), "min_rate": [], "degradation_pct": [],
           "base_min_rate": base}
    for sh in sig_h:
        sv = sig_v[min(range(len(sig_h)), key=lambda k: abs(sig_h[k] - sh))]
        min_rates = []
        for _ in range(10):
            e = np.array([rng.normal(0, sh), rng.normal(0, sh), rng.normal(0, sv)])
            Qp = Q + e
            comps_p = [channel_components(Qp, Theta, i, L) for i in range(M)]
            min_rates.append(mc_eval(Qp, Theta, P, L, M, S, seed, comps=comps_p)[1])
        mn = float(np.mean(min_rates))
        pos["min_rate"].append(mn)
        pos["degradation_pct"].append((base - mn) / base * 100.0)

    # ---- (c) discrete IRS phase ----
    phase = {"bits": ["Continuous", "3-bit", "2-bit", "1-bit", "Random"],
             "min_rate": [], "degradation_pct": []}
    phase["min_rate"].append(base)
    phase["degradation_pct"].append(0.0)
    psi = np.angle(np.diagonal(Theta))
    for b in (3, 2, 1):
        levels = 2 ** b
        psi_q = np.round(psi / (2 * np.pi / levels)) * (2 * np.pi / levels)
        Theta_q = np.diag(np.exp(1j * psi_q))
        comps_q = [channel_components(Q, Theta_q, i, L) for i in range(M)]
        mn = mc_eval(Q, Theta_q, P, L, M, S, seed, comps=comps_q)[1]
        phase["min_rate"].append(mn)
        phase["degradation_pct"].append((base - mn) / base * 100.0)
    # random phase
    min_rates = []
    for _ in range(10):
        Theta_r = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, L)))
        comps_r = [channel_components(Q, Theta_r, i, L) for i in range(M)]
        min_rates.append(mc_eval(Q, Theta_r, P, L, M, S, seed, comps=comps_r)[1])
    mn = float(np.mean(min_rates))
    phase["min_rate"].append(mn)
    phase["degradation_pct"].append((base - mn) / base * 100.0)

    return {"csi": csi, "position": pos, "phase": phase}


# ======================================================================
# Comment 7: ablation + optimized GIRS
# ======================================================================
def comment7(L=8, M=4, S=1000, seed=0, out=5, inner=20):
    centroid = np.array([positions[:M, 0].mean(), positions[:M, 1].mean(), Z_min])
    P_equal = np.full(M, P_HAP / M)

    # full design
    Q, Theta, P, _ = optimize_design(L, M, seed=seed, out=out, inner=inner)
    full = mc_eval(Q, Theta, P, L, M, S, seed)[1]

    # w/o position optimization (fix Q at centroid, keep Theta + P)
    Qf, Thf, Pf, _ = optimize_design(L, M, seed=seed, out=out, inner=inner,
                                     opt_Q=False, Q_init=centroid)
    woQ = mc_eval(Qf, Thf, Pf, L, M, S, seed)[1]

    # w/o phase optimization (random Theta, keep Q + P) -- average over phases
    rng = np.random.default_rng(seed + 1)
    woTh_vals = []
    for _ in range(5):
        Theta_r = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, L)))
        Qr, Thr, Pr, _ = optimize_design(L, M, seed=seed, out=out, inner=inner,
                                         opt_theta=False)
        # overwrite Theta with the random draw (no optimization applied)
        woTh_vals.append(mc_eval(Qr, Theta_r, Pr, L, M, S, seed)[1])
    woTh = float(np.mean(woTh_vals))

    # w/o power optimization (equal power, keep Q + Theta)
    Qp, Thp, Pp, _ = optimize_design(L, M, seed=seed, out=out, inner=inner,
                                     opt_P=False, P_init=P_equal)
    woP = mc_eval(Qp, Thp, P_equal, L, M, S, seed)[1]

    # power-only (fixed Q at centroid, random Theta, optimize P)
    Theta_r0 = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, L)))
    Qpo, Thpo, Ppo, _ = optimize_design(L, M, seed=seed, out=out, inner=inner,
                                        opt_Q=False, opt_theta=False,
                                        Q_init=centroid)
    pow_only = mc_eval(centroid, Theta_r0, Ppo, L, M, S, seed)[1]

    # all-fixed (centroid Q, random Theta, equal power)
    all_fixed = mc_eval(centroid, Theta_r0, P_equal, L, M, S, seed)[1]

    # geometry-phase (LoS geometric alignment toward the user centroid),
    # computed at the optimized UAV position, with the optimized Q and P retained
    ref = np.array([positions[:M, 0].mean(), positions[:M, 1].mean(), 0.0])
    phi_k = (Q[1] - W_s[1]) / np.linalg.norm(Q - W_s)
    phi_ref = (ref[1] - Q[1]) / np.linalg.norm(Q - ref)
    ell_idx = np.arange(L)
    Theta_geo = np.diag(np.exp(1j * (2 * np.pi * d / wavelength) * ell_idx * (phi_k - phi_ref)))
    geo = mc_eval(Q, Theta_geo, P, L, M, S, seed)[1]

    # ---- optimized GIRS benchmark (ground IRS near the GMD cluster) ----
    girs_candidates = []
    for x in (95.0, 100.0, 105.0):
        for y in (-5.0, 0.0, 5.0):
            girs_candidates.append(np.array([x, y, 10.0]))
    girs_best = -np.inf
    girs_Q_best = None
    for gq in girs_candidates:
        Qg, Thg, Pg, _ = optimize_design(L, M, seed=seed, out=out, inner=inner,
                                         opt_Q=False, Q_init=gq)
        mn = mc_eval(gq, Thg, Pg, L, M, S // 4, seed)[1]
        if mn > girs_best:
            girs_best = mn
            girs_Q_best = gq
    girs = girs_best

    schemes = ["Full AFOA", "w/o position", "w/o phase", "w/o power",
               "Power only", "Geometry phase", "All fixed", "Opt. GIRS"]
    rates = [full, woQ, woTh, woP, pow_only, geo, all_fixed, girs]
    gains = [(r - all_fixed) / all_fixed * 100.0 for r in rates]

    return {
        "schemes": schemes,
        "min_rate": rates,
        "gain_over_all_fixed_pct": gains,
        "girs_best_Q": girs_Q_best.tolist(),
    }


# ======================================================================
# Comment 8: end-to-end MEC latency
# ======================================================================
def delay_aware_cpu(C, b, F_max):
    """Solve min_{f} max_i (b_i + C_i/f_i) s.t. sum f <= F_max, f > 0.

    Returns (f, T_opt) via bisection on the common completion time T.
    """
    lo = max(b) + 1e-9
    hi = max(b) + 100.0
    for _ in range(60):
        T = 0.5 * (lo + hi)
        f = C / (T - b)
        if np.sum(f) <= F_max:
            hi = T
        else:
            lo = T
    T = hi
    f = C / (T - b)
    return f, T


def comment8(L=8, M=4, S=800, seed=0, out=5, inner=20):
    Q, Theta, P, _ = optimize_design(L, M, seed=seed, out=out, inner=inner)
    # downlink rates (empirical mean per user), reuse for all CPU schemes
    comps = [channel_components(Q, Theta, i, L) for i in range(M)]
    R_dl, _, _ = mc_eval(Q, Theta, P, L, M, S, seed, comps=comps)

    # heterogeneous task profile: one cell-edge heavy user (index M-1)
    D_in = np.array([1.0e6] * (M - 1) + [1.5e6])          # bits
    c = np.full(M, 200.0)                                  # cycles/bit
    D_out = D_in * 0.5                                     # bits
    R_ul = np.full(M, 2.0e6)                               # uplink bits/s (fixed)
    F_max = 10.0e9                                         # 10 GHz total CPU

    T_ul = D_in / R_ul
    C = c * D_in

    def e2e(f, T_queue=0.0):
        T_comp = C / f
        T_dl = D_out / R_dl
        return T_queue + T_ul + T_comp + T_dl

    schemes = {}
    # equal CPU
    f_eq = np.full(M, F_max / M)
    t_eq = e2e(f_eq)
    schemes["Equal CPU"] = {"T_max": float(t_eq.max()), "T_avg": float(t_eq.mean())}
    # workload-proportional CPU
    f_wl = F_max * C / C.sum()
    t_wl = e2e(f_wl)
    schemes["Workload-proportional"] = {"T_max": float(t_wl.max()),
                                        "T_avg": float(t_wl.mean())}
    # delay-aware CPU
    b = T_ul + D_out / R_dl
    f_da, T_da = delay_aware_cpu(C, b, F_max)
    t_da = e2e(f_da)
    schemes["Delay-aware"] = {"T_max": float(t_da.max()), "T_avg": float(t_da.mean())}

    # fixed-resource baseline (equal power + fixed Q + random phase)
    centroid = np.array([positions[:M, 0].mean(), positions[:M, 1].mean(), Z_min])
    P_equal = np.full(M, P_HAP / M)
    rng = np.random.default_rng(seed + 2)
    Theta_r = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, L)))
    comps_r = [channel_components(centroid, Theta_r, i, L) for i in range(M)]
    R_dl_base, _, _ = mc_eval(centroid, Theta_r, P_equal, L, M, S, seed, comps=comps_r)
    T_dl_base = D_out / R_dl_base
    t_base = T_ul + C / f_eq + T_dl_base
    schemes["Fixed baseline (equal CPU)"] = {"T_max": float(t_base.max()),
                                             "T_avg": float(t_base.mean())}

    # latency composition for the proposed design + delay-aware CPU
    comp = {
        "uplink": float(T_ul.mean()),
        "compute": float((C / f_da).mean()),
        "downlink": float((D_out / R_dl).mean()),
    }

    return {"schemes": schemes, "composition": comp, "R_dl_min": float(R_dl.min())}


# ======================================================================
# Driver
# ======================================================================
if __name__ == "__main__":
    L, M = 8, 4
    print("=== Comment 6 (robustness) ===", flush=True)
    r6 = comment6(L=L, M=M)
    print(" CSI   min_rate:", [f"{v:.4f}" for v in r6["csi"]["min_rate"]], "Mbps")
    print("       degrad%:  ", [f"{v:.2f}" for v in r6["csi"]["degradation_pct"]])
    print(" Pos   min_rate:", [f"{v:.4f}" for v in r6["position"]["min_rate"]], "Mbps")
    print("       degrad%:  ", [f"{v:.2f}" for v in r6["position"]["degradation_pct"]])
    print(" Phase min_rate:", [f"{v:.4f}" for v in r6["phase"]["min_rate"]], "Mbps")
    print("       degrad%:  ", [f"{v:.2f}" for v in r6["phase"]["degradation_pct"]])

    print("=== Comment 7 (ablation + GIRS) ===", flush=True)
    r7 = comment7(L=L, M=M)
    for s, r, g in zip(r7["schemes"], r7["min_rate"], r7["gain_over_all_fixed_pct"]):
        print(f"  {s:22s} min={r:.4f} Mbps  gain={g:.2f}%")

    print("=== Comment 8 (MEC latency) ===", flush=True)
    r8 = comment8(L=L, M=M)
    for k, v in r8["schemes"].items():
        print(f"  {k:28s} T_max={v['T_max']:.4f} s  T_avg={v['T_avg']:.4f} s")

    # ---- save JSON ----
    out_dir = "revised_results"
    os.makedirs(out_dir, exist_ok=True)
    results = {"comment6": r6, "comment7": r7, "comment8": r8}
    with open(os.path.join(out_dir, "comments_678_full_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- figures ----
    # Figure: robustness (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    axes[0].plot(r6["csi"]["nmse_db"], r6["csi"]["degradation_pct"], "o-", color="#1f77b4")
    axes[0].set_xlabel("NMSE (dB)")
    axes[0].set_ylabel("Rate degradation (%)")
    axes[0].set_title("(a) CSI estimation error")
    axes[0].grid(linestyle="--", alpha=0.4)

    axes[1].plot(r6["position"]["sig_h"], r6["position"]["degradation_pct"],
                 "s-", color="#d62728")
    axes[1].set_xlabel(r"Horizontal error $\sigma_{q,h}$ (m)")
    axes[1].set_ylabel("Rate degradation (%)")
    axes[1].set_title("(b) UAV position error")
    axes[1].grid(linestyle="--", alpha=0.4)

    labels = r6["phase"]["bits"]
    axes[2].bar(range(len(labels)), r6["phase"]["degradation_pct"], color="#2ca02c")
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=30, ha="right")
    axes[2].set_ylabel("Rate degradation (%)")
    axes[2].set_title("(c) Discrete phase control")
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig("fig_robustness_full.pdf", dpi=300, bbox_inches="tight")

    # Figure: ablation + GIRS
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = np.arange(len(r7["schemes"]))
    rates_mbps = np.array(r7["min_rate"]) / 1e6
    ax.bar(x, rates_mbps, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(r7["schemes"], rotation=30, ha="right")
    ax.set_ylabel("Empirical min. ergodic rate (Mbps)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for b, v in zip(ax.patches, rates_mbps):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig("fig_ablation_full.pdf", dpi=300, bbox_inches="tight")

    # Figure: MEC latency
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    keys = list(r8["schemes"].keys())
    Tmax = [r8["schemes"][k]["T_max"] for k in keys]
    Tavg = [r8["schemes"][k]["T_avg"] for k in keys]
    x = np.arange(len(keys))
    w = 0.38
    axes[0].bar(x - w / 2, Tmax, w, label="Max delay", color="#1f77b4")
    axes[0].bar(x + w / 2, Tavg, w, label="Avg delay", color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(keys, rotation=30, ha="right")
    axes[0].set_ylabel("E2E latency (s)")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].set_title("(a) Communication + CPU strategies")

    comp_keys = ["uplink", "compute", "downlink"]
    comp_vals = [r8["composition"][k] for k in comp_keys]
    axes[1].bar(comp_keys, comp_vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1].set_ylabel("Mean latency (s)")
    axes[1].set_title("(b) Latency composition (proposed)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("fig_latency_full.pdf", dpi=300, bbox_inches="tight")

    print("Saved figures and results.")
    print("Done.")
