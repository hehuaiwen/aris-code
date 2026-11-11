"""
Experiments for Reviewer 1, Comments 6--8.

All experiments reuse the clean scalar Rician channel model and the closed-form
max-min power allocation from scalar_rician_model.py (which is import-safe).

  Comment 6 (practical realism): imperfect CSI (several sigma levels), UAV
      position errors, and discrete IRS phase resolution (1/2/3-bit), each
      compared against the perfect-information baseline on the SIC-feasible
      Monte Carlo rate.
  Comment 7 (ablation study): remove UAV-position optimization, IRS phase
      optimization (random phase), and power optimization (equal power) one at
      a time; also compare airborne IRS vs a carefully optimized ground IRS.
  Comment 8 (MEC end-to-end latency): transmission + computation time for
      heterogeneous task sizes / computing capabilities; compare rate-max-min
      vs a latency-aware (min-max total delay) power allocation.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from scalar_rician_model import (
    rho0, SIGMA2, W_BAND, Z_MIN, Z_MAX, K_DEF_DB, L_DEF, M_DEF, PHAP_DEF_DB,
    BETA1, BETA2, W_S, USER_POSITIONS, kappa_from_db,
    channel_stats, channel_stats_for_users, uav_position,
    max_min_power, jensen_rates, sample_gains, sic_rate,
)

S = 20000
SEED = 2026
N_ERR = 10          # independent perturbation draws for robustness averaging


# ----------------------------------------------------------------------
# Comment 6 : channel model extensions + robustness evaluation
# ----------------------------------------------------------------------
def channel_stats_quant(Q, w_i, K_db, L, phase_bits=None):
    """Channel stats with B-bit IRS phase quantization.

    The coherent IRS LoS amplitude b_i is scaled by eta = sinc(1/2^B)
    (expected coherent-combining factor for a uniform phase error in
    [-pi/2^B, pi/2^B]); phase_bits=None -> continuous ideal phase.
    The incoherent IRS NLoS term is unaffected by phase quantization.
    """
    kappa = kappa_from_db(K_db)
    d_si = np.linalg.norm(W_S - w_i)
    d_ui = np.linalg.norm(Q - w_i)
    d_us = np.linalg.norm(Q - W_S)
    a_i = np.sqrt(kappa) / (d_si ** (BETA1 / 2.0))
    b_i = np.sqrt(kappa * rho0 * L * L) / ((d_ui ** (BETA2 / 2.0)) * d_us)
    if phase_bits is not None:
        eta = float(np.sinc(1.0 / (2.0 ** int(phase_bits))))
        b_i = b_i * eta
    L_i = (a_i + b_i) ** 2
    gamma_i = (rho0 - kappa) / (d_si ** BETA1)
    tau_i = L * rho0 * (rho0 - kappa) / ((d_ui ** BETA2) * (d_us ** 2))
    s2_i = gamma_i + tau_i
    return L_i + s2_i, L_i, s2_i


def channel_stats_random_phase(Q, w_i, K_db, L):
    """Channel stats (single user) when IRS phases are random (no coherent
    combining): the coherent IRS LoS term is lost, leaving only the direct LoS
    a_i^2 and the diffuse IRS NLoS tau_i."""
    kappa = kappa_from_db(K_db)
    d_si = np.linalg.norm(W_S - w_i)
    d_ui = np.linalg.norm(Q - w_i)
    d_us = np.linalg.norm(Q - W_S)
    a_i = np.sqrt(kappa) / (d_si ** (BETA1 / 2.0))
    L_i = a_i ** 2
    gamma_i = (rho0 - kappa) / (d_si ** BETA1)
    tau_i = L * rho0 * (rho0 - kappa) / ((d_ui ** BETA2) * (d_us ** 2))
    s2_i = gamma_i + tau_i
    return L_i + s2_i, L_i, s2_i


def channel_stats_random_phase_for_users(Q, W_u, K_db, L):
    M = len(W_u)
    Om = np.zeros(M); Lv = np.zeros(M); s2 = np.zeros(M)
    for i in range(M):
        Om[i], Lv[i], s2[i] = channel_stats_random_phase(Q, W_u[i], K_db, L)
    return Om, Lv, s2


def eval_mc_design(Lv, s2, order, P, S, rng):
    """SIC-feasible min MC rate for a design (order, P) over physical gains."""
    gains = sample_gains(Lv, s2, S, rng)          # physical user order
    R = sic_rate(gains[np.asarray(order)], P)     # reorder to decoding order
    return float(np.min(R))


def comment6(M, L, K_db, PHAP_db, S, seed):
    P_HAP = 10.0 ** (PHAP_db / 10.0)
    Q = uav_position(M)
    W_u = USER_POSITIONS[:M]
    Om, Lv, s2 = channel_stats_for_users(Q, W_u, K_db, L)

    # perfect baseline (continuous phase, perfect CSI/position)
    order0 = np.argsort(Om)
    _, P0 = max_min_power(Om[order0], P_HAP)
    rng = np.random.default_rng(seed)
    perfect = eval_mc_design(Lv, s2, order0, P0, S, rng)

    out = {"perfect_mc_min_bps": perfect}

    # ---- (a) imperfect CSI : perturb estimated Omega (dB) ----
    csi = {}
    for sigma_db in [0.5, 1.0, 2.0, 3.0]:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(N_ERR):
            Om_db = 10.0 * np.log10(np.maximum(Om, 1e-12))
            Om_hat = 10.0 ** ((Om_db + rng.normal(0.0, sigma_db, M)) / 10.0)
            order = np.argsort(Om_hat)
            _, P = max_min_power(Om_hat[order], P_HAP)
            vals.append(eval_mc_design(Lv, s2, order, P, S, rng))
        csi[str(sigma_db)] = {
            "mc_min_bps": float(np.mean(vals)),
            "degrad_pct": 100.0 * (perfect - np.mean(vals)) / perfect,
        }
    out["csi"] = csi

    # ---- (b) UAV position error : design at nominal Q, evaluate at Q+delta ----
    pos = {}
    for sigma_pos in [1.0, 2.0, 5.0, 10.0]:
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(N_ERR):
            Q_act = Q + rng.normal(0.0, sigma_pos, 3)
            _, L_act, s2_act = channel_stats_for_users(Q_act, W_u, K_db, L)
            vals.append(eval_mc_design(L_act, s2_act, order0, P0, S, rng))
        pos[str(sigma_pos)] = {
            "mc_min_bps": float(np.mean(vals)),
            "degrad_pct": 100.0 * (perfect - np.mean(vals)) / perfect,
        }
    out["position"] = pos

    # ---- (c) discrete IRS phase resolution (1/2/3-bit) ----
    phase = {}
    for B in [1, 2, 3]:
        Om_q = np.zeros(M); L_q = np.zeros(M); s2_q = np.zeros(M)
        for i in range(M):
            Om_q[i], L_q[i], s2_q[i] = channel_stats_quant(Q, W_u[i], K_db, L, phase_bits=B)
        order = np.argsort(Om_q)
        _, P = max_min_power(Om_q[order], P_HAP)
        rng = np.random.default_rng(seed)
        mc = eval_mc_design(L_q, s2_q, order, P, S, rng)
        phase[str(B)] = {
            "mc_min_bps": float(mc),
            "degrad_pct": 100.0 * (perfect - mc) / perfect,
        }
    out["phase"] = phase

    return out


# ----------------------------------------------------------------------
# Comment 7 : ablation study
# ----------------------------------------------------------------------
def optimize_uav_position(M, K_db, L, PHAP_db, channel_fn,
                          xr=(0.0, 30.0), yr=(0.0, 30.0), step=2.5, zs=(60.0, 80.0, 100.0)):
    """Grid-optimized UAV (IRS) position maximizing the min Jensen rate."""
    P_HAP = 10.0 ** (PHAP_db / 10.0)
    W_u = USER_POSITIONS[:M]
    best, bestQ = -1.0, None
    for x in np.arange(xr[0], xr[1] + 1e-9, step):
        for y in np.arange(yr[0], yr[1] + 1e-9, step):
            for z in zs:
                Q = np.array([x, y, z])
                Om, _, _ = channel_fn(Q, W_u, K_db, L)
                order = np.argsort(Om)
                _, P = max_min_power(Om[order], P_HAP)
                mn = float(np.min(jensen_rates(Om[order], P)))
                if mn > best:
                    best, bestQ = mn, Q
    return bestQ, best


def eval_ablation(Q, W_u, K_db, L, P_HAP, S, rng, channel_fn, power_mode):
    M = len(W_u)
    Om, Lv, s2 = channel_fn(Q, W_u, K_db, L)
    order = np.argsort(Om)
    Om_s = Om[order]
    if power_mode == "maxmin":
        _, P = max_min_power(Om_s, P_HAP)
    elif power_mode == "equal":
        P = np.full(M, P_HAP / M)
    else:
        raise ValueError(power_mode)
    mc = eval_mc_design(Lv, s2, order, P, S, rng)
    jensen_min = float(np.min(jensen_rates(Om_s, P)))
    return {"mc_min_bps": mc, "jensen_min_bps": jensen_min}


def comment7(M, L, K_db, PHAP_db, S, seed):
    P_HAP = 10.0 ** (PHAP_db / 10.0)
    W_u = USER_POSITIONS[:M]
    rng = np.random.default_rng(seed)

    Q_opt, _ = optimize_uav_position(M, K_db, L, PHAP_db, channel_stats_for_users)

    full = eval_ablation(Q_opt, W_u, K_db, L, P_HAP, S, rng,
                         channel_stats_for_users, "maxmin")
    fixed_uav = eval_ablation(np.array([0.0, 0.0, 80.0]), W_u, K_db, L, P_HAP, S, rng,
                              channel_stats_for_users, "maxmin")
    random_phase = eval_ablation(Q_opt, W_u, K_db, L, P_HAP, S, rng,
                                 channel_stats_random_phase_for_users, "maxmin")
    equal_power = eval_ablation(Q_opt, W_u, K_db, L, P_HAP, S, rng,
                                channel_stats_for_users, "equal")

    Q_ground, _ = optimize_uav_position(M, K_db, L, PHAP_db, channel_stats_for_users,
                                        zs=(3.0,))
    ground_irs = eval_ablation(Q_ground, W_u, K_db, L, P_HAP, S, rng,
                               channel_stats_for_users, "maxmin")

    base = full["mc_min_bps"]
    out = {
        "Q_opt": Q_opt.tolist(), "Q_ground": Q_ground.tolist(),
        "full": full, "fixed_uav": fixed_uav, "random_phase": random_phase,
        "equal_power": equal_power, "ground_irs": ground_irs,
    }
    for k in ["fixed_uav", "random_phase", "equal_power", "ground_irs"]:
        out[k]["rel_pct"] = 100.0 * (out[k]["mc_min_bps"] - base) / base
    return out


# ----------------------------------------------------------------------
# Comment 8 : MEC end-to-end latency
# ----------------------------------------------------------------------
def required_power_for_targets(Omega, g):
    """Minimum power (in mW) achieving per-user SINR targets g_r = 2^(R_r/W)-1."""
    M = len(Omega)
    P = np.zeros(M)
    for r in range(M - 1, -1, -1):
        P[r] = g[r] * (np.sum(P[r + 1:M]) + SIGMA2 / Omega[r])
    return P


def latency_max_min_power(Omega, P_HAP, s_bits, cyc, f_hz, iters=400):
    """Minimize max_i (s_i/R_i + c_i/f_i) over power (epigraph bisection).

    For a candidate worst delay T, the required rate of user i is
    R_i^req = s_i / (T - c_i) with c_i = s_i*cyc/f_i (requires T > c_i for all i).
    The feasibility of the resulting per-user SINR targets reduces to the
    closed-form NOMA power allocation; the minimal feasible T is found by
    bisection on the total-power budget.
    """
    M = len(Omega)
    c = s_bits * cyc / f_hz                              # computation delay per user
    lo = float(np.max(c)) + 1e-6
    hi = lo
    with np.errstate(over="ignore"):
        while True:
            R_req = s_bits / (hi - c)
            g = 2.0 ** (R_req / W_BAND) - 1.0
            g = np.clip(g, 0.0, 1e15)
            P = required_power_for_targets(Omega, g)
            if np.all(P >= 0) and np.sum(P) <= P_HAP:
                break
            hi *= 2.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            R_req = s_bits / (mid - c)
            g = np.clip(2.0 ** (R_req / W_BAND) - 1.0, 0.0, 1e15)
            P = required_power_for_targets(Omega, g)
            if np.all(P >= 0) and np.sum(P) <= P_HAP:
                hi = mid
            else:
                lo = mid
    T = hi
    R_req = s_bits / (T - c)
    g = np.clip(2.0 ** (R_req / W_BAND) - 1.0, 0.0, 1e15)
    P = required_power_for_targets(Omega, g)
    return P, T


def comment8(M, L, K_db, PHAP_db, seed):
    P_HAP = 10.0 ** (PHAP_db / 10.0)
    Q = uav_position(M)
    W_u = USER_POSITIONS[:M]
    Om, _, _ = channel_stats_for_users(Q, W_u, K_db, L)
    order = np.argsort(Om)
    Om_s = Om[order]

    # heterogeneous task sizes / computing capabilities (physical user order)
    s_bits = np.array([1.0e6] * (M - 1) + [1.5e6])     # edge user larger task
    cyc = 200.0                                          # cycles/bit
    f_hz = np.array([2.0e9] * (M - 1) + [0.4e9])         # edge user lower compute
    s_s = s_bits[order]
    f_s = f_hz[order]

    def summarize(P):
        R = jensen_rates(Om_s, P)
        tx = s_s / np.maximum(R, 1e-9)
        comp = s_s * cyc / f_s
        tot = tx + comp
        return {
            "rates_bps": R.tolist(),
            "tx_s": tx.tolist(),
            "comp_s": comp.tolist(),
            "total_s": tot.tolist(),
            "max_total_s": float(np.max(tot)),
            "min_total_s": float(np.min(tot)),
        }

    _, P_rmm = max_min_power(Om_s, P_HAP)
    rmm = summarize(P_rmm)

    P_lat, T_lat = latency_max_min_power(Om_s, P_HAP, s_s, cyc, f_s)
    lat = summarize(P_lat)

    return {
        "order": order.tolist(),
        "s_bits": s_s.tolist(),
        "f_hz": f_s.tolist(),
        "cyc_per_bit": cyc,
        "rate_maxmin": rmm,
        "latency_maxmin": lat,
        "max_total_reduction_pct":
            100.0 * (rmm["max_total_s"] - lat["max_total_s"]) / rmm["max_total_s"],
        "T_latency_obj_s": float(T_lat),
    }


# ----------------------------------------------------------------------
# Run everything and save results + figures
# ----------------------------------------------------------------------
if __name__ == "__main__":
    out_dir = "revised_results"
    os.makedirs(out_dir, exist_ok=True)
    all_results = {}

    # ---- Comment 6 ----
    c6 = comment6(M_DEF, L_DEF, K_DEF_DB, PHAP_DEF_DB, S, SEED)
    print("[Comment6] perfect MC min = %.4f Mbps" % (c6["perfect_mc_min_bps"] / 1e6))
    for sigma, v in c6["csi"].items():
        print(f"  CSI sigma={sigma} dB : {v['mc_min_bps']/1e6:.4f} Mbps  degrad {v['degrad_pct']:.2f}%")
    for sigma, v in c6["position"].items():
        print(f"  pos sigma={sigma} m  : {v['mc_min_bps']/1e6:.4f} Mbps  degrad {v['degrad_pct']:.2f}%")
    for B, v in c6["phase"].items():
        print(f"  phase {B}-bit       : {v['mc_min_bps']/1e6:.4f} Mbps  degrad {v['degrad_pct']:.2f}%")
    all_results["comment6"] = c6

    # ---- Comment 7 ----
    c7 = comment7(M_DEF, L_DEF, K_DEF_DB, PHAP_DEF_DB, S, SEED)
    print("\n[Comment7 ablation]")
    for k in ["full", "fixed_uav", "random_phase", "equal_power", "ground_irs"]:
        v = c7[k]
        rel = v.get("rel_pct", 0.0)
        print(f"  {k:13s} MC min={v['mc_min_bps']/1e6:.4f} Mbps  Jensen={v['jensen_min_bps']/1e6:.4f}  rel={rel:+.2f}%")
    all_results["comment7"] = c7

    # ---- Comment 8 ----
    c8 = comment8(M_DEF, L_DEF, K_DEF_DB, PHAP_DEF_DB, SEED)
    print("\n[Comment8 MEC latency]")
    print(f"  rate max-min  : max total delay = {c8['rate_maxmin']['max_total_s']*1e3:.2f} ms")
    print(f"  latency max-min: max total delay = {c8['latency_maxmin']['max_total_s']*1e3:.2f} ms")
    print(f"  max-delay reduction = {c8['max_total_reduction_pct']:.2f}%")
    all_results["comment8"] = c8

    with open(os.path.join(out_dir, "comments_678_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ---- Figure : robustness (3 panels) ----
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    ax = axes[0]
    sig = [float(k) for k in c6["csi"].keys()]
    vals = [c6["csi"][k]["mc_min_bps"] / 1e6 for k in c6["csi"].keys()]
    ax.plot(sig, vals, "o-", color="#1f77b4")
    ax.axhline(c6["perfect_mc_min_bps"] / 1e6, color="k", linestyle="--", label="Perfect")
    ax.set_xlabel(r"CSI error std $\sigma_{\mathrm{CSI}}$ (dB)")
    ax.set_ylabel("Minimum rate (Mbps)")
    ax.set_title("(a) Imperfect CSI")
    ax.grid(linestyle="--", alpha=0.4)

    ax = axes[1]
    sig = [float(k) for k in c6["position"].keys()]
    vals = [c6["position"][k]["mc_min_bps"] / 1e6 for k in c6["position"].keys()]
    ax.plot(sig, vals, "o-", color="#d62728")
    ax.axhline(c6["perfect_mc_min_bps"] / 1e6, color="k", linestyle="--", label="Perfect")
    ax.set_xlabel(r"UAV position error std $\sigma_{\mathrm{p}}$ (m)")
    ax.set_title("(b) UAV position error")
    ax.grid(linestyle="--", alpha=0.4)

    ax = axes[2]
    Bs = [1, 2, 3]
    vals = [c6["phase"][str(b)]["mc_min_bps"] / 1e6 for b in Bs]
    ax.plot(Bs, vals, "o-", color="#2ca02c")
    ax.axhline(c6["perfect_mc_min_bps"] / 1e6, color="k", linestyle="--", label="Continuous")
    ax.set_xlabel("IRS phase resolution (bits)")
    ax.set_xticks(Bs)
    ax.set_title("(c) Discrete IRS phase")
    ax.grid(linestyle="--", alpha=0.4)

    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("fig_robustness.pdf", dpi=300, bbox_inches="tight")
    print("\nSaved fig_robustness.pdf")

    # ---- Figure : ablation (bar chart) ----
    labels = ["Full\n(proposed)", "Fixed\nUAV pos.", "Random\nIRS phase",
              "Equal\npower", "Ground\nIRS"]
    keys = ["full", "fixed_uav", "random_phase", "equal_power", "ground_irs"]
    vals = [c7[k]["mc_min_bps"] / 1e6 for k in keys]
    colors = ["#2ca02c", "#7f7f7f", "#ff7f0e", "#9467bd", "#d62728"]
    fig2, ax2 = plt.subplots(figsize=(7, 3.6))
    bars = ax2.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Minimum rate (Mbps)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    fig2.tight_layout()
    fig2.savefig("fig_ablation.pdf", dpi=300, bbox_inches="tight")
    print("Saved fig_ablation.pdf")

    # ---- Figure : MEC latency (per-user total delay, two allocations) ----
    M = M_DEF
    users = [f"U{i+1}" for i in range(M)]
    x = np.arange(M)
    width = 0.36
    rmm_tot = np.array(c8["rate_maxmin"]["total_s"]) * 1e3
    lat_tot = np.array(c8["latency_maxmin"]["total_s"]) * 1e3
    fig3, ax3 = plt.subplots(figsize=(7, 3.6))
    ax3.bar(x - width / 2, rmm_tot, width, color="#1f77b4", label="Rate max-min")
    ax3.bar(x + width / 2, lat_tot, width, color="#d62728", label="Latency max-min")
    ax3.set_xticks(x)
    ax3.set_xticklabels(users, fontsize=8)
    ax3.set_ylabel("Total task completion delay (ms)")
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    fig3.tight_layout()
    fig3.savefig("fig_latency.pdf", dpi=300, bbox_inches="tight")
    print("Saved fig_latency.pdf")
    print("All done.")
