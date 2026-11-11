"""
Monte Carlo validation of the Jensen-based surrogate (Reviewer 1, Comment 1).

For a given design (UAV position Q, IRS phase Theta, power allocation P, SIC order),
we compare the Jensen upper bound  Rbar_i = W log2(1 + Omega_i p_i / (Omega_i I_i + sigma^2))
against the true empirical ergodic rate obtained by averaging the instantaneous rate over
S independent NLoS channel realizations.

Channel model (scalar Rician, consistent with Appendix A of the paper):
    LoS  power:  L_i   = ( a_i + b_i )^2
    NLoS power:  s_i^2 = gamma_i + tau_i
    average:     Omega_i = L_i + s_i^2
with
    a_i   = sqrt(kappa1) / d_si^(beta1/2)
    b_i   = sqrt(kappa2 * rho0 * L^2) / ( d_ui^(beta2/2) * d_us )
    gamma_i = (rho0 - kappa1) / d_si^beta1
    tau_i   = L * rho0 * (rho0 - kappa2) / ( d_ui^beta2 * d_us^2 )
Instantaneous gain: |H_i^(s)|^2 = | sqrt(L_i) + w_i^(s) |^2,  w_i ~ CN(0, s_i^2),
so E{|H_i|^2} = Omega_i. The power allocation is the closed-form max-min solution
(bisection on the common surrogate rate tau) for a fixed statistical SIC order.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# System parameters (consistent with the paper, Table 1)
# ----------------------------------------------------------------------
rho0 = 0.001            # channel power gain at reference distance (-30 dB)
SIGMA2 = 1e-8           # noise power (-80 dBm)
W_BAND = 1e6            # bandwidth (1 MHz)
Z_MIN, Z_MAX = 30.0, 60.0

# Defaults (held fixed unless swept)
K_DEF_DB = 10.0         # Rician factor (dB) of the AIRS-GMD LoS link (K2); the
                        # direct HAP-GMD link is blocked -> pure NLoS (kappa1 = 0)
L_DEF = 16              # number of IRS elements
M_DEF = 5               # number of users
PHAP_DEF_DB = 30.0      # HAP transmit power budget (dBm)
BETA1, BETA2 = 3.5, 2.0   # BETA1: blocked/NLoS ground link; BETA2: aerial LoS link

# HAP position (matches paper: (0,0,20) m)
W_S = np.array([0.0, 0.0, 20.0])

# GMDs distributed over a ~350 m^2 region located ~100 m from the HAP, matching
# the "HAP at a considerable distance / obstacles blocking the direct path"
# scenario in Section II (the direct HAP-GMD link is blocked, so the AIRS
# reflected path dominates).
USER_POSITIONS = np.array([
    [100.0,   0.0, 0.0],
    [106.0,   4.0, 0.0],
    [104.0,  -5.0, 0.0],
    [ 97.0,   6.0, 0.0],
    [109.0,  -2.0, 0.0],
    [ 93.0,   5.0, 0.0],
    [101.0,  -8.0, 0.0],
    [107.0,  -6.0, 0.0],
    [ 95.0,  -4.0, 0.0],
    [103.0,   8.0, 0.0],
    [ 99.0,   2.0, 0.0],
    [108.0,   3.0, 0.0],
    [ 92.0,  -7.0, 0.0],
    [105.0,   7.0, 0.0],
])


def kappa_from_db(K_db):
    K_lin = 10.0 ** (K_db / 10.0)
    return K_lin * rho0 / (K_lin + 1.0)


def channel_stats(Q, w_i, K_db, L):
    """Return (Omega_i, L_i, s2_i) for user i at position w_i.

    Direct HAP-GMD link is blocked (obstacle) -> pure NLoS: kappa1 = 0.
    K_db is the Rician factor (dB) of the AIRS-GMD LoS link (kappa2).
    """
    kappa1 = 0.0
    kappa2 = kappa_from_db(K_db)
    d_si = np.linalg.norm(W_S - w_i)      # HAP -> user
    d_ui = np.linalg.norm(Q - w_i)        # UAV -> user
    d_us = np.linalg.norm(Q - W_S)        # UAV -> HAP

    a_i = np.sqrt(kappa1) / (d_si ** (BETA1 / 2.0))                     # direct LoS (= 0, blocked)
    b_i = np.sqrt(kappa2 * rho0 * L * L) / ((d_ui ** (BETA2 / 2.0)) * d_us)  # reflected LoS
    L_i = (a_i + b_i) ** 2                                         # coherent LoS power

    gamma_i = (rho0 - kappa1) / (d_si ** BETA1)                        # direct NLoS power
    tau_i = L * rho0 * (rho0 - kappa2) / ((d_ui ** BETA2) * (d_us ** 2))  # reflected NLoS power
    s2_i = gamma_i + tau_i

    return L_i + s2_i, L_i, s2_i


def max_min_power(Omega, P_HAP, tol=1e-9, iters=200):
    """Closed-form max-min power allocation for fixed Omega (sorted ascending).

    Returns (tau, P) where tau is the common surrogate rate (bps/Hz before W scaling
    is applied) and P is the power allocation in mW.
    """
    M = len(Omega)
    # bisection over tau (in bps/Hz, i.e., log2 units; W_BAND applied outside)
    lo, hi = 0.0, 200.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        g = 2.0 ** mid - 1.0
        P = np.zeros(M)
        P[M - 1] = g * SIGMA2 / Omega[M - 1]
        for r in range(M - 2, -1, -1):
            P[r] = g * (np.sum(P[r + 1:M]) + SIGMA2 / Omega[r])
        total = np.sum(P)
        if total > P_HAP:
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
    """Per-user Jensen surrogate rate (in bps), for Omega sorted ascending."""
    M = len(Omega)
    Rbar = np.zeros(M)
    for i in range(M):
        inter = np.sum(P[i + 1:M])
        Rbar[i] = W_BAND * np.log2(1.0 + Omega[i] * P[i] / (Omega[i] * inter + SIGMA2))
    return Rbar


def mc_evaluate(Omega, L_vec, s2_vec, P, S, rng):
    """Empirical ergodic rate (bps) for a fixed design, over S realizations.

    Uses the scalar Rician model: |H_i|^2 = | sqrt(L_i) + w |^2, w ~ CN(0, s2_i).
    The SIC order is fixed by the statistical gains Omega (ascending).
    """
    M = len(Omega)
    Rsum = np.zeros(M)
    amp = np.sqrt(np.maximum(L_vec, 0.0))
    std = np.sqrt(np.maximum(s2_vec, 0.0))
    # vectorized over realizations
    for i in range(M):
        # generate S complex Gaussian realizations for user i
        wr = rng.normal(0.0, std[i] / np.sqrt(2.0), S)
        wi = rng.normal(0.0, std[i] / np.sqrt(2.0), S)
        gain = (amp[i] + wr) ** 2 + wi ** 2   # |sqrt(L_i) + w|^2
        inter = np.sum(P[i + 1:M])
        sinr = gain * P[i] / (gain * inter + SIGMA2)
        Rsum[i] = np.sum(W_BAND * np.log2(1.0 + sinr))
    return Rsum / S


def run_case(K_db, L, M, PHAP_db, S, rng):
    """Run one case: compute Jensen and MC min rates and the gap metrics."""
    Q = uav_position(M)
    W_u = USER_POSITIONS[:M]
    P_HAP = 10.0 ** (PHAP_db / 10.0)   # mW

    Omega = np.zeros(M)
    L_vec = np.zeros(M)
    s2_vec = np.zeros(M)
    for i in range(M):
        Omega[i], L_vec[i], s2_vec[i] = channel_stats(Q, W_u[i], K_db, L)

    # statistical SIC order: ascending Omega
    order = np.argsort(Omega)
    Omega_s = Omega[order]
    L_s = L_vec[order]
    s2_s = s2_vec[order]

    tau_hz, P = max_min_power(Omega_s, P_HAP)
    Rbar = jensen_rates(Omega_s, P)          # all ~ tau_hz * W_BAND
    R_mc = mc_evaluate(Omega_s, L_s, s2_s, P, S, rng)

    eps_i = 100.0 * (Rbar - R_mc) / np.maximum(R_mc, 1e-12)
    eps_mean = float(np.mean(eps_i))
    eps_max = float(np.max(eps_i))
    min_Rbar = float(np.min(Rbar))
    min_Rmc = float(np.min(R_mc))
    eps_min = 100.0 * (min_Rbar - min_Rmc) / max(min_Rmc, 1e-12)

    return {
        "min_Rbar_bps": min_Rbar,
        "min_Rmc_bps": min_Rmc,
        "eps_mean_pct": eps_mean,
        "eps_max_pct": eps_max,
        "eps_min_pct": eps_min,
        "per_user_gap_pct": eps_i.tolist(),
    }


def uav_position(M):
    """UAV at the centroid of the first M users, at a fixed height."""
    W_u = USER_POSITIONS[:M]
    cen = np.mean(W_u, axis=0)
    cen[2] = 0.5 * (Z_MIN + Z_MAX)
    return cen


# ----------------------------------------------------------------------
# Sweep definitions
# ----------------------------------------------------------------------
SWEEPS = {
    "K": {
        "xlabel": "Rician factor $K_2$ (dB)",
        "values": [0.0, 3.0, 5.0, 10.0, 15.0, 20.0],
        "kwargs": lambda v: dict(K_db=v, L=L_DEF, M=M_DEF, PHAP_db=PHAP_DEF_DB),
    },
    "L": {
        "xlabel": "Number of IRS elements $L$",
        "values": [4, 8, 16, 32, 64],
        "kwargs": lambda v: dict(K_db=K_DEF_DB, L=v, M=M_DEF, PHAP_db=PHAP_DEF_DB),
    },
    "M": {
        "xlabel": "Number of users $M$",
        "values": [2, 3, 4, 5, 6, 8],
        "kwargs": lambda v: dict(K_db=K_DEF_DB, L=L_DEF, M=v, PHAP_db=PHAP_DEF_DB),
    },
    "P": {
        "xlabel": "Transmit power budget $P_{\\mathrm{HAP}}$ (dBm)",
        "values": [10.0, 20.0, 30.0, 40.0],
        "kwargs": lambda v: dict(K_db=K_DEF_DB, L=L_DEF, M=M_DEF, PHAP_db=v),
    },
}

S = 20000
seed = 2026

results = {}
for key, spec in SWEEPS.items():
    rng = np.random.default_rng(seed)
    xs = spec["values"]
    min_Rbar = []
    min_Rmc = []
    eps_mean = []
    eps_max = []
    eps_min = []
    per_user = []
    for v in xs:
        kw = spec["kwargs"](v)
        r = run_case(S=S, rng=rng, **kw)
        min_Rbar.append(r["min_Rbar_bps"])
        min_Rmc.append(r["min_Rmc_bps"])
        eps_mean.append(r["eps_mean_pct"])
        eps_max.append(r["eps_max_pct"])
        eps_min.append(r["eps_min_pct"])
        per_user.append(r["per_user_gap_pct"])
        print(f"[{key}={v}] min_Rbar={r['min_Rbar_bps']/1e6:.3f} Mbps, "
              f"min_Rmc={r['min_Rmc_bps']/1e6:.3f} Mbps, "
              f"eps_mean={r['eps_mean_pct']:.3f}%, eps_max={r['eps_max_pct']:.3f}%, "
              f"eps_min={r['eps_min_pct']:.3f}%")
    results[key] = {
        "xlabel": spec["xlabel"],
        "x": xs,
        "min_Rbar_bps": min_Rbar,
        "min_Rmc_bps": min_Rmc,
        "eps_mean_pct": eps_mean,
        "eps_max_pct": eps_max,
        "eps_min_pct": eps_min,
        "per_user_gap_pct": per_user,
    }

# ----------------------------------------------------------------------
# Save raw results
# ----------------------------------------------------------------------
out_dir = "jensen_validation_results"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "jensen_mc_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# ----------------------------------------------------------------------
# Plot: 2 rows x 4 columns (top: rates; bottom: relative gap)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))

for col, (key, spec) in enumerate(SWEEPS.items()):
    x = results[key]["x"]
    ax1 = axes[0, col]
    ax2 = axes[1, col]

    ax1.plot(x, np.array(results[key]["min_Rbar_bps"]) / 1e6, "o-", color="#1f77b4",
             label="Jensen (surrogate)")
    ax1.plot(x, np.array(results[key]["min_Rmc_bps"]) / 1e6, "s--", color="#d62728",
             label="Monte Carlo")
    ax1.set_xlabel(spec["xlabel"])
    ax1.set_ylabel("Minimum rate (Mbps)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_xticks(x)

    ax2.plot(x, results[key]["eps_mean_pct"], "o-", color="#2ca02c", label="Mean gap")
    ax2.plot(x, results[key]["eps_max_pct"], "s--", color="#ff7f0e", label="Max gap")
    ax2.set_xlabel(spec["xlabel"])
    ax2.set_ylabel("Relative Jensen gap (%)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_xticks(x)

axes[0, 0].set_ylabel("Minimum rate (Mbps)")
axes[0, 0].legend(fontsize=7, loc="best")
axes[1, 0].legend(fontsize=7, loc="best")
for c in range(4):
    axes[0, c].set_title(f"({chr(97+c)})", fontsize=10)

fig.tight_layout()
out_fig = "jensen_mc_validation.pdf"
fig.savefig(out_fig, dpi=300, bbox_inches="tight")
print("Figure saved to", os.path.abspath(out_fig))


# ----------------------------------------------------------------------
# Effectiveness under the true (Monte Carlo) rate: proposed vs benchmarks
# ----------------------------------------------------------------------
def direct_channel_stats(Q, w_i, K_db):
    """Direct-link-only channel stats (for NOMA without IRS)."""
    kappa1 = 0.0   # direct link blocked -> pure NLoS
    d_si = np.linalg.norm(W_S - w_i)
    a_i = np.sqrt(kappa1) / (d_si ** (BETA1 / 2.0))
    L_i = a_i ** 2
    gamma_i = (rho0 - kappa1) / (d_si ** BETA1)
    return L_i + gamma_i, L_i, gamma_i


def oma_max_min_power(Omega, P_HAP):
    """OMA max-min power: equalize Omega_i p_i across users."""
    Omega = np.asarray(Omega, dtype=float)
    c = P_HAP / np.sum(1.0 / Omega)
    P = c / Omega
    return P, (W_BAND / len(Omega)) * np.log2(1.0 + c / SIGMA2)


def oma_mc_rate(L_vec, s2_vec, P, S, rng):
    M = len(P)
    Rsum = np.zeros(M)
    amp = np.sqrt(np.maximum(np.asarray(L_vec, float), 0.0))
    std = np.sqrt(np.maximum(np.asarray(s2_vec, float), 0.0))
    for i in range(M):
        wr = rng.normal(0.0, std[i] / np.sqrt(2.0), S)
        wi = rng.normal(0.0, std[i] / np.sqrt(2.0), S)
        gain = (amp[i] + wr) ** 2 + wi ** 2
        Rsum[i] = np.sum((W_BAND / M) * np.log2(1.0 + gain * P[i] / SIGMA2))
    return Rsum / S


def effectiveness(K_db, L, M, PHAP_db, S, seed):
    rng = np.random.default_rng(seed)
    Q = uav_position(M)
    W_u = USER_POSITIONS[:M]
    P_HAP = 10.0 ** (PHAP_db / 10.0)

    # --- proposed: NOMA with AIRS ---
    Om, Lv, s2 = np.zeros(M), np.zeros(M), np.zeros(M)
    for i in range(M):
        Om[i], Lv[i], s2[i] = channel_stats(Q, W_u[i], K_db, L)
    order = np.argsort(Om)
    Om_s, L_s, s2_s = Om[order], Lv[order], s2[order]
    _, P_prop = max_min_power(Om_s, P_HAP)
    Rmc_prop = mc_evaluate(Om_s, L_s, s2_s, P_prop, S, rng)

    # --- OMA with AIRS ---
    P_oma, _ = oma_max_min_power(Om_s, P_HAP)
    Rmc_oma = oma_mc_rate(L_s, s2_s, P_oma, S, rng)

    # --- NOMA without IRS ---
    Om2, Lv2, s22 = np.zeros(M), np.zeros(M), np.zeros(M)
    for i in range(M):
        Om2[i], Lv2[i], s22[i] = direct_channel_stats(Q, W_u[i], K_db)
    order2 = np.argsort(Om2)
    Om2_s, L2_s, s22_s = Om2[order2], Lv2[order2], s22[order2]
    _, P_noirs = max_min_power(Om2_s, P_HAP)
    Rmc_noirs = mc_evaluate(Om2_s, L2_s, s22_s, P_noirs, S, rng)

    min_prop = float(np.min(Rmc_prop))
    min_oma = float(np.min(Rmc_oma))
    min_noirs = float(np.min(Rmc_noirs))
    return {
        "proposed_mc_min_bps": min_prop,
        "oma_mc_min_bps": min_oma,
        "noirs_mc_min_bps": min_noirs,
        "improve_vs_oma_pct": 100.0 * (min_prop - min_oma) / min_oma,
        "improve_vs_noirs_pct": 100.0 * (min_prop - min_noirs) / min_noirs,
    }


eff = effectiveness(K_DEF_DB, L_DEF, M_DEF, PHAP_DEF_DB, S, seed)
print("\n=== Effectiveness under true MC rate (nominal setting) ===")
print(f"Proposed (NOMA-AIRS) min MC rate : {eff['proposed_mc_min_bps']/1e6:.3f} Mbps")
print(f"OMA-AIRS          min MC rate : {eff['oma_mc_min_bps']/1e6:.3f} Mbps")
print(f"NOMA without IRS  min MC rate : {eff['noirs_mc_min_bps']/1e6:.3f} Mbps")
print(f"Improvement vs OMA-AIRS        : {eff['improve_vs_oma_pct']:.2f}%")
print(f"Improvement vs NOMA without IRS: {eff['improve_vs_noirs_pct']:.2f}%")

with open(os.path.join(out_dir, "effectiveness.json"), "w") as f:
    json.dump(eff, f, indent=2)
print("Effectiveness saved.")
