"""
Reviewer 6 Comment 7 -- sensitivity sweeps with 95% confidence intervals.

Runs the worst-user Monte-Carlo ergodic rate of the proposed AFOA over:
  - the number of users  M in {2,3,4,5,6}  (L=8, P_HAP=30 dBm), and
  - the HAP power budget P_HAP in {10,20,30} dBm (L=8, M=4),
each averaged over N_top independent random user topologies drawn uniformly
from the service area x in [92,109] m, y in [-2,2] m, z=0.

For every topology the design (Q, Theta, P) is obtained with the same
penalty-SCA/SDP solver used elsewhere, and the empirical ergodic worst-user
rate is evaluated with S=1000 Monte Carlo channel realizations. The script
reports the mean, the standard deviation across topologies, the 95% confidence
interval of the mean, and the solver success rate.

Run from the 代码/ directory with:
    DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib ~/venvs/wireless_paper/bin/python ci_sweep.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as st

import runtime_scaling as RS

W_BAND = 1.0e6
SIGMA2 = 1.0e-11
BETA1 = 3.5
BETA2 = 2.0


def channel_components(Q, Theta, i, L, W_k, beta1=BETA1, beta2=BETA2):
    gamma_k_i = (RS.rho0 - RS.kappa1) / (np.linalg.norm(RS.W_s - W_k[i]) ** beta1)
    tau_k_i = L * RS.rho0 * (RS.rho0 - RS.kappa2)
    phi_k = (Q[1] - RS.W_s[1]) / np.linalg.norm(Q - RS.W_s)
    psi_k = (Q[2] - RS.W_s[2]) / np.linalg.norm(Q - RS.W_s)
    phi_ki = (W_k[i][1] - Q[1]) / np.linalg.norm(Q - W_k[i])
    hat_s_ki = np.sqrt(RS.kappa1 / np.linalg.norm(RS.W_s - W_k[i]) ** beta1) * RS.s_ki.reshape(-1, 1)
    g_bar_k = np.exp(-1j * (2 * np.pi * RS.d / RS.wavelength) * np.arange(L) * phi_k).reshape(-1, 1)
    g_title_k = np.exp(-1j * (2 * np.pi * RS.d / RS.wavelength) * np.arange(RS.N) * psi_k).reshape(1, -1)
    hat_h_k_i = np.sqrt(RS.kappa2 / np.linalg.norm(Q - W_k[i]) ** beta2) * np.exp(
        -1j * (2 * np.pi * RS.d / RS.wavelength) * np.arange(L) * phi_ki).reshape(-1, 1)
    G_k = np.sqrt(RS.rho0 / np.linalg.norm(Q - RS.W_s) ** 2) * g_bar_k * g_title_k
    gain = np.conj(hat_s_ki.T) + np.conj(hat_h_k_i.T) @ Theta @ G_k
    LoS = float(np.linalg.norm(gain) ** 2)
    NLoS = float(gamma_k_i + tau_k_i / (
        (np.linalg.norm(Q - W_k[i]) ** beta2) * (np.linalg.norm(Q - RS.W_s) ** 2)))
    return LoS, NLoS


def sample_gain(LoS, NLoS, rng):
    n = rng.normal(0.0, np.sqrt(NLoS / 2.0), 2)
    return np.abs(np.sqrt(LoS) + (n[0] + 1j * n[1])) ** 2


def sic_feasible_rates(g, P, M):
    rates = np.zeros(M)
    for i in range(M):
        inter = float(np.sum(P[i + 1:M]))
        rates[i] = min(W_BAND * np.log2(1 + g[k] * P[i] / (g[k] * inter + SIGMA2))
                       for k in range(i, M))
    return rates


def mc_min_rate(Q, Theta, P, L, M, S, seed, W_k):
    comps = [channel_components(Q, Theta, i, L, W_k) for i in range(M)]
    rng = np.random.default_rng(seed)
    mn = np.empty(S)
    for s in range(S):
        g = np.array([sample_gain(c[0], c[1], rng) for c in comps])
        mn[s] = sic_feasible_rates(g, P, M).min()
    return float(mn.mean())


def random_topology(M, rng):
    x = rng.uniform(92.0, 109.0, M)
    y = rng.uniform(-2.0, 2.0, M)
    return np.column_stack([x, y, np.zeros(M)])


def design_min_rate(L, M, P_HAP_dbm, seed, S=1000):
    """One topology: solve + MC-evaluate; return min rate (bps) or None on failure."""
    rng = np.random.default_rng(seed)
    topo = random_topology(M, rng)
    RS.positions = topo  # monkey-patch module-level positions used by AR_IRS
    try:
        irs = RS.AR_IRS()
        irs.bcd_optimization(max_iter=5, L=L, W=W_BAND, M=M, P_HAP=P_HAP_dbm,
                             seed=seed, inner_max=20)
        return mc_min_rate(irs.Q, irs.Theta, irs.P, L, M, S, seed, topo)
    except Exception as e:
        print(f"  [fail] L={L} M={M} P={P_HAP_dbm} seed={seed}: {e}", flush=True)
        return None


def sweep(name, points, L, M_fn, P_fn, N_top=5, S=1000):
    out = {"name": name, "points": list(points), "L": L, "N_top": N_top,
           "mean_bps": [], "std_bps": [], "ci95_bps": [], "success_rate": []}
    for pt in points:
        vals = []
        ok = 0
        for t in range(N_top):
            seed = 1000 + t
            v = design_min_rate(L, M_fn(pt), P_fn(pt), seed, S)
            if v is not None:
                vals.append(v)
                ok += 1
        vals = np.array(vals)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        tcrit = st.t.ppf(0.975, len(vals) - 1) if len(vals) > 1 else 0.0
        ci = float(tcrit * std / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        out["mean_bps"].append(mean)
        out["std_bps"].append(std)
        out["ci95_bps"].append(ci)
        out["success_rate"].append(ok / N_top)
        print(f"  {name} point={pt}: mean={mean/1e3:.2f} kbps  "
              f"CI95=+/-{ci/1e3:.2f} kbps  succ={ok}/{N_top}", flush=True)
    return out


if __name__ == "__main__":
    N_TOP = 5
    S = 1000
    print("=== difM: worst-user rate vs number of users (L=8, P=30 dBm) ===", flush=True)
    difM = sweep("difM", [2, 3, 4, 5, 6], L=8,
                 M_fn=lambda pt: pt, P_fn=lambda pt: 30.0, N_top=N_TOP, S=S)

    print("=== difP: worst-user rate vs HAP power (L=8, M=4) ===", flush=True)
    difP = sweep("difP", [10, 20, 30], L=8,
                 M_fn=lambda pt: 4, P_fn=lambda pt: float(pt), N_top=N_TOP, S=S)

    os.makedirs("revised_results", exist_ok=True)
    with open("revised_results/ci_sweep_results.json", "w") as f:
        json.dump({"difM": difM, "difP": difP}, f, indent=2)

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, d in zip(axes, [difM, difP]):
        x = np.array(d["points"], dtype=float)
        mean = np.array(d["mean_bps"]) / 1e6
        ci = np.array(d["ci95_bps"]) / 1e6
        ax.errorbar(x, mean, yerr=ci, marker="o", capsize=3, color="#1f77b4")
        ax.set_xlabel("Number of users $M$" if d["name"] == "difM"
                      else "HAP power $P_{\\rm HAP}$ (dBm)")
        ax.set_ylabel("Worst-user ergodic rate (Mbps)")
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_title("(a) vs $M$" if d["name"] == "difM" else "(b) vs $P_{\\rm HAP}$")
    fig.tight_layout()
    fig.savefig("fig_ci_sweep.pdf", dpi=300, bbox_inches="tight")
    print("saved revised_results/ci_sweep_results.json and fig_ci_sweep.pdf")
