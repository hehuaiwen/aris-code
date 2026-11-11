"""
Reviewer 1 Comment 2 follow-up: rigorous Fixed vs Dynamic SIC decoding-order
comparison on the scalar Rician model (Appendix A).

  Fixed   : sort users by the statistical gain Omega at the *common* initial UAV
            position Q0, obtain the permutation pi0, and keep pi0 frozen for the
            whole outer BCD loop.
  Dynamic : start from the same pi0; after every outer Q-update the statistical
            gains Omega are recomputed at the new Q, and the permutation is
            re-sorted. A reorder is counted whenever the sorted permutation
            changes; the new order is used in the next outer iteration.

Both rules share the same channel parameters, power budget, topology, initial
UAV position and random seeds; only the ordering-update rule differs. Because
the scalar model assumes coherent phase alignment (no explicit IRS phase
variable), the outer BCD optimizes the UAV position Q and the closed-form
max-min power P only. Each design is finally evaluated with the SIC-feasible
minimum rate over S = 20000 independent channel realizations.

Outputs: revised_results/comment2_fixed_dynamic.json and
         fig_order_fixed_dynamic.pdf.
"""
import os
import json
import time
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scalar_rician_model as R

# ----------------------------------------------------------------------
# Configuration (L = 16 fixed, consistent with the Section 5.5 enumeration)
# ----------------------------------------------------------------------
K_DB = R.K_DEF_DB          # 10 dB (AIRS--GMD Rician factor)
L = R.L_DEF                # 16 (user decision: L = 16 only)
PHAP_DB = R.PHAP_DEF_DB    # 30 dBm
S = R.S                    # 20000 Monte Carlo realizations
BASE_SEED = R.SEED         # 2026
M_LIST = [3, 4]
N_TOPOLOGIES = 20          # random topologies
N_INITS = 3                # common initial UAV positions per topology
N_CRITICAL = 5             # "critical" near-tied-Omega topologies
N_OUTER = 6                # max outer BCD iterations (same for both rules)
N_RESTARTS = 3             # Nelder-Mead multi-starts per Q subproblem

P_HAP = 10.0 ** (PHAP_DB / 10.0)
BOUNDS = np.array([[85.0, 115.0], [-15.0, 15.0], [R.Z_MIN, R.Z_MAX]])


def gen_random_topology(M, rng):
    """M users in the unified deployment region x in [92,109], y in [-2,2]."""
    W_u = np.zeros((M, 3))
    W_u[:, 0] = rng.uniform(92.0, 109.0, M)
    W_u[:, 1] = rng.uniform(-2.0, 2.0, M)
    W_u[:, 2] = 0.0
    return W_u


def centroid_init(W_u):
    cen = np.mean(W_u, axis=0)
    cen[2] = 0.5 * (R.Z_MIN + R.Z_MAX)
    return cen


def random_init(rng):
    return np.array([rng.uniform(85.0, 115.0),
                     rng.uniform(-15.0, 15.0),
                     rng.uniform(R.Z_MIN, R.Z_MAX)])


def jensen_min_rate(Omega, order):
    Om = Omega[np.asarray(order, dtype=int)]
    tau, P = R.max_min_power(Om, P_HAP)
    return float(np.min(R.jensen_rates(Om, P)))


def optimize_Q(Q0, W_u, order, rng):
    """Local max-min-Jensen Q optimization with the given order held fixed."""
    order = np.asarray(order, dtype=int)

    def neg_min(q):
        Omega, _, _ = R.channel_stats_for_users(np.asarray(q, float), W_u, K_DB, L)
        Om = Omega[order]
        tau, P = R.max_min_power(Om, P_HAP)
        return -float(np.min(R.jensen_rates(Om, P)))

    best = None
    for s in range(N_RESTARTS):
        q0 = np.array(Q0, dtype=float).copy()
        if s > 0:
            q0 = np.array([rng.uniform(*b) for b in BOUNDS])
        res = minimize(neg_min, q0, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-6, "fatol": 1e-9})
        q = np.clip(res.x, BOUNDS[:, 0], BOUNDS[:, 1])
        val = neg_min(q)
        if best is None or val < best[1]:
            best = (q, val)
    return best[0]


def run_bcd(W_u, Q0, seed, dynamic):
    rng = np.random.default_rng(seed)
    Q = np.array(Q0, dtype=float).copy()

    Omega0, _, _ = R.channel_stats_for_users(Q, W_u, K_DB, L)
    order = np.argsort(Omega0).copy()          # ascending statistical gain
    obj_hist = [jensen_min_rate(Omega0, order)]  # initial value at Q0
    reorder_flags = []

    n_switch = 0
    t0 = time.time()
    for t in range(N_OUTER):
        Q = optimize_Q(Q, W_u, order, rng)
        Omega, L_vec, s2_vec = R.channel_stats_for_users(Q, W_u, K_DB, L)
        order_new = np.argsort(Omega)
        switched = False
        if dynamic and not np.array_equal(order_new, order):
            switched = True
            n_switch += 1
        if dynamic:
            order = order_new.copy()
        obj_hist.append(jensen_min_rate(Omega, order))
        reorder_flags.append(bool(switched))
    runtime = time.time() - t0

    # Final SIC-feasible Monte Carlo evaluation (S = 20000 realizations).
    rng_mc = np.random.default_rng(seed + 100000)
    gains = R.sample_gains(L_vec, s2_vec, S, rng_mc)
    min_mc, _, _ = R.mc_min_rate_for_order(Omega, gains, P_HAP, order=order)

    return {
        "Q": Q.tolist(),
        "order": order.tolist(),
        "n_switch": int(n_switch),
        "min_mc_bps": float(min_mc),
        "obj_hist": [float(v) for v in obj_hist],
        "reorder_flags": reorder_flags,
        "runtime": float(runtime),
    }


def relative_adjacent_gap(W_u, Q0):
    """Min relative gap between adjacent (sorted) Omega values at Q0."""
    Omega, _, _ = R.channel_stats_for_users(Q0, W_u, K_DB, L)
    o = np.sort(Omega)
    gaps = (o[1:] - o[:-1]) / np.maximum(o[1:], 1e-30)
    return float(np.min(gaps))


def find_critical_topologies(M, n_target, seed, max_attempts=2000):
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(max_attempts):
        W_u = gen_random_topology(M, rng)
        Q0 = centroid_init(W_u)
        if relative_adjacent_gap(W_u, Q0) <= 0.05:
            found.append(W_u)
            if len(found) >= n_target:
                break
    return found


def evaluate_pairs(W_list, inits, seed):
    """Run fixed & dynamic for every (topology, init) pair; return records."""
    records = []
    for ti, W_u in enumerate(W_list):
        for ii, Q0 in enumerate(inits):
            s = seed + 1000 * ti + 11 * ii
            f = run_bcd(W_u, Q0, s, dynamic=False)
            d = run_bcd(W_u, Q0, s, dynamic=True)
            records.append({
                "topo": ti,
                "init": ii,
                "R_fix_bps": f["min_mc_bps"],
                "R_dyn_bps": d["min_mc_bps"],
                "n_switch": d["n_switch"],
                "runtime_fix": f["runtime"],
                "runtime_dyn": d["runtime"],
                "obj_fix": f["obj_hist"],
                "obj_dyn": d["obj_hist"],
                "reorder_flags": d["reorder_flags"],
                "order_fix": f["order"],
                "order_dyn": d["order"],
            })
    return records


def summarize(records, n_topologies, n_inits):
    """Per-topology-averaged statistics with topology as the independent unit."""
    Rf = np.array([r["R_fix_bps"] for r in records])
    Rd = np.array([r["R_dyn_bps"] for r in records])
    D = Rd - Rf
    # per-topology means (3 inits averaged first)
    topo_D = []
    topo_Rf = []
    topo_Rd = []
    n_reorder = 0
    for t in range(n_topologies):
        idx = [k for k, r in enumerate(records) if r["topo"] == t]
        topo_D.append(np.mean([D[k] for k in idx]))
        topo_Rf.append(np.mean([Rf[k] for k in idx]))
        topo_Rd.append(np.mean([Rd[k] for k in idx]))
        if any(records[k]["n_switch"] >= 1 for k in idx):
            n_reorder += 1
    topo_D = np.array(topo_D)
    mean_D = float(np.mean(topo_D))
    std_D = float(np.std(topo_D, ddof=1))
    # 95% CI via t-like 1.96 * std/sqrt(n) (n=20 topologies)
    ci = 1.96 * std_D / np.sqrt(n_topologies)
    return {
        "n_pairs": len(records),
        "mean_R_fix_bps": float(np.mean(Rf)),
        "median_R_fix_bps": float(np.median(Rf)),
        "mean_R_dyn_bps": float(np.mean(Rd)),
        "median_R_dyn_bps": float(np.median(Rd)),
        "mean_diff_bps": float(mean_D),
        "mean_diff_pct": float(100.0 * mean_D / np.mean(Rf)),
        "ci95_diff_bps": float(ci),
        "ci95_lo_bps": float(mean_D - ci),
        "ci95_hi_bps": float(mean_D + ci),
        "n_topologies_reordered": int(n_reorder),
        "reorder_ratio": float(n_reorder / n_topologies),
        "mean_runtime_fix_s": float(np.mean([r["runtime_fix"] for r in records])),
        "mean_runtime_dyn_s": float(np.mean([r["runtime_dyn"] for r in records])),
    }


if __name__ == "__main__":
    out_dir = "revised_results"
    os.makedirs(out_dir, exist_ok=True)

    results = {}

    # ---- random topologies ----
    for M in M_LIST:
        rng = np.random.default_rng(BASE_SEED + 500 + M)
        W_list = [gen_random_topology(M, rng) for _ in range(N_TOPOLOGIES)]
        # 3 common initial positions (deterministic across the topology set)
        init_rng = np.random.default_rng(BASE_SEED + 900 + M)
        inits = [centroid_init(W_list[0])] + [random_init(init_rng) for _ in range(N_INITS - 1)]
        records = evaluate_pairs(W_list, inits, BASE_SEED + 2000 + M)
        results[f"random_M{M}"] = {
            "summary": summarize(records, N_TOPOLOGIES, N_INITS),
            "records": records,
        }
        sm = results[f"random_M{M}"]["summary"]
        print(f"[random M={M}] pairs={sm['n_pairs']} "
              f"R_fix={sm['mean_R_fix_bps']/1e6:.4f} R_dyn={sm['mean_R_dyn_bps']/1e6:.4f} Mbps "
              f"diff={sm['mean_diff_pct']:+.3f}% CI=[{sm['ci95_lo_bps']/1e6:+.4f},"
              f"{sm['ci95_hi_bps']/1e6:+.4f}] reorder_ratio={sm['reorder_ratio']:.2f} "
              f"n_switch_max={max(r['n_switch'] for r in records)}")

    # ---- critical (near-tied Omega) topologies ----
    for M in M_LIST:
        crit = find_critical_topologies(M, N_CRITICAL, BASE_SEED + 7000 + M)
        print(f"[critical M={M}] found {len(crit)} topologies, "
              f"min_gaps=" + ", ".join(f"{relative_adjacent_gap(w, centroid_init(w))*100:.2f}%" for w in crit))
        inits = [centroid_init(crit[0])] + [random_init(np.random.default_rng(BASE_SEED + 8000 + M))
                                            for _ in range(N_INITS - 1)]
        records = evaluate_pairs(crit, inits, BASE_SEED + 9000 + M)
        results[f"critical_M{M}"] = {
            "summary": summarize(records, len(crit), N_INITS),
            "records": records,
            "min_gaps_pct": [relative_adjacent_gap(w, centroid_init(w)) * 100.0 for w in crit],
        }
        sm = results[f"critical_M{M}"]["summary"]
        print(f"[critical M={M}] R_fix={sm['mean_R_fix_bps']/1e6:.4f} "
              f"R_dyn={sm['mean_R_dyn_bps']/1e6:.4f} diff={sm['mean_diff_pct']:+.3f}% "
              f"reorder_ratio={sm['reorder_ratio']:.2f} "
              f"n_switch_max={max(r['n_switch'] for r in records)}")

    with open(os.path.join(out_dir, "comment2_fixed_dynamic.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Figure: (a) paired scatter (Fixed vs Dynamic), (b) representative
    # trajectory with reorder points marked.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))

    # (a) scatter for random topologies, M=3 and M=4
    ax = axes[0]
    colors = {3: "#2ca02c", 4: "#1f77b4"}
    for M in M_LIST:
        recs = results[f"random_M{M}"]["records"]
        x = [r["R_fix_bps"] / 1e6 for r in recs]
        y = [r["R_dyn_bps"] / 1e6 for r in recs]
        ax.scatter(x, y, s=22, color=colors[M], alpha=0.6, label=f"M={M}")
    lims = [min(min([r['R_fix_bps'] for r in results[f'random_M{M}']['records']]) / 1e6
                for M in M_LIST),
            max(max([r['R_dyn_bps'] for r in results[f'random_M{M}']['records']]) / 1e6
                for M in M_LIST)]
    pad = 0.02 * (lims[1] - lims[0])
    ax.plot([lims[0]-pad, lims[1]+pad], [lims[0]-pad, lims[1]+pad], "k--", lw=1, label="Fixed = Dynamic")
    ax.set_xlim(lims[0]-pad, lims[1]+pad)
    ax.set_ylim(lims[0]-pad, lims[1]+pad)
    ax.set_xlabel("Fixed-order min rate (Mbps)")
    ax.set_ylabel("Dynamic-order min rate (Mbps)")
    ax.set_title("(a) Paired results")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) trajectory of a representative scenario that actually reorders
    ax = axes[1]
    best = None
    for key in [f"critical_M{M}" for M in M_LIST]:
        for r in results[key]["records"]:
            if r["n_switch"] >= 1 and (best is None or r["n_switch"] > best["n_switch"]):
                best = r
    if best is not None:
        n = len(best["obj_fix"])
        x = np.arange(n)
        ax.plot(x, np.array(best["obj_fix"]) / 1e6, "o-", color="#1f77b4", label="Fixed")
        ax.plot(x, np.array(best["obj_dyn"]) / 1e6, "s-", color="#2ca02c", label="Dynamic")
        for t, flag in enumerate(best["reorder_flags"]):
            if flag:
                ax.axvline(t + 1, color="r", linestyle=":", lw=1)
        ax.set_xlabel("Outer BCD iteration")
        ax.set_ylabel("Surrogate min rate (Mbps)")
        ax.set_title(f"(b) Trajectory (N_switch={best['n_switch']})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No reordering observed", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)

    fig.tight_layout()
    fig.savefig("fig_order_fixed_dynamic.pdf", dpi=300, bbox_inches="tight")
    print("\nSaved revised_results/comment2_fixed_dynamic.json and fig_order_fixed_dynamic.pdf")
