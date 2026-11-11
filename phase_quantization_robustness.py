"""
Reviewer 1 Comment 6 follow-up (review item 4): phase-quantization vs random-phase.

The reviewer noted that 1-bit quantization and a random-phase baseline retained
nearly identical performance at L=8 (96.4% vs 95.7%), suggesting the phase
optimization contributes little at small L. This script re-evaluates the phase
robustness at BOTH L=8 and L=32, averaging the random-phase baseline over 100
independent phase vectors and reporting mean +- std, so that the L-dependence of
the quantization loss can be quantified.

Reuses optimize_design / channel_components / mc_eval from
vector_airs_model. The dominant cost is the L=32 design (~330 s);
everything else is Monte Carlo evaluation only.

Outputs: revised_results/comment6_phase.json and fig_phase_L.pdf.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vector_airs_model as R


def phase_sweep(L, M=4, S=5000, seed=0, out=5, inner=30, N_rand=100):
    Q, Theta, P, _ = R.optimize_design(L, M, seed=seed, out=out, inner=inner)
    comps = [R.channel_components(Q, Theta, i, L) for i in range(M)]
    base = R.mc_eval(Q, Theta, P, L, M, S, seed, comps=comps)[1]

    # ---- discrete phase control ----
    psi = np.angle(np.diagonal(Theta))
    quant = {}
    for b in (3, 2, 1):
        levels = 2 ** b
        psi_q = np.round(psi / (2 * np.pi / levels)) * (2 * np.pi / levels)
        Theta_q = np.diag(np.exp(1j * psi_q))
        comps_q = [R.channel_components(Q, Theta_q, i, L) for i in range(M)]
        quant[b] = R.mc_eval(Q, Theta_q, P, L, M, S, seed, comps=comps_q)[1]

    # ---- random phase, averaged over N_rand independent vectors ----
    rng = np.random.default_rng(seed + 12345)
    rand_vals = np.empty(N_rand)
    for k in range(N_rand):
        Theta_r = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, L)))
        comps_r = [R.channel_components(Q, Theta_r, i, L) for i in range(M)]
        rand_vals[k] = R.mc_eval(Q, Theta_r, P, L, M, S, seed, comps=comps_r)[1]

    return {
        "L": L,
        "M": M,
        "S": S,
        "N_rand": N_rand,
        "base_bps": float(base),
        "quant_bps": {str(b): float(quant[b]) for b in (3, 2, 1)},
        "random_mean_bps": float(rand_vals.mean()),
        "random_std_bps": float(rand_vals.std()),
    }


def summarize(res):
    b = res["base_bps"]
    out = {
        "L": res["L"],
        "continuous_retention_pct": 100.0,
        "3bit_retention_pct": 100.0 * res["quant_bps"]["3"] / b,
        "2bit_retention_pct": 100.0 * res["quant_bps"]["2"] / b,
        "1bit_retention_pct": 100.0 * res["quant_bps"]["1"] / b,
        "random_retention_pct": 100.0 * res["random_mean_bps"] / b,
        "random_retention_std_pct": 100.0 * res["random_std_bps"] / b,
        "gap_1bit_vs_random_pct": 100.0 * (res["quant_bps"]["1"] - res["random_mean_bps"]) / b,
    }
    return out


if __name__ == "__main__":
    out_dir = "revised_results"
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for L in (8, 32):
        print(f"=== L={L} : optimize_design + phase sweep ===", flush=True)
        r = phase_sweep(L)
        results[str(L)] = r
        s = summarize(r)
        print(f"  L={L}: base={r['base_bps']/1e6:.4f} Mbps")
        print(f"    3-bit={s['3bit_retention_pct']:.2f}%  "
              f"2-bit={s['2bit_retention_pct']:.2f}%  "
              f"1-bit={s['1bit_retention_pct']:.2f}%")
        print(f"    random={s['random_retention_pct']:.2f}% +/- "
              f"{s['random_retention_std_pct']:.2f}%  "
              f"(1-bit vs random gap={s['gap_1bit_vs_random_pct']:.2f}%)")

    with open(os.path.join(out_dir, "comment6_phase.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- figure: retention vs phase resolution for L=8 and L=32 ----
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    labels = ["Continuous", "3-bit", "2-bit", "1-bit", "Random"]
    x = np.arange(len(labels))
    w = 0.36
    for xoff, L, color in [(-w / 2, 8, "#1f77b4"), (w / 2, 32, "#d62728")]:
        s = summarize(results[str(L)])
        vals = [s["continuous_retention_pct"], s["3bit_retention_pct"],
                s["2bit_retention_pct"], s["1bit_retention_pct"],
                s["random_retention_pct"]]
        ax.bar(x + xoff, vals, w, label=f"$L={L}$", color=color)
        # error bar on the random baseline only
        ax.errorbar(x[-1] + xoff, s["random_retention_pct"],
                    yerr=s["random_retention_std_pct"], fmt="none",
                    ecolor="k", capsize=3, elinewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Retained min. rate (%)")
    ax.set_ylim(65, 102)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title("Phase-control robustness versus $L$")
    fig.tight_layout()
    fig.savefig("fig_phase_L.pdf", dpi=300, bbox_inches="tight")
    print("\nSaved revised_results/comment6_phase.json and fig_phase_L.pdf")
