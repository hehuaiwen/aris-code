"""
Reviewer 1 Comment 8 follow-up: sensitivity of the end-to-end MEC latency to
(i) the MEC CPU capacity F_max and (ii) the task input size D_in.

Reuses a single optimized downlink design (L=8, M=4) from the full AR_IRS
model; only the CPU-allocation / task-size parameters are swept, so this is
CPU-only and fast.

Outputs: revised_results/comment8_sweeps.json and fig_latency_sweep.pdf.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vector_airs_model as R


def latency_sweeps(L=8, M=4, S=800, seed=0, out=5, inner=20):
    Q, Theta, P, _ = R.optimize_design(L, M, seed=seed, out=out, inner=inner)
    comps = [R.channel_components(Q, Theta, i, L) for i in range(M)]
    R_dl, _, _ = R.mc_eval(Q, Theta, P, L, M, S, seed, comps=comps)

    c = 200.0          # cycles/bit
    R_ul = 2.0e6       # uplink bits/s (fixed)

    def max_e2e(D_in, F_max):
        D_in = np.asarray(D_in, dtype=float)
        D_out = D_in * 0.5
        C = c * D_in
        b = D_in / R_ul + D_out / R_dl
        f, T = R.delay_aware_cpu(C, b, F_max)
        return float(np.max(T)), float(np.mean(T)), float(T)

    # ---- (i) F_max sweep (heterogeneous profile: one 1.5 Mbit edge user) ----
    D_het = np.array([1.0e6] * (M - 1) + [1.5e6])
    F_sweep = [5.0e9, 10.0e9, 15.0e9, 20.0e9]
    fmax_res = {"F_max_GHz": [], "T_max": [], "T_avg": []}
    for F in F_sweep:
        tmax, tavg, _ = max_e2e(D_het, F)
        fmax_res["F_max_GHz"].append(F / 1e9)
        fmax_res["T_max"].append(tmax)
        fmax_res["T_avg"].append(tavg)

    # ---- (ii) task-size sweep (uniform tasks, F_max = 10 GHz) ----
    D_sweep = [0.5e6, 1.0e6, 1.5e6, 2.0e6]
    dsize_res = {"D_in_Mbit": [], "T_max": [], "T_avg": []}
    for Din in D_sweep:
        tmax, tavg, _ = max_e2e(np.full(M, Din), 10.0e9)
        dsize_res["D_in_Mbit"].append(Din / 1e6)
        dsize_res["T_max"].append(tmax)
        dsize_res["T_avg"].append(tavg)

    # ---- figures ----
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    axes[0].plot(fmax_res["F_max_GHz"], fmax_res["T_max"], "o-", color="#1f77b4",
                 label="Max E2E latency")
    axes[0].plot(fmax_res["F_max_GHz"], fmax_res["T_avg"], "s--", color="#ff7f0e",
                 label="Avg E2E latency")
    axes[0].set_xlabel(r"MEC CPU capacity $F_{\max}$ (GHz)")
    axes[0].set_ylabel("E2E latency (s)")
    axes[0].legend(fontsize=8)
    axes[0].grid(linestyle="--", alpha=0.4)
    axes[0].set_title("(a) CPU-capacity sensitivity")

    axes[1].plot(dsize_res["D_in_Mbit"], dsize_res["T_max"], "o-", color="#1f77b4",
                 label="Max E2E latency")
    axes[1].plot(dsize_res["D_in_Mbit"], dsize_res["T_avg"], "s--", color="#ff7f0e",
                 label="Avg E2E latency")
    axes[1].set_xlabel(r"Task input size $D_i^{\mathrm{in}}$ (Mbit)")
    axes[1].set_ylabel("E2E latency (s)")
    axes[1].legend(fontsize=8)
    axes[1].grid(linestyle="--", alpha=0.4)
    axes[1].set_title("(b) Task-size sensitivity")
    fig.tight_layout()
    fig.savefig("fig_latency_sweep.pdf", dpi=300, bbox_inches="tight")

    return {"F_max_sweep": fmax_res, "task_size_sweep": dsize_res,
            "R_dl_min_bps": float(R_dl.min())}


if __name__ == "__main__":
    out_dir = "revised_results"
    os.makedirs(out_dir, exist_ok=True)
    res = latency_sweeps()
    print("=== F_max sweep (heterogeneous 1.5 Mbit edge) ===")
    for g, tm, ta in zip(res["F_max_sweep"]["F_max_GHz"],
                         res["F_max_sweep"]["T_max"],
                         res["F_max_sweep"]["T_avg"]):
        print(f"  F_max={g:>3.0f} GHz  T_max={tm*1e3:6.2f} ms  T_avg={ta*1e3:6.2f} ms")
    print("=== task-size sweep (uniform, F_max=10 GHz) ===")
    for d, tm, ta in zip(res["task_size_sweep"]["D_in_Mbit"],
                         res["task_size_sweep"]["T_max"],
                         res["task_size_sweep"]["T_avg"]):
        print(f"  D_in={d:>3.1f} Mbit  T_max={tm*1e3:6.2f} ms  T_avg={ta*1e3:6.2f} ms")
    with open(os.path.join(out_dir, "comment8_sweeps.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\nSaved revised_results/comment8_sweeps.json and fig_latency_sweep.pdf")
