import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("revised_results/comment6_phase.json") as f:
    results = json.load(f)

def summarize(res):
    b = res["base_bps"]
    return {
        "continuous": 100.0,
        "3bit": 100.0 * res["quant_bps"]["3"] / b,
        "2bit": 100.0 * res["quant_bps"]["2"] / b,
        "1bit": 100.0 * res["quant_bps"]["1"] / b,
        "random": 100.0 * res["random_mean_bps"] / b,
        "random_std": 100.0 * res["random_std_bps"] / b,
    }

labels = ["Continuous", "3-bit", "2-bit", "1-bit", "Random"]
x = np.arange(len(labels))
w = 0.36
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for xoff, L, color in [(-w/2, 8, "#1f77b4"), (w/2, 32, "#d62728")]:
    s = summarize(results[str(L)])
    vals = [s["continuous"], s["3bit"], s["2bit"], s["1bit"], s["random"]]
    ax.bar(x + xoff, vals, w, label=f"$L={L}$", color=color)
    ax.errorbar(x[-1] + xoff, s["random"], yerr=s["random_std"],
                fmt="none", ecolor="k", capsize=3, elinewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Retained min. rate (%)")
ax.set_ylim(60, 103)
ax.legend(fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_title("Phase-control robustness versus $L$")
for xoff, L in [(-w/2, 8), (w/2, 32)]:
    s = summarize(results[str(L)])
    vals = [s["continuous"], s["3bit"], s["2bit"], s["1bit"], s["random"]]
    for xi, v in zip(x + xoff, vals):
        ax.text(xi, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=6)
fig.tight_layout()
fig.savefig("fig_phase_L.pdf", dpi=300, bbox_inches="tight")
print("regenerated fig_phase_L.pdf")
for L in (8, 32):
    s = summarize(results[str(L)])
    print(f"L={L}: 3bit={s['3bit']:.2f} 2bit={s['2bit']:.2f} 1bit={s['1bit']:.2f} "
          f"random={s['random']:.2f}+/-{s['random_std']:.2f}")
