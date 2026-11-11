"""Regenerate fig_order_fixed_dynamic.pdf from comment2_fixed_dynamic.json.

Panel (a): paired scatter of fixed-order vs dynamic-order minimum rates
(the two rules coincide, so all points lie on the diagonal).
Panel (b): surrogate objective trajectory of one representative pair, showing
that the fixed and dynamic optimization trajectories coincide (N_switch = 0).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("revised_results/comment2_fixed_dynamic.json"))

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))

# ---- (a) paired scatter ----
ax = axes[0]
colors = {3: "#2ca02c", 4: "#1f77b4"}
allx, ally = [], []
for M in [3, 4]:
    recs = d[f"random_M{M}"]["records"]
    x = [r["R_fix_bps"] / 1e6 for r in recs]
    y = [r["R_dyn_bps"] / 1e6 for r in recs]
    allx += x; ally += y
    ax.scatter(x, y, s=18, color=colors[M], alpha=0.55, label=f"M={M}")
lo = min(allx + ally); hi = max(allx + ally)
pad = 0.03 * (hi - lo)
ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="Fixed = Dynamic")
ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
ax.set_xlabel("Fixed-order min rate (Mbps)")
ax.set_ylabel("Dynamic-order min rate (Mbps)")
ax.set_title("(a) Paired results (all on diagonal)")
ax.legend(fontsize=7, loc="lower right")
ax.grid(alpha=0.3)

# ---- (b) representative trajectory ----
ax = axes[1]
rep = d["critical_M4"]["records"][0]
obj_f = np.array(rep["obj_fix"]) / 1e6
obj_d = np.array(rep["obj_dyn"]) / 1e6
x = np.arange(len(obj_f))
ax.plot(x, obj_f, "o-", color="#1f77b4", label="Fixed")
ax.plot(x, obj_d, "s--", color="#2ca02c", label="Dynamic")
ax.set_xlabel("Outer BCD iteration")
ax.set_ylabel("Surrogate min rate (Mbps)")
ax.set_title(f"(b) Trajectory ($N_\\mathrm{{switch}}=0$)")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("fig_order_fixed_dynamic.pdf", dpi=300, bbox_inches="tight")
print("saved fig_order_fixed_dynamic.pdf")
