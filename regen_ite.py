"""Regenerate the convergence figure (duibi_ite.pdf) from the fresh
new_data/difPHAP_ris_iteL10.csv and difL_ris_ite.csv (regenerated with the
correct units: sigma^2=1e-11 W, W=1 MHz, GAIN_SCALE rescaling).

The raw objective (tau) is in nats/s; we convert to Mbit/s as in plot_all.py.
Output is written directly to ../paper/figs/duibi_ite.pdf.
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "../paper/figs"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

NATS2MBPS = 1.0 / np.log(2) / 1e6  # nats/s -> Mbit/s


def load_numeric_rows(name):
    rows = list(csv.reader(open(f"new_data/{name}.csv")))
    out = []
    for r in rows:
        try:
            out.append(np.array([float(x) for x in r]))
        except ValueError:
            out.append(None)  # bracketed power-allocation row (skipped)
    return out


# difPHAP_ris_iteL10 rows: [P(10), tau(10), P(20), tau(20), P(30), tau(30), iter]
ite = load_numeric_rows("difPHAP_ris_iteL10")
# difL_ris_ite rows: [P(8), tau(8), ..., P(32), tau(32), iter]
dl = load_numeric_rows("difL_ris_ite")

x = dl[-1]  # iteration index 1..max_iter
y1 = ite[1] * NATS2MBPS   # L=10, P_HAP=10 dBm
y2 = ite[5] * NATS2MBPS   # L=10, P_HAP=30 dBm
y3 = dl[13] * NATS2MBPS   # L=32, P_HAP=30 dBm

fig, ax = plt.subplots(figsize=(7.2, 4.8))
styles = [
    dict(color="#0a7ac7", marker="^", linestyle="--"),
    dict(color="#de8a00", marker="o", linestyle="-"),
    dict(color="#bf2c00", marker="*", linestyle="-"),
]
labels = [
    r"$L=10,\ P_{\mathrm{max}}=10$ dBm",
    r"$L=10,\ P_{\mathrm{max}}=30$ dBm",
    r"$L=32,\ P_{\mathrm{max}}=30$ dBm",
]
for y, s, lab in zip((y1, y2, y3), styles, labels):
    ax.plot(x, y, linewidth=2, markersize=8, markevery=1, label=lab, **s)

ax.set_xlabel("Iteration index", fontsize=14)
ax.set_ylabel("Minimum rate (Mbit/s)", fontsize=14)
ax.set_xlim(1, 20)
ax.set_ylim(0, 1.6)
ax.set_xticks(np.arange(0, 21, 4))
ax.set_yticks(np.arange(0, 1.61, 0.2))
ax.grid(True, alpha=0.35, linestyle="--")
ax.legend(fontsize=11, loc="best")
ax.tick_params(labelsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "duibi_ite.pdf"), format="pdf")
plt.close(fig)
print("saved duibi_ite.pdf ->", os.path.abspath(os.path.join(OUT_DIR, "duibi_ite.pdf")))
print("converged rates (Mbit/s):", float(y1[-1]), float(y2[-1]), float(y3[-1]))
