"""Plot the four fast baseline schemes (no-IRS NOMA/OMA + GIRS NOMA/OMA).

Reads fresh new_data/*.csv (regenerated with sigmma=1e-11, W=1e6) and draws the
five comparison figures with only the baselines, so the AIRS (AR-IRS) curves can
be added later once the optimize_Q SCA issue is fixed. The y-axis is converted
from nats/s (raw tau) to Mbit/s.

Output directory: ./baseline_figs/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from new_func.myfunc import readdata

OUT = "./baseline_figs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

# Subset of the 6-scheme legend (baselines only), keeping the original colors.
LABELS = {
    "no_irs_na": "NOMA without IRS",
    "no_irs_if": "OMA without IRS",
    "fix_irs": "NOMA with GIRS",
    "fix_irs_if": "OMA with GIRS",
}
STYLE = {
    "no_irs_na": dict(marker="p", color="#bf00a8", linestyle="--"),
    "no_irs_if": dict(marker="s", color="#1aa600", linestyle="-"),
    "fix_irs": dict(marker="*", color="#bf2c00", linestyle="-"),
    "fix_irs_if": dict(marker="+", color="#bf9300", linestyle="-"),
}
ORDER = ["no_irs_na", "no_irs_if", "fix_irs", "fix_irs_if"]

NATS2MBPS = 1.0 / np.log(2) / 1e6   # nats/s -> Mbit/s


def read_one(base):
    d = readdata(base)
    x = np.array(d[1], dtype=float)
    y = np.array(d[0], dtype=float) * NATS2MBPS
    return x, y


def line_fig(bases, fname, xlabel):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for b in ORDER:
        if b not in bases:
            continue
        x, y = read_one(bases[b])
        ax.plot(x, y, linewidth=2, markersize=8, label=LABELS[b], **STYLE[b])
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Minimum rate (Mbit/s)", fontsize=14)
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.legend(fontsize=10, loc="best")
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), format="pdf")
    plt.close(fig)
    print("saved", fname)


sw = ["no_irs_na", "no_irs_if", "fix_irs", "fix_irs_if"]

line_fig({b: f"difM_{b}" for b in sw}, "baseline_difM.pdf",
         "Number of users $M$")
line_fig({b: f"difL_{b}" for b in sw}, "baseline_difL.pdf",
         "Number of IRS elements $L$")
line_fig({b: f"difP_{b}" for b in sw}, "baseline_difP.pdf",
         "HAP max power $P_{HAP}$ (dBm)")
line_fig({b: f"difBeta_{b}" for b in sw}, "baseline_difBeta.pdf",
         r"Path loss exponents $\beta_1,\beta_2$")
line_fig({b: f"difW_{b}" for b in sw}, "baseline_difW.pdf",
         "Bandwidth $W$ (MHz)")

print("DONE ->", os.path.abspath(OUT))
