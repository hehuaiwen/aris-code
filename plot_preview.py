"""Preview the regenerated main results into a separate folder for review.

Reads the new_data/*.csv files (Sep 18 regeneration) and draws the six main
comparison figures + the convergence figure with auto-scaled y-axes, so the
advisor can inspect the honest values before deciding what to keep.

Output directory: ./preview_figs/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from new_func.myfunc import readdata

OUT = "./preview_figs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

LABELS = [
    "NOMA with AR-IRS",
    "OMA with AR-IRS",
    "NOMA without IRS",
    "OMA without IRS",
    "NOMA with GR-IRS",
    "OMA with GR-IRS",
]
MARKERS = ["^", "o", "p", "s", "*", "+"]
COLORS = ["#0a7ac7", "#de8a00", "#bf00a8", "#1aa600", "#bf2c00", "#bf9300"]
LINESTYLES = ["--", "-", "--", "-", "--", "-"]


def read_sweep(files):
    """files: list of 6 csv basenames. returns (x, [y0..y5])."""
    y = []
    x = None
    for f in files:
        d = readdata(f)          # shape (2, N): [y, x]
        y.append(d[0])
        if x is None:
            x = d[-1]
    return np.array(x, dtype=float), y


def read_ite(name):
    """Parse the mixed-format ite csv. returns (iter_idx, [minrate arrays])."""
    with open("./new_data/" + name + ".csv") as f:
        lines = [l.strip() for l in f if l.strip()]
    non_bracket = [l for l in lines if "[" not in l]
    idx = np.array([float(v) for v in non_bracket[-1].split(",")])
    minrates = [np.array([float(v) for v in l.split(",")]) for l in non_bracket[:-1]]
    return idx, minrates


def line_fig(x, ys, fname, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for i, y in enumerate(ys):
        ax.plot(x, y, marker=MARKERS[i], color=COLORS[i], linestyle=LINESTYLES[i],
                linewidth=2, markersize=8, label=LABELS[i])
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.legend(fontsize=10, loc="best")
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), format="pdf")
    plt.close(fig)
    print("saved", fname)


# --- difM ---
x, ys = read_sweep(["difM_irs", "difM_irs_if", "difM_no_irs_na",
                    "difM_no_irs_if", "difM_fix_irs", "difM_fix_irs_if"])
line_fig(x, ys, "preview_difM_RATE.pdf", "Number of users $M$", "Minimum rate (bit/s)")

# --- difL ---
x, ys = read_sweep(["difL_irs", "difL_irs_if", "difL_no_irs_na",
                    "difL_no_irs_if", "difL_fix_irs", "difL_fix_irs_if"])
line_fig(x, ys, "preview_difL_RATE.pdf", "Number of IRS elements $L$", "Minimum rate (bit/s)")

# --- difP ---
x, ys = read_sweep(["difP_irs", "difP_irs_if", "difP_no_irs_na",
                    "difP_no_irs_if", "difP_fix_irs", "difP_fix_irs_if"])
line_fig(x, ys, "preview_difP_RATE.pdf", "HAP max power $P_{HAP}$ (dBm)", "Minimum rate (bit/s)")

# --- difBeta ---
x, ys = read_sweep(["difBet_irs", "difBeta_irs_if", "difBeta_no_irs_na",
                    "difBeta_no_irs_if", "difBeta_fix_irs", "difBeta_fix_irs_if"])
line_fig(x, ys, "preview_difBeta_RATE.pdf", r"Path loss exponents $\beta_1,\beta_2$", "Minimum rate (bit/s)")

# --- difW ---
x, ys = read_sweep(["difW_irs", "difW_irs_if", "difW_no_irs_na",
                    "difW_no_irs_if", "difW_fix_irs", "difW_fix_irs_if"])
line_fig(x, ys, "preview_difW_RATE.pdf", "Bandwidth $W$ (Hz)", "Minimum rate (bit/s)")

# --- convergence (ite) ---
_, l10 = read_ite("difPHAP_ris_iteL10")   # L=10, P in [10,20,30]
_, l32 = read_ite("difPHAP_ris_ite")      # L=32, P in [10,20,30]
_, lSweep = read_ite("difL_ris_ite")      # L in [8..32], P=30
idx = np.arange(1, len(l10[0]) + 1)
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.plot(idx, l10[0], marker="^", color=COLORS[0], linestyle="--", linewidth=2,
        markersize=8, label="L=10, $P_{HAP}=10$ dBm")
ax.plot(idx, l10[2], marker="o", color=COLORS[1], linestyle="-", linewidth=2,
        markersize=8, label="L=10, $P_{HAP}=30$ dBm")
ax.plot(idx, lSweep[6], marker="p", color=COLORS[2], linestyle="-", linewidth=2,
        markersize=8, label="L=32, $P_{HAP}=30$ dBm")
ax.set_xlabel("Iteration index", fontsize=14)
ax.set_ylabel("Minimum rate (bit/s)", fontsize=14)
ax.grid(True, alpha=0.35, linestyle="--")
ax.legend(fontsize=10, loc="best")
ax.tick_params(labelsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "preview_ite.pdf"), format="pdf")
plt.close(fig)
print("saved preview_ite.pdf")

print("DONE ->", os.path.abspath(OUT))
