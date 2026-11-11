"""Plot the six comparison figures (difM/difL/difP/difBeta/difW) from the
regenerated new_data/*.csv files (sigmma=1e-11, W=1e6, GAIN_SCALE rescaling).

Y-axis is converted from nats/s (raw tau) to Mbit/s. Output: ./main_figs/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from new_func.myfunc import readdata

OUT = "./main_figs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

LABELS = {
    "irs": "NOMA with AR-IRS",
    "irs_if": "OMA with AR-IRS",
    "no_irs_na": "NOMA without IRS",
    "no_irs_if": "OMA without IRS",
    "fix_irs": "NOMA with GR-IRS",
    "fix_irs_if": "OMA with GR-IRS",
}
STYLE = {
    "irs": dict(marker="^", color="#0a7ac7", linestyle="--"),
    "irs_if": dict(marker="o", color="#de8a00", linestyle="-"),
    "no_irs_na": dict(marker="p", color="#bf00a8", linestyle="--"),
    "no_irs_if": dict(marker="s", color="#1aa600", linestyle="-"),
    "fix_irs": dict(marker="*", color="#bf2c00", linestyle="-"),
    "fix_irs_if": dict(marker="+", color="#bf9300", linestyle="-"),
}
ORDER = ["irs", "irs_if", "no_irs_na", "no_irs_if", "fix_irs", "fix_irs_if"]

NATS2MBPS = 1.0 / np.log(2) / 1e6   # nats/s -> Mbit/s


def line_fig(prefix, fname, xlabel):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    handles = []
    for b in ORDER:
        d = readdata(f"{prefix}_{b}")
        x = np.array(d[1], dtype=float)
        y = np.array(d[0], dtype=float) * NATS2MBPS
        ax.plot(x, y, linewidth=2, markersize=8, **STYLE[b])
        handles.append(Line2D([], [], linewidth=2, markersize=8,
                              label=LABELS[b], **STYLE[b]))
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Minimum rate (Mbit/s)", fontsize=14)
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.legend(handles=handles, fontsize=10, loc="best", handlelength=3.5)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), format="pdf")
    plt.close(fig)
    print("saved", fname)


line_fig("difM", "difM.pdf", "Number of users $M$")
line_fig("difL", "difL.pdf", "Number of IRS elements $L$")
line_fig("difP", "difP.pdf", "HAP max power $P_{HAP}$ (dBm)")
line_fig("difBeta", "difBeta.pdf", r"Path loss exponents $\beta_1,\beta_2$")
line_fig("difW", "difW.pdf", "Bandwidth $W$ (MHz)")

print("DONE ->", os.path.abspath(OUT))
