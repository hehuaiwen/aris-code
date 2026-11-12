"""
Re-render the Reviewer5 comment-10 mobility figures as two SEPARATE figures,
keeping the original two-figure layout and style (titles, legends, CI shading,
colors, markers) unchanged, only adjusting the aspect ratio for clearer paper
placement.  Reads the raw per-trajectory per-slot CSV (no experiment re-run).

Outputs (vector): fig1_slot_rates.pdf / .eps and fig2_paired_diff.pdf / .eps,
plus PNG previews.  Fonts are embedded (Type 42) and set to a serif (Times)
family for IEEE compatibility.

Confidence intervals treat trajectories as the independent statistical unit
(t-distribution, df = N_test - 1), matching the experiment script.
"""
import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as _tdist

RES_DIR = "reviewer5_mobility_results/full"

# IEEE single-column width (~3.5 in), taller than the original 6.4x3.6 so the
# curves / legend / CI shading are not horizontally stretched.
FIG1_SIZE = (3.5, 2.9)
FIG2_SIZE = (3.5, 2.7)


def ci95(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(x.mean())
    if n < 2:
        return m, m, m
    se = float(x.std(ddof=1) / np.sqrt(n))
    h = float(se * _tdist.ppf(0.975, n - 1))
    return m, m - h, m + h


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "font.size": 9.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load():
    with open(os.path.join(RES_DIR, "summary.json")) as f:
        summary = json.load(f)
    cfg = summary["config"]
    N_test, T = cfg["N_test"], cfg["T"]
    rows = list(csv.DictReader(
        open(os.path.join(RES_DIR, "per_traj_slot.csv"))))
    keys = ["adaptive", "fixed", "girs"]
    D = {k: np.array([float(r[f"r_{k}"]) for r in rows]).reshape(N_test, T)
         for k in keys}
    return N_test, T, D


def fig1(N_test, T, D):
    fig, ax = plt.subplots(figsize=FIG1_SIZE)
    slots = np.arange(T)
    style = {"adaptive": ("#1f77b4", "AIRS-adaptive", "-o"),
             "fixed": ("#2ca02c", "AIRS-fixed-position", "-s"),
             "girs": ("#8c564b", "GIRS-fixed-position", "-^")}
    for k in style:
        col, lab, mk = style[k]
        m = D[k].mean(axis=0) / 1e6
        per_slot = D[k].T  # (T, N_test)
        lo = np.array([ci95(per_slot[t])[1] for t in range(T)]) / 1e6
        hi = np.array([ci95(per_slot[t])[2] for t in range(T)]) / 1e6
        ax.plot(slots, m, mk, color=col, label=lab, lw=1.6, ms=4)
        ax.fill_between(slots, lo, hi, color=col, alpha=0.18)
    ax.set_xlabel("Hotspot-motion slot")
    ax.set_ylabel("Worst-user SIC-feasible rate (Mbps)")
    ax.set_xticks(slots)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    ax.set_title("Moving hotspot: scheme rates (mean ± 95% CI over trajectories)")
    fig.tight_layout()
    return fig


def fig2(N_test, T, D):
    fig, ax = plt.subplots(figsize=FIG2_SIZE)
    traj_mean = {k: D[k].mean(axis=1) for k in D}
    paired = (traj_mean["adaptive"] - traj_mean["girs"]) / 1e6
    x = np.arange(N_test)
    ax.scatter(x, paired, s=22, color="#1f77b4", zorder=3)
    ax.axhline(0.0, color="k", lw=0.9, linestyle="--")
    m, lo, hi = ci95(paired)
    ax.axhline(m, color="#d62728", lw=1.2, label="mean paired diff")
    ax.axhspan(lo, hi, color="#d62728", alpha=0.15,
               label="95% CI (over trajectories)")
    ax.set_xlabel("Test trajectory index")
    ax.set_ylabel("AIRS-adaptive − GIRS-fixed (Mbps)")
    ax.set_xticks(x)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    ax.set_title("Paired per-trajectory rate difference vs fixed GIRS")
    fig.tight_layout()
    return fig


def main():
    setup_style()
    N_test, T, D = load()

    for name, maker in (("fig1_slot_rates", fig1),
                        ("fig2_paired_diff", fig2)):
        fig = maker(N_test, T, D)
        base = os.path.join(RES_DIR, name)
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".eps", bbox_inches="tight")
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {base}.pdf / .eps / .png")

    # verification numbers
    overall = {k: float(D[k].mean()) for k in D}
    p = ci95(D["adaptive"].mean(axis=1) - D["girs"].mean(axis=1))
    print("=== verification ===")
    for k in overall:
        print(f"  {k:12s} overall mean = {overall[k]/1e6:.4f} Mbps")
    print(f"  adaptive - girs paired diff: mean={p[0]/1e6:.4f} Mbps "
          f"95% CI [{p[1]/1e6:.4f}, {p[2]/1e6:.4f}]")


if __name__ == "__main__":
    main()
