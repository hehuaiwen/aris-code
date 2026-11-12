"""
Render the Reviewer5 comment-10 mobility results as a SINGLE side-by-side
two-panel figure (a) per-slot rates and (b) paired difference, for compact
single-column placement.  Reads the raw per-trajectory per-slot CSV (no
experiment re-run).

Clarity pass: larger serif fonts (10 pt), thicker lines, larger markers, and
600-dpi PNG preview; the PDF/EPS stay vector (Type 42 embedded fonts).

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
OUT = os.path.join(RES_DIR, "fig_mobility_combined")

FIG_SIZE = (7.0, 2.9)   # full single-column width, two panels side by side


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
        "font.size": 10.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.0,
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


def main():
    setup_style()
    N_test, T, D = load()
    slots = np.arange(T)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)

    # ---- (a) per-slot worst-user rate ----
    style = {"adaptive": ("#1f77b4", "AIRS-adaptive", "-o"),
             "fixed": ("#2ca02c", "AIRS-fixed-position", "-s"),
             "girs": ("#8c564b", "GIRS-fixed-position", "-^")}
    for k in style:
        col, lab, mk = style[k]
        m = D[k].mean(axis=0) / 1e6
        per_slot = D[k].T  # (T, N_test)
        lo = np.array([ci95(per_slot[t])[1] for t in range(T)]) / 1e6
        hi = np.array([ci95(per_slot[t])[2] for t in range(T)]) / 1e6
        ax1.plot(slots, m, mk, color=col, label=lab, lw=2.0, ms=5.5)
        ax1.fill_between(slots, lo, hi, color=col, alpha=0.20)
    ax1.set_xlabel("Hotspot-motion slot")
    ax1.set_ylabel("Worst-user SIC-feasible rate (Mbps)")
    ax1.set_xticks(slots)
    ax1.grid(linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8.5, loc="best")
    ax1.set_title("(a) Per-slot rates", loc="left", fontweight="bold")

    # ---- (b) paired per-trajectory difference vs fixed GIRS ----
    traj_mean = {k: D[k].mean(axis=1) for k in D}
    paired = (traj_mean["adaptive"] - traj_mean["girs"]) / 1e6
    x = np.arange(N_test)
    ax2.scatter(x, paired, s=30, color="#1f77b4", zorder=3)
    ax2.axhline(0.0, color="k", lw=1.0, linestyle="--")
    m, lo, hi = ci95(paired)
    ax2.axhline(m, color="#d62728", lw=1.4, label="mean paired diff")
    ax2.axhspan(lo, hi, color="#d62728", alpha=0.16,
                label="95% CI (over trajectories)")
    ax2.set_xlabel("Test trajectory index")
    ax2.set_ylabel("AIRS-adaptive $-$ GIRS-fixed (Mbps)")
    ax2.set_xticks(x)
    ax2.grid(linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8.5, loc="best")
    ax2.set_title("(b) Paired difference", loc="left", fontweight="bold")

    fig.tight_layout()
    for ext in (".pdf", ".eps"):
        fig.savefig(OUT + ext, bbox_inches="tight")
    fig.savefig(OUT + ".png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}.pdf / .eps / .png")

    # verification numbers
    m, lo, hi = ci95(traj_mean["adaptive"] - traj_mean["girs"])
    print(f"adaptive-girs paired diff: {m/1e6:.5f} Mbps "
          f"95% CI [{lo/1e6:.5f}, {hi/1e6:.5f}]")
    m, lo, hi = ci95(traj_mean["adaptive"] - traj_mean["fixed"])
    print(f"adaptive-fixed paired diff: {m/1e6:.5f} Mbps "
          f"95% CI [{lo/1e6:.5f}, {hi/1e6:.5f}]")


if __name__ == "__main__":
    main()
