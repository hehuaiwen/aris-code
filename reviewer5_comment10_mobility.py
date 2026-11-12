"""
Reviewer 5 (2nd round) Comment 10 -- moving-user-hotspot AIRS vs fixed GIRS.

Supplementary simulation requested by the reviewer.  It compares, under a
moving user hotspot, three deployments that all reuse the paper's existing
full vector-channel model, AFOA (AR_IRS penalty-SCA / SDP / power) optimizer,
grid-searched GIRS and SIC-feasible rate:

  1. AIRS-adaptive        : per-slot re-optimize UAV position, IRS phase, power.
  2. AIRS-fixed-position  : optimize UAV position on slot 0 only; afterwards
                            fix that position and re-optimize phase + power.
  3. GIRS-fixed-position  : GIRS mount point is fixed for the whole trajectory
                            (chosen on training trajectories) while phase and
                            power are re-optimized per slot.

IMPORTANT research boundaries (per the reviewer's brief):

  * The metric is the paper's CURRENT norm-based benchmark SIC-feasible
    worst-user rate (r[s,t] = min_i mean_n R[i,s,t,n]); it is NOT the fixed
    common-beam projected physical link of Eq. (9).  Do not present it as a
    validated fixed-beam physical rate.
  * No explicit building blockage or flight dynamics are modelled.  Do not
    call the position changes a "blockage experiment" and do not claim flight
    time / energy / real-time deployability are considered.
  * Results are reported as-is.  If AIRS does not beat GIRS, report that and
    recommend narrowing the paper's claims.

Reproducibility: fixed seeds, explicit config, saved raw per-trajectory
per-slot results + summary + figures.  Nothing in the paper or the existing
result files is overwritten; outputs go to a new directory.

Reuses (unchanged) `vector_airs_model.py` (the full vector AIRS model)
and `runtime_scaling.py` (the AR_IRS BCD optimizer).  The optimizer hardcodes
the global `runtime_scaling.positions[:M]` as the user positions; to evaluate a
moving hotspot we temporarily overwrite that slice, then evaluate with the
user positions passed explicitly through `channel_components(..., W_k=...)`
so the Monte Carlo metric never depends on global state.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as _tdist

import vector_airs_model as R
import runtime_scaling as RS


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
class Config:
    # system (matches Section 5.13 / full vector model)
    M = 4
    L = 8

    # optimizer protocol (outer BCD iterations, inner SCA iterations)
    out = 5
    inner = 20
    opt_seed = 0         # seed for the deterministic BCD optimizer (same for
                         # every scheme/slot -> identical Theta/P warm starts)
    eval_seed = 12345    # seed for the Monte Carlo evaluation (identical
                         # external channel samples across schemes/slots)

    # hotspot motion (moving hotspot centre; users = centre + fixed offsets)
    center_x_range = (90.0, 110.0)   # centre x, keeps all users inside [85,115]
    center_y_range = (-13.0, 13.0)   # centre y, keeps all users inside [-15,15]

    # GIRS candidate mount points (ground surface z=10 m, expanded to cover the
    # whole moving service area, not just the initial user cluster)
    girs_x = (90.0, 100.0, 110.0)
    girs_y = (-10.0, 0.0, 10.0)
    girs_z = 10.0

    # experiment scale -- smoke vs full (see CLI)
    N_train = 8
    N_test = 12
    T = 8
    S = 2000

    # trajectory seed (fixed for reproducibility)
    traj_seed = 20261112


def make_modes():
    smoke = dict(N_train=3, N_test=3, T=6, S=500)
    full = dict(N_train=8, N_test=12, T=8, S=2000)
    return {"smoke": smoke, "full": full}


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def user_offsets():
    """Fixed relative-position template (users 0..M-1 around their centroid).

    Derived from the paper's default full-vector user positions; the template
    is held constant for every trajectory/slot -- only the common hotspot
    centre moves.
    """
    base = R.positions[:Config.M].astype(float)
    centroid = base.mean(axis=0)
    return base - centroid


def hotspot_centre_path(rng, T, cfg):
    """One trajectory: hotspot centre does a deterministic out-and-back move.

    A random start point c0 and a random turning point c1 are drawn (seeded);
    the centre travels linearly c0 -> c1 -> c0 (triangular in the slot index),
    so every user stays inside the legal service area for the whole trajectory.
    """
    c0 = np.array([rng.uniform(*cfg.center_x_range),
                   rng.uniform(*cfg.center_y_range), 0.0])
    c1 = np.array([rng.uniform(*cfg.center_x_range),
                   rng.uniform(*cfg.center_y_range), 0.0])
    centres = []
    for t in range(T):
        frac = t / (T - 1) if T > 1 else 0.0
        tri = 1.0 - abs(2.0 * frac - 1.0)   # 0 -> 1 -> 0
        centres.append(c0 + tri * (c1 - c0))
    return np.array(centres)                # (T, 3)


def set_users(W):
    """Overwrite the optimizer's global user slice (hack: the AR_IRS BCD reads
    runtime_scaling.positions[:M] directly and offers no per-call W_k)."""
    RS.positions[:Config.M] = W


# ----------------------------------------------------------------------
# Optimizer / evaluator wrappers
# ----------------------------------------------------------------------
def optimize(cfg, opt_Q, Q_init=None):
    Q, Theta, P, _ = R.optimize_design(
        cfg.L, cfg.M, seed=cfg.opt_seed, out=cfg.out, inner=cfg.inner,
        opt_Q=opt_Q, Q_init=Q_init)
    return Q, Theta, P


def evaluate(cfg, Q, Theta, P, W_now):
    comps = [R.channel_components(Q, Theta, i, cfg.L, W_k=W_now)
             for i in range(cfg.M)]
    _, mn, _ = R.mc_eval(Q, Theta, P, cfg.L, cfg.M, cfg.S, cfg.eval_seed,
                          comps=comps)
    return float(mn)


# ----------------------------------------------------------------------
# Confidence interval (independent unit = trajectory)
# ----------------------------------------------------------------------
def ci95(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(x.mean())
    if n < 2:
        return m, m, m
    se = float(x.std(ddof=1) / np.sqrt(n))
    h = float(se * _tdist.ppf(0.975, n - 1))
    return m, m - h, m + h


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------
def run(mode="smoke", out_dir=None):
    cfg = Config()
    for k, v in make_modes()[mode].items():
        setattr(cfg, k, v)

    rng = np.random.default_rng(cfg.traj_seed)
    offsets = user_offsets()

    # GIRS candidate grid
    girs_candidates = []
    for x in cfg.girs_x:
        for y in cfg.girs_y:
            girs_candidates.append(np.array([x, y, cfg.girs_z]))

    # trajectory generation (train + test share the same template & seeds rule)
    def gen_trajs(n):
        return [hotspot_centre_path(rng, cfg.T, cfg) for _ in range(n)]

    train_centres = gen_trajs(cfg.N_train)
    test_centres = gen_trajs(cfg.N_test)

    def W_at(centres, tr, t):
        return centres[tr][t] + offsets  # (M, 3)

    # ---- GIRS mount-point selection on TRAINING trajectories only ----
    girs_Q = None
    best_score = -np.inf
    girs_scores = {}
    for gq in girs_candidates:
        scores = []
        for tr in range(cfg.N_train):
            for t in range(cfg.T):
                W_now = W_at(train_centres, tr, t)
                set_users(W_now)
                Qg, Thg, Pg = optimize(cfg, opt_Q=False, Q_init=gq)
                scores.append(evaluate(cfg, Qg, Thg, Pg, W_now))
        s = float(np.mean(scores))
        girs_scores[tuple(gq)] = s
        if s > best_score:
            best_score = s
            girs_Q = gq
    girs_Q = np.array(girs_Q, dtype=float)

    # ---- evaluate the three schemes on TEST trajectories ----
    # r[scheme][tr][t], plus per-trajectory UAV displacement for adaptive
    keys = ["adaptive", "fixed", "girs"]
    r = {k: np.full((cfg.N_test, cfg.T), np.nan) for k in keys}
    disp_adaptive = np.zeros(cfg.N_test)
    Q_fixed = np.zeros((cfg.N_test, 3))
    Q_adaptive_prev = None

    for tr in range(cfg.N_test):
        Q_adaptive_prev = None
        for t in range(cfg.T):
            W_now = W_at(test_centres, tr, t)
            set_users(W_now)

            # 1) AIRS-adaptive
            Qa, Tha, Pa = optimize(cfg, opt_Q=True)
            r["adaptive"][tr, t] = evaluate(cfg, Qa, Tha, Pa, W_now)
            if Q_adaptive_prev is not None:
                disp_adaptive[tr] += float(np.linalg.norm(Qa - Q_adaptive_prev))
            Q_adaptive_prev = Qa

            # 2) AIRS-fixed-position
            if t == 0:
                Qf0, _, _ = optimize(cfg, opt_Q=True)
                Q_fixed[tr] = Qf0
            Qf, Thf, Pf = optimize(cfg, opt_Q=False, Q_init=Q_fixed[tr])
            r["fixed"][tr, t] = evaluate(cfg, Qf, Thf, Pf, W_now)

            # 3) GIRS-fixed-position
            Qg, Thg, Pg = optimize(cfg, opt_Q=False, Q_init=girs_Q)
            r["girs"][tr, t] = evaluate(cfg, Qg, Thg, Pg, W_now)

    # ---- summary statistics (independent unit = trajectory) ----
    summary = {"mode": mode, "config": {
        "M": cfg.M, "L": cfg.L, "N_train": cfg.N_train, "N_test": cfg.N_test,
        "T": cfg.T, "S": cfg.S, "out": cfg.out, "inner": cfg.inner,
        "opt_seed": cfg.opt_seed, "eval_seed": cfg.eval_seed,
        "traj_seed": cfg.traj_seed,
        "center_x_range": list(cfg.center_x_range),
        "center_y_range": list(cfg.center_y_range),
        "girs_x": list(cfg.girs_x), "girs_y": list(cfg.girs_y),
        "girs_z": cfg.girs_z,
        "girs_Q": girs_Q.tolist(),
    }}

    # (i) per-scheme mean over trajectories and slots
    overall = {k: float(np.nanmean(r[k])) for k in keys}

    # (ii) per-trajectory slot-averaged r, mean + 95% CI
    traj_mean = {k: r[k].mean(axis=1) for k in keys}  # (N_test,)
    traj_ci = {k: ci95(traj_mean[k]) for k in keys}

    # (iii) AIRS-adaptive vs GIRS-fixed paired rate difference + 95% CI
    paired_diff = traj_mean["adaptive"] - traj_mean["girs"]
    paired_ci = ci95(paired_diff)

    # (iv) fraction of test trajectories where adaptive beats GIRS
    frac_win = float(np.mean(paired_diff > 0.0))

    # (v) mean per-slot UAV displacement for AIRS-adaptive
    mean_disp = float(np.mean(disp_adaptive) / (cfg.T - 1)) if cfg.T > 1 else 0.0

    summary["overall_mean"] = overall
    summary["trajectory_mean_ci"] = {
        k: {"mean": traj_ci[k][0], "lo": traj_ci[k][1], "hi": traj_ci[k][2]}
        for k in keys}
    summary["paired_diff_adaptive_minus_girs_ci"] = {
        "mean": paired_ci[0], "lo": paired_ci[1], "hi": paired_ci[2]}
    summary["fraction_adaptive_beats_girs"] = frac_win
    summary["mean_adaptive_uav_displacement_m"] = mean_disp
    summary["girs_training_scores"] = {
        "candidate_" + "_".join(f"{v:.0f}" for v in k): v
        for k, v in girs_scores.items()}

    return cfg, r, test_centres, summary


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def save(cfg, r, test_centres, summary, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # raw per-trajectory per-slot CSV + JSON
    keys = ["adaptive", "fixed", "girs"]
    rows = []
    for tr in range(cfg.N_test):
        for t in range(cfg.T):
            rows.append({
                "trajectory": tr, "slot": t,
                "centre_x": float(test_centres[tr][t][0]),
                "centre_y": float(test_centres[tr][t][1]),
                **{f"r_{k}": float(r[k][tr, t]) for k in keys},
            })
    with open(os.path.join(out_dir, "per_traj_slot.json"), "w") as f:
        json.dump({"config": summary["config"], "rows": rows}, f, indent=2)

    with open(os.path.join(out_dir, "per_traj_slot.csv"), "w") as f:
        f.write("trajectory,slot,centre_x,centre_y," +
                ",".join(f"r_{k}" for k in keys) + "\n")
        for row in rows:
            vals = ",".join(f"{row['r_' + k]:.10f}" for k in keys)
            f.write(f"{row['trajectory']},{row['slot']},"
                    f"{row['centre_x']:.4f},{row['centre_y']:.4f},{vals}\n")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Figure 1: per-slot worst-user rate (mean + 95% CI) ----
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    slots = np.arange(cfg.T)
    style = {"adaptive": ("#1f77b4", "AIRS-adaptive", "-o"),
             "fixed": ("#2ca02c", "AIRS-fixed-position", "-s"),
             "girs": ("#8c564b", "GIRS-fixed-position", "-^")}
    for k in keys:
        col, lab, mk = style[k]
        m = r[k].mean(axis=0)
        # CI across trajectories at each slot
        per_slot = r[k].T  # (T, N_test)
        lo_arr = np.array([ci95(per_slot[t])[1] for t in range(cfg.T)])
        hi_arr = np.array([ci95(per_slot[t])[2] for t in range(cfg.T)])
        ax.plot(slots, m, mk, color=col, label=lab, lw=1.6, ms=4)
        ax.fill_between(slots, lo_arr, hi_arr, color=col, alpha=0.18)
    ax.set_xlabel("Hotspot-motion slot")
    ax.set_ylabel("Worst-user SIC-feasible rate (Mbps)")
    ax.set_xticks(slots)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    ax.set_title("Moving hotspot: scheme rates (mean ± 95% CI over trajectories)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_slot_rates.pdf"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: paired per-trajectory difference vs GIRS ----
    traj_mean = {k: r[k].mean(axis=1) for k in keys}
    paired = traj_mean["adaptive"] - traj_mean["girs"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(cfg.N_test)
    ax.scatter(x, paired / 1e6, s=22, color="#1f77b4", zorder=3)
    ax.axhline(0.0, color="k", lw=0.9, linestyle="--")
    m, lo, hi = ci95(paired)
    ax.axhline(m / 1e6, color="#d62728", lw=1.2, label="mean paired diff")
    ax.axhspan(lo / 1e6, hi / 1e6, color="#d62728", alpha=0.15,
               label="95% CI (over trajectories)")
    ax.set_xlabel("Test trajectory index")
    ax.set_ylabel("AIRS-adaptive − GIRS-fixed (Mbps)")
    ax.set_xticks(x)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    ax.set_title("Paired per-trajectory rate difference vs fixed GIRS")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_paired_diff.pdf"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # console report
    print("=== Reviewer5 comment10 mobility (%s) ===" % summary["mode"])
    print(f"  GIRS mount point chosen on training: {summary['config']['girs_Q']}")
    for k in keys:
        print(f"  {k:12s} overall mean = {summary['overall_mean'][k]/1e6:.4f} Mbps "
              f"(traj-mean 95% CI [{summary['trajectory_mean_ci'][k]['lo']/1e6:.4f}, "
              f"{summary['trajectory_mean_ci'][k]['hi']/1e6:.4f}])")
    p = summary["paired_diff_adaptive_minus_girs_ci"]
    print(f"  adaptive - girs paired diff: mean={p['mean']/1e6:.4f} Mbps "
          f"95% CI [{p['lo']/1e6:.4f}, {p['hi']/1e6:.4f}]")
    print(f"  fraction adaptive beats girs = {summary['fraction_adaptive_beats_girs']:.3f}")
    print(f"  mean adaptive UAV displacement = {summary['mean_adaptive_uav_displacement_m']:.2f} m/slot")
    print(f"  saved -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        "reviewer5_mobility_results", args.mode)
    cfg, r, centres, summary = run(args.mode)
    save(cfg, r, centres, summary, out_dir)
    print("Done.")
