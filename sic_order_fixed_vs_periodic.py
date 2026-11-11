"""Paired fixed versus periodic SIC ordering using the Fig. 4 channel code."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import t

import runtime_scaling as solver
from vector_airs_model import (
    BETA1, BETA2, P_HAP, P_HAP_DBM, SIGMA2, W_BAND, channel_components,
)


M, L = 4, 8
SEED = 2026
USER_X = (92.0, 109.0)
USER_Y = (-2.0, 2.0)
UAV_LOWER = np.array([85.0, -15.0, solver.Z_min])
UAV_UPPER = np.array([115.0, 15.0, solver.Z_max])
TOL = 1e-5


def channel_powers(topology, position, phases):
    return np.array([
        channel_components(position, phases, i, L, W_k=topology)
        for i in range(M)
    ])


def residual_power(powers):
    return np.r_[np.cumsum(powers[:0:-1])[::-1], 0.0]


def record_design(position, phases, powers, order, gains, cycle):
    # Power and gains are stored in physical-user order, not decoding order.
    p = powers[order]
    omega = gains[order]
    rate = W_BAND * np.log2(1.0 + omega * p / (omega * residual_power(p) + SIGMA2))
    return {
        "cycle": cycle,
        "position_m": position.tolist(),
        "phase_radians": np.angle(np.diag(phases)).tolist(),
        "power_w": powers.tolist(),
        "statistical_gain": gains.tolist(),
        "order": order.tolist(),
        "surrogate_worst_bps": float(rate.min()),
    }


def evaluate(topology, design, test_draws):
    order = design["order"]
    phases = np.diag(np.exp(1j * np.array(design["phase_radians"])))
    los, nlos = channel_powers(topology, np.array(design["position_m"]), phases).T
    gain = np.abs(np.sqrt(los) + np.sqrt(nlos) * test_draws) ** 2
    # Every user ranked at or after i must decode i; SINR increases in gain.
    weakest_decoder = np.minimum.accumulate(gain[:, order][:, ::-1], axis=1)[:, ::-1]
    p = np.array(design["power_w"])[order]
    rates = W_BAND * np.log2(
        1.0 + weakest_decoder * p / (weakest_decoder * residual_power(p) + SIGMA2)
    )
    mean_by_user = np.empty(M)
    mean_by_user[order] = rates.mean(axis=0)
    return mean_by_user.tolist(), float(mean_by_user.min())


def optimize(topology, initial, test_draws, outer, inner, reorder):
    model = solver.AR_IRS()
    position, phases, powers, order, gains = (x.copy() for x in initial)
    trajectory = [record_design(position, phases, powers, order, gains, 0)]
    saved_positions = solver.positions
    switches = []
    try:
        for cycle in range(1, outer + 1):
            solver.positions = topology[order]
            phases, _, _, rank = model.optimize_pi_Theta(
                position, phases, powers[order], L, W_BAND, M, BETA1, BETA2,
                max_iter=inner,
            )
            if model.inner_irs[-1] == 0 or not np.all(np.isfinite(phases)):
                raise RuntimeError(f"IRS phase block failed in cycle {cycle}")
            if not np.allclose(np.abs(np.diag(phases)), 1.0, atol=TOL, rtol=0):
                raise RuntimeError(f"Recovered phases violate unit modulus in cycle {cycle}")

            # Initialize the placement SCA with the current distances and gains
            # in the same decoding order as the position solver's input powers.
            gains = channel_powers(topology, position, phases).sum(axis=1)
            x = {
                "u": np.linalg.norm(solver.positions - position, axis=1),
                "uu": float(np.linalg.norm(solver.W_s - position)),
                "Omega_underline": gains[order] * solver.GAIN_SCALE,
            }
            position, _, _ = model.optimize_Q(
                x, position, powers[order], phases, L, W_BAND, M, BETA1, BETA2,
                max_iter=inner,
            )
            if model.inner_q[-1] == 0 or not np.all(np.isfinite(position)):
                raise RuntimeError(f"Placement block failed in cycle {cycle}")
            if np.any(position < UAV_LOWER - TOL) or np.any(position > UAV_UPPER + TOL):
                raise RuntimeError(f"Placement violates the flight box in cycle {cycle}")
            position = np.clip(position, UAV_LOWER, UAV_UPPER)
            gains = channel_powers(topology, position, phases).sum(axis=1)
            if reorder:
                new_order = np.argsort(gains, kind="stable")
                if not np.array_equal(new_order, order):
                    switches.append(cycle)
                order = new_order

            solver.positions = topology[order]
            solver_gains = np.array([
                model.compute_channel_gain(position, phases, i, L, solver.positions, BETA1, BETA2)
                for i in range(M)
            ]) / solver.GAIN_SCALE
            np.testing.assert_allclose(solver_gains, gains[order], rtol=1e-10, atol=0)
            p, _ = model.optimize_P(
                position, powers[order], phases, L, P_HAP_DBM, W_BAND, M, BETA1, BETA2,
            )
            if not np.all(np.isfinite(p)) or np.any(p <= 0) or p.sum() > P_HAP + TOL:
                raise RuntimeError(f"Power block returned an invalid allocation in cycle {cycle}")
            # Remove only a tolerance-sized solver overshoot before scoring.
            powers[order] = p * min(1.0, P_HAP / p.sum())
            candidate = record_design(position, phases, powers, order, gains, cycle)
            candidate.update(
                phase_iterations=model.inner_irs[-1],
                placement_iterations=model.inner_q[-1],
                lifted_rank_residual=float(rank["rank_error"]),
            )
            trajectory.append(candidate)
    finally:
        solver.positions = saved_positions

    # Same selection rule in both arms, using the surrogate and no test samples.
    selected = max(trajectory, key=lambda row: row["surrogate_worst_bps"])
    user_rates, worst_rate = evaluate(topology, selected, test_draws)
    return {
        "switch_cycles": switches,
        "selected_cycle": selected["cycle"],
        "mc_user_bps": user_rates,
        "mc_worst_bps": worst_rate,
        "trajectory": trajectory,
    }


def paired_interval(differences):
    values = np.asarray(differences, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return {"mean": mean, "ci95": None}
    radius = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "ci95": [mean - radius, mean + radius]}


def run(args):
    topology_rng = np.random.default_rng(SEED)
    records = []
    for i in range(args.topologies):
        start = time.perf_counter()
        topology = np.column_stack([
            topology_rng.uniform(*USER_X, M),
            topology_rng.uniform(*USER_Y, M),
            np.zeros(M),
        ])
        init_seed = SEED + 1000 + i
        test_seed = SEED + 100000 + i
        position = topology.mean(axis=0)
        position[2] = (solver.Z_min + solver.Z_max) / 2
        phases = solver.AR_IRS().initialize_Theta(L, rng=np.random.default_rng(init_seed))
        gains = channel_powers(topology, position, phases).sum(axis=1)
        order = np.argsort(gains, kind="stable")
        powers = np.full(M, 0.1 * P_HAP / (M - 1))
        powers[order[0]] = 0.9 * P_HAP
        initial = (position, phases, powers, order, gains)

        noise = np.random.default_rng(test_seed).standard_normal((args.samples, M, 2))
        test_draws = (noise[..., 0] + 1j * noise[..., 1]) / np.sqrt(2)
        fixed = optimize(topology, initial, test_draws, args.outer, args.inner, reorder=False)
        dynamic = optimize(topology, initial, test_draws, args.outer, args.inner, reorder=True)
        delta = dynamic["mc_worst_bps"] - fixed["mc_worst_bps"]
        if not dynamic["switch_cycles"]:
            np.testing.assert_allclose(
                fixed["mc_user_bps"], dynamic["mc_user_bps"], rtol=1e-6, atol=1e-3,
                err_msg=f"Topologies without a switch must match (pair {i + 1})",
            )
        print(
            f"[{i + 1:2}/{args.topologies}] switches={len(dynamic['switch_cycles'])} "
            f"fixed={fixed['mc_worst_bps'] / 1e6:.6f} "
            f"dynamic={dynamic['mc_worst_bps'] / 1e6:.6f} Mbps "
            f"({time.perf_counter() - start:.1f}s)",
            flush=True,
        )
        records.append({
            "topology": i + 1,
            "users_m": topology.tolist(),
            "initial_phase_seed": init_seed,
            "test_seed": test_seed,
            "fixed": fixed,
            "periodic": dynamic,
            "difference_bps": delta,
            "difference_pct_of_fixed": 100.0 * delta / fixed["mc_worst_bps"],
        })
    switches_per_topology = [len(row["periodic"]["switch_cycles"]) for row in records]
    summary = {
        "topologies_with_switches": sum(n > 0 for n in switches_per_topology),
        "total_switches": sum(switches_per_topology),
        "total_periodic_cycles": args.topologies * args.outer,
        "fixed_mean_mbps": float(np.mean([row["fixed"]["mc_worst_bps"] for row in records]) / 1e6),
        "periodic_mean_mbps": float(np.mean([row["periodic"]["mc_worst_bps"] for row in records]) / 1e6),
        "paired_periodic_minus_fixed_mbps": paired_interval([row["difference_bps"] / 1e6 for row in records]),
        "paired_periodic_minus_fixed_pct": paired_interval([row["difference_pct_of_fixed"] for row in records]),
    }
    return {
        "settings": {
            "channel_implementation": "Fig. 4 full-vector code in runtime_scaling.py and vector_airs_model.py",
            "M": M, "L": L, "N": solver.N, "K1": solver.K1, "K2": solver.K2,
            "power_dbm": P_HAP_DBM, "bandwidth_hz": W_BAND, "noise_w": SIGMA2,
            "beta1": BETA1, "beta2": BETA2, "topology_seed": SEED,
            "users_x_m": USER_X, "users_y_m": USER_Y,
            "uav_lower_m": UAV_LOWER.tolist(), "uav_upper_m": UAV_UPPER.tolist(),
            "topologies": args.topologies, "outer_cycles": args.outer,
            "inner_cap": args.inner, "test_samples_per_topology": args.samples,
            "selected_by": "highest worst-user Jensen surrogate among visited feasible designs",
            "test_metric": "minimum across users of mean Eq. (12) SIC-feasible rate",
            "order_indices": "zero-based physical-user IDs, from first decoded to last decoded",
        },
        "summary": summary,
        "per_topology": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topologies", type=int, default=20)
    parser.add_argument("--outer", type=int, default=5)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("revised_results") / "review5_full_vector_sic_order.json",
    )
    args = parser.parse_args()
    if min(args.topologies, args.outer, args.inner, args.samples) < 1:
        parser.error("topologies, iteration limits, and samples must be positive")
    result = run(args)
    # Publish only a complete run.
    temp = args.output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
