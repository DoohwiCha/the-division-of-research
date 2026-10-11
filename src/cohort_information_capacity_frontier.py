"""Estimate preliminary LLM roster-capacity curves and frontier points.

This module uses only frozen, cached experiments.  It answers the first
question required before a larger frontier experiment: does effective book
capacity display a stable scaling curve as analyst seats are added?

The exact identity behind the curve is

    N_eff(K) = K / (1 + (K - 1) * kappa_K),

where kappa_K is the mean squared off-diagonal book correlation.  A constant
kappa approximation therefore has a transparent saturation level, 1/kappa.
The approximation is fitted to date-level K=2,3,4 measurements and is not
treated as evidence beyond the observed range.

Run:
    PYTHONPATH=src python3 src/cohort_information_capacity_frontier.py
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from _paths import DATA, FIGURES, RESULTS
from cohort_information_diversity import CHANNELS, PILOT_MODELS, _atomic_json
from cohort_information_diversity_random_placebo import PACKET_LABELS


SCALING_PATH = Path(RESULTS) / "information_diversity_scaling.json"
RANDOM_RESULT_PATH = Path(RESULTS) / "information_diversity_random_placebo_results.json"
RANDOM_PLAN_PATH = Path(RESULTS) / "information_diversity_random_placebo_plan.json"
LONG_ONLY_PATH = Path(RESULTS) / "information_diversity_longonly_results.json"
PILOT_CACHE = Path(DATA) / "cohort_agents_information_diversity.json"
HELDOUT_CACHE = Path(DATA) / "cohort_agents_information_diversity_confirmatory.json"
RANDOM_CACHE = Path(DATA) / "cohort_agents_information_diversity_random_placebo.json"

OUTPUT_PATH = Path(RESULTS) / "information_capacity_frontier_preliminary.json"
CURVE_CSV = Path(RESULTS) / "information_capacity_curve_preliminary.csv"
FRONTIER_CSV = Path(RESULTS) / "information_capacity_frontier_preliminary.csv"
FIGURE_PATH = Path(FIGURES) / "fig_information_capacity_preliminary.pdf"

BOOTSTRAP_DRAWS = 50_000
TEAM_SIZES = (1, 2, 3, 4)
RANDOM_CONDITIONS = tuple(f"random_{label}" for label in PACKET_LABELS)
DESIGN_KEYS = {
    "shared": "shared_full_models_m4",
    "random": "random_models_m4",
    "domain": "structured_models_m4",
}
LONG_ONLY_KEYS = {
    "shared": "shared_models_m4",
    "random": "random_models_m4",
    "domain": "structured_models_m4",
}
PERFORMANCE_KEYS = {"shared": "shared", "random": "random", "domain": "structured"}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _kappa(k: int, n_eff: float) -> float:
    if k <= 1:
        return 0.0
    return float((k / n_eff - 1.0) / (k - 1.0))


def _curve(k: np.ndarray | float, kappa: float) -> np.ndarray | float:
    return k / (1.0 + (k - 1.0) * kappa)


def _fit_kappa(values: np.ndarray) -> tuple[float, float]:
    k = np.asarray((2.0, 3.0, 4.0))
    result = minimize_scalar(
        lambda value: float(np.mean((values - _curve(k, value)) ** 2)),
        bounds=(1e-8, 0.999999),
        method="bounded",
    )
    fitted = np.asarray(_curve(k, float(result.x)))
    return float(result.x), float(np.sqrt(np.mean((values - fitted) ** 2)))


def _bootstrap_curve(rows: list[dict], design: str, seed: int) -> dict:
    key = "models_shared_rich_m{}" if design == "shared" else "information_same_model_m{}"
    values = np.asarray([
        [float(row[key.format(k)]["N_eff"]) for k in (2, 3, 4)]
        for row in rows
    ])
    means = values.mean(axis=0)
    fitted_kappa, rmse = _fit_kappa(means)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(BOOTSTRAP_DRAWS, len(rows)))
    boot_means = values[indices].mean(axis=1)
    boot_kappa = np.empty(BOOTSTRAP_DRAWS)
    boot_rmse = np.empty(BOOTSTRAP_DRAWS)
    for index, sample in enumerate(boot_means):
        boot_kappa[index], boot_rmse[index] = _fit_kappa(sample)

    levels = np.column_stack([np.ones(BOOTSTRAP_DRAWS), boot_means])
    marginal = np.diff(levels, axis=1)
    second_difference = marginal[:, 1:] - marginal[:, :-1]
    observed_levels = np.r_[1.0, means]
    observed_marginal = np.diff(observed_levels)
    return {
        "n_dates": len(rows),
        "levels": {
            str(k): {
                "mean": float(observed_levels[k - 1]),
                "date_bootstrap_95": [
                    float(value) for value in np.quantile(levels[:, k - 1], [0.025, 0.975])
                ],
            }
            for k in TEAM_SIZES
        },
        "marginal_capacity": {
            f"{k-1}_to_{k}": {
                "mean": float(observed_marginal[k - 2]),
                "date_bootstrap_95": [
                    float(value)
                    for value in np.quantile(marginal[:, k - 2], [0.025, 0.975])
                ],
            }
            for k in (2, 3, 4)
        },
        "second_differences": {
            "at_K2": {
                "mean": float(observed_marginal[1] - observed_marginal[0]),
                "date_bootstrap_95": [
                    float(value)
                    for value in np.quantile(second_difference[:, 0], [0.025, 0.975])
                ],
            },
            "at_K3": {
                "mean": float(observed_marginal[2] - observed_marginal[1]),
                "date_bootstrap_95": [
                    float(value)
                    for value in np.quantile(second_difference[:, 1], [0.025, 0.975])
                ],
            },
        },
        "constant_kappa_fit": {
            "kappa": fitted_kappa,
            "kappa_date_bootstrap_95": [
                float(value) for value in np.quantile(boot_kappa, [0.025, 0.975])
            ],
            "implied_saturation_1_over_kappa": float(1.0 / fitted_kappa),
            "implied_saturation_date_bootstrap_95": [
                float(value)
                for value in np.quantile(1.0 / boot_kappa, [0.025, 0.975])
            ],
            "rmse_K2_to_K4": rmse,
            "rmse_date_bootstrap_95": [
                float(value) for value in np.quantile(boot_rmse, [0.025, 0.975])
            ],
            "scope": "descriptive fit over observed K=2,3,4; saturation is model-implied, not observed",
        },
        "identity_kappa_by_K": {
            str(k): _kappa(k, float(means[k - 2])) for k in (2, 3, 4)
        },
    }


def _nmi(packets: list[dict]) -> float:
    edges = [
        (seat, field_id.split(".", 1)[0])
        for seat, packet in enumerate(packets)
        for field_id in packet["field_ids"]
    ]
    n = len(edges)
    seat_count: dict[int, int] = {}
    domain_count: dict[str, int] = {}
    joint_count: dict[tuple[int, str], int] = {}
    for seat, domain in edges:
        seat_count[seat] = seat_count.get(seat, 0) + 1
        domain_count[domain] = domain_count.get(domain, 0) + 1
        joint_count[seat, domain] = joint_count.get((seat, domain), 0) + 1
    mutual_information = sum(
        count / n * math.log(count * n / (seat_count[seat] * domain_count[domain]))
        for (seat, domain), count in joint_count.items()
    )
    seat_entropy = -sum(count / n * math.log(count / n) for count in seat_count.values())
    domain_entropy = -sum(count / n * math.log(count / n) for count in domain_count.values())
    return float(mutual_information / math.sqrt(seat_entropy * domain_entropy))


def _specialization() -> dict:
    plan = _load(RANDOM_PLAN_PATH)
    random_nmi = np.asarray([_nmi(row["packets"]) for row in plan["assignments"]])
    return {
        "definition": "normalized mutual information between analyst seat and research-domain label",
        "shared_full_packet": 0.0,
        "random_partition": {
            "mean": float(random_nmi.mean()),
            "min": float(random_nmi.min()),
            "max": float(random_nmi.max()),
        },
        "domain_assignment": 1.0,
    }


def _usage_rows() -> tuple[list[dict], list[str]]:
    records = []
    for path in (PILOT_CACHE, HELDOUT_CACHE, RANDOM_CACHE):
        records.extend(_load(path))
    random_records = _load(RANDOM_CACHE)
    dates = sorted({row["date"] for row in random_records})
    return records, dates


def _team_input_costs() -> dict:
    records, dates = _usage_rows()
    lookup = {
        (row["date"], row["vendor"], row["condition"], int(row.get("replicate", 0))): row
        for row in records if row.get("parse_ok")
    }
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    values = {
        design: {"prompt_tokens": [], "reported_cost": []}
        for design in ("shared", "random", "domain")
    }

    def append(design: str, selected: list[dict]) -> None:
        values[design]["prompt_tokens"].append(sum(
            float(row.get("usage", {}).get("prompt_tokens", 0.0)) for row in selected
        ))
        values[design]["reported_cost"].append(sum(
            float(row.get("usage", {}).get("cost", 0.0)) for row in selected
        ))

    for date in dates:
        for combination in itertools.combinations(vendors, 4):
            append("shared", [lookup[date, vendor, "rich", 0] for vendor in combination])
            for assignment in itertools.permutations(CHANNELS):
                append("domain", [
                    lookup[date, vendor, condition, 0]
                    for vendor, condition in zip(combination, assignment)
                ])
            for assignment in itertools.permutations(RANDOM_CONDITIONS):
                append("random", [
                    lookup[date, vendor, condition, 0]
                    for vendor, condition in zip(combination, assignment)
                ])
    return {
        design: {
            "mean_team_prompt_tokens": float(np.mean(row["prompt_tokens"])),
            "mean_team_reported_cost": float(np.mean(row["reported_cost"])),
        }
        for design, row in values.items()
    }


def _frontier_points() -> tuple[dict, list[dict]]:
    random_result = _load(RANDOM_RESULT_PATH)
    long_only = _load(LONG_ONLY_PATH)
    signed_rows = random_result["per_date"]
    long_rows = long_only["scopes"]["pooled_34_dates"]["per_date"]
    performance = long_only["scopes"]["pooled_34_dates"]["operating_performance"]["levels"]
    turnover = long_only["turnover"]["levels"]
    usage = _team_input_costs()
    specialization = _specialization()
    specialization_value = {
        "shared": specialization["shared_full_packet"],
        "random": specialization["random_partition"]["mean"],
        "domain": specialization["domain_assignment"],
    }
    points = []
    for design in ("shared", "random", "domain"):
        signed_key = DESIGN_KEYS[design]
        long_key = LONG_ONLY_KEYS[design]
        performance_key = PERFORMANCE_KEYS[design]
        point = {
            "design": design,
            "access_separation": 0.0 if design == "shared" else 1.0,
            "domain_specialization": specialization_value[design],
            "signed_N_eff": float(np.mean([row[signed_key]["N_eff"] for row in signed_rows])),
            "signed_same_side_overlap": float(np.mean([
                row[signed_key]["same_side_overlap"] for row in signed_rows
            ])),
            "signed_order_concentration": float(np.mean([
                row[signed_key]["convex_coorder_adv_weighted"] for row in signed_rows
            ])),
            "signed_forward_return": float(np.mean([
                row[signed_key]["forward_return_mean"] for row in signed_rows
            ])),
            "long_only_active_N_eff": float(np.mean([
                row[long_key]["N_eff"] for row in long_rows
            ])),
            "long_only_active_overlap": float(np.mean([
                row[long_key]["same_side_active_overlap"] for row in long_rows
            ])),
            "annualized_tracking_error": performance[performance_key]["annualized_tracking_error"],
            "information_ratio": performance[performance_key]["information_ratio"],
            "sharpe_ratio": performance[performance_key]["sharpe_ratio"],
            "mean_monthly_turnover": turnover[performance_key]["mean_monthly_turnover"],
            **usage[design],
        }
        points.append(point)

    # Capacity-cost Pareto set: maximize signed N_eff and minimize reported cost.
    for point in points:
        point["capacity_cost_pareto"] = not any(
            other["signed_N_eff"] >= point["signed_N_eff"]
            and other["mean_team_reported_cost"] <= point["mean_team_reported_cost"]
            and (
                other["signed_N_eff"] > point["signed_N_eff"]
                or other["mean_team_reported_cost"] < point["mean_team_reported_cost"]
            )
            for other in points
        )
    return specialization, points


def _figure(curves: dict, points: list[dict]) -> None:
    colors = {"shared": "#59656f", "random": "#4f7f92", "domain": "#3f8068"}
    labels = {"shared": "Shared full packet", "random": "Random split", "domain": "Domain assignment"}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))

    ax = axes[0]
    k_observed = np.asarray(TEAM_SIZES)
    k_smooth = np.linspace(1.0, 7.0, 200)
    for design in ("shared", "domain"):
        row = curves[design]
        y = np.asarray([row["levels"][str(k)]["mean"] for k in TEAM_SIZES])
        kappa = row["constant_kappa_fit"]["kappa"]
        ax.plot(k_smooth, _curve(k_smooth, kappa), color=colors[design], alpha=0.45, lw=1.4)
        ax.plot(k_observed, y, marker="o", color=colors[design], lw=1.8, label=labels[design])
    ax.plot(k_smooth, k_smooth, ls="--", color="#a8afb4", lw=1.0, label="Independent seats")
    ax.axvline(4, color="#c7cdd1", lw=0.8, ls=":")
    ax.text(4.08, 1.08, "observed through K=4", fontsize=7, color="#667078")
    ax.set(xlabel="Nominal analyst seats K", ylabel="Effective books $N_{eff}$", xlim=(1, 7), ylim=(0.9, 7.1))
    ax.set_xticks(range(1, 8))
    ax.grid(axis="y", color="#dfe3e6", lw=0.6)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("A  Roster capacity curve", loc="left", fontsize=9, fontweight="bold")

    ax = axes[1]
    frontier = sorted((point for point in points if point["capacity_cost_pareto"]),
                      key=lambda row: row["mean_team_reported_cost"])
    if len(frontier) > 1:
        ax.plot(
            [row["mean_team_reported_cost"] for row in frontier],
            [row["signed_N_eff"] for row in frontier],
            color="#26343d", lw=1.0, ls="--", zorder=1,
        )
    for point in points:
        ax.scatter(
            point["mean_team_reported_cost"], point["signed_N_eff"],
            s=72, color=colors[point["design"]], edgecolor="white", linewidth=0.8, zorder=2,
        )
        ax.annotate(
            labels[point["design"]],
            (point["mean_team_reported_cost"], point["signed_N_eff"]),
            xytext=(5, 4), textcoords="offset points", fontsize=7,
        )
    ax.set(xlabel="Mean reported API cost per four-seat team (USD)", ylabel="Effective books $N_{eff}$")
    ax.grid(color="#dfe3e6", lw=0.6)
    ax.set_title("B  Information-capacity frontier", loc="left", fontsize=9, fontweight="bold")
    fig.tight_layout(w_pad=2.2)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    scaling = _load(SCALING_PATH)
    rows = scaling["per_date"]
    curves = {
        "shared": _bootstrap_curve(rows, "shared", 2026072801),
        "domain": _bootstrap_curve(rows, "domain", 2026072802),
    }
    specialization, points = _frontier_points()
    result = {
        "schema_version": "information-capacity-frontier-preliminary-v1",
        "analysis_status": "cached-evidence preliminary; no new model calls",
        "curve_identity": "N_eff(K) = K / [1 + (K-1) kappa_K]",
        "curve_samples": "34 date-level means; K=1 is the one-book identity and K=2,3,4 are observed",
        "curves": curves,
        "information_architecture": specialization,
        "frontier_points": points,
        "claim_boundary": (
            "The K=2..4 curve is observed. Values beyond K=4 in the figure are constant-kappa "
            "visual extrapolations and are not used as empirical evidence."
        ),
    }
    _atomic_json(OUTPUT_PATH, result)

    curve_rows = []
    for design, row in curves.items():
        for k in TEAM_SIZES:
            curve_rows.append({
                "design": design,
                "K": k,
                "N_eff": row["levels"][str(k)]["mean"],
                "ci_low": row["levels"][str(k)]["date_bootstrap_95"][0],
                "ci_high": row["levels"][str(k)]["date_bootstrap_95"][1],
                "identity_kappa": 0.0 if k == 1 else row["identity_kappa_by_K"][str(k)],
            })
    pd.DataFrame(curve_rows).to_csv(CURVE_CSV, index=False)
    pd.DataFrame(points).to_csv(FRONTIER_CSV, index=False)
    _figure(curves, points)

    print("Preliminary roster-capacity curves")
    for design, row in curves.items():
        levels = " -> ".join(f"{row['levels'][str(k)]['mean']:.3f}" for k in TEAM_SIZES)
        fit = row["constant_kappa_fit"]
        print(
            f"  {design:6s}: {levels}; kappa={fit['kappa']:.3f}; "
            f"implied saturation={fit['implied_saturation_1_over_kappa']:.2f}; "
            f"RMSE={fit['rmse_K2_to_K4']:.4f}"
        )
    print("Frontier points")
    for row in points:
        print(
            f"  {row['design']:6s}: N_eff={row['signed_N_eff']:.3f}, "
            f"cost=${row['mean_team_reported_cost']:.5f}, "
            f"tokens={row['mean_team_prompt_tokens']:.0f}, pareto={row['capacity_cost_pareto']}"
        )
    print(f"wrote {OUTPUT_PATH}")
    print(f"wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
