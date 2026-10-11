"""Analyze the frozen Information-Capacity Curve experiment.

All model subsets and packet-to-model assignments are averaged within a date.
The date is then the sole unit of inference.  This avoids treating the many
overlapping roster configurations as independent observations.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA, RESULTS
from cohort_information_capacity_experiment import (
    ARCHITECTURES,
    CACHE_PATH,
    PLAN_PATH,
    TEAM_SIZES,
)
from cohort_information_diversity import PILOT_MODELS, _atomic_json, _sign_flip_p, _strict_complete
from cohort_information_diversity_random_placebo import load_packets
from cohort_information_diversity_scaling import IMPACT_DELTA
from cohort_crossvendor_hardening import matched_null


PILOT_SHARED_CACHE = Path(DATA) / "cohort_agents_information_diversity.json"
HELDOUT_SHARED_CACHE = Path(DATA) / "cohort_agents_information_diversity_confirmatory.json"
FINAL_RESULT_PATH = Path(RESULTS) / "information_capacity_experiment_results.json"
FINAL_SUMMARY_PATH = Path(RESULTS) / "information_capacity_experiment_summary.csv"
BOOTSTRAP_DRAWS = 20_000


METRICS = (
    "N_eff",
    "N_eff_over_null",
    "kappa_squared_correlation",
    "mean_abs_corr",
    "mean_signed_corr",
    "same_side_overlap",
    "surviving_gross",
    "convex_order_concentration",
    "common_book_beta",
    "common_book_beta_squared",
    "consensus_forward_return",
    "full_consensus_alignment",
    "team_prompt_tokens",
    "team_reported_cost",
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _book(row: dict, symbols: list[str]) -> np.ndarray:
    raw = {item["symbol"]: float(item["weight"]) for item in row["weights"]}
    vector = np.asarray([raw.get(symbol, 0.0) for symbol in symbols], dtype=float)
    vector -= vector.mean()
    gross = float(np.abs(vector).sum())
    if gross <= 1e-12:
        raise ValueError(
            f"zero-gross book: {row.get('date')} {row.get('vendor')} {row.get('condition')}"
        )
    return vector / gross


def _same_side_overlap(matrix: np.ndarray) -> float:
    values = []
    for left, right in itertools.combinations(range(matrix.shape[1]), 2):
        x, y = matrix[:, left], matrix[:, right]
        same = np.sign(x) == np.sign(y)
        values.append(float(np.minimum(np.abs(x), np.abs(y))[same].sum()))
    return float(np.mean(values))


def _common_beta(matrix: np.ndarray) -> tuple[float, float]:
    values = []
    for index in range(matrix.shape[1]):
        other = np.mean(np.delete(matrix, index, axis=1), axis=1)
        if np.std(other) <= 1e-12:
            values.append(0.0)
        else:
            values.append(float(np.corrcoef(matrix[:, index], other)[0, 1]))
    array = np.asarray(values)
    return float(array.mean()), float(np.square(array).mean())


def _metrics(
    matrix: np.ndarray,
    full_consensus: np.ndarray,
    forward: np.ndarray,
    adv: np.ndarray,
    null_mean: float,
    usage_rows: list[dict],
) -> dict:
    k = matrix.shape[1]
    corr = np.corrcoef(matrix.T)
    iu = np.triu_indices(k, 1)
    neff = float(k * k / np.square(corr).sum())
    squared = np.square(corr[iu])
    kappa_squared = float(squared.mean())
    identity_value = float(k / (1.0 + (k - 1) * kappa_squared))
    if not np.isclose(neff, identity_value, atol=1e-12):
        raise AssertionError("configuration-level N_eff identity failed")
    beta, beta_squared = _common_beta(matrix)
    depth = adv / np.median(adv)
    scale = np.power(depth, IMPACT_DELTA)
    aggregate = matrix.sum(axis=1)
    order_numerator = np.sum(np.power(np.abs(aggregate), 1.0 + IMPACT_DELTA) / scale)
    order_denominator = np.sum(
        np.power(np.abs(matrix), 1.0 + IMPACT_DELTA) / scale[:, None]
    )
    consensus = matrix.mean(axis=1)
    if np.std(consensus) <= 1e-12 or np.std(full_consensus) <= 1e-12:
        alignment = 0.0
    else:
        alignment = float(np.corrcoef(consensus, full_consensus)[0, 1])
    return {
        "N_eff": neff,
        "N_eff_over_null": float(neff / null_mean),
        "kappa_squared_correlation": kappa_squared,
        "mean_abs_corr": float(np.abs(corr[iu]).mean()),
        "mean_signed_corr": float(corr[iu].mean()),
        "same_side_overlap": _same_side_overlap(matrix),
        "surviving_gross": float(np.abs(aggregate).sum() / k),
        "convex_order_concentration": float(order_numerator / order_denominator),
        "common_book_beta": beta,
        "common_book_beta_squared": beta_squared,
        "consensus_forward_return": float(consensus.dot(forward)),
        "full_consensus_alignment": alignment,
        "team_prompt_tokens": float(sum(
            row.get("usage", {}).get("prompt_tokens", 0.0) or 0.0 for row in usage_rows
        )),
        "team_reported_cost": float(sum(
            row.get("usage", {}).get("cost", 0.0) or 0.0 for row in usage_rows
        )),
    }


def _mean_metrics(rows: list[dict]) -> dict:
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}


def _selected_rows(rows: list[dict], sample: str) -> list[dict]:
    if sample == "all":
        return rows
    return [row for row in rows if row.get("sample") == sample]


def _load_inputs(sample: str, allow_partial: bool) -> tuple[list[dict], list[dict], dict]:
    packet_rows = _selected_rows(load_packets(), sample)
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"{CACHE_PATH} is missing; collect the frozen experiment first")
    capacity = _selected_rows(_load(CACHE_PATH), sample)
    bad = [row for row in capacity if not _strict_complete(row)]
    if bad and not allow_partial:
        raise ValueError(f"capacity cache has {len(bad)} incomplete rows in sample={sample}")
    capacity = [row for row in capacity if _strict_complete(row)]
    shared = []
    for path, current_sample in (
        (PILOT_SHARED_CACHE, "pilot"),
        (HELDOUT_SHARED_CACHE, "held-out"),
    ):
        if sample not in ("all", current_sample):
            continue
        for row in _load(path):
            if row.get("condition") == "rich" and int(row.get("replicate", 0)) == 0:
                if not _strict_complete(row):
                    raise ValueError(f"incomplete shared control in {path.name}")
                shared.append({**row, "sample": current_sample})
    plan = _load(PLAN_PATH)
    expected = sum(
        1 for assignment in plan["assignments"]
        if sample == "all" or assignment["sample"] == sample
        for _packet in assignment["packets"]
        for _model in PILOT_MODELS
    )
    if len(capacity) != expected and not allow_partial:
        raise ValueError(
            f"sample={sample} is incomplete: expected {expected} capacity rows, found {len(capacity)}"
        )
    if allow_partial:
        complete_dates = {
            date for date in {row["date"] for row in capacity}
            if sum(row["date"] == date for row in capacity) == 378
        }
        if not complete_dates:
            raise ValueError("no complete 378-call date is available for partial analysis")
        packet_rows = [row for row in packet_rows if row["snapshot"]["date"] in complete_dates]
        capacity = [row for row in capacity if row["date"] in complete_dates]
        shared = [row for row in shared if row["date"] in complete_dates]
    return packet_rows, capacity, {"shared": shared, "plan": plan}


def _date_rows(sample: str, allow_partial: bool = False) -> list[dict]:
    packet_rows, capacity, extras = _load_inputs(sample, allow_partial)
    shared = extras["shared"]
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    capacity_lookup = {
        (row["date"], row["vendor"], row["condition"]): row for row in capacity
    }
    shared_lookup = {(row["date"], row["vendor"]): row for row in shared}
    null_means = {
        k: matched_null(k, [f"A{index}" for index in range(14)], B=10_000, seed=2026071700 + k)["mean"]
        for k in TEAM_SIZES
    }
    output = []
    for item in packet_rows:
        snapshot = item["snapshot"]
        date = snapshot["date"]
        symbols = snapshot["symbols"]
        forward = np.asarray([snapshot["forward_return_21d"][symbol] for symbol in symbols])
        adv = np.asarray([
            snapshot["features"]["market"][symbol]["adv_bn"] for symbol in symbols
        ])
        shared_books = {
            vendor: _book(shared_lookup[date, vendor], symbols) for vendor in vendors
        }
        full_consensus = np.mean(list(shared_books.values()), axis=0)
        for k in TEAM_SIZES:
            shared_configurations = []
            for combination in itertools.combinations(vendors, k):
                rows = [shared_lookup[date, vendor] for vendor in combination]
                matrix = np.column_stack([shared_books[vendor] for vendor in combination])
                shared_configurations.append(
                    _metrics(matrix, full_consensus, forward, adv, null_means[k], rows)
                )
            row = {
                "sample": item["sample"],
                "snapshot_id": snapshot["snapshot_id"],
                "date": date,
                "return_regime": snapshot["return_regime"],
                "volatility_regime": snapshot["volatility_regime"],
                "team_size": k,
                "shared": _mean_metrics(shared_configurations),
                "shared_configurations": len(shared_configurations),
            }
            for architecture in ARCHITECTURES:
                configurations = []
                books = {
                    (vendor, packet): _book(
                        capacity_lookup[
                            date, vendor, f"capacity_{architecture}_k{k}_p{packet}"
                        ],
                        symbols,
                    )
                    for vendor in vendors for packet in range(1, k + 1)
                }
                for combination in itertools.combinations(vendors, k):
                    for assignment in itertools.permutations(range(1, k + 1)):
                        rows = [
                            capacity_lookup[
                                date, vendor, f"capacity_{architecture}_k{k}_p{packet}"
                            ]
                            for vendor, packet in zip(combination, assignment)
                        ]
                        matrix = np.column_stack([
                            books[vendor, packet]
                            for vendor, packet in zip(combination, assignment)
                        ])
                        configurations.append(
                            _metrics(matrix, full_consensus, forward, adv, null_means[k], rows)
                        )
                row[architecture] = _mean_metrics(configurations)
                row[f"{architecture}_configurations"] = len(configurations)
            output.append(row)
            print(f"  analyzed {date} K={k}", flush=True)
    return output


def _bootstrap(values: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "date_bootstrap_95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "n_dates": len(values),
    }


def _paired(rows: list[dict], treatment: str, control: str, metric: str, seed: int) -> dict:
    difference = np.asarray([
        row[treatment][metric] - row[control][metric] for row in rows
    ], dtype=float)
    result = _bootstrap(difference, seed)
    result.update({
        "treatment": treatment,
        "control": control,
        "metric": metric,
        "treatment_mean": float(np.mean([row[treatment][metric] for row in rows])),
        "control_mean": float(np.mean([row[control][metric] for row in rows])),
        "dates_positive": int(np.sum(difference > 0)),
        "dates_negative": int(np.sum(difference < 0)),
        "sign_flip_p_two_sided": _sign_flip_p(difference),
    })
    return result


def _summaries(date_rows: list[dict]) -> dict:
    designs = ("shared", "mixed", "coherent")
    levels = {}
    comparisons = {}
    seed = 2026071800
    for k in TEAM_SIZES:
        rows = [row for row in date_rows if row["team_size"] == k]
        levels[str(k)] = {}
        for design in designs:
            levels[str(k)][design] = {}
            for metric in METRICS:
                levels[str(k)][design][metric] = _bootstrap(
                    np.asarray([row[design][metric] for row in rows]), seed
                )
                seed += 1
        comparisons[str(k)] = {}
        for treatment, control in (
            ("mixed", "shared"),
            ("coherent", "mixed"),
            ("coherent", "shared"),
        ):
            key = f"{treatment}_minus_{control}"
            comparisons[str(k)][key] = {}
            for metric in (
                "N_eff", "same_side_overlap", "surviving_gross",
                "convex_order_concentration", "consensus_forward_return",
                "full_consensus_alignment", "team_prompt_tokens", "team_reported_cost",
            ):
                comparisons[str(k)][key][metric] = _paired(
                    rows, treatment, control, metric, seed
                )
                seed += 1

    by_date = {(row["date"], row["team_size"]): row for row in date_rows}
    dates = sorted({row["date"] for row in date_rows})
    marginal = {}
    concavity = {}
    for design in designs:
        marginal[design] = {}
        date_marginals = {}
        for k in range(2, 8):
            values = np.asarray([
                by_date[date, k][design]["N_eff"]
                - (1.0 if k == 2 else by_date[date, k - 1][design]["N_eff"])
                for date in dates
            ])
            date_marginals[k] = values
            marginal[design][f"{k-1}_to_{k}"] = {
                **_bootstrap(values, seed),
                "dates_positive": int(np.sum(values > 0)),
                "dates_negative": int(np.sum(values < 0)),
            }
            seed += 1
        concavity[design] = {}
        for k in range(3, 8):
            second = date_marginals[k] - date_marginals[k - 1]
            concavity[design][f"at_K{k-1}"] = {
                **_bootstrap(second, seed),
                "dates_negative": int(np.sum(second < 0)),
                "dates_positive": int(np.sum(second > 0)),
                "sign_flip_p_two_sided": _sign_flip_p(second),
            }
            seed += 1
    return {
        "levels": levels,
        "paired_comparisons": comparisons,
        "marginal_capacity": marginal,
        "curve_concavity": concavity,
    }


def _flat_rows(date_rows: list[dict]) -> list[dict]:
    output = []
    for row in date_rows:
        common = {key: row[key] for key in (
            "sample", "snapshot_id", "date", "return_regime", "volatility_regime", "team_size"
        )}
        for design in ("shared", "mixed", "coherent"):
            output.append({**common, "design": design, **row[design]})
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=("pilot", "held-out", "all"), default="all")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    date_rows = _date_rows(args.sample, args.allow_partial)
    summaries = _summaries(date_rows)
    result = {
        "schema_version": "information-capacity-experiment-results-v1",
        "sample": args.sample,
        "design_status": (
            "partial engineering check; not inferential evidence"
            if args.allow_partial else "pre-frozen before model collection"
        ),
        "date_is_inference_unit": True,
        "team_information_union_fixed": True,
        "configuration_averaging": (
            "all model subsets and all packet-to-model permutations are averaged within date"
        ),
        "N_eff_identity": (
            "N_eff(K)=K/[1+(K-1)kappa_K] is verified for every roster "
            "configuration before configuration-level metrics are averaged within date"
        ),
        "n_dates": len({row["date"] for row in date_rows}),
        "per_date": date_rows,
        **summaries,
    }
    if args.allow_partial:
        output_path = FINAL_RESULT_PATH.with_name(
            f"information_capacity_experiment_results_{args.sample.replace('-', '_')}_partial.json"
        )
        summary_path = FINAL_SUMMARY_PATH.with_name(
            f"information_capacity_experiment_summary_{args.sample.replace('-', '_')}_partial.csv"
        )
    elif args.sample == "all":
        output_path = FINAL_RESULT_PATH
        summary_path = FINAL_SUMMARY_PATH
    else:
        output_path = FINAL_RESULT_PATH.with_name(
            f"information_capacity_experiment_results_{args.sample.replace('-', '_')}.json"
        )
        summary_path = FINAL_SUMMARY_PATH.with_name(
            f"information_capacity_experiment_summary_{args.sample.replace('-', '_')}.csv"
        )
    _atomic_json(output_path, result)
    pd.DataFrame(_flat_rows(date_rows)).to_csv(summary_path, index=False)
    print("Information-Capacity Curve")
    for k in TEAM_SIZES:
        row = summaries["levels"][str(k)]
        print(
            f"  K={k}: shared={row['shared']['N_eff']['mean']:.3f}, "
            f"mixed={row['mixed']['N_eff']['mean']:.3f}, "
            f"coherent={row['coherent']['N_eff']['mean']:.3f}"
        )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
