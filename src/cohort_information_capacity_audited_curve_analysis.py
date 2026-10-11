"""Analyze the complete audit-corrected K=1..7 information-access curve."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA, RESULTS
from cohort_crossvendor_hardening import matched_null
from cohort_information_capacity_analysis import METRICS, _book, _mean_metrics, _metrics
from cohort_information_capacity_audited_analysis import RESULT_PATH as K7_RESULT_PATH
from cohort_information_capacity_audited_curve import CACHE_PATH, PLAN_PATH, TEAM_SIZES
from cohort_information_capacity_integrity import (
    CACHE_PATH as SHARED_CACHE_PATH,
    INTEGRITY_CORRECTION_ID,
    _load_packets,
)
from cohort_information_diversity import PILOT_MODELS, _atomic_json, _sign_flip_p


RESULT_PATH = Path(RESULTS) / "information_capacity_audited_curve_results.json"
SUMMARY_PATH = Path(RESULTS) / "information_capacity_audited_curve_summary.csv"
BOOTSTRAP_DRAWS = 20_000
DESIGNS = ("shared", "mixed", "coherent")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _current(row: dict) -> bool:
    return bool(
        row.get("parse_ok")
        and row.get("integrity_correction_id") == INTEGRITY_CORRECTION_ID
        and row.get("response_id")
        and row.get("request_payload_sha256")
    )


def _configuration_metrics(
    *,
    k: int,
    architecture: str,
    vendors: list[str],
    lookup: dict[tuple[str, int, str], dict],
    symbols: list[str],
    full_consensus: np.ndarray,
    forward: np.ndarray,
    adv: np.ndarray,
    null_mean: float,
) -> tuple[dict, int]:
    books = {
        (vendor, packet): _book(lookup[architecture, packet, vendor], symbols)
        for vendor in vendors
        for packet in range(1, k + 1)
    }
    configurations = []
    for combination in itertools.combinations(vendors, k):
        for assignment in itertools.permutations(range(1, k + 1)):
            rows = [
                lookup[architecture, packet, vendor]
                for vendor, packet in zip(combination, assignment)
            ]
            matrix = np.column_stack([
                books[vendor, packet]
                for vendor, packet in zip(combination, assignment)
            ])
            configurations.append(
                _metrics(matrix, full_consensus, forward, adv, null_mean, rows)
            )
    return _mean_metrics(configurations), len(configurations)


def _curve_rows() -> tuple[list[dict], dict[int, dict]]:
    shared_rows = [
        row for row in _load(SHARED_CACHE_PATH)
        if row.get("architecture") == "shared" and _current(row)
    ]
    split_rows = [row for row in _load(CACHE_PATH) if _current(row)]
    if len(shared_rows) != 34 * 7:
        raise ValueError(f"expected 238 corrected shared rows, found {len(shared_rows)}")
    expected_split = 34 * 7 * 2 * sum(TEAM_SIZES)
    if len(split_rows) != expected_split:
        raise ValueError(f"expected {expected_split} corrected curve rows, found {len(split_rows)}")

    shared_lookup = {(row["date"], row["vendor"]): row for row in shared_rows}
    split_lookup = {
        (
            row["date"], int(row["team_size"]), row["architecture"],
            int(row["packet_index"]), row["vendor"],
        ): row
        for row in split_rows
    }
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    null_benchmarks = {
        k: matched_null(
            k, [f"A{index}" for index in range(14)], B=10_000,
            seed=2026071700 + k,
        )
        for k in range(2, 8)
    }
    null_means = {k: values["mean"] for k, values in null_benchmarks.items()}
    output = []
    for item in _load_packets():
        snapshot = item["snapshot"]
        date = snapshot["date"]
        symbols = snapshot["symbols"]
        date_shared = {vendor: shared_lookup[date, vendor] for vendor in vendors}
        shared_books = {vendor: _book(date_shared[vendor], symbols) for vendor in vendors}
        full_consensus = np.mean(list(shared_books.values()), axis=0)
        forward = np.asarray([snapshot["forward_return_21d"][symbol] for symbol in symbols])
        adv = np.asarray([snapshot["features"]["market"][symbol]["adv_bn"] for symbol in symbols])

        for k in TEAM_SIZES:
            shared_configurations = []
            for combination in itertools.combinations(vendors, k):
                rows = [date_shared[vendor] for vendor in combination]
                matrix = np.column_stack([shared_books[vendor] for vendor in combination])
                shared_configurations.append(
                    _metrics(matrix, full_consensus, forward, adv, null_means[k], rows)
                )
            lookup = {
                (architecture, packet, vendor):
                    split_lookup[date, k, architecture, packet, vendor]
                for architecture in ("mixed_audited", "coherent_audited")
                for packet in range(1, k + 1)
                for vendor in vendors
            }
            mixed, mixed_n = _configuration_metrics(
                k=k, architecture="mixed_audited", vendors=vendors, lookup=lookup,
                symbols=symbols, full_consensus=full_consensus, forward=forward,
                adv=adv, null_mean=null_means[k],
            )
            coherent, coherent_n = _configuration_metrics(
                k=k, architecture="coherent_audited", vendors=vendors, lookup=lookup,
                symbols=symbols, full_consensus=full_consensus, forward=forward,
                adv=adv, null_mean=null_means[k],
            )
            output.append({
                "sample": item["sample"],
                "snapshot_id": snapshot["snapshot_id"],
                "date": date,
                "return_regime": snapshot["return_regime"],
                "volatility_regime": snapshot["volatility_regime"],
                "team_size": k,
                "shared": _mean_metrics(shared_configurations),
                "mixed": mixed,
                "coherent": coherent,
                "configuration_counts": {
                    "shared": len(shared_configurations),
                    "mixed": mixed_n,
                    "coherent": coherent_n,
                },
            })
            print(f"  analyzed corrected curve {date} K={k}", flush=True)

    k7 = _load(K7_RESULT_PATH)
    for row in k7["per_date"]:
        output.append({
            **{key: row[key] for key in (
                "sample", "snapshot_id", "date", "return_regime", "volatility_regime",
                "team_size",
            )},
            "shared": row["shared"],
            "mixed": row["mixed"],
            "coherent": row["coherent"],
            "configuration_counts": row["configuration_counts"],
        })
    return (
        sorted(output, key=lambda row: (row["date"], row["team_size"])),
        null_benchmarks,
    )


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
    values = np.asarray([
        row[treatment][metric] - row[control][metric] for row in rows
    ], dtype=float)
    result = _bootstrap(values, seed)
    result.update({
        "treatment": treatment,
        "control": control,
        "metric": metric,
        "treatment_mean": float(np.mean([row[treatment][metric] for row in rows])),
        "control_mean": float(np.mean([row[control][metric] for row in rows])),
        "dates_positive": int(np.sum(values > 0)),
        "dates_negative": int(np.sum(values < 0)),
        "sign_flip_p_two_sided": _sign_flip_p(values),
    })
    return result


def _summaries(rows: list[dict]) -> dict:
    seed = 2026072160
    levels = {}
    comparisons = {}
    for k in range(2, 8):
        current = [row for row in rows if row["team_size"] == k]
        levels[str(k)] = {}
        for design in DESIGNS:
            levels[str(k)][design] = {}
            for metric in METRICS:
                levels[str(k)][design][metric] = _bootstrap(
                    np.asarray([row[design][metric] for row in current]), seed
                )
                seed += 1
        comparisons[str(k)] = {}
        for treatment, control in (
            ("mixed", "shared"), ("coherent", "mixed"), ("coherent", "shared"),
        ):
            key = f"{treatment}_minus_{control}"
            comparisons[str(k)][key] = {}
            for metric in (
                "N_eff", "same_side_overlap", "surviving_gross",
                "convex_order_concentration", "consensus_forward_return",
                "full_consensus_alignment", "team_prompt_tokens", "team_reported_cost",
            ):
                comparisons[str(k)][key][metric] = _paired(
                    current, treatment, control, metric, seed
                )
                seed += 1

    by_date = {(row["date"], row["team_size"]): row for row in rows}
    dates = sorted({row["date"] for row in rows})
    marginal = {}
    concavity = {}
    for design in DESIGNS:
        marginal[design] = {}
        transitions = {}
        for k in range(2, 8):
            values = np.asarray([
                by_date[date, k][design]["N_eff"]
                - (1.0 if k == 2 else by_date[date, k - 1][design]["N_eff"])
                for date in dates
            ])
            transitions[k] = values
            marginal[design][f"{k-1}_to_{k}"] = {
                **_bootstrap(values, seed),
                "dates_positive": int(np.sum(values > 0)),
                "dates_negative": int(np.sum(values < 0)),
            }
            seed += 1
        concavity[design] = {}
        for k in range(3, 8):
            second = transitions[k] - transitions[k - 1]
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


def _flat_rows(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        common = {key: row[key] for key in (
            "sample", "snapshot_id", "date", "return_regime", "volatility_regime", "team_size"
        )}
        for design in DESIGNS:
            output.append({**common, "design": design, **row[design]})
    return output


def main() -> None:
    rows, null_benchmarks = _curve_rows()
    result = {
        "schema_version": "information-capacity-audited-curve-results-v1",
        "integrity_correction_id": INTEGRITY_CORRECTION_ID,
        "design_status": "post-audit corrected curve; all 34 dates form one pooled panel",
        "date_is_inference_unit": True,
        "team_information_union_fixed": True,
        "configuration_averaging": (
            "all model subsets and packet-to-model permutations are averaged within date"
        ),
        "n_dates": len({row["date"] for row in rows}),
        "random_book_benchmarks": {
            str(k): values for k, values in null_benchmarks.items()
        },
        "per_date": rows,
        **_summaries(rows),
        "plan_path": str(PLAN_PATH.relative_to(Path(DATA).parent)),
        "K7_result_path": str(K7_RESULT_PATH.relative_to(Path(DATA).parent)),
    }
    _atomic_json(RESULT_PATH, result)
    pd.DataFrame(_flat_rows(rows)).to_csv(SUMMARY_PATH, index=False)
    print("Audit-corrected Information-Capacity Curve")
    for k in range(2, 8):
        level = result["levels"][str(k)]
        print(
            f"  K={k}: shared={level['shared']['N_eff']['mean']:.3f}, "
            f"mixed={level['mixed']['N_eff']['mean']:.3f}, "
            f"coherent={level['coherent']['N_eff']['mean']:.3f}"
        )
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
