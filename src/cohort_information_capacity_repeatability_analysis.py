"""Analyze persistent access-path diversity in the repeatability panel."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from _paths import DATA, RESULTS
from cohort_crossvendor_hardening import matched_null
from cohort_information_capacity_analysis import _book, _same_side_overlap
from cohort_information_capacity_integrity import _load_packets
from cohort_information_capacity_integrity import _sha256_file
from cohort_information_capacity_repeatability import (
    CACHE_PATH,
    CONDITIONS,
    PLAN_PATH,
    REPLICATES,
    SELECTED_DATES,
    TEAM_SIZE,
    TEMPERATURES,
)
from cohort_information_diversity import PILOT_MODELS, _atomic_json
from cohort_information_diversity_scaling import IMPACT_DELTA


RESULT_PATH = Path(RESULTS) / "information_capacity_repeatability_results.json"
BOOTSTRAP_DRAWS = 50_000
STABLE_BACKEND_VENDORS = ("openai", "google", "mistral", "deepseek", "anthropic")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _row_key(row: dict) -> tuple:
    return (
        row["date"], row["condition"], row["vendor"],
        float(row["temperature"]), int(row["replicate"]),
    )


def _normalized_mean(vectors: list[np.ndarray]) -> np.ndarray:
    vector = np.mean(vectors, axis=0)
    vector -= vector.mean()
    gross = float(np.abs(vector).sum())
    if gross <= 1e-12:
        raise ValueError("repeat-average book has zero gross")
    return vector / gross


def _metrics(matrix: np.ndarray, adv: np.ndarray, null_mean: float) -> dict:
    corr = np.corrcoef(matrix.T)
    iu = np.triu_indices(matrix.shape[1], 1)
    aggregate = matrix.sum(axis=1)
    depth = adv / np.median(adv)
    scale = np.power(depth, IMPACT_DELTA)
    numerator = np.sum(np.power(np.abs(aggregate), 1.0 + IMPACT_DELTA) / scale)
    denominator = np.sum(
        np.power(np.abs(matrix), 1.0 + IMPACT_DELTA) / scale[:, None]
    )
    neff = float(matrix.shape[1] ** 2 / np.square(corr).sum())
    return {
        "N_eff": neff,
        "N_eff_over_null": float(neff / null_mean),
        "mean_abs_corr": float(np.abs(corr[iu]).mean()),
        "mean_signed_corr": float(corr[iu].mean()),
        "same_side_overlap": _same_side_overlap(matrix),
        "surviving_gross": float(np.abs(aggregate).sum() / matrix.shape[1]),
        "convex_order_concentration": float(numerator / denominator),
    }


def _bootstrap(values: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "date_bootstrap_95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "dates_positive": int(np.sum(values > 0)),
        "dates_negative": int(np.sum(values < 0)),
        "n_dates": int(len(values)),
    }


def _cell_reliability(vectors: list[np.ndarray]) -> dict:
    pair_corr = []
    sign_agreement = []
    for left, right in itertools.combinations(vectors, 2):
        pair_corr.append(float(np.corrcoef(left, right)[0, 1]))
        active = (np.abs(left) > 1e-12) | (np.abs(right) > 1e-12)
        sign_agreement.append(float(np.mean(np.sign(left[active]) == np.sign(right[active]))))
    center = np.mean(vectors, axis=0)
    dispersion = float(np.mean([np.linalg.norm(vector - center) for vector in vectors]))
    return {
        "mean_pairwise_correlation": float(np.mean(pair_corr)),
        "mean_sign_agreement": float(np.mean(sign_agreement)),
        "mean_l2_distance_from_cell_mean": dispersion,
    }


def _variance_decomposition(cube: np.ndarray) -> dict:
    """Decompose unit-book dispersion into between-cell and within-cell parts.

    cube has shape (replicate, route, asset). Each route retains the same packet
    within a date and temperature, so its replicate mean is a persistent path.
    """
    persistent = cube.mean(axis=0)
    grand = persistent.mean(axis=0)
    between = float(np.mean(np.sum(np.square(persistent - grand), axis=1)))
    within = float(np.mean(np.sum(np.square(cube - persistent[None, :, :]), axis=2)))
    total = between + within
    return {
        "between_access_path_variance": between,
        "within_cell_stochastic_variance": within,
        "persistent_share": float(between / total) if total > 0 else 0.0,
    }


def main() -> None:
    plan = _load(PLAN_PATH)
    rows = _load(CACHE_PATH)
    expected = int(plan["n_calls"])
    if len(rows) != expected:
        raise ValueError(f"repeatability cache incomplete: expected {expected}, found {len(rows)}")
    lookup = {_row_key(row): row for row in rows}
    if len(lookup) != expected:
        raise ValueError("repeatability cache has duplicate cell keys")
    planned_lookup = {_row_key(row): row for row in plan["job_manifest"]}
    if set(lookup) != set(planned_lookup):
        raise ValueError("repeatability cache keys differ from the frozen manifest")
    for key, row in lookup.items():
        planned = planned_lookup[key]
        for field in (
            "packet_index", "requested_model", "prompt_sha256", "packet_sha256",
            "response_format_sha256", "request_payload_sha256", "request_seed",
        ):
            if row.get(field) != planned.get(field):
                raise ValueError(f"frozen-manifest mismatch for {key}: {field}")

    valid_lookup = {
        key: row for key, row in lookup.items()
        if row.get("parse_ok") and row.get("response_id")
    }
    invalid_rows = [row for key, row in lookup.items() if key not in valid_lookup]

    packet_by_date = {
        row["snapshot"]["date"]: row["snapshot"] for row in _load_packets()
    }
    vendors = [vendor for vendor, _model in PILOT_MODELS]
    null_mean = matched_null(
        TEAM_SIZE,
        [f"A{index}" for index in range(14)],
        B=10_000,
        seed=2026072270,
    )["mean"]

    per_run = []
    per_persistent = []
    cell_rows = []
    decomposition_rows = []
    vectors = {}
    balanced_replicates = {}
    complete_replicate_blocks = 0
    total_replicate_blocks = len(SELECTED_DATES) * len(TEMPERATURES) * len(REPLICATES)
    for date, _return_regime, _volatility_regime in SELECTED_DATES:
        snapshot = packet_by_date[date]
        symbols = snapshot["symbols"]
        adv = np.asarray([
            snapshot["features"]["market"][symbol]["adv_bn"] for symbol in symbols
        ])
        for temperature in TEMPERATURES:
            usable_replicates = [
                replicate for replicate in REPLICATES
                if all(
                    (date, condition, vendor, temperature, replicate) in valid_lookup
                    for condition in CONDITIONS for vendor in vendors
                )
            ]
            if len(usable_replicates) < 2:
                raise ValueError(
                    f"fewer than two balanced replicates for {date}, T={temperature}"
                )
            balanced_replicates[date, temperature] = usable_replicates
            complete_replicate_blocks += len(usable_replicates)
            for condition in CONDITIONS:
                for vendor in vendors:
                    vectors[date, temperature, condition, vendor] = [
                        _book(
                            valid_lookup[date, condition, vendor, temperature, replicate],
                            symbols,
                        )
                        for replicate in usable_replicates
                    ]
                    cell_rows.append({
                        "date": date,
                        "temperature": temperature,
                        "condition": condition,
                        "vendor": vendor,
                        **_cell_reliability(vectors[date, temperature, condition, vendor]),
                    })

                cube = np.asarray([
                    [
                        vectors[date, temperature, condition, vendor][repeat_index]
                        for vendor in vendors
                    ]
                    for repeat_index in range(len(usable_replicates))
                ])
                decomposition_rows.append({
                    "date": date,
                    "temperature": temperature,
                    "condition": condition,
                    **_variance_decomposition(cube),
                })
                for repeat_index, replicate in enumerate(usable_replicates):
                    matrix = np.column_stack([
                        vectors[date, temperature, condition, vendor][repeat_index]
                        for vendor in vendors
                    ])
                    per_run.append({
                        "date": date,
                        "return_regime": snapshot["return_regime"],
                        "volatility_regime": snapshot["volatility_regime"],
                        "temperature": temperature,
                        "replicate": replicate,
                        "condition": condition,
                        **_metrics(matrix, adv, null_mean),
                    })

                persistent_matrix = np.column_stack([
                    _normalized_mean(vectors[date, temperature, condition, vendor])
                    for vendor in vendors
                ])
                per_persistent.append({
                    "date": date,
                    "return_regime": snapshot["return_regime"],
                    "volatility_regime": snapshot["volatility_regime"],
                    "temperature": temperature,
                    "condition": condition,
                    "replicates_averaged": len(usable_replicates),
                    **_metrics(persistent_matrix, adv, null_mean),
                })

    summary = {}
    seed = 2026072280
    for temperature in TEMPERATURES:
        label = f"temperature_{temperature:.1f}"
        summary[label] = {"levels": {}, "persistent_levels": {}, "contrasts": {}}
        for condition in CONDITIONS:
            raw_date_levels = []
            persistent_levels = []
            for date, _ret, _vol in SELECTED_DATES:
                raw_date_levels.append(float(np.mean([
                    row["N_eff"] for row in per_run
                    if row["date"] == date
                    and row["temperature"] == temperature
                    and row["condition"] == condition
                ])))
                persistent_levels.append(next(
                    row["N_eff"] for row in per_persistent
                    if row["date"] == date
                    and row["temperature"] == temperature
                    and row["condition"] == condition
                ))
            summary[label]["levels"][condition] = _bootstrap(np.asarray(raw_date_levels), seed)
            seed += 1
            summary[label]["persistent_levels"][condition] = _bootstrap(
                np.asarray(persistent_levels), seed
            )
            seed += 1

        for treatment in (
            "repeat_same_small", "repeat_mixed_split", "repeat_coherent_split",
        ):
            raw_differences = []
            persistent_differences = []
            for date, _ret, _vol in SELECTED_DATES:
                raw_treatment = np.mean([
                    row["N_eff"] for row in per_run
                    if row["date"] == date and row["temperature"] == temperature
                    and row["condition"] == treatment
                ])
                raw_shared = np.mean([
                    row["N_eff"] for row in per_run
                    if row["date"] == date and row["temperature"] == temperature
                    and row["condition"] == "repeat_shared_full"
                ])
                persistent_treatment = next(
                    row["N_eff"] for row in per_persistent
                    if row["date"] == date and row["temperature"] == temperature
                    and row["condition"] == treatment
                )
                persistent_shared = next(
                    row["N_eff"] for row in per_persistent
                    if row["date"] == date and row["temperature"] == temperature
                    and row["condition"] == "repeat_shared_full"
                )
                raw_differences.append(float(raw_treatment - raw_shared))
                persistent_differences.append(float(persistent_treatment - persistent_shared))
            raw = _bootstrap(np.asarray(raw_differences), seed)
            persistent = _bootstrap(np.asarray(persistent_differences), seed + 1)
            seed += 2
            raw_mean = raw["mean"]
            summary[label]["contrasts"][f"{treatment}_minus_shared"] = {
                "raw_draw_average": raw,
                "persistent_repeat_average": persistent,
                "persistent_effect_retention": (
                    float(persistent["mean"] / raw_mean) if abs(raw_mean) > 1e-12 else None
                ),
            }

        complete_dates = [
            date for date, _ret, _vol in SELECTED_DATES
            if len(balanced_replicates[date, temperature]) == len(REPLICATES)
        ]
        summary[label]["complete_panel_sensitivity"] = {
            "n_dates": len(complete_dates),
            "dates": complete_dates,
            "contrasts": {},
        }
        for treatment in (
            "repeat_same_small", "repeat_mixed_split", "repeat_coherent_split",
        ):
            differences = []
            for date in complete_dates:
                treatment_value = next(
                    row["N_eff"] for row in per_persistent
                    if row["date"] == date and row["temperature"] == temperature
                    and row["condition"] == treatment
                )
                shared_value = next(
                    row["N_eff"] for row in per_persistent
                    if row["date"] == date and row["temperature"] == temperature
                    and row["condition"] == "repeat_shared_full"
                )
                differences.append(float(treatment_value - shared_value))
            summary[label]["complete_panel_sensitivity"]["contrasts"][
                f"{treatment}_minus_shared"
            ] = _bootstrap(np.asarray(differences), seed)
            seed += 1

    reliability_summary = {}
    decomposition_summary = {}
    for temperature in TEMPERATURES:
        label = f"temperature_{temperature:.1f}"
        reliability_summary[label] = {}
        decomposition_summary[label] = {}
        for condition in CONDITIONS:
            selected_cells = [
                row for row in cell_rows
                if row["temperature"] == temperature and row["condition"] == condition
            ]
            reliability_summary[label][condition] = {
                metric: float(np.mean([row[metric] for row in selected_cells]))
                for metric in (
                    "mean_pairwise_correlation", "mean_sign_agreement",
                    "mean_l2_distance_from_cell_mean",
                )
            }
            selected_decomp = [
                row for row in decomposition_rows
                if row["temperature"] == temperature and row["condition"] == condition
            ]
            decomposition_summary[label][condition] = {
                metric: float(np.mean([row[metric] for row in selected_decomp]))
                for metric in (
                    "between_access_path_variance", "within_cell_stochastic_variance",
                    "persistent_share",
                )
            }

    stable_null_mean = matched_null(
        len(STABLE_BACKEND_VENDORS),
        [f"A{index}" for index in range(14)],
        B=10_000,
        seed=2026072271,
    )["mean"]
    stable_summary = {}
    for temperature in TEMPERATURES:
        label = f"temperature_{temperature:.1f}"
        stable_summary[label] = {"levels": {}, "contrasts": {}}
        stable_date_levels = {condition: [] for condition in CONDITIONS}
        for date, _ret, _vol in SELECTED_DATES:
            snapshot = packet_by_date[date]
            symbols = snapshot["symbols"]
            adv = np.asarray([
                snapshot["features"]["market"][symbol]["adv_bn"] for symbol in symbols
            ])
            for condition in CONDITIONS:
                matrix = np.column_stack([
                    _normalized_mean(vectors[date, temperature, condition, vendor])
                    for vendor in STABLE_BACKEND_VENDORS
                ])
                stable_date_levels[condition].append(
                    _metrics(matrix, adv, stable_null_mean)["N_eff"]
                )
        for condition in CONDITIONS:
            stable_summary[label]["levels"][condition] = _bootstrap(
                np.asarray(stable_date_levels[condition]), seed
            )
            seed += 1
        for treatment in (
            "repeat_same_small", "repeat_mixed_split", "repeat_coherent_split",
        ):
            differences = np.asarray(stable_date_levels[treatment]) - np.asarray(
                stable_date_levels["repeat_shared_full"]
            )
            stable_summary[label]["contrasts"][f"{treatment}_minus_shared"] = _bootstrap(
                differences, seed
            )
            seed += 1

    result = {
        "schema_version": "information-capacity-repeatability-results-v1",
        "plan_path": str(PLAN_PATH.relative_to(Path(DATA).parent)),
        "provenance": {
            "plan_sha256": _sha256_file(PLAN_PATH),
            "cache_path": str(CACHE_PATH.relative_to(Path(DATA).parent)),
            "cache_sha256": _sha256_file(CACHE_PATH),
        },
        "n_rows": len(rows),
        "n_dates": len(SELECTED_DATES),
        "team_size": TEAM_SIZE,
        "null_mean": null_mean,
        "analysis_population": {
            "planned_rows": expected,
            "valid_rows": len(valid_lookup),
            "invalid_rows": len(invalid_rows),
            "balanced_replicate_blocks": complete_replicate_blocks,
            "planned_replicate_blocks": total_replicate_blocks,
            "rule": (
                "Within each date-temperature panel, retain only replicate indices for "
                "which all four conditions and all seven routes produced valid books."
            ),
            "balanced_replicates_by_panel": {
                f"{date}|{temperature:.1f}": replicates
                for (date, temperature), replicates in balanced_replicates.items()
            },
            "invalid_cells": [
                {
                    "date": row["date"],
                    "condition": row["condition"],
                    "vendor": row["vendor"],
                    "temperature": row["temperature"],
                    "replicate": row["replicate"],
                    "requested_model": row["requested_model"],
                    "last_error": row.get("request_attempts", [{}])[-1].get("error"),
                }
                for row in invalid_rows
            ],
        },
        "summary": summary,
        "cell_reliability": reliability_summary,
        "variance_decomposition": decomposition_summary,
        "stable_backend_sensitivity": {
            "selection_rule": (
                "requested-model routes whose realized backend provider is at least "
                "95 percent concentrated in one provider across valid repeat calls"
            ),
            "vendors": list(STABLE_BACKEND_VENDORS),
            "team_size": len(STABLE_BACKEND_VENDORS),
            "null_mean": stable_null_mean,
            "summary": stable_summary,
        },
        "collection": {
            "total_reported_cost": float(sum(
                row.get("usage", {}).get("cost", 0.0) or 0.0 for row in rows
            )),
            "prompt_tokens": int(sum(
                row.get("usage", {}).get("prompt_tokens", 0) or 0 for row in rows
            )),
            "completion_tokens": int(sum(
                row.get("usage", {}).get("completion_tokens", 0) or 0 for row in rows
            )),
            "retried_rows": int(sum(int(row.get("format_retry_count", 0)) > 0 for row in rows)),
        },
        "per_run": per_run,
        "per_persistent_date": per_persistent,
        "per_cell_reliability": cell_rows,
        "per_date_variance_decomposition": decomposition_rows,
    }
    _atomic_json(RESULT_PATH, result)

    print("Repeatability result")
    for temperature in TEMPERATURES:
        label = f"temperature_{temperature:.1f}"
        levels = summary[label]["persistent_levels"]
        print(
            f"  T={temperature:.1f}: persistent N_eff "
            f"shared={levels['repeat_shared_full']['mean']:.3f}, "
            f"same-small={levels['repeat_same_small']['mean']:.3f}, "
            f"mixed={levels['repeat_mixed_split']['mean']:.3f}, "
            f"coherent={levels['repeat_coherent_split']['mean']:.3f}"
        )
        for treatment in ("repeat_mixed_split", "repeat_coherent_split"):
            contrast = summary[label]["contrasts"][f"{treatment}_minus_shared"]
            print(
                f"    {treatment}: persistent gap="
                f"{contrast['persistent_repeat_average']['mean']:+.3f}, "
                f"retention={contrast['persistent_effect_retention']:.3f}"
            )
        stable = stable_summary[label]
        print(
            "    stable-backend K=5: "
            f"shared={stable['levels']['repeat_shared_full']['mean']:.3f}, "
            f"mixed={stable['levels']['repeat_mixed_split']['mean']:.3f}, "
            f"coherent={stable['levels']['repeat_coherent_split']['mean']:.3f}"
        )
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
