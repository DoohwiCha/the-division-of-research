"""Analyze the SEC- and prompt-corrected K=7 integrity rerun."""
from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA, RESULTS
from cohort_crossvendor_hardening import matched_null
from cohort_information_capacity_analysis import (
    METRICS,
    _book,
    _mean_metrics,
    _metrics,
)
from cohort_information_capacity_experiment import PILOT_MODELS
from cohort_information_capacity_integrity import (
    CACHE_PATH,
    INTEGRITY_CORRECTION_ID,
    MASKED_CACHE_PATH,
    PLAN_PATH,
    TEAM_SIZE,
    _job_key,
    _load_packets,
)
from cohort_information_diversity import _atomic_json, _sign_flip_p
from cohort_information_diversity_scaling import IMPACT_DELTA


RESULT_PATH = Path(RESULTS) / "information_capacity_integrity_results.json"
SUMMARY_PATH = Path(RESULTS) / "information_capacity_integrity_summary.csv"
EVENT_DIFF_PATH = Path(RESULTS) / "information_diversity_integrity_event_diff.json"
OLD_RESULT_PATH = Path(RESULTS) / "information_capacity_experiment_results.json"
BOOTSTRAP_DRAWS = 20_000
AI_NARRATIVE = re.compile(r"\bAI\b|artificial intelligence|AI-driven|AI leadership", re.I)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _complete_rows(path: Path, expected: int) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = _load(path)
    usable = [row for row in rows if row.get("parse_ok")]
    stale = [
        row for row in usable
        if row.get("integrity_correction_id") != INTEGRITY_CORRECTION_ID
        or not row.get("response_id")
        or not row.get("request_payload_sha256")
    ]
    if len(rows) != expected or len(usable) != expected or stale:
        raise ValueError(
            f"{path.name}: expected {expected} complete current rows, "
            f"found rows={len(rows)}, usable={len(usable)}, stale={len(stale)}"
        )
    if len({_job_key(row) for row in rows}) != expected:
        raise ValueError(f"{path.name} contains duplicate job keys")
    return rows


def _contains_ai_narrative(row: dict) -> bool:
    text = " ".join([str(row.get("thesis", "")), *map(str, row.get("primary_signals", []))])
    return bool(AI_NARRATIVE.search(text))


def _configuration_metrics(
    *,
    architecture: str,
    vendors: list[str],
    lookup: dict[tuple[str, str, int], dict],
    symbols: list[str],
    full_consensus: np.ndarray,
    forward: np.ndarray,
    adv: np.ndarray,
    null_mean: float,
    macro_masked_lookup: dict[str, dict] | None = None,
    exclude_explicit_narratives: bool = False,
) -> tuple[dict, int]:
    books = {
        (vendor, packet): _book(lookup[vendor, architecture, packet], symbols)
        for vendor in vendors
        for packet in range(1, TEAM_SIZE + 1)
    }
    if macro_masked_lookup is not None:
        for vendor in vendors:
            books[vendor, TEAM_SIZE] = _book(macro_masked_lookup[vendor], symbols)
    configurations = []
    for assignment in itertools.permutations(range(1, TEAM_SIZE + 1)):
        rows = [lookup[vendor, architecture, packet] for vendor, packet in zip(vendors, assignment)]
        if macro_masked_lookup is not None:
            rows = [
                macro_masked_lookup[vendor] if packet == TEAM_SIZE else row
                for vendor, packet, row in zip(vendors, assignment, rows)
            ]
        if exclude_explicit_narratives and any(
            packet == TEAM_SIZE and _contains_ai_narrative(row)
            for packet, row in zip(assignment, rows)
        ):
            continue
        matrix = np.column_stack([
            books[vendor, packet]
            for vendor, packet in zip(vendors, assignment)
        ])
        configurations.append(
            _metrics(matrix, full_consensus, forward, adv, null_mean, rows)
        )
    if not configurations:
        raise ValueError("no configurations survive the requested sensitivity")
    return _mean_metrics(configurations), len(configurations)


def _date_rows(main_rows: list[dict], masked_rows: list[dict]) -> list[dict]:
    packet_rows = _load_packets()
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    by_date = {}
    for row in main_rows:
        by_date.setdefault(row["date"], []).append(row)
    masked_by_date = {}
    for row in masked_rows:
        masked_by_date.setdefault(row["date"], []).append(row)
    null_mean = matched_null(
        TEAM_SIZE,
        [f"A{index}" for index in range(14)],
        B=10_000,
        seed=2026071700 + TEAM_SIZE,
    )["mean"]
    output = []
    for item in packet_rows:
        snapshot = item["snapshot"]
        date = snapshot["date"]
        symbols = snapshot["symbols"]
        rows = by_date[date]
        lookup = {
            (row["vendor"], row["architecture"], int(row["packet_index"])): row
            for row in rows
        }
        masked_lookup = {row["vendor"]: row for row in masked_by_date[date]}
        shared_rows = [lookup[vendor, "shared", 0] for vendor in vendors]
        shared_matrix = np.column_stack([_book(row, symbols) for row in shared_rows])
        full_consensus = shared_matrix.mean(axis=1)
        forward = np.asarray([snapshot["forward_return_21d"][symbol] for symbol in symbols])
        adv = np.asarray([snapshot["features"]["market"][symbol]["adv_bn"] for symbol in symbols])
        shared = _metrics(
            shared_matrix, full_consensus, forward, adv, null_mean, shared_rows
        )
        mixed, mixed_n = _configuration_metrics(
            architecture="mixed",
            vendors=vendors,
            lookup=lookup,
            symbols=symbols,
            full_consensus=full_consensus,
            forward=forward,
            adv=adv,
            null_mean=null_mean,
        )
        coherent, coherent_n = _configuration_metrics(
            architecture="coherent",
            vendors=vendors,
            lookup=lookup,
            symbols=symbols,
            full_consensus=full_consensus,
            forward=forward,
            adv=adv,
            null_mean=null_mean,
        )
        coherent_clean, coherent_clean_n = _configuration_metrics(
            architecture="coherent",
            vendors=vendors,
            lookup=lookup,
            symbols=symbols,
            full_consensus=full_consensus,
            forward=forward,
            adv=adv,
            null_mean=null_mean,
            exclude_explicit_narratives=True,
        )
        coherent_macro_masked, masked_n = _configuration_metrics(
            architecture="coherent",
            vendors=vendors,
            lookup=lookup,
            symbols=symbols,
            full_consensus=full_consensus,
            forward=forward,
            adv=adv,
            null_mean=null_mean,
            macro_masked_lookup=masked_lookup,
        )
        output.append({
            "sample": item["sample"],
            "snapshot_id": snapshot["snapshot_id"],
            "date": date,
            "return_regime": snapshot["return_regime"],
            "volatility_regime": snapshot["volatility_regime"],
            "team_size": TEAM_SIZE,
            "shared": shared,
            "mixed": mixed,
            "coherent": coherent,
            "coherent_without_explicit_ai_narratives": coherent_clean,
            "coherent_macro_ticker_masked": coherent_macro_masked,
            "configuration_counts": {
                "shared": 1,
                "mixed": mixed_n,
                "coherent": coherent_n,
                "coherent_without_explicit_ai_narratives": coherent_clean_n,
                "coherent_macro_ticker_masked": masked_n,
            },
        })
        print(f"  analyzed integrity rerun {date}", flush=True)
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


def _level(rows: list[dict], design: str, metric: str, seed: int) -> dict:
    return _bootstrap(np.asarray([row[design][metric] for row in rows]), seed)


def _paired(rows: list[dict], treatment: str, control: str, metric: str, seed: int) -> dict:
    values = np.asarray([
        row[treatment][metric] - row[control][metric]
        for row in rows
    ])
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


def _summaries(date_rows: list[dict], affected_dates: set[str]) -> dict:
    designs = (
        "shared",
        "mixed",
        "coherent",
        "coherent_without_explicit_ai_narratives",
        "coherent_macro_ticker_masked",
    )
    seed = 2026072120
    levels = {}
    comparisons = {}
    for design in designs:
        levels[design] = {}
        for metric in METRICS:
            levels[design][metric] = _level(date_rows, design, metric, seed)
            seed += 1
    for treatment, control in (
        ("mixed", "shared"),
        ("coherent", "shared"),
        ("coherent", "mixed"),
        ("coherent_without_explicit_ai_narratives", "shared"),
        ("coherent_macro_ticker_masked", "shared"),
        ("coherent_macro_ticker_masked", "coherent"),
    ):
        key = f"{treatment}_minus_{control}"
        comparisons[key] = {}
        for metric in (
            "N_eff", "same_side_overlap", "surviving_gross",
            "convex_order_concentration", "consensus_forward_return",
        ):
            comparisons[key][metric] = _paired(date_rows, treatment, control, metric, seed)
            seed += 1
    period_sensitivity = {}
    for label, rows in (
        ("SEC_affected_dates", [row for row in date_rows if row["date"] in affected_dates]),
        ("SEC_unaffected_dates", [row for row in date_rows if row["date"] not in affected_dates]),
        ("development_dates", [row for row in date_rows if row["sample"] == "development"]),
        ("complement_dates", [row for row in date_rows if row["sample"] == "complement"]),
    ):
        period_sensitivity[label] = {
            "n_dates": len(rows),
            "mixed_minus_shared_N_eff": _paired(rows, "mixed", "shared", "N_eff", seed),
            "coherent_minus_shared_N_eff": _paired(rows, "coherent", "shared", "N_eff", seed + 1),
            "coherent_minus_shared_overlap": _paired(
                rows, "coherent", "shared", "same_side_overlap", seed + 2
            ),
        }
        seed += 3
    return {
        "levels": levels,
        "paired_comparisons": comparisons,
        "period_sensitivity": period_sensitivity,
    }


def _collection_diagnostics(main_rows: list[dict], masked_rows: list[dict]) -> dict:
    all_rows = [*main_rows, *masked_rows]
    main_macro = [
        row for row in main_rows
        if row["architecture"] == "coherent" and int(row["packet_index"]) == TEAM_SIZE
    ]
    masked_mentions = [row for row in masked_rows if _contains_ai_narrative(row)]
    main_mentions = [row for row in main_macro if _contains_ai_narrative(row)]
    retry_rows = [row for row in all_rows if int(row.get("format_retry_count", 0)) > 0]
    retry_by_vendor = {
        vendor: {
            "n": sum(row["vendor"] == vendor for row in all_rows),
            "retried": sum(
                row["vendor"] == vendor and int(row.get("format_retry_count", 0)) > 0
                for row in all_rows
            ),
        }
        for vendor, _ in PILOT_MODELS
    }
    return {
        "main_rows": len(main_rows),
        "masked_macro_rows": len(masked_rows),
        "response_id_coverage": sum(bool(row.get("response_id")) for row in all_rows) / len(all_rows),
        "response_provider_coverage": sum(bool(row.get("response_provider")) for row in all_rows) / len(all_rows),
        "system_fingerprint_coverage": sum(bool(row.get("system_fingerprint")) for row in all_rows) / len(all_rows),
        "seed_requested_rows": sum(bool(row.get("seed_supported")) for row in all_rows),
        "retry_rows": len(retry_rows),
        "retry_rate": len(retry_rows) / len(all_rows),
        "retry_by_vendor": retry_by_vendor,
        "all_retry_attempts_use_frozen_payload": all(
            row.get("request_payload_sha256")
            and all(attempt.get("attempt") for attempt in row.get("request_attempts", []))
            for row in all_rows
        ),
        "macro_only_responses": len(main_macro),
        "macro_responses_with_explicit_ai_narrative": len(main_mentions),
        "macro_ticker_masked_responses_with_explicit_ai_narrative": len(masked_mentions),
        "macro_ai_mentions_by_vendor": {
            vendor: sum(row["vendor"] == vendor for row in main_mentions)
            for vendor, _ in PILOT_MODELS
        },
    }


def _old_comparison() -> dict:
    old = _load(OLD_RESULT_PATH)["levels"][str(TEAM_SIZE)]
    return {
        design: {
            metric: old[design][metric]["mean"]
            for metric in ("N_eff", "same_side_overlap", "convex_order_concentration")
        }
        for design in ("shared", "mixed", "coherent")
    }


def _flat_rows(date_rows: list[dict]) -> list[dict]:
    output = []
    designs = (
        "shared", "mixed", "coherent",
        "coherent_without_explicit_ai_narratives", "coherent_macro_ticker_masked",
    )
    for row in date_rows:
        common = {key: row[key] for key in (
            "sample", "snapshot_id", "date", "return_regime", "volatility_regime", "team_size"
        )}
        for design in designs:
            output.append({**common, "design": design, **row[design]})
    return output


def main() -> None:
    main_rows = _complete_rows(CACHE_PATH, 34 * 7 * 15)
    masked_rows = _complete_rows(MASKED_CACHE_PATH, 34 * 7)
    event_diff = _load(EVENT_DIFF_PATH)
    date_rows = _date_rows(main_rows, masked_rows)
    summaries = _summaries(date_rows, set(event_diff["affected_dates"]))
    result = {
        "schema_version": "information-capacity-integrity-results-v1",
        "integrity_correction_id": INTEGRITY_CORRECTION_ID,
        "design_status": "post-audit integrity rerun; all 34 dates treated as one pooled panel",
        "team_size": TEAM_SIZE,
        "n_dates": len(date_rows),
        "impact_exponent": IMPACT_DELTA,
        "event_data_correction": {
            key: event_diff[key]
            for key in (
                "n_symbol_dates", "n_changed_symbol_dates", "n_changed_field_values",
                "changed_field_counts", "affected_dates", "affected_symbols",
            )
        },
        "predecessor_K7_levels": _old_comparison(),
        "collection_diagnostics": _collection_diagnostics(main_rows, masked_rows),
        "per_date": date_rows,
        **summaries,
        "plan_path": str(PLAN_PATH.relative_to(Path(DATA).parent)),
    }
    _atomic_json(RESULT_PATH, result)
    pd.DataFrame(_flat_rows(date_rows)).to_csv(SUMMARY_PATH, index=False)
    levels = result["levels"]
    print("Corrected K=7 integrity rerun")
    for design in (
        "shared", "mixed", "coherent",
        "coherent_without_explicit_ai_narratives", "coherent_macro_ticker_masked",
    ):
        print(
            f"  {design}: N_eff={levels[design]['N_eff']['mean']:.3f}, "
            f"overlap={levels[design]['same_side_overlap']['mean']:.3f}, "
            f"Q={levels[design]['convex_order_concentration']['mean']:.3f}"
        )
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
