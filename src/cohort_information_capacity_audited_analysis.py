"""Analyze the audit-corrected K=7 architecture against the unified shared arm."""
from __future__ import annotations

import itertools
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA, RESULTS
from cohort_crossvendor_hardening import matched_null
from cohort_information_capacity_analysis import METRICS, _book, _mean_metrics, _metrics
from cohort_information_capacity_audited_architecture import (
    CACHE_PATH as AUDITED_CACHE_PATH,
    PLAN_PATH,
)
from cohort_information_capacity_integrity import (
    CACHE_PATH as INTEGRITY_CACHE_PATH,
    CATALOG_PATH,
    INTEGRITY_CORRECTION_ID,
    MASKED_CACHE_PATH,
    TEAM_SIZE,
    _load_packets,
)
from cohort_information_diversity import PILOT_MODELS, _atomic_json, _sign_flip_p


RESULT_PATH = Path(RESULTS) / "information_capacity_audited_architecture_results.json"
SUMMARY_PATH = Path(RESULTS) / "information_capacity_audited_architecture_summary.csv"
EVENT_DIFF_PATH = Path(RESULTS) / "information_diversity_integrity_event_diff.json"
BOOTSTRAP_DRAWS = 20_000
AI_NARRATIVE = re.compile(r"\bAI\b|artificial intelligence|AI-driven|AI leadership", re.I)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _current(row: dict) -> bool:
    return bool(
        row.get("parse_ok")
        and row.get("integrity_correction_id") == INTEGRITY_CORRECTION_ID
        and row.get("response_id")
        and row.get("request_payload_sha256")
    )


def _contains_ai_narrative(row: dict) -> bool:
    text = " ".join([
        str(row.get("thesis", "")),
        *map(str, row.get("primary_signals", [])),
    ])
    return bool(AI_NARRATIVE.search(text))


def _configuration_metrics(
    architecture: str,
    vendors: list[str],
    lookup: dict[tuple[str, str, int], dict],
    symbols: list[str],
    full_consensus: np.ndarray,
    forward: np.ndarray,
    adv: np.ndarray,
    null_mean: float,
    exclude_explicit_narratives: bool = False,
) -> tuple[dict, int]:
    books = {
        (vendor, packet): _book(lookup[vendor, architecture, packet], symbols)
        for vendor in vendors
        for packet in range(1, TEAM_SIZE + 1)
    }
    configurations = []
    for assignment in itertools.permutations(range(1, TEAM_SIZE + 1)):
        rows = [
            lookup[vendor, architecture, packet]
            for vendor, packet in zip(vendors, assignment)
        ]
        if exclude_explicit_narratives and any(
            _contains_ai_narrative(row) for row in rows
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
        raise ValueError(f"{architecture}: no configurations survive the narrative screen")
    return _mean_metrics(configurations), len(configurations)


def _broadcast_metrics(
    architecture: str,
    vendors: list[str],
    lookup: dict[tuple[str, str, int], dict],
    symbols: list[str],
    full_consensus: np.ndarray,
    forward: np.ndarray,
    adv: np.ndarray,
    null_mean: float,
) -> tuple[dict, int]:
    """Broadcast one short packet to every route, then average across packets."""
    configurations = []
    for packet in range(1, TEAM_SIZE + 1):
        rows = [lookup[vendor, architecture, packet] for vendor in vendors]
        matrix = np.column_stack([_book(row, symbols) for row in rows])
        configurations.append(
            _metrics(matrix, full_consensus, forward, adv, null_mean, rows)
        )
    return _mean_metrics(configurations), len(configurations)


def _within_route_metrics(
    architecture: str,
    vendors: list[str],
    lookup: dict[tuple[str, str, int], dict],
    symbols: list[str],
    full_consensus: np.ndarray,
    forward: np.ndarray,
    adv: np.ndarray,
    null_mean: float,
) -> dict:
    """Measure a seven-packet split team while holding the model route fixed."""
    by_vendor = {}
    for vendor in vendors:
        rows = [lookup[vendor, architecture, packet] for packet in range(1, TEAM_SIZE + 1)]
        matrix = np.column_stack([_book(row, symbols) for row in rows])
        by_vendor[vendor] = _metrics(
            matrix, full_consensus, forward, adv, null_mean, rows
        )
    return _mean_metrics(list(by_vendor.values()))


def _date_rows() -> list[dict]:
    shared_rows = [
        row for row in _load(INTEGRITY_CACHE_PATH)
        if row.get("architecture") == "shared" and _current(row)
    ]
    audited_rows = [row for row in _load(AUDITED_CACHE_PATH) if _current(row)]
    if len(shared_rows) != 34 * 7:
        raise ValueError(f"expected 238 unified shared rows, found {len(shared_rows)}")
    if len(audited_rows) != 34 * 2 * 7 * 7:
        raise ValueError(f"expected 3332 audited rows, found {len(audited_rows)}")
    shared_lookup = {(row["date"], row["vendor"]): row for row in shared_rows}
    audited_lookup = {
        (row["date"], row["vendor"], row["architecture"], int(row["packet_index"])): row
        for row in audited_rows
    }
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    null_mean = matched_null(
        TEAM_SIZE,
        [f"A{index}" for index in range(14)],
        B=10_000,
        seed=2026071700 + TEAM_SIZE,
    )["mean"]
    output = []
    for item in _load_packets():
        snapshot = item["snapshot"]
        date = snapshot["date"]
        symbols = snapshot["symbols"]
        date_shared = [shared_lookup[date, vendor] for vendor in vendors]
        shared_matrix = np.column_stack([_book(row, symbols) for row in date_shared])
        full_consensus = shared_matrix.mean(axis=1)
        forward = np.asarray([snapshot["forward_return_21d"][symbol] for symbol in symbols])
        adv = np.asarray([snapshot["features"]["market"][symbol]["adv_bn"] for symbol in symbols])
        shared = _metrics(
            shared_matrix, full_consensus, forward, adv, null_mean, date_shared
        )
        lookup = {
            (vendor, architecture, packet):
                audited_lookup[date, vendor, architecture, packet]
            for vendor in vendors
            for architecture in ("mixed_audited", "coherent_audited")
            for packet in range(1, TEAM_SIZE + 1)
        }
        mixed, mixed_n = _configuration_metrics(
            "mixed_audited", vendors, lookup, symbols, full_consensus, forward, adv, null_mean
        )
        coherent, coherent_n = _configuration_metrics(
            "coherent_audited", vendors, lookup, symbols, full_consensus, forward, adv, null_mean
        )
        mixed_broadcast, mixed_broadcast_n = _broadcast_metrics(
            "mixed_audited", vendors, lookup, symbols, full_consensus, forward, adv, null_mean
        )
        coherent_broadcast, coherent_broadcast_n = _broadcast_metrics(
            "coherent_audited", vendors, lookup, symbols, full_consensus, forward, adv, null_mean
        )
        mixed_within_route = _within_route_metrics(
            "mixed_audited", vendors, lookup, symbols, full_consensus, forward, adv, null_mean
        )
        coherent_within_route = _within_route_metrics(
            "coherent_audited", vendors, lookup, symbols, full_consensus, forward, adv, null_mean
        )
        mixed_without_explicit_ai, mixed_clean_n = _configuration_metrics(
            "mixed_audited", vendors, lookup, symbols, full_consensus, forward, adv, null_mean,
            exclude_explicit_narratives=True,
        )
        coherent_without_explicit_ai, coherent_clean_n = _configuration_metrics(
            "coherent_audited", vendors, lookup, symbols, full_consensus, forward, adv, null_mean,
            exclude_explicit_narratives=True,
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
            "mixed_short_packet_broadcast": mixed_broadcast,
            "coherent_short_packet_broadcast": coherent_broadcast,
            "mixed_within_route": mixed_within_route,
            "coherent_within_route": coherent_within_route,
            "mixed_without_explicit_ai_narratives": mixed_without_explicit_ai,
            "coherent_without_explicit_ai_narratives": coherent_without_explicit_ai,
            "configuration_counts": {
                "shared": 1,
                "mixed": mixed_n,
                "coherent": coherent_n,
                "mixed_short_packet_broadcast": mixed_broadcast_n,
                "coherent_short_packet_broadcast": coherent_broadcast_n,
                "mixed_without_explicit_ai_narratives": mixed_clean_n,
                "coherent_without_explicit_ai_narratives": coherent_clean_n,
            },
        })
        print(f"  analyzed audited architecture {date}", flush=True)
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


def _latest_route_registration() -> dict:
    catalog = _load(CATALOG_PATH)
    latest = max(catalog["models"], key=lambda row: int(row["created"]))
    registered = datetime.fromtimestamp(int(latest["created"]), tz=timezone.utc)
    return {
        "vendor": latest["vendor"],
        "requested_model": latest["requested_model"],
        "canonical_slug": latest["canonical_slug"],
        "registered_at_utc": registered.isoformat(),
        "registration_date": registered.date().isoformat(),
        "selection_rule": "decision date strictly after the latest recorded route registration",
    }


def _summaries(
    date_rows: list[dict],
    numeric_affected_dates: set[str],
    latest_registration: dict,
) -> dict:
    seed = 2026072150
    levels = {}
    for design in ("shared", "mixed", "coherent"):
        levels[design] = {}
        for metric in METRICS:
            levels[design][metric] = _bootstrap(
                np.asarray([row[design][metric] for row in date_rows]), seed
            )
            seed += 1
    comparisons = {}
    for treatment, control in (
        ("mixed", "shared"),
        ("coherent", "shared"),
        ("coherent", "mixed"),
    ):
        key = f"{treatment}_minus_{control}"
        comparisons[key] = {}
        for metric in (
            "N_eff", "same_side_overlap", "surviving_gross",
            "convex_order_concentration", "consensus_forward_return",
            "full_consensus_alignment", "team_prompt_tokens", "team_reported_cost",
        ):
            comparisons[key][metric] = _paired(
                date_rows, treatment, control, metric, seed
            )
            seed += 1
    sensitivity = {}
    sensitivity_samples = (
        ("SEC_numeric_affected_dates", [
            row for row in date_rows if row["date"] in numeric_affected_dates
        ]),
        ("SEC_numeric_unaffected_dates", [
            row for row in date_rows if row["date"] not in numeric_affected_dates
        ]),
        ("development_dates", [
            row for row in date_rows if row["sample"] == "development"
        ]),
        ("complement_dates", [
            row for row in date_rows if row["sample"] == "complement"
        ]),
        ("post_latest_route_registration_dates", [
            row for row in date_rows
            if row["date"] > latest_registration["registration_date"]
        ]),
    )
    for label, rows in sensitivity_samples:
        sensitivity[label] = {
            "n_dates": len(rows),
            "dates": [row["date"] for row in rows],
            "mixed_minus_shared_N_eff": _paired(rows, "mixed", "shared", "N_eff", seed),
            "coherent_minus_shared_N_eff": _paired(
                rows, "coherent", "shared", "N_eff", seed + 1
            ),
            "coherent_minus_shared_overlap": _paired(
                rows, "coherent", "shared", "same_side_overlap", seed + 2
            ),
            "coherent_minus_shared_order_concentration": _paired(
                rows, "coherent", "shared", "convex_order_concentration", seed + 3
            ),
        }
        seed += 4
    sensitivity["post_latest_route_registration_dates"]["route_registration"] = (
        latest_registration
    )
    mechanism_levels = {}
    for design in (
        "mixed_short_packet_broadcast",
        "coherent_short_packet_broadcast",
        "mixed_without_explicit_ai_narratives",
        "coherent_without_explicit_ai_narratives",
    ):
        mechanism_levels[design] = {}
        for metric in METRICS:
            mechanism_levels[design][metric] = _bootstrap(
                np.asarray([row[design][metric] for row in date_rows]), seed
            )
            seed += 1
    mechanism_comparisons = {}
    for treatment, control in (
        ("mixed", "mixed_short_packet_broadcast"),
        ("coherent", "coherent_short_packet_broadcast"),
        ("mixed_without_explicit_ai_narratives", "mixed"),
        ("coherent_without_explicit_ai_narratives", "coherent"),
    ):
        key = f"{treatment}_minus_{control}"
        mechanism_comparisons[key] = {}
        for metric in (
            "N_eff", "same_side_overlap", "surviving_gross",
            "convex_order_concentration", "full_consensus_alignment",
        ):
            mechanism_comparisons[key][metric] = _paired(
                date_rows, treatment, control, metric, seed
            )
            seed += 1
    within_route = {}
    for design in ("mixed", "coherent"):
        row_key = f"{design}_within_route"
        within_route[design] = {
            "definition": (
                "seven distinct packets sent through one fixed model route; "
                "date-level values average the seven routes"
            ),
            "levels": {
                metric: _bootstrap(
                    np.asarray([row[row_key][metric] for row in date_rows]), seed + offset
                )
                for offset, metric in enumerate(METRICS)
            },
        }
        seed += len(METRICS)
    return {
        "levels": levels,
        "paired_comparisons": comparisons,
        "sample_sensitivity": sensitivity,
        "mechanism_controls": {
            "levels": mechanism_levels,
            "paired_comparisons": mechanism_comparisons,
            "short_packet_broadcast_definition": (
                "all seven routes receive the same reduced packet; results average the seven "
                "packet-specific teams within date"
            ),
        },
        "within_route_split_diagnostic": within_route,
    }


def _collection_diagnostics() -> dict:
    rows = [
        row for row in _load(INTEGRITY_CACHE_PATH)
        if row.get("architecture") == "shared" and _current(row)
    ] + [row for row in _load(AUDITED_CACHE_PATH) if _current(row)]
    retries = [row for row in rows if int(row.get("format_retry_count", 0)) > 0]
    split_rows = [row for row in rows if row.get("architecture") != "shared"]
    return {
        "rows": len(rows),
        "response_id_coverage": sum(bool(row.get("response_id")) for row in rows) / len(rows),
        "response_provider_coverage": sum(bool(row.get("response_provider")) for row in rows) / len(rows),
        "system_fingerprint_coverage": sum(bool(row.get("system_fingerprint")) for row in rows) / len(rows),
        "seed_requested_rows": sum(bool(row.get("seed_supported")) for row in rows),
        "retried_rows": len(retries),
        "retry_rate": len(retries) / len(rows),
        "terminal_failures": 0,
        "split_responses_with_explicit_ai_narrative": sum(
            _contains_ai_narrative(row) for row in split_rows
        ),
        "retry_by_vendor": {
            vendor: {
                "n": sum(row["vendor"] == vendor for row in rows),
                "retried": sum(
                    row["vendor"] == vendor and int(row.get("format_retry_count", 0)) > 0
                    for row in rows
                ),
            }
            for vendor, _ in PILOT_MODELS
        },
    }


def _legacy_macro_audit() -> dict:
    visible_rows = [
        row for row in _load(INTEGRITY_CACHE_PATH)
        if row.get("architecture") == "coherent"
        and int(row.get("packet_index", 0)) == TEAM_SIZE
        and _current(row)
    ]
    masked_rows = [row for row in _load(MASKED_CACHE_PATH) if _current(row)]
    visible = {(row["date"], row["vendor"]): row for row in visible_rows}
    masked = {(row["date"], row["vendor"]): row for row in masked_rows}
    paired_keys = sorted(set(visible).intersection(masked))
    return {
        "status": "discarded design audit; not part of the final treatment estimates",
        "planned_pairs": 34 * 7,
        "successful_matched_pairs": len(paired_keys),
        "visible_ticker_explicit_ai_narratives": sum(
            _contains_ai_narrative(visible[key]) for key in paired_keys
        ),
        "masked_ticker_explicit_ai_narratives": sum(
            _contains_ai_narrative(masked[key]) for key in paired_keys
        ),
        "action": (
            "the macro-only seat was removed; every final split seat receives "
            "asset-specific evidence"
        ),
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


def main() -> None:
    event_diff = _load(EVENT_DIFF_PATH)
    numeric_fields = {
        "days_since_latest_8k", "days_since_periodic_filing", "earnings_release_90d",
        "eight_k_30d", "eight_k_90d", "material_event_90d",
    }
    numeric_affected_dates = {
        row["date"]
        for row in event_diff["rows"]
        if numeric_fields.intersection(row["differences"])
    }
    date_rows = _date_rows()
    latest_registration = _latest_route_registration()
    result = {
        "schema_version": "information-capacity-audited-architecture-results-v2",
        "integrity_correction_id": INTEGRITY_CORRECTION_ID,
        "design_status": "post-audit integrity rerun; all 34 dates form one pooled panel",
        "team_size": TEAM_SIZE,
        "n_dates": len(date_rows),
        "SEC_event_correction": {
            "numeric_affected_dates": sorted(numeric_affected_dates),
            "n_numeric_affected_dates": len(numeric_affected_dates),
            "n_changed_numeric_values": sum(
                count for field, count in event_diff["changed_field_counts"].items()
                if field in numeric_fields
            ),
            "n_changed_values_including_labels": event_diff["n_changed_field_values"],
        },
        "design_audit": {
            "legacy_macro_ticker_sensitivity": _legacy_macro_audit(),
            "final_explicit_narrative_screen": (
                "main estimates are accompanied by a configuration-level exclusion sensitivity"
            ),
            "latest_route_registration": latest_registration,
        },
        "collection_diagnostics": _collection_diagnostics(),
        "per_date": date_rows,
        **_summaries(date_rows, numeric_affected_dates, latest_registration),
        "plan_path": str(PLAN_PATH.relative_to(Path(DATA).parent)),
    }
    _atomic_json(RESULT_PATH, result)
    pd.DataFrame(_flat_rows(date_rows)).to_csv(SUMMARY_PATH, index=False)
    print("Audit-corrected K=7 result")
    for design in ("shared", "mixed", "coherent"):
        level = result["levels"][design]
        print(
            f"  {design}: N_eff={level['N_eff']['mean']:.3f}, "
            f"overlap={level['same_side_overlap']['mean']:.3f}, "
            f"Q={level['convex_order_concentration']['mean']:.3f}"
        )
    for comparison in ("mixed_minus_shared", "coherent_minus_shared", "coherent_minus_mixed"):
        values = result["paired_comparisons"][comparison]
        print(
            f"  {comparison}: dN={values['N_eff']['mean']:+.3f} "
            f"CI={values['N_eff']['date_bootstrap_95']}; "
            f"dO={values['same_side_overlap']['mean']:+.3f}"
        )
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
