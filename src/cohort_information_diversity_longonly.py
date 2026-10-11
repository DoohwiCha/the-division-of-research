"""Collect and analyze the long-only mandate-transfer experiment.

The existing information-architecture experiment uses dollar-neutral active
books.  This extension asks whether the same ordering survives a fully
invested long-only mandate.  Raw holdings are never compared directly because
their common positive market exposure would mechanically inflate similarity.
Every portfolio is instead expressed as active weights relative to the same
equal-weight benchmark before independence and overlap are measured.

The collection is staged but the complete 34-date design is frozen at once:

    # Freeze and inspect the complete design without making API calls.
    PYTHONPATH=src python3 src/cohort_information_diversity_longonly.py --dry-run

    # Collect the 12-date pilot first (756 calls).
    OPENROUTER_API_KEY=... PYTHONPATH=src \
      python3 src/cohort_information_diversity_longonly.py --sample pilot

    # Extend only after inspecting the pre-specified pilot output (1,386 calls).
    OPENROUTER_API_KEY=... PYTHONPATH=src \
      python3 src/cohort_information_diversity_longonly.py --sample held-out

    # Reanalyze whichever complete stages are present.
    PYTHONPATH=src python3 src/cohort_information_diversity_longonly.py --analyze-only
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from _paths import DATA, RESULTS
from cohort_analyze import mean_pair_corr, n_eff
from cohort_capacity_delta import gross_survival
from cohort_crossvendor import OPENROUTER_URL, _extract_json, _post
from cohort_crossvendor_hardening import book_permutation_null, same_side_overlap
from cohort_equity_locked import _clean_weights
from cohort_information_diversity import (
    BOOTSTRAP_DRAWS,
    CHANNELS,
    PILOT_MODELS,
    TEMPERATURE,
    _atomic_json,
    _credit_exhausted,
    _sha256,
    _sign_flip_p,
)
from pit_correction import (
    cache_record_is_current,
    record as pit_correction_record,
    require_enabled as require_pit_correction,
    response_marker as pit_response_marker,
)
from cohort_information_diversity_random_placebo import (
    PACKET_LABELS,
    _packet_text as _random_packet_text,
    _partition,
    load_packets,
)


CACHE_PATH = Path(DATA) / "cohort_agents_information_diversity_longonly.json"
PLAN_PATH = Path(RESULTS) / "information_diversity_longonly_plan.json"
RESULT_PATH = Path(RESULTS) / "information_diversity_longonly_results.json"
SUMMARY_PATH = Path(RESULTS) / "information_diversity_longonly_summary.csv"

PROMPT_SPEC_VERSION = "information-diversity-long-only-active-v1"
PLAN_SCHEMA_VERSION = "information-diversity-long-only-plan-v1"
RESULT_SCHEMA_VERSION = "information-diversity-long-only-results-v3"
RANDOM_CONDITIONS = tuple(f"random_{label}" for label in PACKET_LABELS)
CONDITIONS = ("rich", *CHANNELS, *RANDOM_CONDITIONS)
DEFAULT_MAX_WORKERS = int(os.environ.get("INFORMATION_DIVERSITY_MAX_WORKERS", "5"))
PERMUTATION_DRAWS = 2_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _usage(response: dict) -> dict:
    raw = response.get("usage") or {}
    keep = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
    return {key: raw[key] for key in keep if raw.get(key) is not None}


def _structured_packet_text(snapshot: dict, condition: str) -> str:
    if condition == "rich":
        return snapshot["tables"]["rich"]
    label = {
        "market": "MARKET BEHAVIOR AND LIQUIDITY",
        "fundamentals": "POINT-IN-TIME FUNDAMENTALS AND VALUATION",
        "events": "RECENT SEC FILING EVENTS",
        "risk": "RISK AND MACRO CONTEXT",
    }[condition]
    return label + "\n" + snapshot["tables"][condition]


def _source_note(condition: str) -> str:
    if condition == "rich":
        return "You receive the desk's complete research packet."
    if condition in CHANNELS:
        label = {
            "market": "market behavior and liquidity",
            "fundamentals": "fundamentals and valuation",
            "events": "recent public-filing events",
            "risk": "risk and macro context",
        }[condition]
        return f"You receive only the desk's {label} research packet."
    label = condition.removeprefix("random_")
    return (
        f"You receive randomly partitioned research packet {label}. Across packets A-D, "
        "the four analysts jointly receive the complete research packet with no duplicated "
        "fields. Each packet mixes all four research domains, and its label has no economic role."
    )


def build_prompt(snapshot: dict, condition: str, packet_text: str) -> str:
    symbols = snapshot["symbols"]
    example = symbols[0]
    benchmark_weight = 1.0 / len(symbols)
    return (
        "You are an institutional U.S. equity analyst. Construct a fully invested long-only "
        "portfolio for a one-month horizon. Seek the best expected active risk-adjusted return "
        "relative to the equal-weight portfolio of the listed stocks. Do not change positions "
        "merely to appear different from other analysts.\n\n"
        f"{_source_note(condition)} The snapshot identifier is {snapshot['snapshot_id']}; its "
        "calendar date is intentionally hidden. Use only the supplied packet. Do not import current "
        "facts, outside knowledge, or unavailable research channels.\n\n"
        "COMMON UNIVERSE AND BENCHMARK\n"
        f"{snapshot['common_core']}\n"
        f"The benchmark holds every listed stock at {benchmark_weight:.8f}.\n\n"
        "PERMITTED INFORMATION\n"
        f"{packet_text}\n\n"
        "Return a non-negative decimal portfolio weight for every listed ticker exactly once. "
        "Weights must sum to 1. You may assign zero weight, but the portfolio must express an active "
        "view and must not reproduce the equal-weight benchmark. The evaluation uses benchmark-relative "
        "active weights (portfolio weight minus equal weight) to compare independent decisions.\n\n"
        "Reply with one JSON object and no markdown. Use this schema: "
        f'{{"weights":[{{"symbol":"{example}","weight":0.10}},...],'
        '"thesis":"at most 35 words","primary_signals":["signal 1","signal 2"]}}.'
    )


def make_design(packets: list[dict]) -> list[dict]:
    design = []
    for item in packets:
        snapshot = item["snapshot"]
        partition = _partition(snapshot)
        random_packets = []
        for label in PACKET_LABELS:
            text = _random_packet_text(snapshot, label, partition[label])
            random_packets.append({
                "label": label,
                "condition": f"random_{label}",
                "field_ids": partition[label],
                "packet_sha256": _sha256(text),
            })
        design.append({
            "sample": item["sample"],
            "snapshot_id": snapshot["snapshot_id"],
            "date": snapshot["date"],
            "random_packets": random_packets,
        })
    return design


def _packet_for_condition(snapshot: dict, condition: str, design_row: dict) -> str:
    if condition in ("rich", *CHANNELS):
        return _structured_packet_text(snapshot, condition)
    label = condition.removeprefix("random_")
    row = next(item for item in design_row["random_packets"] if item["label"] == label)
    return _random_packet_text(snapshot, label, row["field_ids"])


def make_jobs(packets: list[dict], design: list[dict]) -> list[dict]:
    design_by_date = {row["date"]: row for row in design}
    jobs = []
    for item in packets:
        snapshot = item["snapshot"]
        design_row = design_by_date[snapshot["date"]]
        for condition in CONDITIONS:
            packet_text = _packet_for_condition(snapshot, condition, design_row)
            prompt = build_prompt(snapshot, condition, packet_text)
            for vendor, model in PILOT_MODELS:
                jobs.append({
                    "experiment": "information_diversity_long_only",
                    "sample": item["sample"],
                    "snapshot_id": snapshot["snapshot_id"],
                    "date": snapshot["date"],
                    "return_regime": snapshot["return_regime"],
                    "volatility_regime": snapshot["volatility_regime"],
                    "condition": condition,
                    "replicate": 0,
                    "temperature": TEMPERATURE,
                    "vendor": vendor,
                    "requested_model": model,
                    "symbols": snapshot["symbols"],
                    "prompt": prompt,
                    "prompt_spec_version": PROMPT_SPEC_VERSION,
                    "prompt_sha256": _sha256(prompt),
                    "packet_sha256": _sha256(packet_text),
                })
    return jobs


def _job_key(row: dict) -> tuple:
    return (
        row["snapshot_id"],
        row["condition"],
        row["vendor"],
    )


def write_plan(packets: list[dict], design: list[dict], jobs: list[dict]) -> dict:
    assignment_payload = json.dumps(design, sort_keys=True, separators=(",", ":"))
    manifest = [{
        "sample": job["sample"],
        "snapshot_id": job["snapshot_id"],
        "condition": job["condition"],
        "vendor": job["vendor"],
        "requested_model": job["requested_model"],
        "prompt_sha256": job["prompt_sha256"],
        "packet_sha256": job["packet_sha256"],
    } for job in jobs]
    manifest_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    calls_by_sample = {
        sample: sum(job["sample"] == sample for job in jobs)
        for sample in ("pilot", "held-out")
    }
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "frozen_before_collection": True,
        "hypothesis": (
            "Separating access to research fields increases independent benchmark-relative "
            "active weights and reduces same-side active overlap under a long-only mandate."
        ),
        "mandate": {
            "portfolio": "fully invested long-only",
            "horizon": "one month",
            "benchmark": "equal weight over the common 14-stock universe",
            "analysis_representation": "portfolio weights minus equal benchmark weights",
            "raw_holdings_similarity_used": False,
            "weight_normalization": (
                "every complete non-negative vector with positive gross weight is normalized "
                "exactly to one; the raw sum and L1 adjustment are retained"
            ),
        },
        "design": {
            "n_dates": len(packets),
            "pilot_dates": sum(row["sample"] == "pilot" for row in packets),
            "heldout_dates": sum(row["sample"] == "held-out" for row in packets),
            "n_models": len(PILOT_MODELS),
            "conditions": list(CONDITIONS),
            "n_calls": len(jobs),
            "calls_by_sample": calls_by_sample,
            "temperature": TEMPERATURE,
            "models": [{"vendor": vendor, "model": model} for vendor, model in PILOT_MODELS],
            "prompt_spec_version": PROMPT_SPEC_VERSION,
        },
        "primary_tests": {
            "structured_minus_shared_N_eff": "positive",
            "structured_minus_shared_active_overlap": "negative",
            "structured_minus_random_N_eff": "positive",
            "structured_minus_random_active_overlap": "negative",
        },
        "quality_checks": {
            "active_return": "descriptive; no directional alpha hypothesis",
            "active_share": "reported separately so geometric changes are not confused with risk scale",
            "permutation_null": "preserves each long-only book's weights and permutes ticker identity",
        },
        "format_pilot_amendment": {
            "date": "2026-07-16",
            "timing": "after 157 valid and 9 invalid format-pilot responses; before outcome analysis",
            "reason": (
                "Meta and Mistral repeatedly returned all 14 non-negative relative weights but "
                "made arithmetic sum errors. Scale carries no portfolio information under a "
                "fully invested mandate, so complete positive vectors are deterministically normalized."
            ),
            "applies_to": "all models, dates, and information conditions",
            "outcome_metrics_examined_before_amendment": False,
        },
        "random_assignment_sha256": _sha256(assignment_payload),
        "job_manifest_sha256": _sha256(manifest_payload),
        "random_assignments": design,
        "pit_correction": pit_correction_record(),
    }
    if CACHE_PATH.exists() and PLAN_PATH.exists():
        previous = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        if (
            previous.get("random_assignment_sha256") != plan["random_assignment_sha256"]
            or previous.get("job_manifest_sha256") != plan["job_manifest_sha256"]
        ):
            require_pit_correction("refusing to change the long-only plan after collection began")
            plan["supersedes"] = {
                "random_assignment_sha256": previous.get("random_assignment_sha256"),
                "job_manifest_sha256": previous.get("job_manifest_sha256"),
            }
    _atomic_json(PLAN_PATH, plan)
    return plan


def _load_records() -> dict[tuple, dict]:
    if not CACHE_PATH.exists():
        return {}
    rows = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {_job_key(row): row for row in rows}


def _strict_complete(row: dict) -> bool:
    return bool(
        row.get("parse_ok")
        and cache_record_is_current(row)
        and not row.get("missing_symbols_zero_filled")
        and not row.get("duplicate_symbols_last_value_used")
        and not row.get("unknown_symbols_dropped")
        and row.get("long_only_valid")
        and abs(float(row.get("normalized_weight_sum", 0.0)) - 1.0) <= 1e-10
    )


def call_openrouter(job: dict) -> dict:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://icaif-cohort.local",
        "X-Title": "information-access-long-only",
    }
    last_error = "unknown error"
    repair_format = False
    format_retry_count = 0
    for attempt in range(5):
        try:
            prompt = job["prompt"]
            if repair_format:
                allowed = ", ".join(job["symbols"])
                prompt += (
                    "\n\nFORMAT CORRECTION: Return every requested symbol exactly once. Use only "
                    "non-negative decimal weights that sum to 1. Do not return the equal-weight portfolio. "
                    f"The only permitted symbols are: {allowed}. Do not substitute ETFs, ticker aliases, "
                    "or any symbol outside this list."
                )
            payload = {
                "model": job["requested_model"],
                "temperature": job["temperature"],
                "max_tokens": 1400,
                "messages": [{"role": "user", "content": prompt}],
            }
            response = _post(OPENROUTER_URL, payload, headers)
            obj = _extract_json(response["choices"][0]["message"]["content"])
            weights, diagnostics = _clean_weights(obj, job["symbols"])
            format_issues = (
                diagnostics.get("missing_symbols_zero_filled")
                or diagnostics.get("duplicate_symbols_last_value_used")
                or diagnostics.get("unknown_symbols_dropped")
            )
            raw = np.asarray([row["weight"] for row in weights], dtype=float)
            raw_sum = float(raw.sum())
            benchmark = 1.0 / len(raw)
            invalid = (
                format_issues
                or float(raw.min()) < -1e-10
                or raw_sum <= 0
            )
            if invalid:
                format_retry_count += 1
                repair_format = True
                raise ValueError(
                    "invalid long-only portfolio: "
                    f"sum={raw_sum:.6f}, min={raw.min():.6f}, diagnostics={diagnostics}"
                )
            normalized = np.maximum(raw, 0.0) / raw_sum
            active_share = 0.5 * float(np.abs(normalized - benchmark).sum())
            if active_share <= 1e-6:
                format_retry_count += 1
                repair_format = True
                raise ValueError("response reproduces the equal-weight benchmark")
            cleaned = [
                {"symbol": symbol, "weight": float(value)}
                for symbol, value in zip(job["symbols"], normalized)
            ]
            return {
                **{key: job[key] for key in (
                    "experiment", "sample", "snapshot_id", "date", "return_regime",
                    "volatility_regime", "condition", "replicate", "temperature", "vendor",
                    "requested_model", "prompt_spec_version", "prompt_sha256", "packet_sha256",
                )},
                "model": job["requested_model"],
                "response_model": response.get("model"),
                "collected_at_utc": _utc_now(),
                "weights": cleaned,
                "thesis": obj.get("thesis", ""),
                "primary_signals": obj.get("primary_signals", []),
                "parse_ok": True,
                "long_only_valid": True,
                "format_retry_count": format_retry_count,
                "raw_weight_sum": raw_sum,
                "normalized_weight_sum": float(normalized.sum()),
                "normalization_l1_adjustment": float(np.abs(normalized - raw).sum()),
                "active_share": active_share,
                "max_portfolio_weight": float(normalized.max()),
                "usage": _usage(response),
                **pit_response_marker(),
                **diagnostics,
            }
        except Exception as exc:  # pragma: no cover - network-dependent
            last_error = f"{type(exc).__name__}: {exc}"[:500]
            if _credit_exhausted(exc):
                break
            if attempt < 4:
                time.sleep((2 ** attempt) + random.random())
    return {
        **{key: job[key] for key in (
            "experiment", "sample", "snapshot_id", "date", "return_regime",
            "volatility_regime", "condition", "replicate", "temperature", "vendor",
            "requested_model", "prompt_spec_version", "prompt_sha256", "packet_sha256",
        )},
        "model": job["requested_model"],
        "collected_at_utc": _utc_now(),
        "parse_ok": False,
        "long_only_valid": False,
        "error": last_error,
        **pit_response_marker(),
    }


def collect(jobs: list[dict], max_workers: int, max_new_calls: int | None) -> None:
    records = _load_records()
    pending = []
    for job in jobs:
        cached = records.get(_job_key(job), {})
        current = (
            _strict_complete(cached)
            and cached.get("prompt_sha256") == job["prompt_sha256"]
            and cached.get("packet_sha256") == job["packet_sha256"]
            and cached.get("requested_model") == job["requested_model"]
        )
        if not current:
            pending.append(job)
    if max_new_calls is not None:
        pending = pending[:max_new_calls]
    print(
        f"long-only mandate transfer: {len(jobs)} selected, "
        f"{len(records)} cached, {len(pending)} pending"
    )
    if not pending:
        return
    completed = failures = 0
    batch_size = max(1, max_workers)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        stop_for_credit = False
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_openrouter, job) for job in batch]
            for future in as_completed(futures):
                row = future.result()
                records[_job_key(row)] = row
                completed += 1
                failures += int(not row.get("parse_ok"))
                stop_for_credit = stop_for_credit or _credit_exhausted(row)
        _atomic_json(CACHE_PATH, sorted(records.values(), key=_job_key))
        if completed % 10 == 0 or completed == len(pending) or stop_for_credit:
            print(f"  {completed}/{len(pending)} completed ({failures} failures)", flush=True)
        if stop_for_credit:
            print("  stopping collection: OpenRouter credit exhausted", flush=True)
            break


def _holdings_matrix(
    records: Iterable[dict], symbols: list[str], label_keys: tuple[str, ...]
) -> pd.DataFrame:
    rows = [row for row in records if _strict_complete(row)]
    columns = [":".join(str(row[key]) for key in label_keys) for row in rows]
    if len(columns) != len(set(columns)):
        raise ValueError(f"duplicate holdings columns for {label_keys}")
    frame = pd.DataFrame(0.0, index=symbols, columns=columns)
    for row, column in zip(rows, columns):
        for weight in row["weights"]:
            frame.loc[weight["symbol"], column] = float(weight["weight"])
    if (frame < -1e-12).any().any():
        raise ValueError("negative holding in a long-only cache")
    if not np.allclose(frame.sum(axis=0), 1.0, atol=1e-10):
        raise ValueError("long-only holdings do not sum to one")
    return frame


def _active_books(holdings: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    benchmark = 1.0 / holdings.shape[0]
    raw = holdings - benchmark
    gross = raw.abs().sum(axis=0)
    if (gross <= 1e-10).any():
        raise ValueError("equal-weight response has no active book")
    normalized = raw.divide(gross, axis=1)
    if not np.allclose(normalized.sum(axis=0), 0.0, atol=1e-10):
        raise AssertionError("benchmark-relative active weights are not dollar neutral")
    return normalized, 0.5 * gross


def _roster_metrics(holdings: pd.DataFrame, forward_returns: dict[str, float]) -> dict:
    active, active_share = _active_books(holdings)
    m = active.shape[1]
    if m < 2:
        raise ValueError("a roster needs at least two portfolios")
    corr = np.corrcoef(active.to_numpy(dtype=float).T)
    iu = np.triu_indices(m, 1)
    forward = pd.Series(forward_returns, dtype=float).reindex(active.index)
    raw_active = holdings.subtract(1.0 / holdings.shape[0])
    active_return = raw_active.T.dot(forward)
    return {
        "M": int(m),
        "N_eff": float(n_eff(corr)),
        "mean_abs_corr": float(np.abs(corr[iu]).mean()),
        "mean_signed_corr": float(mean_pair_corr(corr)),
        "same_side_active_overlap": float(same_side_overlap(active)),
        "surviving_active_gross": float(gross_survival(active)),
        "active_share_mean": float(active_share.mean()),
        "max_portfolio_weight_mean": float(holdings.max(axis=0).mean()),
        "active_return_mean": float(active_return.mean()),
        "active_return_sd_across_books": float(active_return.std(ddof=1)),
    }


METRICS = (
    "N_eff",
    "mean_abs_corr",
    "mean_signed_corr",
    "same_side_active_overlap",
    "surviving_active_gross",
    "active_share_mean",
    "max_portfolio_weight_mean",
    "active_return_mean",
    "active_return_sd_across_books",
)


def _mean(rows: list[dict]) -> dict:
    return {key: float(np.mean([row[key] for row in rows])) for key in METRICS}


def _paired(rows: list[dict], treatment: str, control: str, metric: str, seed: int) -> dict:
    treatment_values = np.asarray([row[treatment][metric] for row in rows], dtype=float)
    control_values = np.asarray([row[control][metric] for row in rows], dtype=float)
    differences = treatment_values - control_values
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(differences), size=(BOOTSTRAP_DRAWS, len(differences))
    )
    bootstrap = differences[indices].mean(axis=1)
    return {
        "metric": metric,
        "n_dates": len(rows),
        "treatment_mean": float(treatment_values.mean()),
        "control_mean": float(control_values.mean()),
        "mean_difference": float(differences.mean()),
        "bootstrap_95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "dates_treatment_higher": int(np.sum(differences > 0)),
        "dates_treatment_lower": int(np.sum(differences < 0)),
        "dates_equal": int(np.sum(differences == 0)),
        "sign_flip_p_two_sided": _sign_flip_p(differences),
    }


def _scope_rows(packets: list[dict], records: list[dict], sample: str) -> list[dict]:
    selected_packets = [item for item in packets if item["sample"] == sample]
    selected_records = [row for row in records if row["sample"] == sample]
    expected = len(selected_packets) * len(PILOT_MODELS) * len(CONDITIONS)
    if not selected_records:
        return []
    usable = [row for row in selected_records if _strict_complete(row)]
    if len(usable) != expected:
        raise ValueError(
            f"{sample} long-only analysis requires {expected} complete rows; found {len(usable)}"
        )
    by_date: dict[str, list[dict]] = {}
    for row in usable:
        by_date.setdefault(row["date"], []).append(row)
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    vendor_combinations = list(itertools.combinations(vendors, 4))
    channel_assignments = list(itertools.permutations(CHANNELS))
    random_assignments = list(itertools.permutations(RANDOM_CONDITIONS))
    date_rows = []

    for date_index, item in enumerate(selected_packets):
        snapshot = item["snapshot"]
        date = snapshot["date"]
        rows = by_date[date]
        symbols = snapshot["symbols"]
        forward = snapshot["forward_return_21d"]

        rich_rows = [row for row in rows if row["condition"] == "rich"]
        rich_holdings = _holdings_matrix(rich_rows, symbols, ("vendor",))
        rich_metrics = _roster_metrics(rich_holdings, forward)
        rich_active, _ = _active_books(rich_holdings)
        rich_metrics["permutation_null"] = book_permutation_null(
            rich_active,
            B=PERMUTATION_DRAWS,
            seed=2026071700 + date_index,
        )

        same_model_structured = []
        same_model_random = []
        provider_same_model = {}
        for vendor in vendors:
            structured = [
                row for row in rows if row["vendor"] == vendor and row["condition"] in CHANNELS
            ]
            random_rows = [
                row for row in rows
                if row["vendor"] == vendor and row["condition"] in RANDOM_CONDITIONS
            ]
            structured_metrics = _roster_metrics(
                _holdings_matrix(structured, symbols, ("condition",)), forward
            )
            random_metrics = _roster_metrics(
                _holdings_matrix(random_rows, symbols, ("condition",)), forward
            )
            same_model_structured.append(structured_metrics)
            same_model_random.append(random_metrics)
            provider_same_model[vendor] = {
                "structured": structured_metrics,
                "random": random_metrics,
            }

        shared_models = []
        structured_models = []
        random_models = []
        for combination in vendor_combinations:
            chosen = set(combination)
            shared = [
                row for row in rows if row["condition"] == "rich" and row["vendor"] in chosen
            ]
            shared_models.append(_roster_metrics(
                _holdings_matrix(shared, symbols, ("vendor",)), forward
            ))
            for assignment in channel_assignments:
                assigned = []
                for vendor, condition in zip(combination, assignment):
                    match = [
                        row for row in rows
                        if row["vendor"] == vendor and row["condition"] == condition
                    ]
                    if len(match) != 1:
                        raise ValueError(f"missing structured assignment {date} {vendor} {condition}")
                    assigned.extend(match)
                structured_models.append(_roster_metrics(
                    _holdings_matrix(assigned, symbols, ("vendor",)), forward
                ))
            for assignment in random_assignments:
                assigned = []
                for vendor, condition in zip(combination, assignment):
                    match = [
                        row for row in rows
                        if row["vendor"] == vendor and row["condition"] == condition
                    ]
                    if len(match) != 1:
                        raise ValueError(f"missing random assignment {date} {vendor} {condition}")
                    assigned.extend(match)
                random_models.append(_roster_metrics(
                    _holdings_matrix(assigned, symbols, ("vendor",)), forward
                ))

        date_rows.append({
            "sample": sample,
            "snapshot_id": snapshot["snapshot_id"],
            "date": date,
            "return_regime": snapshot["return_regime"],
            "volatility_regime": snapshot["volatility_regime"],
            "rich_models_m7": rich_metrics,
            "shared_models_m4": _mean(shared_models),
            "structured_models_m4": _mean(structured_models),
            "random_models_m4": _mean(random_models),
            "structured_same_model_m4": _mean(same_model_structured),
            "random_same_model_m4": _mean(same_model_random),
            "provider_same_model": provider_same_model,
        })
    return date_rows


def _comparisons(rows: list[dict], seed: int) -> dict:
    specs = {
        "structured_vs_shared": ("structured_models_m4", "shared_models_m4"),
        "random_vs_shared": ("random_models_m4", "shared_models_m4"),
        "structured_vs_random": ("structured_models_m4", "random_models_m4"),
        "structured_vs_random_same_model": (
            "structured_same_model_m4", "random_same_model_m4"
        ),
    }
    output = {}
    for comparison, (treatment, control) in specs.items():
        output[comparison] = {}
        for metric in METRICS:
            output[comparison][metric] = _paired(rows, treatment, control, metric, seed)
            seed += 1
    return output


def _provider_comparisons(rows: list[dict], seed: int) -> dict:
    output = {}
    for vendor, _ in PILOT_MODELS:
        provider_rows = [
            {
                "structured": row["provider_same_model"][vendor]["structured"],
                "random": row["provider_same_model"][vendor]["random"],
            }
            for row in rows
        ]
        output[vendor] = {}
        for metric in METRICS:
            output[vendor][metric] = _paired(
                provider_rows, "structured", "random", metric, seed
            )
            seed += 1
    return output


PERFORMANCE_DESIGNS = {
    "shared": "shared_models_m4",
    "random": "random_models_m4",
    "structured": "structured_models_m4",
}
PERFORMANCE_METRICS = (
    "annualized_active_return",
    "annualized_tracking_error",
    "information_ratio",
    "annualized_total_return",
    "annualized_total_volatility",
    "sharpe_ratio",
)


def _performance_metrics(
    active_returns: np.ndarray,
    total_returns: np.ndarray,
    risk_free_returns: np.ndarray,
) -> dict:
    tracking_error = float(active_returns.std(ddof=1) * np.sqrt(12.0))
    total_volatility = float(total_returns.std(ddof=1) * np.sqrt(12.0))
    wealth = np.cumprod(1.0 + total_returns)
    running_peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return {
        "annualized_active_return": float(active_returns.mean() * 12.0),
        "annualized_tracking_error": tracking_error,
        "information_ratio": float(
            active_returns.mean() / active_returns.std(ddof=1) * np.sqrt(12.0)
        ),
        "annualized_total_return": float(total_returns.mean() * 12.0),
        "annualized_total_volatility": total_volatility,
        "sharpe_ratio": float(
            (total_returns - risk_free_returns).mean()
            / total_returns.std(ddof=1) * np.sqrt(12.0)
        ),
        "compound_annual_growth": float(
            np.prod(1.0 + total_returns) ** (12.0 / len(total_returns)) - 1.0
        ),
        "maximum_drawdown": float(np.min(wealth / running_peak - 1.0)),
        "active_return_hit_rate": float(np.mean(active_returns > 0.0)),
    }


def _bootstrap_performance_metrics(
    active_returns: np.ndarray,
    total_returns: np.ndarray,
    risk_free_returns: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorized metrics used in the paired date bootstrap."""
    active = active_returns[indices]
    total = total_returns[indices]
    risk_free = risk_free_returns[indices]
    active_sd = active.std(axis=1, ddof=1)
    total_sd = total.std(axis=1, ddof=1)
    return {
        "annualized_active_return": active.mean(axis=1) * 12.0,
        "annualized_tracking_error": active_sd * np.sqrt(12.0),
        "information_ratio": np.divide(
            active.mean(axis=1) * np.sqrt(12.0),
            active_sd,
            out=np.full_like(active_sd, np.nan),
            where=active_sd > 0.0,
        ),
        "annualized_total_return": total.mean(axis=1) * 12.0,
        "annualized_total_volatility": total_sd * np.sqrt(12.0),
        "sharpe_ratio": np.divide(
            (total - risk_free).mean(axis=1) * np.sqrt(12.0),
            total_sd,
            out=np.full_like(total_sd, np.nan),
            where=total_sd > 0.0,
        ),
    }


def _performance_analysis(rows: list[dict], packets: list[dict], seed: int) -> dict:
    snapshots = {item["snapshot"]["date"]: item["snapshot"] for item in packets}
    ordered = sorted(rows, key=lambda row: row["date"])
    benchmark = np.asarray([
        np.mean(list(snapshots[row["date"]]["forward_return_21d"].values()))
        for row in ordered
    ])
    monthly_risk_free = np.asarray([
        snapshots[row["date"]]["features"]["macro"]["three_month_yield"] / 1200.0
        for row in ordered
    ])
    series = {}
    levels = {}
    for label, design in PERFORMANCE_DESIGNS.items():
        active = np.asarray([row[design]["active_return_mean"] for row in ordered])
        total = benchmark + active
        series[label] = {"active": active, "total": total}
        levels[label] = _performance_metrics(active, total, monthly_risk_free)

    comparisons = {}
    specs = (
        ("structured_vs_shared", "structured", "shared"),
        ("structured_vs_random", "structured", "random"),
        ("random_vs_shared", "random", "shared"),
    )
    rng = np.random.default_rng(seed)
    n = len(ordered)
    for name, treatment, control in specs:
        point = {
            metric: levels[treatment][metric] - levels[control][metric]
            for metric in PERFORMANCE_METRICS
        }
        indices = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
        treatment_draws = _bootstrap_performance_metrics(
            series[treatment]["active"],
            series[treatment]["total"],
            monthly_risk_free,
            indices,
        )
        control_draws = _bootstrap_performance_metrics(
            series[control]["active"],
            series[control]["total"],
            monthly_risk_free,
            indices,
        )
        comparisons[name] = {
            metric: {
                "mean_difference": point[metric],
                "date_bootstrap_95": [
                    float(value) for value in np.quantile(
                        treatment_draws[metric] - control_draws[metric],
                        [0.025, 0.975],
                    )
                ],
            }
            for metric in PERFORMANCE_METRICS
        }

    return {
        "status": "descriptive post-collection operating-performance analysis",
        "frequency": f"{n} monthly 21-session forward-return windows",
        "annualization": "arithmetic monthly mean and standard deviation annualized by 12 and sqrt(12)",
        "risk_free_rate": "point-in-time 3-month Treasury yield divided by 12",
        "portfolio": "date-level mean matched four-model portfolio for each design",
        "levels": levels,
        "comparisons": comparisons,
        "per_date": [
            {
                "date": row["date"],
                "benchmark_return": float(benchmark[index]),
                "monthly_risk_free": float(monthly_risk_free[index]),
                **{
                    f"{label}_active_return": float(series[label]["active"][index])
                    for label in PERFORMANCE_DESIGNS
                },
            }
            for index, row in enumerate(ordered)
        ],
    }


def _row_portfolio(row: dict, symbols: list[str]) -> np.ndarray:
    values = {item["symbol"]: float(item["weight"]) for item in row["weights"]}
    return np.asarray([values[symbol] for symbol in symbols], dtype=float)


def _turnover_analysis(packets: list[dict], records: list[dict]) -> dict:
    snapshots = sorted((item["snapshot"] for item in packets), key=lambda row: row["date"])
    dates = [snapshot["date"] for snapshot in snapshots]
    symbols = snapshots[0]["symbols"]
    vectors = {
        (row["date"], row["vendor"], row["condition"]): _row_portfolio(row, symbols)
        for row in records if _strict_complete(row)
    }
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    transition_values = {
        design: [[] for _ in range(len(dates) - 1)]
        for design in ("shared", "random", "structured")
    }
    for combination in itertools.combinations(vendors, 4):
        shared = np.asarray([
            np.mean([vectors[(date, vendor, "rich")] for vendor in combination], axis=0)
            for date in dates
        ])
        for index, value in enumerate(0.5 * np.abs(np.diff(shared, axis=0)).sum(axis=1)):
            transition_values["shared"][index].append(float(value))
        for structured_assignment, random_assignment in zip(
            itertools.permutations(CHANNELS), itertools.permutations(RANDOM_CONDITIONS)
        ):
            structured = np.asarray([
                np.mean([
                    vectors[(date, vendor, condition)]
                    for vendor, condition in zip(combination, structured_assignment)
                ], axis=0)
                for date in dates
            ])
            random_split = np.asarray([
                np.mean([
                    vectors[(date, vendor, condition)]
                    for vendor, condition in zip(combination, random_assignment)
                ], axis=0)
                for date in dates
            ])
            for design, portfolios in (
                ("structured", structured), ("random", random_split)
            ):
                values = 0.5 * np.abs(np.diff(portfolios, axis=0)).sum(axis=1)
                for index, value in enumerate(values):
                    transition_values[design][index].append(float(value))

    transition_means = {
        design: np.asarray([np.mean(values) for values in rows])
        for design, rows in transition_values.items()
    }
    comparisons = {}
    for name, treatment, control in (
        ("structured_vs_shared", "structured", "shared"),
        ("structured_vs_random", "structured", "random"),
        ("random_vs_shared", "random", "shared"),
    ):
        differences = transition_means[treatment] - transition_means[control]
        comparisons[name] = {
            "mean_difference": float(differences.mean()),
            "transitions_treatment_lower": int(np.sum(differences < 0)),
            "n_transitions": len(differences),
        }
    return {
        "status": "descriptive; adjacent turnover transitions are not treated as independent inference units",
        "definition": "one-way turnover: half the absolute change in fully invested portfolio weights",
        "levels": {
            design: {
                "mean_monthly_turnover": float(values.mean()),
                "median_monthly_turnover": float(np.median(values)),
            }
            for design, values in transition_means.items()
        },
        "comparisons": comparisons,
        "per_transition": [
            {
                "from": dates[index],
                "to": dates[index + 1],
                **{
                    design: float(values[index])
                    for design, values in transition_means.items()
                },
            }
            for index in range(len(dates) - 1)
        ],
    }


def analyze(packets: list[dict], jobs: list[dict], plan: dict) -> dict:
    records = list(_load_records().values())
    current_jobs = {_job_key(job): job for job in jobs}
    stale = [
        _job_key(row) for row in records
        if _job_key(row) not in current_jobs
        or any(
            row.get(key) != current_jobs[_job_key(row)].get(key)
            for key in ("prompt_sha256", "packet_sha256", "requested_model")
        )
    ]
    if stale:
        raise ValueError(f"long-only cache has {len(stale)} stale records")

    scopes = {}
    all_rows = []
    for offset, sample in enumerate(("pilot", "held-out")):
        rows = _scope_rows(packets, records, sample)
        if rows:
            all_rows.extend(rows)
            scopes[sample] = {
                "n_dates": len(rows),
                "per_date": rows,
                "paired_date_comparisons": _comparisons(rows, 2026071750 + 100 * offset),
                "provider_same_model_comparisons": _provider_comparisons(
                    rows, 2026072100 + 100 * offset
                ),
                "operating_performance": _performance_analysis(
                    rows, packets, 2026072500 + 100 * offset
                ),
            }
    if not scopes:
        raise ValueError("no complete long-only sample is available for analysis")
    if len(scopes) == 2:
        scopes["pooled_34_dates"] = {
            "n_dates": len(all_rows),
            "per_date": sorted(all_rows, key=lambda row: row["date"]),
            "paired_date_comparisons": _comparisons(all_rows, 2026071950),
            "provider_same_model_comparisons": _provider_comparisons(all_rows, 2026072300),
            "operating_performance": _performance_analysis(
                all_rows, packets, 2026072700
            ),
        }

    usable = [row for row in records if _strict_complete(row)]
    usage = [row.get("usage", {}) for row in usable]
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "design": plan,
        "collection": {
            "n_records": len(records),
            "n_parse_ok": len(usable),
            "n_parse_failures": len(records) - len(usable),
            "prompt_tokens": int(sum(row.get("prompt_tokens", 0) for row in usage)),
            "completion_tokens": int(sum(row.get("completion_tokens", 0) for row in usage)),
            "reported_cost": float(sum(row.get("cost", 0.0) for row in usage)),
            "mean_weight_sum_adjustment": float(np.mean([
                row["normalization_l1_adjustment"] for row in usable
            ])),
        },
        "scopes": scopes,
        "turnover": _turnover_analysis(packets, usable) if len(scopes) == 3 else None,
    }
    _atomic_json(RESULT_PATH, result)

    summary_rows = []
    for scope, scope_result in scopes.items():
        for comparison, metrics in scope_result["paired_date_comparisons"].items():
            for metric, row in metrics.items():
                summary_rows.append({
                    "scope": scope,
                    "comparison": comparison,
                    "metric": metric,
                    "n_dates": row["n_dates"],
                    "treatment_mean": row["treatment_mean"],
                    "control_mean": row["control_mean"],
                    "mean_difference": row["mean_difference"],
                    "ci_low": row["bootstrap_95"][0],
                    "ci_high": row["bootstrap_95"][1],
                    "dates_treatment_higher": row["dates_treatment_higher"],
                    "dates_treatment_lower": row["dates_treatment_lower"],
                    "sign_flip_p": row["sign_flip_p_two_sided"],
                })
    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)
    return result


def print_result(result: dict) -> None:
    print("\nLong-only mandate transfer")
    for scope, scope_result in result["scopes"].items():
        print(f"  {scope}: {scope_result['n_dates']} dates")
        for comparison in ("structured_vs_shared", "structured_vs_random"):
            for metric in ("N_eff", "same_side_active_overlap", "active_return_mean"):
                row = scope_result["paired_date_comparisons"][comparison][metric]
                print(
                    f"    {comparison:25s} {metric:25s}: "
                    f"{row['treatment_mean']:.4f} vs {row['control_mean']:.4f}; "
                    f"delta={row['mean_difference']:+.4f} "
                    f"[{row['bootstrap_95'][0]:+.4f},{row['bootstrap_95'][1]:+.4f}], "
                    f"p={row['sign_flip_p_two_sided']:.4f}"
                )


def self_test() -> None:
    symbols = [f"A{index:02d}" for index in range(14)]
    benchmark = np.full(14, 1.0 / 14.0)
    active = np.asarray([
        [0.030, -0.020, 0.015, -0.010, 0.005, -0.005, 0.010, -0.010,
         0.005, -0.005, 0.000, -0.005, -0.005, -0.005],
        [0.028, -0.018, 0.014, -0.012, 0.006, -0.006, 0.011, -0.009,
         0.004, -0.004, 0.001, -0.004, -0.006, -0.005],
        [-0.010, 0.025, -0.005, 0.020, -0.010, 0.010, -0.005, 0.010,
         -0.005, 0.005, -0.005, -0.010, -0.010, 0.000],
        [0.005, 0.000, -0.015, 0.010, 0.020, -0.010, -0.005, -0.005,
         0.010, -0.010, 0.005, -0.005, 0.000, 0.000],
    ]).T
    active -= active.mean(axis=0)
    if not np.allclose(active.sum(axis=0), 0.0):
        raise AssertionError("self-test active books must sum to zero")
    holdings = pd.DataFrame(benchmark[:, None] + active, index=symbols)
    metrics = _roster_metrics(holdings, {symbol: 0.0 for symbol in symbols})
    if not (1.0 <= metrics["N_eff"] <= 4.0):
        raise AssertionError("invalid self-test effective dimension")

    # A signed active book can always be embedded in long-only holdings by
    # shrinking it around a positive benchmark. Correlation and normalized
    # overlap are unchanged, proving that post-hoc conversion is not a mandate test.
    rng = np.random.default_rng(20260716)
    signed = rng.normal(size=(14, 4))
    signed -= signed.mean(axis=0)
    signed /= np.abs(signed).sum(axis=0)
    scale = np.asarray([
        0.95 * min(benchmark[signed[:, column] < 0] / -signed[signed[:, column] < 0, column])
        for column in range(signed.shape[1])
    ])
    embedded = pd.DataFrame(benchmark[:, None] + signed * scale, index=symbols)
    recovered, _ = _active_books(embedded)
    original = signed / np.abs(signed).sum(axis=0)
    if not np.allclose(recovered.to_numpy(), original, atol=1e-12):
        raise AssertionError("post-hoc embedding invariance failed")
    print(
        "self-test passed: long-only active geometry is valid; "
        "post-hoc embedding is invariant and therefore not treated as new evidence"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=("pilot", "held-out", "all"), default="pilot")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-new-calls", type=int)
    args = parser.parse_args()

    self_test()
    if args.self_test:
        return
    packets = load_packets()
    design = make_design(packets)
    jobs = make_jobs(packets, design)
    plan = write_plan(packets, design, jobs)
    selected_samples = {"pilot", "held-out"} if args.sample == "all" else {args.sample}
    selected_jobs = [job for job in jobs if job["sample"] in selected_samples]
    if args.dry_run:
        print(f"planned total calls: {plan['design']['n_calls']}")
        print(f"calls by sample: {json.dumps(plan['design']['calls_by_sample'], sort_keys=True)}")
        print(f"selected calls: {len(selected_jobs)}")
        prompt_lengths = [len(job["prompt"]) for job in selected_jobs]
        print(
            f"prompt characters: min={min(prompt_lengths):,}, "
            f"mean={np.mean(prompt_lengths):,.0f}, max={max(prompt_lengths):,}"
        )
        forbidden = [job for job in jobs if "long/short" in job["prompt"].lower()]
        if forbidden:
            raise AssertionError("long-only prompt contains a long/short mandate")
        return
    if not args.analyze_only:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit("OPENROUTER_API_KEY is required for collection")
        collect(selected_jobs, args.max_workers, args.max_new_calls)
        if args.collect_only:
            return
    result = analyze(packets, jobs, plan)
    print_result(result)


if __name__ == "__main__":
    main()
