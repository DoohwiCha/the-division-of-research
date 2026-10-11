"""Collect and analyze the information-diversity pilot.

The experiment separates three sources of apparent analyst diversity:

* model diversity: different model families see the same rich packet;
* sampling diversity: repeated draws from one model see the same rich packet;
* information diversity: one model receives four source-specific packets.

Every agent has the same universe, one-month objective, long/short mandate,
book-construction rule, and JSON output schema.  The specialized conditions change only
the permitted information packet.  Collection is resumable and the analysis
uses the date as the statistical unit.

Dry run:
    PYTHONPATH=src python3 src/cohort_information_diversity.py --dry-run

Live collection:
    OPENROUTER_API_KEY=... PYTHONPATH=src \
      python3 src/cohort_information_diversity.py

Offline analysis:
    PYTHONPATH=src python3 src/cohort_information_diversity.py --analyze-only
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
from cohort_analyze import mean_pair_corr, n_eff, normalize
from cohort_capacity_delta import gross_survival
from cohort_crossvendor import OPENROUTER_MODELS, OPENROUTER_URL, _extract_json, _post
from cohort_crossvendor_hardening import matched_null, same_side_overlap
from cohort_equity_locked import _clean_weights
from pit_correction import (
    cache_record_is_current,
    record as pit_correction_record,
    response_marker as pit_response_marker,
)


PACKET_PATH = Path(DATA) / "information_diversity_packets.json"
CACHE_PATH = Path(DATA) / "cohort_agents_information_diversity.json"
PLAN_PATH = Path(RESULTS) / "information_diversity_collection_plan.json"
RESULT_PATH = Path(RESULTS) / "information_diversity_results.json"
SUMMARY_PATH = Path(RESULTS) / "information_diversity_summary.csv"

PROMPT_SPEC_VERSION = "information-diversity-neutral-mandate-v1"
TEMPERATURE = 0.7
RICH_REPLICATES = 4
CHANNELS = ("market", "fundamentals", "events", "risk")
CONDITIONS = ("simple", "rich", *CHANNELS)
DEFAULT_MAX_WORKERS = int(os.environ.get("INFORMATION_DIVERSITY_MAX_WORKERS", "5"))
BOOTSTRAP_DRAWS = 20_000

# The original Qwen 2.5 route used by the frozen panels currently
# rejects chat-completion requests on its remaining OpenRouter provider.  The
# pilot is a new collection, so it freezes a currently routable Qwen model
# without changing any historical cache or paper result.
PILOT_MODELS = [
    (vendor, "qwen/qwen3-235b-a22b-2507" if vendor == "alibaba" else model)
    for vendor, model in OPENROUTER_MODELS
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def load_packets() -> dict:
    if not PACKET_PATH.exists():
        raise FileNotFoundError(
            f"{PACKET_PATH} is missing; run cohort_information_diversity_data.py first"
        )
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    if not packet.get("design_frozen_before_llm_collection"):
        raise ValueError("information-diversity design is not marked frozen")
    if len(packet.get("snapshots", [])) != 12:
        raise ValueError("expected 12 preselected snapshots")
    return packet


def _packet_text(snapshot: dict, condition: str) -> str:
    if condition == "simple":
        return "STANDARD MARKET PACKET\n" + snapshot["tables"]["simple"]
    if condition == "rich":
        return snapshot["tables"]["rich"]
    if condition in CHANNELS:
        label = {
            "market": "MARKET BEHAVIOR AND LIQUIDITY",
            "fundamentals": "POINT-IN-TIME FUNDAMENTALS AND VALUATION",
            "events": "RECENT SEC FILING EVENTS",
            "risk": "RISK AND MACRO CONTEXT",
        }[condition]
        return label + "\n" + snapshot["tables"][condition]
    raise ValueError(condition)


def build_prompt(snapshot: dict, condition: str) -> str:
    source_note = {
        "simple": "You receive the desk's standard compact market packet.",
        "rich": "You receive the union of the desk's four research packets.",
        "market": "You receive only the market-behavior and liquidity research packet.",
        "fundamentals": "You receive only the point-in-time fundamentals and valuation research packet.",
        "events": "You receive only the recent public-filing event research packet.",
        "risk": "You receive only the risk and macro-context research packet.",
    }[condition]
    symbols = snapshot["symbols"]
    example = symbols[0]
    return (
        "You are an institutional U.S. equity analyst. Construct a one-month active long/short book "
        "that seeks the best expected risk-adjusted return from the supplied evidence. Do not create "
        "different positions merely to appear diversified.\n\n"
        f"{source_note} The snapshot identifier is {snapshot['snapshot_id']}; its calendar date is intentionally "
        "hidden. Use only the supplied packet. Do not import current facts, outside knowledge, or unavailable "
        "research channels.\n\n"
        "COMMON UNIVERSE\n"
        f"{snapshot['common_core']}\n\n"
        "PERMITTED INFORMATION\n"
        f"{_packet_text(snapshot, condition)}\n\n"
        "Apply the same mandate in every condition. Give every listed ticker a signed conviction score from "
        "-1 to +1, with both positive and negative scores. Positive means long and negative means short. These "
        "are research scores, not final capital weights: the fixed evaluation pipeline subtracts their mean and "
        "rescales absolute exposure to 1, imposing the same dollar-neutral book construction on every response.\n\n"
        "Reply with one JSON object and no markdown. Use this schema: "
        f'{{"weights":[{{"symbol":"{example}","weight":0.10}},...],'
        '"thesis":"at most 35 words","primary_signals":["signal 1","signal 2"]}}.'
    )


def make_jobs(packet: dict) -> list[dict]:
    jobs = []
    for snapshot in packet["snapshots"]:
        condition_replicates = [("simple", 0)]
        condition_replicates.extend(("rich", replicate) for replicate in range(RICH_REPLICATES))
        condition_replicates.extend((channel, 0) for channel in CHANNELS)
        for condition, replicate in condition_replicates:
            prompt = build_prompt(snapshot, condition)
            packet_text = _packet_text(snapshot, condition)
            for vendor, model in PILOT_MODELS:
                jobs.append({
                    "experiment": "information_diversity",
                    "snapshot_id": snapshot["snapshot_id"],
                    "date": snapshot["date"],
                    "return_regime": snapshot["return_regime"],
                    "volatility_regime": snapshot["volatility_regime"],
                    "condition": condition,
                    "replicate": replicate,
                    "temperature": TEMPERATURE,
                    "vendor": vendor,
                    "requested_model": model,
                    "symbols": snapshot["symbols"],
                    "prompt": prompt,
                    "prompt_spec_version": PROMPT_SPEC_VERSION,
                    "prompt_sha256": _sha256(prompt),
                    "packet_sha256": _sha256(packet_text),
                    "require_complete_symbols": True,
                })
    return jobs


def _job_key(row: dict) -> tuple:
    return (
        row["snapshot_id"],
        row["condition"],
        int(row["replicate"]),
        row["vendor"],
    )


def _load_records(path: Path = CACHE_PATH) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {_job_key(row): row for row in rows}


def _usage(response: dict) -> dict:
    raw = response.get("usage") or {}
    keep = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
    return {key: raw[key] for key in keep if raw.get(key) is not None}


def _credit_exhausted(value: object) -> bool:
    if isinstance(value, dict):
        text = str(value.get("error", ""))
    else:
        text = f"{type(value).__name__}: {value}"
    lowered = text.lower()
    return (
        "http error 402" in lowered
        or "http 402" in lowered
        or "status code 402" in lowered
        or "insufficient credit" in lowered
        or "key limit exceeded" in lowered
        or "weekly limit" in lowered
    )


def call_openrouter(job: dict) -> dict:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://icaif-cohort.local",
        "X-Title": "information-access-study",
    }
    last_error = "unknown error"
    repair_format = False
    format_retry_count = 0
    for attempt in range(5):
        try:
            prompt = job["prompt"]
            if repair_format:
                exact_symbols = ", ".join(job["symbols"])
                prompt += (
                    "\n\nFORMAT CORRECTION: The prior response did not satisfy the book constraints. "
                    "Return every symbol in COMMON UNIVERSE exactly once, including zero-conviction "
                    f"symbols. The weights array must contain exactly {len(job['symbols'])} entries "
                    f"in this order: {exact_symbols}. Copy these ticker strings exactly and do not add "
                    "or substitute any symbol. Express a nonzero active view with at least one positive "
                    "and one negative score. Keep the same mandate and JSON schema."
                )
            payload = {
                "model": job["requested_model"],
                "temperature": job["temperature"],
                "max_tokens": 1400,
                "messages": [{"role": "user", "content": prompt}],
            }
            response = _post(OPENROUTER_URL, payload, headers)
            text = response["choices"][0]["message"]["content"]
            obj = _extract_json(text)
            weights, diagnostics = _clean_weights(obj, job["symbols"])
            format_issues = (
                diagnostics.get("missing_symbols_zero_filled")
                or diagnostics.get("duplicate_symbols_last_value_used")
                or diagnostics.get("unknown_symbols_dropped")
            )
            if job.get("require_complete_symbols") and format_issues:
                format_retry_count += 1
                repair_format = True
                raise ValueError(
                    "response must contain every requested symbol exactly once; "
                    f"diagnostics={diagnostics}"
                )
            raw_values = np.asarray([row["weight"] for row in weights], dtype=float)
            if float(np.abs(raw_values - raw_values.mean()).sum()) <= 1e-12:
                format_retry_count += 1
                repair_format = True
                raise ValueError("response does not express a nonzero active book")
            return {
                **{key: job[key] for key in (
                    "experiment", "snapshot_id", "date", "return_regime", "volatility_regime",
                    "condition", "replicate", "temperature", "vendor", "requested_model",
                    "prompt_spec_version", "prompt_sha256", "packet_sha256",
                )},
                "model": job["requested_model"],
                "response_model": response.get("model"),
                "collected_at_utc": _utc_now(),
                "weights": weights,
                "thesis": obj.get("thesis", ""),
                "primary_signals": obj.get("primary_signals", []),
                "parse_ok": True,
                "format_retry_count": format_retry_count,
                "raw_net": float(raw_values.sum()),
                "raw_gross": float(np.abs(raw_values).sum()),
                "raw_max_abs_weight": float(np.abs(raw_values).max()),
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
            "experiment", "snapshot_id", "date", "return_regime", "volatility_regime",
            "condition", "replicate", "temperature", "vendor", "requested_model",
            "prompt_spec_version", "prompt_sha256", "packet_sha256",
        )},
        "model": job["requested_model"],
        "collected_at_utc": _utc_now(),
        "parse_ok": False,
        "error": last_error,
        **pit_response_marker(),
    }


def _strict_complete(row: dict) -> bool:
    try:
        values = np.asarray(
            [float(item["weight"]) for item in row.get("weights", [])],
            dtype=float,
        )
        nonzero_active = bool(
            len(values)
            and np.isfinite(values).all()
            and float(np.abs(values - values.mean()).sum()) > 1e-12
        )
    except (KeyError, TypeError, ValueError):
        nonzero_active = False
    return bool(
        row.get("parse_ok")
        and cache_record_is_current(row)
        and not row.get("missing_symbols_zero_filled")
        and not row.get("duplicate_symbols_last_value_used")
        and not row.get("unknown_symbols_dropped")
        and nonzero_active
    )


def write_plan(packet: dict, jobs: list[dict]) -> dict:
    by_condition: dict[str, int] = {}
    for job in jobs:
        by_condition[job["condition"]] = by_condition.get(job["condition"], 0) + 1
    plan = {
        "schema_version": "information-diversity-collection-plan-v1",
        "frozen_before_collection": True,
        "packet_path": str(PACKET_PATH.relative_to(Path(DATA).parent)),
        "packet_sha256": hashlib.sha256(PACKET_PATH.read_bytes()).hexdigest(),
        "prompt_spec_version": PROMPT_SPEC_VERSION,
        "temperature": TEMPERATURE,
        "rich_replicates": RICH_REPLICATES,
        "n_dates": len(packet["snapshots"]),
        "dates": [row["date"] for row in packet["snapshots"]],
        "n_models": len(PILOT_MODELS),
        "models": [{"vendor": vendor, "model": model} for vendor, model in PILOT_MODELS],
        "channels": list(CHANNELS),
        "n_calls": len(jobs),
        "calls_by_condition": by_condition,
        "analysis_unit": "date; within-date rosters are averaged before paired inference",
        "matched_roster_size": 4,
        "complete_symbol_rows_required": True,
        "comparisons": {
            "richness": "seven-model common-rich roster versus seven-model common-simple roster",
            "information_vs_sampling": "four channel-specific books versus four rich-packet repeated draws from the same model",
            "information_vs_models": "four channel-specific books from one model versus four models sharing one rich packet",
            "combined": "four models assigned one distinct channel each, averaged over every channel permutation",
        },
        "pit_correction": pit_correction_record(),
    }
    _atomic_json(PLAN_PATH, plan)
    return plan


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
    print(f"information-diversity: {len(jobs)} planned, {len(records)} cached, {len(pending)} pending")
    if not pending:
        return
    done = failures = 0
    batch_size = max(1, max_workers)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        stop_for_credit = False
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_openrouter, job) for job in batch]
            for future in as_completed(futures):
                row = future.result()
                records[_job_key(row)] = row
                done += 1
                failures += int(not row.get("parse_ok"))
                stop_for_credit = stop_for_credit or _credit_exhausted(row)
        _atomic_json(CACHE_PATH, sorted(records.values(), key=_job_key))
        if done % 10 == 0 or done == len(pending) or stop_for_credit:
            print(f"  {done}/{len(pending)} completed ({failures} failures)", flush=True)
        if stop_for_credit:
            print("  stopping collection: OpenRouter credit exhausted", flush=True)
            break


def _matrix(records: Iterable[dict], symbols: list[str], label_keys: tuple[str, ...]) -> pd.DataFrame:
    rows = [row for row in records if row.get("parse_ok") and row.get("weights")]
    columns = [":".join(str(row[key]) for key in label_keys) for row in rows]
    if len(columns) != len(set(columns)):
        raise ValueError(f"duplicate matrix columns for {label_keys}")
    frame = pd.DataFrame(0.0, index=symbols, columns=columns)
    for row, column in zip(rows, columns):
        for weight in row["weights"]:
            if weight["symbol"] in frame.index:
                frame.loc[weight["symbol"], column] = float(weight["weight"])
    frame = normalize(frame)
    return frame.loc[:, frame.std() > 1e-12]


def _roster_metrics(
    frame: pd.DataFrame,
    forward_returns: dict[str, float],
    nulls: dict[int, dict],
) -> dict:
    m = frame.shape[1]
    if m < 2:
        raise ValueError("a roster needs at least two usable books")
    corr = np.corrcoef(frame.to_numpy(dtype=float).T)
    iu = np.triu_indices(m, 1)
    if m not in nulls:
        nulls[m] = matched_null(m, list(frame.index), B=10_000, seed=2026071500 + m)
    fwd = pd.Series(forward_returns, dtype=float).reindex(frame.index)
    realized = frame.T.dot(fwd)
    return {
        "M": int(m),
        "N_eff": float(n_eff(corr)),
        "N_eff_over_null": float(n_eff(corr) / nulls[m]["mean"]),
        "mean_abs_corr": float(np.abs(corr[iu]).mean()),
        "mean_signed_corr": float(mean_pair_corr(corr)),
        "same_side_overlap": float(same_side_overlap(frame)),
        "surviving_gross": float(gross_survival(frame)),
        "forward_return_mean": float(realized.mean()),
        "forward_return_sd_across_books": float(realized.std(ddof=1)),
    }


def _summarize(rows: list[dict], keys: tuple[str, ...]) -> dict:
    out: dict[str, object] = {"n": len(rows)}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=float)
        out[key] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "p2_5": float(np.quantile(values, 0.025)),
            "p97_5": float(np.quantile(values, 0.975)),
        }
    return out


METRICS = (
    "N_eff", "N_eff_over_null", "mean_abs_corr", "mean_signed_corr",
    "same_side_overlap", "surviving_gross", "forward_return_mean",
    "forward_return_sd_across_books",
)


def _sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    if len(values) <= 16:
        means = []
        for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
            means.append(abs(float(np.mean(values * np.asarray(signs, dtype=float)))))
        return float(sum(value >= observed - 1e-15 for value in means) / len(means))
    rng = np.random.default_rng(2026071511)
    draws = rng.choice((-1.0, 1.0), size=(100_000, len(values)))
    simulated = np.abs((draws * values).mean(axis=1))
    return float((np.sum(simulated >= observed) + 1) / (len(simulated) + 1))


def _paired_effect(rows: list[dict], treatment: str, control: str, metric: str, seed: int) -> dict:
    values = np.asarray([row[treatment][metric] - row[control][metric] for row in rows], dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([
        rng.choice(values, size=len(values), replace=True).mean() for _ in range(BOOTSTRAP_DRAWS)
    ])
    return {
        "treatment": treatment,
        "control": control,
        "metric": metric,
        "n_dates": len(values),
        "treatment_mean": float(np.mean([row[treatment][metric] for row in rows])),
        "control_mean": float(np.mean([row[control][metric] for row in rows])),
        "mean_difference": float(values.mean()),
        "bootstrap_95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "dates_treatment_higher": int(np.sum(values > 0)),
        "dates_treatment_lower": int(np.sum(values < 0)),
        "exact_sign_flip_p_two_sided": _sign_flip_p(values),
    }


def _records_for(
    rows: list[dict], condition: str, replicate: int | None = None, vendors: set[str] | None = None
) -> list[dict]:
    selected = [row for row in rows if row["condition"] == condition and row.get("parse_ok")]
    if replicate is not None:
        selected = [row for row in selected if int(row["replicate"]) == replicate]
    if vendors is not None:
        selected = [row for row in selected if row["vendor"] in vendors]
    return selected


def analyze(packet: dict) -> dict:
    records = list(_load_records().values())
    current_jobs = {_job_key(job): job for job in make_jobs(packet)}
    stale = []
    for row in records:
        job = current_jobs.get(_job_key(row))
        if not job or any(
            row.get(key) != job.get(key)
            for key in ("prompt_sha256", "packet_sha256", "requested_model")
        ):
            stale.append(_job_key(row))
    if stale:
        raise ValueError(f"cache contains {len(stale)} stale records; rerun collection")
    expected = len(packet["snapshots"]) * len(PILOT_MODELS) * (1 + RICH_REPLICATES + len(CHANNELS))
    usable = [row for row in records if _strict_complete(row)]
    if len(usable) != expected:
        raise ValueError(f"analysis requires {expected} usable records; found {len(usable)}")
    by_snapshot = {row["snapshot_id"]: row for row in packet["snapshots"]}
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    vendor_combinations = list(itertools.combinations(vendors, len(CHANNELS)))
    channel_permutations = list(itertools.permutations(CHANNELS))
    nulls: dict[int, dict] = {}
    date_rows = []
    vendor_rows = []

    for snapshot_id, snapshot in by_snapshot.items():
        date_records = [row for row in usable if row["snapshot_id"] == snapshot_id]
        symbols = snapshot["symbols"]
        forward = snapshot["forward_return_21d"]
        simple = _matrix(_records_for(date_records, "simple", 0), symbols, ("vendor",))
        simple_metrics = _roster_metrics(simple, forward, nulls)
        rich_rosters = []
        for replicate in range(RICH_REPLICATES):
            frame = _matrix(_records_for(date_records, "rich", replicate), symbols, ("vendor",))
            rich_rosters.append(_roster_metrics(frame, forward, nulls))

        stochastic_rows = []
        information_rows = []
        for vendor in vendors:
            stochastic_records = [
                row for row in date_records if row["condition"] == "rich" and row["vendor"] == vendor
            ]
            information_records = [
                row for row in date_records if row["condition"] in CHANNELS and row["vendor"] == vendor
            ]
            stochastic = _roster_metrics(
                _matrix(stochastic_records, symbols, ("replicate",)), forward, nulls
            )
            information = _roster_metrics(
                _matrix(information_records, symbols, ("condition",)), forward, nulls
            )
            stochastic_rows.append(stochastic)
            information_rows.append(information)
            vendor_rows.append({
                "snapshot_id": snapshot_id,
                "date": snapshot["date"],
                "vendor": vendor,
                "stochastic": stochastic,
                "information": information,
            })

        model_m4_rows = []
        combined_m4_rows = []
        for combination in vendor_combinations:
            chosen = set(combination)
            model_records = _records_for(date_records, "rich", 0, chosen)
            model_m4_rows.append(_roster_metrics(
                _matrix(model_records, symbols, ("vendor",)), forward, nulls
            ))
            for assignment in channel_permutations:
                assigned = []
                for vendor, channel in zip(combination, assignment):
                    match = [
                        row for row in date_records
                        if row["vendor"] == vendor and row["condition"] == channel
                    ]
                    if len(match) != 1:
                        raise ValueError(f"missing channel assignment {snapshot_id} {vendor} {channel}")
                    assigned.extend(match)
                combined_m4_rows.append(_roster_metrics(
                    _matrix(assigned, symbols, ("vendor",)), forward, nulls
                ))

        rich_summary = _summarize(rich_rosters, METRICS)
        stochastic_summary = _summarize(stochastic_rows, METRICS)
        information_summary = _summarize(information_rows, METRICS)
        model_summary = _summarize(model_m4_rows, METRICS)
        combined_summary = _summarize(combined_m4_rows, METRICS)

        def means(summary: dict) -> dict:
            return {key: summary[key]["mean"] for key in METRICS}

        date_rows.append({
            "snapshot_id": snapshot_id,
            "date": snapshot["date"],
            "return_regime": snapshot["return_regime"],
            "volatility_regime": snapshot["volatility_regime"],
            "simple_models_m7": simple_metrics,
            "rich_models_m7": means(rich_summary),
            "sampling_same_model_m4": means(stochastic_summary),
            "information_same_model_m4": means(information_summary),
            "models_shared_rich_m4": means(model_summary),
            "models_specialized_m4": means(combined_summary),
            "within_date_distributions": {
                "rich_models_m7": rich_summary,
                "sampling_same_model_m4": stochastic_summary,
                "information_same_model_m4": information_summary,
                "models_shared_rich_m4": model_summary,
                "models_specialized_m4": combined_summary,
            },
        })

    comparisons = {}
    specs = {
        "rich_vs_simple": ("rich_models_m7", "simple_models_m7"),
        "information_vs_sampling": ("information_same_model_m4", "sampling_same_model_m4"),
        "information_vs_models": ("information_same_model_m4", "models_shared_rich_m4"),
        "specialized_models_vs_shared_models": ("models_specialized_m4", "models_shared_rich_m4"),
    }
    seed = 2026071520
    for comparison, (treatment, control) in specs.items():
        comparisons[comparison] = {}
        for metric in METRICS:
            comparisons[comparison][metric] = _paired_effect(
                date_rows, treatment, control, metric, seed
            )
            seed += 1

    usage = [row.get("usage", {}) for row in usable]
    result = {
        "schema_version": "information-diversity-results-v1",
        "design": json.loads(PLAN_PATH.read_text(encoding="utf-8")),
        "collection": {
            "n_records": len(records),
            "n_parse_ok": len(usable),
            "n_parse_failures": len(records) - len(usable),
            "prompt_tokens": int(sum(row.get("prompt_tokens", 0) for row in usage)),
            "completion_tokens": int(sum(row.get("completion_tokens", 0) for row in usage)),
            "reported_cost": float(sum(row.get("cost", 0.0) for row in usage)),
            "raw_output_diagnostics": {
                "mean_abs_net": float(np.mean([abs(row["raw_net"]) for row in usable])),
                "mean_gross": float(np.mean([row["raw_gross"] for row in usable])),
                "share_max_score_above_1": float(np.mean([row["raw_max_abs_weight"] > 1.000001 for row in usable])),
            },
        },
        "nulls": {str(key): value for key, value in nulls.items()},
        "per_date": date_rows,
        "per_vendor_date": vendor_rows,
        "paired_date_comparisons": comparisons,
    }
    _atomic_json(RESULT_PATH, result)
    summary_rows = []
    for comparison, metrics in comparisons.items():
        for metric, row in metrics.items():
            summary_rows.append({
                "comparison": comparison,
                "metric": metric,
                "treatment_mean": row["treatment_mean"],
                "control_mean": row["control_mean"],
                "mean_difference": row["mean_difference"],
                "ci_low": row["bootstrap_95"][0],
                "ci_high": row["bootstrap_95"][1],
                "dates_treatment_higher": row["dates_treatment_higher"],
                "dates_treatment_lower": row["dates_treatment_lower"],
                "sign_flip_p": row["exact_sign_flip_p_two_sided"],
            })
    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)
    return result


def print_result(result: dict) -> None:
    print("\nInformation-diversity pilot")
    collection = result["collection"]
    print(
        f"  usable books={collection['n_parse_ok']}/{collection['n_records']}; "
        f"prompt tokens={collection['prompt_tokens']:,}; completion tokens={collection['completion_tokens']:,}"
    )
    selected = (
        ("rich_vs_simple", "N_eff"),
        ("information_vs_sampling", "N_eff"),
        ("information_vs_models", "N_eff"),
        ("information_vs_models", "same_side_overlap"),
        ("specialized_models_vs_shared_models", "N_eff"),
        ("specialized_models_vs_shared_models", "same_side_overlap"),
        ("specialized_models_vs_shared_models", "forward_return_mean"),
    )
    for comparison, metric in selected:
        row = result["paired_date_comparisons"][comparison][metric]
        print(
            f"  {comparison:38s} {metric:22s}: "
            f"{row['treatment_mean']:.4f} vs {row['control_mean']:.4f}; "
            f"delta={row['mean_difference']:+.4f} "
            f"[{row['bootstrap_95'][0]:+.4f},{row['bootstrap_95'][1]:+.4f}], "
            f"p={row['exact_sign_flip_p_two_sided']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-new-calls", type=int)
    args = parser.parse_args()

    packet = load_packets()
    jobs = make_jobs(packet)
    plan = write_plan(packet, jobs)
    if args.dry_run:
        prompt_lengths = [len(job["prompt"]) for job in jobs]
        print(f"planned calls: {plan['n_calls']}")
        print(f"calls by condition: {json.dumps(plan['calls_by_condition'], sort_keys=True)}")
        print(
            f"prompt characters: min={min(prompt_lengths):,}, "
            f"mean={np.mean(prompt_lengths):,.0f}, max={max(prompt_lengths):,}"
        )
        return
    if not args.analyze_only:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit("OPENROUTER_API_KEY is required for collection")
        collect(jobs, args.max_workers, args.max_new_calls)
        if args.collect_only:
            return
    result = analyze(packet)
    print_result(result)


if __name__ == "__main__":
    main()
