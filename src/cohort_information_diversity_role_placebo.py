"""Collect and analyze a role-label placebo for information specialization.

Each of four agents receives the identical full research packet.  Only its
organizational lens (market, fundamentals, filings, or risk) changes.  The
placebo distinguishes information separation from role naming alone.

Dry run:
    PYTHONPATH=src python3 src/cohort_information_diversity_role_placebo.py --dry-run

Live collection:
    OPENROUTER_API_KEY=... PYTHONPATH=src \
      python3 src/cohort_information_diversity_role_placebo.py

Offline analysis:
    PYTHONPATH=src python3 src/cohort_information_diversity_role_placebo.py --analyze-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _paths import DATA, RESULTS
from cohort_information_diversity import (
    BOOTSTRAP_DRAWS,
    CHANNELS,
    METRICS,
    PILOT_MODELS,
    TEMPERATURE,
    _atomic_json,
    _credit_exhausted,
    _job_key,
    _matrix,
    _paired_effect,
    _records_for,
    _roster_metrics,
    _strict_complete,
    _summarize,
    build_prompt,
    call_openrouter,
    load_packets,
)
from pit_correction import record as pit_correction_record


CACHE_PATH = Path(DATA) / "cohort_agents_information_diversity_role_placebo.json"
PLAN_PATH = Path(RESULTS) / "information_diversity_role_placebo_plan.json"
RESULT_PATH = Path(RESULTS) / "information_diversity_role_placebo_results.json"
PROMPT_SPEC_VERSION = "information-diversity-role-placebo-v1"
DEFAULT_MAX_WORKERS = int(os.environ.get("INFORMATION_DIVERSITY_MAX_WORKERS", "5"))

ROLE_LABELS = {
    "market": "market behavior and liquidity",
    "fundamentals": "fundamentals and valuation",
    "events": "public-filing events",
    "risk": "risk and macro context",
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_role_prompt(snapshot: dict, role: str) -> str:
    base = build_prompt(snapshot, "rich")
    marker = "You are an institutional U.S. equity analyst."
    replacement = (
        "You are an institutional U.S. equity analyst. "
        f"Your primary research lens is {ROLE_LABELS[role]}. "
        "You still receive the complete research packet and may use every supplied field."
    )
    if marker not in base:
        raise ValueError("base prompt marker changed")
    return base.replace(marker, replacement, 1)


def make_jobs(packet: dict) -> list[dict]:
    jobs = []
    for snapshot in packet["snapshots"]:
        packet_text = snapshot["tables"]["rich"]
        for role in CHANNELS:
            prompt = build_role_prompt(snapshot, role)
            for vendor, model in PILOT_MODELS:
                jobs.append({
                    "experiment": "information_diversity_role_placebo",
                    "snapshot_id": snapshot["snapshot_id"],
                    "date": snapshot["date"],
                    "return_regime": snapshot["return_regime"],
                    "volatility_regime": snapshot["volatility_regime"],
                    "condition": f"role_{role}",
                    "role": role,
                    "replicate": 0,
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


def _load_records() -> dict[tuple, dict]:
    if not CACHE_PATH.exists():
        return {}
    rows = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {_job_key(row): row for row in rows}


def write_plan(packet: dict, jobs: list[dict]) -> None:
    plan = {
        "schema_version": "information-diversity-role-placebo-plan-v1",
        "analysis_status": "post-hoc mechanism validation",
        "frozen_before_collection": True,
        "frozen_before_role_placebo_collection": True,
        "prompt_spec_version": PROMPT_SPEC_VERSION,
        "n_dates": len(packet["snapshots"]),
        "dates": [row["date"] for row in packet["snapshots"]],
        "n_models": len(PILOT_MODELS),
        "models": [{"vendor": vendor, "model": model} for vendor, model in PILOT_MODELS],
        "roles": list(CHANNELS),
        "n_calls": len(jobs),
        "temperature": TEMPERATURE,
        "matched_roster_size": 4,
        "complete_symbol_rows_required": True,
        "placebo_definition": (
            "four role-labeled calls from one model; every call receives the identical full packet"
        ),
        "primary_comparison": (
            "four restricted-information specialists versus four role-labeled full-packet agents"
        ),
        "analysis_unit": "date; model families are averaged within date before paired inference",
        "pit_correction": pit_correction_record(),
    }
    _atomic_json(PLAN_PATH, plan)


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
    print(f"role placebo: {len(jobs)} planned, {len(records)} cached, {len(pending)} pending")
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
                row["role"] = row["condition"].removeprefix("role_")
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


def _mean_metrics(rows: list[dict]) -> dict:
    summary = _summarize(rows, METRICS)
    return {key: summary[key]["mean"] for key in METRICS}


def analyze(packet: dict) -> dict:
    records = list(_load_records().values())
    jobs = {_job_key(job): job for job in make_jobs(packet)}
    stale = [
        _job_key(row)
        for row in records
        if _job_key(row) not in jobs
        or any(
            row.get(key) != jobs[_job_key(row)].get(key)
            for key in ("prompt_sha256", "packet_sha256", "requested_model")
        )
    ]
    if stale:
        raise ValueError(f"role-placebo cache contains {len(stale)} stale records")
    expected = len(packet["snapshots"]) * len(PILOT_MODELS) * len(CHANNELS)
    usable = [row for row in records if _strict_complete(row)]
    if len(usable) != expected:
        raise ValueError(f"role-placebo analysis requires {expected} usable records; found {len(usable)}")

    visible = json.loads(
        (Path(DATA) / "cohort_agents_information_diversity.json").read_text(encoding="utf-8")
    )
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    nulls: dict[int, dict] = {}
    date_rows = []
    vendor_rows = []
    for snapshot in packet["snapshots"]:
        sid = snapshot["snapshot_id"]
        symbols = snapshot["symbols"]
        forward = snapshot["forward_return_21d"]
        current = [row for row in usable if row["snapshot_id"] == sid]
        original = [row for row in visible if row["snapshot_id"] == sid and _strict_complete(row)]
        placebo_metrics = []
        information_metrics = []
        sampling_metrics = []
        for vendor in vendors:
            placebo = [row for row in current if row["vendor"] == vendor]
            information = [
                row for row in original
                if row["vendor"] == vendor and row["condition"] in CHANNELS
            ]
            sampling = [
                row for row in original
                if row["vendor"] == vendor and row["condition"] == "rich"
            ]
            metrics = {
                "role_labeled_full_packet_m4": _roster_metrics(
                    _matrix(placebo, symbols, ("condition",)), forward, nulls
                ),
                "information_same_model_m4": _roster_metrics(
                    _matrix(information, symbols, ("condition",)), forward, nulls
                ),
                "sampling_same_model_m4": _roster_metrics(
                    _matrix(sampling, symbols, ("replicate",)), forward, nulls
                ),
            }
            placebo_metrics.append(metrics["role_labeled_full_packet_m4"])
            information_metrics.append(metrics["information_same_model_m4"])
            sampling_metrics.append(metrics["sampling_same_model_m4"])
            vendor_rows.append({
                "snapshot_id": sid,
                "date": snapshot["date"],
                "vendor": vendor,
                **metrics,
            })
        date_rows.append({
            "snapshot_id": sid,
            "date": snapshot["date"],
            "return_regime": snapshot["return_regime"],
            "volatility_regime": snapshot["volatility_regime"],
            "role_labeled_full_packet_m4": _mean_metrics(placebo_metrics),
            "information_same_model_m4": _mean_metrics(information_metrics),
            "sampling_same_model_m4": _mean_metrics(sampling_metrics),
        })

    specs = {
        "information_vs_role_labels": (
            "information_same_model_m4", "role_labeled_full_packet_m4"
        ),
        "role_labels_vs_repeated_draws": (
            "role_labeled_full_packet_m4", "sampling_same_model_m4"
        ),
        "information_vs_repeated_draws": (
            "information_same_model_m4", "sampling_same_model_m4"
        ),
    }
    comparisons = {}
    seed = 2026071610
    for name, (treatment, control) in specs.items():
        comparisons[name] = {}
        for metric in METRICS:
            comparisons[name][metric] = _paired_effect(
                date_rows, treatment, control, metric, seed
            )
            seed += 1

    usage = [row.get("usage", {}) for row in usable]
    result = {
        "schema_version": "information-diversity-role-placebo-results-v1",
        "design": json.loads(PLAN_PATH.read_text(encoding="utf-8")),
        "collection": {
            "n_records": len(records),
            "n_parse_ok": len(usable),
            "n_parse_failures": len(records) - len(usable),
            "prompt_tokens": int(sum(row.get("prompt_tokens", 0) for row in usage)),
            "completion_tokens": int(sum(row.get("completion_tokens", 0) for row in usage)),
            "reported_cost": float(sum(row.get("cost", 0.0) for row in usage)),
        },
        "per_date": date_rows,
        "per_vendor_date": vendor_rows,
        "paired_date_comparisons": comparisons,
    }
    _atomic_json(RESULT_PATH, result)
    return result


def print_result(result: dict) -> None:
    print("\nInformation-diversity role-label placebo")
    collection = result["collection"]
    print(f"  usable books={collection['n_parse_ok']}/{collection['n_records']}")
    for comparison in (
        "information_vs_role_labels",
        "role_labels_vs_repeated_draws",
        "information_vs_repeated_draws",
    ):
        nrow = result["paired_date_comparisons"][comparison]["N_eff"]
        orow = result["paired_date_comparisons"][comparison]["same_side_overlap"]
        print(
            f"  {comparison:32s}: "
            f"dN={nrow['mean_difference']:+.3f} "
            f"[{nrow['bootstrap_95'][0]:+.3f},{nrow['bootstrap_95'][1]:+.3f}], "
            f"dO={orow['mean_difference']:+.3f} "
            f"[{orow['bootstrap_95'][0]:+.3f},{orow['bootstrap_95'][1]:+.3f}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-new-calls", type=int)
    args = parser.parse_args()
    packet = load_packets()
    jobs = make_jobs(packet)
    write_plan(packet, jobs)
    if args.dry_run:
        print(f"role placebo dry run: {len(jobs)} calls")
        print(f"plan: {PLAN_PATH}")
        return
    if not args.analyze_only:
        collect(jobs, args.max_workers, args.max_new_calls)
        if args.collect_only:
            return
    result = analyze(packet)
    print_result(result)


if __name__ == "__main__":
    main()
