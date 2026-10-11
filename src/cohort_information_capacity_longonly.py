"""Collect the mandate-matched long-only Information-Capacity Curve.

This experiment reuses the frozen dates, model routes, 32-field union, team
sizes, and mixed/coherent packet assignments from the signed-book capacity
experiment.  Only the portfolio mandate and output constraint change: every
response is a fully invested long-only portfolio evaluated as active weights
relative to the common equal-weight benchmark.

Existing full-packet long-only responses supply the shared-access control.
The new cache contains only the mixed and coherent K=2,...,7 responses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path

from _paths import DATA, RESULTS
from cohort_information_capacity_experiment import (
    ARCHITECTURES,
    PLAN_PATH as SIGNED_PLAN_PATH,
    TEAM_SIZES,
    _packet_text,
)
from cohort_information_diversity import (
    PILOT_MODELS,
    TEMPERATURE,
    _atomic_json,
    _credit_exhausted,
    _sha256,
)
from cohort_information_diversity_longonly import (
    CACHE_PATH as SHARED_LONGONLY_CACHE_PATH,
    _strict_complete,
    call_openrouter,
)
from cohort_information_diversity_random_placebo import load_packets
from pit_correction import record as pit_correction_record, require_enabled as require_pit_correction


CACHE_PATH = Path(DATA) / "cohort_agents_information_capacity_longonly.json"
PLAN_PATH = Path(RESULTS) / "information_capacity_longonly_plan.json"
PLAN_SCHEMA_VERSION = "information-capacity-long-only-plan-v1"
PROMPT_SPEC_VERSION = "information-capacity-long-only-neutral-packet-v1"
DEFAULT_MAX_WORKERS = 16


def _job_key(row: dict) -> tuple:
    return row["snapshot_id"], row["condition"], row["vendor"]


def _build_prompt(snapshot: dict, packet_text: str) -> str:
    symbols = snapshot["symbols"]
    example = symbols[0]
    benchmark_weight = 1.0 / len(symbols)
    return (
        "You are an institutional U.S. equity analyst. Construct a fully invested long-only "
        "portfolio for a one-month horizon. Seek the best expected active risk-adjusted return "
        "relative to the equal-weight portfolio of the listed stocks. Do not change positions "
        "merely to appear different from other analysts.\n\n"
        "You receive one packet from a research workflow. Its field composition was fixed before "
        "model collection. Use only the supplied packet. Do not infer missing fields, import current "
        "facts, or use outside knowledge. The calendar date is intentionally hidden.\n\n"
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


def _load_signed_plan() -> dict:
    plan = json.loads(SIGNED_PLAN_PATH.read_text(encoding="utf-8"))
    design = plan.get("design", {})
    if design.get("n_dates") != 34 or design.get("team_sizes") != list(TEAM_SIZES):
        raise ValueError("signed capacity plan does not contain the expected 34-date K=2,...,7 design")
    if design.get("architectures") != list(ARCHITECTURES):
        raise ValueError("signed capacity architecture list changed")
    if not plan.get("frozen_before_collection"):
        raise ValueError("signed capacity plan is not marked frozen")
    return plan


def make_jobs(packet_rows: list[dict], signed_plan: dict) -> list[dict]:
    snapshots = {row["snapshot"]["date"]: row for row in packet_rows}
    jobs = []
    for assignment in signed_plan["assignments"]:
        item = snapshots[assignment["date"]]
        snapshot = item["snapshot"]
        if item["sample"] != assignment["sample"]:
            raise AssertionError(f"sample mismatch at {assignment['date']}")
        for packet in assignment["packets"]:
            packet_text = _packet_text(snapshot, packet["field_ids"])
            if _sha256(packet_text) != packet["packet_sha256"]:
                raise AssertionError(
                    f"rendered packet drift at {assignment['date']} "
                    f"{assignment['architecture']} K={assignment['team_size']} "
                    f"P={packet['packet_index']}"
                )
            prompt = _build_prompt(snapshot, packet_text)
            for vendor, model in PILOT_MODELS:
                jobs.append({
                    "experiment": "information_capacity_curve_long_only",
                    "sample": item["sample"],
                    "snapshot_id": snapshot["snapshot_id"],
                    "date": snapshot["date"],
                    "return_regime": snapshot["return_regime"],
                    "volatility_regime": snapshot["volatility_regime"],
                    "condition": (
                        f"capacity_longonly_{assignment['architecture']}_"
                        f"k{assignment['team_size']}_p{packet['packet_index']}"
                    ),
                    "replicate": 0,
                    "architecture": assignment["architecture"],
                    "team_size": assignment["team_size"],
                    "packet_index": packet["packet_index"],
                    "fine_domain_nmi": assignment["fine_domain_nmi"],
                    "broad_domain_nmi": assignment["broad_domain_nmi"],
                    "field_ids": packet["field_ids"],
                    "temperature": TEMPERATURE,
                    "vendor": vendor,
                    "requested_model": model,
                    "symbols": snapshot["symbols"],
                    "prompt": prompt,
                    "prompt_spec_version": PROMPT_SPEC_VERSION,
                    "prompt_sha256": _sha256(prompt),
                    "packet_sha256": packet["packet_sha256"],
                    "require_complete_symbols": True,
                })
    return jobs


def _manifest(jobs: list[dict]) -> list[dict]:
    keys = (
        "sample", "snapshot_id", "architecture", "team_size", "packet_index",
        "condition", "vendor", "requested_model", "prompt_sha256", "packet_sha256",
    )
    return [{key: job[key] for key in keys} for job in jobs]


def write_plan(packet_rows: list[dict], signed_plan: dict, jobs: list[dict]) -> dict:
    manifest = json.dumps(_manifest(jobs), sort_keys=True, separators=(",", ":"))
    shared_rows = json.loads(SHARED_LONGONLY_CACHE_PATH.read_text(encoding="utf-8"))
    shared = [
        row for row in shared_rows
        if row.get("condition") == "rich" and int(row.get("replicate", 0)) == 0
    ]
    if len(shared) != 238 or not all(_strict_complete(row) for row in shared):
        raise ValueError("the 34-date by 7-model full-packet long-only control is incomplete")
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "frozen_before_collection": True,
        "research_question": (
            "Under the same frozen capacity-curve design, does separating information access "
            "increase independent benchmark-relative active books in a fully invested long-only mandate?"
        ),
        "mandate": {
            "portfolio": "fully invested long-only",
            "horizon": "one month",
            "benchmark": "equal weight over the common 14-stock universe",
            "analysis_representation": "portfolio weights minus equal benchmark weights",
            "raw_holdings_similarity_used": False,
        },
        "design": {
            "n_dates": len(packet_rows),
            "pilot_dates": sum(row["sample"] == "pilot" for row in packet_rows),
            "heldout_dates": sum(row["sample"] == "held-out" for row in packet_rows),
            "team_sizes": list(TEAM_SIZES),
            "architectures": list(ARCHITECTURES),
            "n_models": len(PILOT_MODELS),
            "models": [{"vendor": vendor, "model": model} for vendor, model in PILOT_MODELS],
            "full_packet_fields": 32,
            "team_union_equals_full_packet": True,
            "each_field_appears_once_per_partitioned_team": True,
            "packet_size_profile_matched_between_architectures": True,
            "architecture_and_role_labels_hidden_from_prompt": True,
            "temperature": TEMPERATURE,
            "prompt_spec_version": PROMPT_SPEC_VERSION,
            "n_new_calls": len(jobs),
            "new_calls_by_sample": {
                sample: sum(job["sample"] == sample for job in jobs)
                for sample in ("pilot", "held-out")
            },
            "shared_full_packet_control": (
                "238 complete responses reused from cohort_agents_information_diversity_longonly.json"
            ),
        },
        "source_signed_plan": {
            "path": str(SIGNED_PLAN_PATH.relative_to(Path(DATA).parent)),
            "assignment_sha256": signed_plan["assignment_sha256"],
            "job_manifest_sha256": signed_plan["job_manifest_sha256"],
        },
        "pre_specified_tests": {
            "coherent_minus_mixed_N_eff": "positive at matched K",
            "mixed_minus_shared_N_eff": "positive at matched K",
            "coherent_minus_mixed_same_side_active_overlap": "negative at matched K",
            "coherent_minus_mixed_order_concentration": "negative at matched K",
            "active_return": "descriptive; no directional alpha hypothesis",
        },
        "inference_unit": (
            "date; model subsets and packet assignments are averaged within date before inference"
        ),
        "job_manifest_sha256": hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
        "pit_correction": pit_correction_record(),
    }
    if CACHE_PATH.exists() and PLAN_PATH.exists():
        previous = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        if previous.get("job_manifest_sha256") != plan["job_manifest_sha256"]:
            require_pit_correction(
                "refusing to change the frozen long-only capacity plan after collection began"
            )
        if previous.get("source_signed_plan") != plan["source_signed_plan"]:
            require_pit_correction("signed source plan changed after long-only collection began")
        if (
            previous.get("job_manifest_sha256") != plan["job_manifest_sha256"]
            or previous.get("source_signed_plan") != plan["source_signed_plan"]
        ):
            plan["supersedes"] = {
                "job_manifest_sha256": previous.get("job_manifest_sha256"),
                "source_signed_plan": previous.get("source_signed_plan"),
            }
    _atomic_json(PLAN_PATH, plan)
    return plan


def _load_records() -> dict[tuple, dict]:
    if not CACHE_PATH.exists():
        return {}
    rows = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {_job_key(row): row for row in rows}


EXTRA_KEYS = (
    "architecture", "team_size", "packet_index", "fine_domain_nmi",
    "broad_domain_nmi", "field_ids",
)


def _call(job: dict) -> dict:
    row = call_openrouter(job)
    row.update({key: job[key] for key in EXTRA_KEYS})
    return row


def collect(jobs: list[dict], sample: str, max_workers: int, max_new_calls: int | None) -> None:
    records = _load_records()
    eligible = [job for job in jobs if sample == "all" or job["sample"] == sample]
    pending = []
    for job in eligible:
        cached = records.get(_job_key(job), {})
        current = (
            _strict_complete(cached)
            and cached.get("prompt_sha256") == job["prompt_sha256"]
            and cached.get("packet_sha256") == job["packet_sha256"]
            and cached.get("requested_model") == job["requested_model"]
            and cached.get("architecture") == job["architecture"]
            and int(cached.get("team_size", -1)) == job["team_size"]
        )
        if not current:
            pending.append(job)
    if max_new_calls is not None:
        pending = pending[:max_new_calls]
    print(
        f"long-only capacity: {len(jobs)} planned, {len(eligible)} eligible, "
        f"{len(records)} cached, {len(pending)} pending",
        flush=True,
    )
    if not pending:
        return
    completed = failures = 0
    credit_exhausted = False
    save_interval = max_workers * 5
    next_save = save_interval
    next_report = save_interval
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        job_iter = iter(pending)
        in_flight = {
            executor.submit(_call, job): job
            for job in islice(job_iter, max_workers)
        }
        while in_flight:
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                in_flight.pop(future)
                row = future.result()
                records[_job_key(row)] = row
                completed += 1
                failures += int(not row.get("parse_ok"))
                credit_exhausted |= _credit_exhausted(row)
            if completed >= next_save or completed == len(pending) or credit_exhausted:
                _atomic_json(CACHE_PATH, sorted(records.values(), key=_job_key))
                while next_save <= completed:
                    next_save += save_interval
            if completed >= next_report or completed == len(pending) or credit_exhausted:
                print(
                    f"  {completed}/{len(pending)} completed ({failures} terminal failures)",
                    flush=True,
                )
                while next_report <= completed:
                    next_report += save_interval
            if credit_exhausted:
                for future in in_flight:
                    future.cancel()
                print(
                    "  stopped at the OpenRouter credit/key limit; rerun after raising it",
                    flush=True,
                )
                break
            for _ in finished:
                try:
                    job = next(job_iter)
                except StopIteration:
                    break
                in_flight[executor.submit(_call, job)] = job
    _atomic_json(CACHE_PATH, sorted(records.values(), key=_job_key))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--sample", choices=("pilot", "held-out", "all"), default="pilot")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-new-calls", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet_rows = load_packets()
    signed_plan = _load_signed_plan()
    jobs = make_jobs(packet_rows, signed_plan)
    plan = write_plan(packet_rows, signed_plan, jobs)
    print(
        f"frozen {len(jobs)} new long-only calls; "
        f"source_assignment={signed_plan['assignment_sha256'][:12]}, "
        f"manifest={plan['job_manifest_sha256'][:12]}",
        flush=True,
    )
    if args.freeze_only:
        return
    collect(jobs, args.sample, args.max_workers, args.max_new_calls)


if __name__ == "__main__":
    main()
