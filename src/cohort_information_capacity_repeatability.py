"""Collect a frozen repeatability panel for the K=7 access experiment.

The panel is deliberately small enough to rerun while preserving the primary
identification test. Six dates cross return (strong/middle/weak) and volatility
(normal/high) regimes. Within a date, route, packet, and temperature, three
fresh calls isolate generation noise from persistent access-path differences.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path

from _paths import DATA, RESULTS
from cohort_information_capacity_audited_architecture import (
    PLAN_PATH as ARCHITECTURE_PLAN_PATH,
    _frozen_assignment,
)
from cohort_information_capacity_experiment import FIELD_BY_ID, PILOT_MODELS
from cohort_information_capacity_integrity import (
    CATALOG_PATH,
    INTEGRITY_CORRECTION_ID,
    _atomic_json,
    _base_job,
    _complete,
    _credit_exhausted,
    _load_packets,
    _model_catalog,
    _payload,
    _sha256_file,
    _sha256_text,
    _stable_json,
    call_openrouter,
)


PLAN_PATH = Path(RESULTS) / "information_capacity_repeatability_plan.json"
CACHE_PATH = Path(DATA) / "cohort_agents_information_capacity_repeatability.json"
SCHEMA_VERSION = "information-capacity-repeatability-plan-v1"
REQUEST_SPEC_VERSION = "openrouter-json-schema-repeatability-v1"
BASE_SEED = 2026072200
TEAM_SIZE = 7
TEMPERATURES = (0.0, 0.7)
REPLICATES = (1, 2, 3)
CONDITIONS = (
    "repeat_shared_full",
    "repeat_same_small",
    "repeat_mixed_split",
    "repeat_coherent_split",
)

# Selected before repeat collection by crossing the three return regimes with
# the two volatility regimes. Development/complement labels are both present.
SELECTED_DATES = (
    ("2023-08-31", "strong", "high"),
    ("2024-01-31", "strong", "normal"),
    ("2024-04-30", "middle", "high"),
    ("2025-01-31", "middle", "normal"),
    ("2025-04-30", "weak", "high"),
    ("2026-04-30", "weak", "normal"),
)


def _repeat_seed(job: dict) -> int:
    identity = "|".join(str(job[key]) for key in (
        "date", "condition", "packet_index", "vendor", "temperature", "replicate",
    ))
    return BASE_SEED + int(_sha256_text(identity)[:8], 16) % 100_000_000


def _finalize_job(
    job: dict,
    *,
    condition: str,
    temperature: float,
    replicate: int,
) -> dict:
    job = dict(job)
    job.update({
        "experiment": "information_capacity_repeatability",
        "condition": condition,
        "architecture": condition.removeprefix("repeat_"),
        "temperature": float(temperature),
        "replicate": int(replicate),
        "request_spec_version": REQUEST_SPEC_VERSION,
    })
    job["request_seed"] = _repeat_seed(job)
    job["request_payload_sha256"] = _sha256_text(_stable_json(_payload(job)))
    return job


def _manifest(jobs: list[dict]) -> list[dict]:
    keys = (
        "sample", "snapshot_id", "date", "return_regime", "volatility_regime",
        "condition", "architecture", "team_size", "packet_index", "vendor",
        "requested_model", "canonical_model_slug", "field_ids", "temperature",
        "replicate", "prompt_sha256", "packet_sha256", "response_format_sha256",
        "request_payload_sha256", "request_seed", "seed_supported",
    )
    return [{key: job[key] for key in keys} for job in jobs]


def make_jobs() -> tuple[list[dict], list[dict]]:
    packet_rows = {row["snapshot"]["date"]: row for row in _load_packets()}
    catalog = _model_catalog()
    catalog_by_model = {row["requested_model"]: row for row in catalog["models"]}
    assignment = _frozen_assignment()
    mixed = [list(row["field_ids"]) for row in assignment["mixed"]]
    coherent = [list(row["field_ids"]) for row in assignment["coherent"]]
    all_fields = list(FIELD_BY_ID)
    jobs = []
    date_assignments = []

    for date_index, (date, return_regime, volatility_regime) in enumerate(SELECTED_DATES):
        item = packet_rows.get(date)
        if item is None:
            raise ValueError(f"selected repeatability date is unavailable: {date}")
        snapshot = item["snapshot"]
        if (
            snapshot["return_regime"] != return_regime
            or snapshot["volatility_regime"] != volatility_regime
        ):
            raise ValueError(f"frozen regime label changed for {date}")

        same_small_packet = date_index % TEAM_SIZE
        mixed_packets = [(vendor_index + date_index) % TEAM_SIZE for vendor_index in range(TEAM_SIZE)]
        coherent_packets = [
            (vendor_index + 2 * date_index + 1) % TEAM_SIZE
            for vendor_index in range(TEAM_SIZE)
        ]
        date_assignments.append({
            "date": date,
            "return_regime": return_regime,
            "volatility_regime": volatility_regime,
            "same_small_packet_index": same_small_packet + 1,
            "mixed_packet_by_vendor": {
                vendor: mixed_packets[index] + 1
                for index, (vendor, _model) in enumerate(PILOT_MODELS)
            },
            "coherent_packet_by_vendor": {
                vendor: coherent_packets[index] + 1
                for index, (vendor, _model) in enumerate(PILOT_MODELS)
            },
        })

        for temperature in TEMPERATURES:
            for replicate in REPLICATES:
                for vendor_index, (vendor, model) in enumerate(PILOT_MODELS):
                    specifications = (
                        ("repeat_shared_full", "shared", 0, all_fields),
                        (
                            "repeat_same_small",
                            "mixed_audited",
                            same_small_packet + 1,
                            mixed[same_small_packet],
                        ),
                        (
                            "repeat_mixed_split",
                            "mixed_audited",
                            mixed_packets[vendor_index] + 1,
                            mixed[mixed_packets[vendor_index]],
                        ),
                        (
                            "repeat_coherent_split",
                            "coherent_audited",
                            coherent_packets[vendor_index] + 1,
                            coherent[coherent_packets[vendor_index]],
                        ),
                    )
                    for condition, base_architecture, packet_index, field_ids in specifications:
                        base = _base_job(
                            item=item,
                            architecture=base_architecture,
                            packet_index=packet_index,
                            field_ids=list(field_ids),
                            vendor=vendor,
                            model=model,
                            catalog_by_model=catalog_by_model,
                        )
                        jobs.append(_finalize_job(
                            base,
                            condition=condition,
                            temperature=temperature,
                            replicate=replicate,
                        ))
    return jobs, date_assignments


def write_plan(jobs: list[dict], date_assignments: list[dict]) -> dict:
    manifest = _manifest(jobs)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "frozen_before_collection": True,
        "purpose": (
            "separate persistent information-path diversity from within-cell generation noise"
        ),
        "selection_rule": (
            "six month-ends cross strong/middle/weak return regimes with high/normal volatility"
        ),
        "selected_dates": [date for date, _ret, _vol in SELECTED_DATES],
        "team_size": TEAM_SIZE,
        "conditions": list(CONDITIONS),
        "temperatures": list(TEMPERATURES),
        "replicates": list(REPLICATES),
        "n_calls": len(jobs),
        "date_assignments": date_assignments,
        "assignment_sha256": _sha256_text(_stable_json(date_assignments)),
        "job_manifest": manifest,
        "job_manifest_sha256": _sha256_text(_stable_json(manifest)),
        "source_architecture_plan": {
            "path": str(ARCHITECTURE_PLAN_PATH.relative_to(Path(DATA).parent)),
            "sha256": _sha256_file(ARCHITECTURE_PLAN_PATH),
        },
        "model_catalog": {
            "path": str(CATALOG_PATH.relative_to(Path(DATA).parent)),
            "sha256": _sha256_file(CATALOG_PATH),
        },
        "integrity_correction_id": INTEGRITY_CORRECTION_ID,
    }
    if PLAN_PATH.exists():
        previous = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        if previous.get("job_manifest_sha256") != plan["job_manifest_sha256"]:
            raise ValueError("repeatability plan already exists with a different manifest")
        return previous
    _atomic_json(PLAN_PATH, plan)
    return plan


def _job_key(row: dict) -> tuple:
    return (
        row["date"], row["condition"], row["vendor"],
        float(row["temperature"]), int(row["replicate"]),
    )


def _complete_repeat(row: dict, job: dict) -> bool:
    return bool(
        _complete(row, job)
        and row.get("condition") == job["condition"]
        and float(row.get("temperature", -1.0)) == float(job["temperature"])
        and int(row.get("replicate", -1)) == int(job["replicate"])
    )


def collect(jobs: list[dict], max_workers: int, max_new_calls: int | None) -> None:
    rows = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else []
    records = {_job_key(row): row for row in rows}
    pending = [job for job in jobs if not _complete_repeat(records.get(_job_key(job), {}), job)]
    if max_new_calls is not None:
        pending = pending[:max_new_calls]
    print(f"repeatability collection: {len(jobs)} planned, {len(pending)} pending")
    if not pending:
        return

    completed = failures = 0
    stop_for_credit = False
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        iterator = iter(pending)
        in_flight = {
            executor.submit(call_openrouter, job): job
            for job in islice(iterator, max_workers)
        }
        while in_flight:
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                job = in_flight.pop(future)
                row = future.result()
                records[_job_key(job)] = row
                completed += 1
                failures += int(not row.get("parse_ok"))
                stop_for_credit = stop_for_credit or _credit_exhausted(row)
                if completed % 25 == 0 or completed == len(pending):
                    print(
                        f"  {completed}/{len(pending)} completed ({failures} failures)",
                        flush=True,
                    )
            if completed % max(25, max_workers * 5) < len(finished) or completed == len(pending):
                _atomic_json(CACHE_PATH, sorted(records.values(), key=_job_key))
            if stop_for_credit:
                for future in in_flight:
                    future.cancel()
                break
            for _ in finished:
                try:
                    job = next(iterator)
                except StopIteration:
                    break
                in_flight[executor.submit(call_openrouter, job)] = job
    _atomic_json(CACHE_PATH, sorted(records.values(), key=_job_key))
    if stop_for_credit:
        raise RuntimeError("OpenRouter credit or key limit reached; cache is resumable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--max-new-calls", type=int)
    args = parser.parse_args()

    jobs, assignments = make_jobs()
    plan = write_plan(jobs, assignments)
    print(
        f"frozen {len(jobs)} repeatability calls; "
        f"assignment={plan['assignment_sha256'][:12]}, "
        f"manifest={plan['job_manifest_sha256'][:12]}"
    )
    if not args.freeze_only:
        collect(jobs, args.max_workers, args.max_new_calls)


if __name__ == "__main__":
    main()
