"""Collect the audit-corrected K=7 information architecture as long-only books."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _paths import DATA, RESULTS
from cohort_information_capacity_audited_architecture import (
    PLAN_PATH as SIGNED_AUDITED_PLAN_PATH,
    make_jobs as make_signed_split_jobs,
)
from cohort_information_capacity_experiment import FIELD_BY_ID, PILOT_MODELS, _packet_text
from cohort_information_capacity_integrity import (
    CATALOG_PATH,
    INTEGRITY_CORRECTION_ID,
    TEAM_SIZE,
    _base_job,
    _load_packets,
    _manifest,
    _model_catalog,
    _payload,
    _request_seed,
    _response_format,
    _sha256_file,
    _sha256_text,
    _stable_json,
    collect,
)
from cohort_information_diversity import _atomic_json


PLAN_PATH = Path(RESULTS) / "information_capacity_audited_longonly_plan.json"
CACHE_PATH = Path(DATA) / "cohort_agents_information_capacity_audited_longonly.json"
SCHEMA_VERSION = "information-capacity-audited-long-only-plan-v1"
PROMPT_SPEC_VERSION = "information-capacity-audited-long-only-v1-unified-object"
REQUEST_SPEC_VERSION = "openrouter-json-schema-audited-long-only-v1"
RECOVERY_SEED_OFFSETS = {
    ("2025-09-30", "mixed_audited", 1, "google"): 1,
}


def _build_prompt(snapshot: dict, packet_text: str) -> str:
    example = snapshot["symbols"][0]
    benchmark_weight = 1.0 / len(snapshot["symbols"])
    return (
        "You are an institutional U.S. equity analyst. Construct a fully invested long-only "
        "portfolio for a one-month horizon. Seek the best expected active risk-adjusted return "
        "relative to the equal-weight portfolio of the listed stocks. Do not create different "
        "positions merely to appear diversified.\n\n"
        "You receive one packet from a research workflow. Its field composition was fixed before "
        "model collection. Use only the supplied packet. Do not infer missing fields, import current "
        "facts, or use outside knowledge. The calendar date is intentionally hidden.\n\n"
        "COMMON UNIVERSE AND BENCHMARK\n"
        f"{snapshot['common_core']}\n"
        f"The benchmark holds every listed stock at {benchmark_weight:.8f}.\n\n"
        "PERMITTED INFORMATION\n"
        f"{packet_text}\n\n"
        "Give every listed ticker a non-negative decimal portfolio weight from 0 to 1. The weights "
        "must sum to 1. You may assign zero weight, but the portfolio must express an active view and "
        "must not reproduce the equal-weight benchmark. The fixed evaluation pipeline normalizes a "
        "complete positive vector to exactly 1 and evaluates benchmark-relative active weights.\n\n"
        "Reply with one JSON object and no markdown. The weights field must be an object containing "
        "every listed ticker exactly once. Use this schema: "
        f'{{"weights":{{"{example}":0.10,"...":0.00}},'
        '"thesis":"at most 35 words","primary_signals":["signal 1","signal 2"]}}.'
    )


def _convert_job(job: dict, snapshot_by_date: dict[str, dict]) -> dict:
    snapshot = snapshot_by_date[job["date"]]
    packet_text = _packet_text(snapshot, job["field_ids"])
    prompt = _build_prompt(snapshot, packet_text)
    response_format = _response_format(snapshot["symbols"])
    request_seed = _request_seed({
        "date": job["date"],
        "architecture": f"longonly-{job['architecture']}",
        "team_size": TEAM_SIZE,
        "packet_index": job["packet_index"],
        "vendor": job["vendor"],
    })
    recovery_key = (
        job["date"], job["architecture"], int(job["packet_index"]), job["vendor"]
    )
    request_seed += RECOVERY_SEED_OFFSETS.get(recovery_key, 0)
    converted = {
        **job,
        "experiment": "information_capacity_audited_long_only",
        "condition": (
            "shared_full_longonly"
            if job["architecture"] == "shared"
            else f"{job['architecture']}_longonly_k7_p{job['packet_index']}"
        ),
        "portfolio_mandate": "long_only",
        "prompt": prompt,
        "prompt_spec_version": PROMPT_SPEC_VERSION,
        "prompt_sha256": _sha256_text(prompt),
        "packet_sha256": _sha256_text(packet_text),
        "response_format": response_format,
        "response_format_sha256": _sha256_text(_stable_json(response_format)),
        "request_spec_version": REQUEST_SPEC_VERSION,
        "request_seed": request_seed,
        "seed_recovery_offset": RECOVERY_SEED_OFFSETS.get(recovery_key, 0),
    }
    converted["request_payload_sha256"] = _sha256_text(_stable_json(_payload(converted)))
    return converted


def make_jobs() -> tuple[list[dict], dict]:
    packet_rows = _load_packets()
    snapshot_by_date = {row["snapshot"]["date"]: row["snapshot"] for row in packet_rows}
    catalog = _model_catalog()
    catalog_by_model = {row["requested_model"]: row for row in catalog["models"]}

    shared = []
    for item in packet_rows:
        for vendor, model in PILOT_MODELS:
            shared.append(_base_job(
                item=item,
                architecture="shared",
                packet_index=0,
                field_ids=list(FIELD_BY_ID),
                vendor=vendor,
                model=model,
                catalog_by_model=catalog_by_model,
            ))
    split, assignment = make_signed_split_jobs()
    jobs = [_convert_job(job, snapshot_by_date) for job in (*shared, *split)]
    return jobs, assignment


def write_plan(jobs: list[dict], assignment: dict) -> dict:
    manifest = _manifest(jobs)
    signed_plan = json.loads(SIGNED_AUDITED_PLAN_PATH.read_text(encoding="utf-8"))
    plan = {
        "schema_version": SCHEMA_VERSION,
        "frozen_before_collection": True,
        "design_status": "post-audit mandate-transfer rerun",
        "integrity_correction_id": INTEGRITY_CORRECTION_ID,
        "portfolio": "fully invested long-only",
        "benchmark": "equal weight over the common 14-stock universe",
        "analysis_representation": "portfolio weights minus equal-weight benchmark",
        "n_dates": 34,
        "team_size": TEAM_SIZE,
        "architectures": ["shared", "mixed_audited", "coherent_audited"],
        "n_calls": len(jobs),
        "prompt_spec_version": PROMPT_SPEC_VERSION,
        "request_spec_version": REQUEST_SPEC_VERSION,
        "retry_policy": "up to five byte-identical request payloads; no augmented prompt",
        "seed_recovery": {
            "reason": (
                "one deterministic Google response exhausted max_tokens by emitting an "
                "unbounded decimal zero; the superseding job changes only its seed"
            ),
            "offsets": [
                {
                    "date": key[0],
                    "architecture": key[1],
                    "packet_index": key[2],
                    "vendor": key[3],
                    "seed_offset": value,
                }
                for key, value in RECOVERY_SEED_OFFSETS.items()
            ],
        },
        "assignment": assignment,
        "source_signed_plan": {
            "path": str(SIGNED_AUDITED_PLAN_PATH.relative_to(Path(DATA).parent)),
            "file_sha256": _sha256_file(SIGNED_AUDITED_PLAN_PATH),
            "assignment_sha256": signed_plan["assignment_sha256"],
            "job_manifest_sha256": signed_plan["job_manifest_sha256"],
        },
        "model_catalog": {
            "path": str(CATALOG_PATH.relative_to(Path(DATA).parent)),
            "sha256": _sha256_file(CATALOG_PATH),
        },
        "job_manifest_sha256": _sha256_text(_stable_json(manifest)),
        "job_manifest": manifest,
    }
    if PLAN_PATH.exists():
        previous = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        if previous.get("job_manifest_sha256") != plan["job_manifest_sha256"]:
            plan["supersedes"] = {
                "file_sha256": _sha256_file(PLAN_PATH),
                "job_manifest_sha256": previous.get("job_manifest_sha256"),
            }
    _atomic_json(PLAN_PATH, plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-new-calls", type=int)
    args = parser.parse_args()
    jobs, assignment = make_jobs()
    plan = write_plan(jobs, assignment)
    print(
        f"frozen {len(jobs)} audited long-only calls; "
        f"manifest={plan['job_manifest_sha256'][:12]}"
    )
    if not args.freeze_only:
        collect(jobs, CACHE_PATH, args.max_workers, args.max_new_calls)


if __name__ == "__main__":
    main()
