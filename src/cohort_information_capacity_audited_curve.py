"""Collect the audit-corrected K=2..6 information-access curve.

K=7 is supplied by ``cohort_information_capacity_audited_architecture.py``.
This extension uses the same corrected SEC history, unified prompt, model
vintage, and strict response schema.  The coherent hierarchy is nested: broad
research desks are split into narrower economic desks as K increases.  Every
seat retains at least one asset-specific field, so no seat can construct a
cross-sectional book from company-name priors alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _paths import DATA, RESULTS
from cohort_information_capacity_audited_architecture import (
    AUDITED_COHERENT_GROUPS,
    PLAN_PATH as K7_PLAN_PATH,
)
from cohort_information_capacity_experiment import (
    FIELD_BY_ID,
    PILOT_MODELS,
    _is_macro,
)
from cohort_information_capacity_integrity import (
    CATALOG_PATH,
    INTEGRITY_CORRECTION_ID,
    _base_job,
    _load_packets,
    _manifest,
    _model_catalog,
    _payload,
    _request_seed,
    _sha256_file,
    _sha256_text,
    _stable_json,
    collect,
)
from cohort_information_diversity import _atomic_json


TEAM_SIZES = (2, 3, 4, 5, 6)
PLAN_PATH = Path(RESULTS) / "information_capacity_audited_curve_plan.json"
CACHE_PATH = Path(DATA) / "cohort_agents_information_capacity_audited_curve.json"
SCHEMA_VERSION = "information-capacity-audited-curve-plan-v1"
MIXED_SEED = 2026072131
RECOVERY_SEED_OFFSETS = {
    ("2023-10-31", "mixed_audited", 5, 2, "google"): 1,
    ("2024-03-31", "mixed_audited", 5, 2, "google"): 1,
    ("2024-08-31", "mixed_audited", 5, 3, "google"): 1,
    ("2024-09-30", "coherent_audited", 3, 2, "google"): 1,
    ("2024-11-30", "coherent_audited", 3, 2, "google"): 1,
}


ATOMS = {
    name: tuple(fields) for name, fields in AUDITED_COHERENT_GROUPS
}


def _join(*names: str) -> list[str]:
    return [field for name in names for field in ATOMS[name]]


COHERENT_HIERARCHY = {
    2: (
        ("market_and_risk", _join(
            "market_trend", "market_liquidity",
            "systematic_risk_and_rates", "idiosyncratic_and_stress_risk",
        )),
        ("company_and_events", _join("valuation", "company_quality", "corporate_events")),
    ),
    3: (
        ("market", _join("market_trend", "market_liquidity")),
        ("company_and_events", _join("valuation", "company_quality", "corporate_events")),
        ("risk_and_rates", _join("systematic_risk_and_rates", "idiosyncratic_and_stress_risk")),
    ),
    4: (
        ("market", _join("market_trend", "market_liquidity")),
        ("company_fundamentals", _join("valuation", "company_quality")),
        ("corporate_events", _join("corporate_events")),
        ("risk_and_rates", _join("systematic_risk_and_rates", "idiosyncratic_and_stress_risk")),
    ),
    5: (
        ("market", _join("market_trend", "market_liquidity")),
        ("valuation", _join("valuation")),
        ("company_quality", _join("company_quality")),
        ("corporate_events", _join("corporate_events")),
        ("risk_and_rates", _join("systematic_risk_and_rates", "idiosyncratic_and_stress_risk")),
    ),
    6: (
        ("market_trend", _join("market_trend")),
        ("market_liquidity", _join("market_liquidity")),
        ("valuation", _join("valuation")),
        ("company_quality", _join("company_quality")),
        ("corporate_events", _join("corporate_events")),
        ("risk_and_rates", _join("systematic_risk_and_rates", "idiosyncratic_and_stress_risk")),
    ),
}


def _validate_groups(groups: list[list[str]], k: int) -> None:
    flat = [field for group in groups for field in group]
    if len(groups) != k or len(flat) != len(set(flat)) or set(flat) != set(FIELD_BY_ID):
        raise AssertionError(f"K={k} does not partition the 32-field union")
    empty = [index for index, group in enumerate(groups, start=1) if all(_is_macro(field) for field in group)]
    if empty:
        raise AssertionError(f"K={k} has macro-only seats: {empty}")


def assignments() -> dict[str, dict[str, list[dict]]]:
    """Load the exact K=2..6 field maps used for collection."""
    if not PLAN_PATH.exists():
        raise FileNotFoundError(f"canonical curve plan is missing: {PLAN_PATH}")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    frozen = plan.get("assignment")
    if not isinstance(frozen, dict):
        raise ValueError("canonical curve plan has no assignment object")
    if _sha256_text(_stable_json(frozen)) != plan.get("assignment_sha256"):
        raise ValueError("canonical curve assignment hash mismatch")
    if set(frozen) != {str(k) for k in TEAM_SIZES}:
        raise ValueError("canonical curve assignment has unexpected team sizes")

    for k in TEAM_SIZES:
        rows = frozen[str(k)]
        coherent = [list(row["field_ids"]) for row in rows.get("coherent", [])]
        mixed = [list(row["field_ids"]) for row in rows.get("mixed", [])]
        _validate_groups(coherent, k)
        _validate_groups(mixed, k)
        expected_coherent = [list(fields) for _, fields in COHERENT_HIERARCHY[k]]
        if coherent != expected_coherent:
            raise ValueError(f"K={k}: canonical coherent assignment differs from the hierarchy")
        if [len(group) for group in mixed] != [len(group) for group in coherent]:
            raise ValueError(f"K={k}: canonical mixed and coherent packet sizes differ")
    return frozen


def _curve_job(job: dict, k: int) -> dict:
    architecture = job["architecture"]
    converted = {
        **job,
        "experiment": "information_capacity_audited_curve",
        "team_size": k,
        "condition": f"{architecture}_k{k}_p{job['packet_index']}",
    }
    converted["request_seed"] = _request_seed({
        "date": converted["date"],
        "architecture": architecture,
        "team_size": k,
        "packet_index": converted["packet_index"],
        "vendor": converted["vendor"],
    })
    recovery_key = (
        converted["date"], architecture, k,
        int(converted["packet_index"]), converted["vendor"],
    )
    converted["request_seed"] += RECOVERY_SEED_OFFSETS.get(recovery_key, 0)
    converted["seed_recovery_offset"] = RECOVERY_SEED_OFFSETS.get(recovery_key, 0)
    converted["request_payload_sha256"] = _sha256_text(_stable_json(_payload(converted)))
    return converted


def make_jobs() -> tuple[list[dict], dict]:
    packet_rows = _load_packets()
    catalog = _model_catalog()
    catalog_by_model = {row["requested_model"]: row for row in catalog["models"]}
    frozen = assignments()
    jobs = []
    for item in packet_rows:
        for k in TEAM_SIZES:
            for label in ("mixed", "coherent"):
                architecture = f"{label}_audited"
                for packet in frozen[str(k)][label]:
                    for vendor, model in PILOT_MODELS:
                        base = _base_job(
                            item=item,
                            architecture=architecture,
                            packet_index=int(packet["seat"]),
                            field_ids=list(packet["field_ids"]),
                            vendor=vendor,
                            model=model,
                            catalog_by_model=catalog_by_model,
                        )
                        jobs.append(_curve_job(base, k))
    return jobs, frozen


def write_plan(jobs: list[dict], frozen: dict) -> dict:
    manifest = _manifest(jobs)
    k7_plan = json.loads(K7_PLAN_PATH.read_text(encoding="utf-8"))
    plan = {
        "schema_version": SCHEMA_VERSION,
        "frozen_before_collection": True,
        "design_status": "post-audit corrected curve; all 34 dates form one pooled panel",
        "integrity_correction_id": INTEGRITY_CORRECTION_ID,
        "n_dates": 34,
        "team_sizes_collected": list(TEAM_SIZES),
        "team_size_7_source": {
            "path": str(K7_PLAN_PATH.relative_to(Path(DATA).parent)),
            "file_sha256": _sha256_file(K7_PLAN_PATH),
            "assignment_sha256": k7_plan["assignment_sha256"],
            "job_manifest_sha256": k7_plan["job_manifest_sha256"],
        },
        "architectures": ["mixed_audited", "coherent_audited"],
        "n_calls": len(jobs),
        "mixed_seed": MIXED_SEED,
        "seed_recovery": {
            "reason": (
                "five deterministic Google responses produced fewer than two usable nonzero "
                "symbols; superseding jobs change only the supported request seed"
            ),
            "offsets": [
                {
                    "date": key[0],
                    "architecture": key[1],
                    "team_size": key[2],
                    "packet_index": key[3],
                    "vendor": key[4],
                    "seed_offset": value,
                }
                for key, value in RECOVERY_SEED_OFFSETS.items()
            ],
        },
        "hierarchy_rule": (
            "economic desks split monotonically as K rises; every seat contains at least "
            "one asset-specific field; K=7 uses the separately frozen audited endpoint"
        ),
        "assignment": frozen,
        "assignment_sha256": _sha256_text(_stable_json(frozen)),
        "job_manifest_sha256": _sha256_text(_stable_json(manifest)),
        "model_catalog": {
            "path": str(CATALOG_PATH.relative_to(Path(DATA).parent)),
            "sha256": _sha256_file(CATALOG_PATH),
        },
        "job_manifest": manifest,
    }
    if PLAN_PATH.exists():
        previous = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        if previous.get("job_manifest_sha256") != plan["job_manifest_sha256"]:
            plan["supersedes"] = {
                "file_sha256": _sha256_file(PLAN_PATH),
                "assignment_sha256": previous.get("assignment_sha256"),
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
    jobs, frozen = make_jobs()
    plan = write_plan(jobs, frozen)
    print(
        f"frozen {len(jobs)} audited K=2..6 calls; "
        f"assignment={plan['assignment_sha256'][:12]}, "
        f"manifest={plan['job_manifest_sha256'][:12]}"
    )
    if not args.freeze_only:
        collect(jobs, CACHE_PATH, args.max_workers, args.max_new_calls)


if __name__ == "__main__":
    main()
