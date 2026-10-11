"""Collect the K=7 information architecture from its frozen assignment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _paths import DATA, RESULTS
from cohort_information_capacity_experiment import (
    FIELD_BY_ID,
    FINE_GROUP_BY_FIELD,
    PILOT_MODELS,
    _nmi,
)
from cohort_information_capacity_integrity import (
    CATALOG_PATH,
    INTEGRITY_CORRECTION_ID,
    PLAN_PATH as INTEGRITY_PLAN_PATH,
    TEAM_SIZE,
    _atomic_json,
    _base_job,
    _load_packets,
    _manifest,
    _model_catalog,
    _sha256_file,
    _sha256_text,
    _stable_json,
    collect,
)


PLAN_PATH = Path(RESULTS) / "information_capacity_audited_architecture_plan.json"
CACHE_PATH = Path(DATA) / "cohort_agents_information_capacity_audited_architecture.json"
SCHEMA_VERSION = "information-capacity-audited-architecture-plan-v1"
MIXED_SEED = 2026072127
MIXED_CANDIDATES = 100_000


AUDITED_COHERENT_GROUPS = (
    (
        "market_trend",
        (
            "market.ret_7d", "market.ret_30d", "market.ret_90d",
            "market.drawdown_90d",
        ),
    ),
    ("market_liquidity", ("market.vol_30d", "market.adv_bn")),
    (
        "valuation",
        (
            "fundamentals.pe", "fundamentals.price_to_sales",
            "fundamentals.price_to_book",
        ),
    ),
    (
        "company_quality",
        (
            "fundamentals.roa", "fundamentals.debt_to_assets",
            "fundamentals.cash_to_assets", "fundamentals.revenue_growth",
            "fundamentals.accounting_age",
        ),
    ),
    (
        "corporate_events",
        (
            "events.eight_k_30d", "events.eight_k_90d",
            "events.earnings_release_90d", "events.material_event_90d",
            "events.days_since_latest_8k", "events.days_since_periodic_filing",
            "events.latest_8k_items",
        ),
    ),
    (
        "systematic_risk_and_rates",
        (
            "risk.market_beta", "risk.downside_beta", "risk.market_corr",
            "risk.ten_year_yield", "risk.three_month_yield",
            "risk.yield_curve_spread", "risk.equity_market_return_3m",
        ),
    ),
    (
        "idiosyncratic_and_stress_risk",
        (
            "risk.idio_vol", "risk.max_drawdown_1y", "risk.vix",
            "risk.high_yield_return_3m",
        ),
    ),
)


def _validate_coherent() -> list[list[str]]:
    groups = [list(fields) for _, fields in AUDITED_COHERENT_GROUPS]
    flattened = [field for group in groups for field in group]
    if len(groups) != TEAM_SIZE or len(flattened) != 32:
        raise AssertionError("audited coherent architecture has the wrong dimensions")
    if len(flattened) != len(set(flattened)) or set(flattened) != set(FIELD_BY_ID):
        raise AssertionError("audited coherent architecture does not partition the field union")
    for name, fields in AUDITED_COHERENT_GROUPS:
        if not any(not field.startswith("risk.") or field in {
            "risk.market_beta", "risk.downside_beta", "risk.market_corr",
            "risk.idio_vol", "risk.max_drawdown_1y",
        } for field in fields):
            raise AssertionError(f"{name} lacks an asset-specific field")
    return groups


def regenerate_mixed_assignment(sizes: list[int]) -> tuple[list[list[str]], float]:
    """Reproduce the minimum-domain-NMI mixed partition frozen before collection."""
    fields = sorted(FIELD_BY_ID)
    rng = np.random.default_rng(MIXED_SEED)
    best: tuple[float, list[list[str]]] | None = None
    for _ in range(MIXED_CANDIDATES):
        shuffled = list(map(str, rng.permutation(fields)))
        groups = []
        cursor = 0
        for size in sizes:
            groups.append(shuffled[cursor:cursor + size])
            cursor += size
        objective = _nmi(groups, FINE_GROUP_BY_FIELD)
        if best is None or objective < best[0]:
            best = (objective, groups)
    if best is None:
        raise RuntimeError("failed to regenerate the mixed partition")
    return best[1], best[0]


def _frozen_assignment() -> dict:
    """Load the exact field map used for collection and verify its hash."""
    if not PLAN_PATH.exists():
        raise FileNotFoundError(f"canonical assignment plan is missing: {PLAN_PATH}")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assignment = plan.get("assignment")
    if not isinstance(assignment, dict):
        raise ValueError("canonical assignment plan has no assignment object")
    if _sha256_text(_stable_json(assignment)) != plan.get("assignment_sha256"):
        raise ValueError("canonical assignment hash mismatch")

    coherent = [list(row["field_ids"]) for row in assignment.get("coherent", [])]
    mixed = [list(row["field_ids"]) for row in assignment.get("mixed", [])]
    if coherent != _validate_coherent():
        raise ValueError("canonical coherent assignment differs from the declared domains")
    if len(mixed) != TEAM_SIZE:
        raise ValueError("canonical mixed assignment has the wrong team size")
    flat = [field for group in mixed for field in group]
    if len(flat) != len(set(flat)) or set(flat) != set(FIELD_BY_ID):
        raise ValueError("canonical mixed assignment does not partition the field union")
    packet_sizes = [len(group) for group in coherent]
    if [len(group) for group in mixed] != packet_sizes:
        raise ValueError("canonical mixed and coherent packet sizes differ")
    if assignment.get("packet_sizes") != packet_sizes:
        raise ValueError("canonical packet-size record is inconsistent")
    return assignment


def make_jobs() -> tuple[list[dict], dict]:
    packet_rows = _load_packets()
    catalog = _model_catalog()
    catalog_by_model = {row["requested_model"]: row for row in catalog["models"]}
    assignment = _frozen_assignment()
    coherent = [list(row["field_ids"]) for row in assignment["coherent"]]
    mixed = [list(row["field_ids"]) for row in assignment["mixed"]]
    jobs = []
    for item in packet_rows:
        for architecture, groups in (
            ("mixed_audited", mixed),
            ("coherent_audited", coherent),
        ):
            for packet_index, field_ids in enumerate(groups, start=1):
                for vendor, model in PILOT_MODELS:
                    jobs.append(_base_job(
                        item=item,
                        architecture=architecture,
                        packet_index=packet_index,
                        field_ids=field_ids,
                        vendor=vendor,
                        model=model,
                        catalog_by_model=catalog_by_model,
                    ))
    return jobs, assignment


def write_plan(jobs: list[dict], assignment: dict) -> dict:
    manifest = _manifest(jobs)
    lineage = [{
        "path": str(INTEGRITY_PLAN_PATH.relative_to(Path(DATA).parent)),
        "file_sha256": _sha256_file(INTEGRITY_PLAN_PATH),
        "assignment_sha256": json.loads(
            INTEGRITY_PLAN_PATH.read_text(encoding="utf-8")
        ).get("assignment_sha256"),
        "job_manifest_sha256": json.loads(
            INTEGRITY_PLAN_PATH.read_text(encoding="utf-8")
        ).get("job_manifest_sha256"),
    }]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "frozen_before_audited_collection": True,
        "design_status": "post-audit correction; not described as prespecified confirmation",
        "integrity_correction_id": INTEGRITY_CORRECTION_ID,
        "reason": (
            "every K=7 seat must receive at least one asset-specific field; macro context is "
            "paired with observed asset sensitivities rather than company-name priors"
        ),
        "n_dates": 34,
        "team_size": TEAM_SIZE,
        "architectures": ["mixed_audited", "coherent_audited"],
        "n_calls": len(jobs),
        "mixed_seed": MIXED_SEED,
        "assignment": assignment,
        "assignment_sha256": _sha256_text(_stable_json(assignment)),
        "job_manifest_sha256": _sha256_text(_stable_json(manifest)),
        "model_catalog": {
            "path": str(CATALOG_PATH.relative_to(Path(DATA).parent)),
            "sha256": _sha256_file(CATALOG_PATH),
        },
        "lineage": lineage,
        "job_manifest": manifest,
    }
    if PLAN_PATH.exists():
        previous = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        if all(previous.get(key) == plan.get(key) for key in (
            "assignment_sha256", "job_manifest_sha256"
        )):
            plan["lineage"] = previous.get("lineage", lineage)
        else:
            plan["lineage"] = [
                *previous.get("lineage", []),
                {
                    "path": str(PLAN_PATH.relative_to(Path(DATA).parent)),
                    "file_sha256": _sha256_file(PLAN_PATH),
                    "assignment_sha256": previous.get("assignment_sha256"),
                    "job_manifest_sha256": previous.get("job_manifest_sha256"),
                },
            ]
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
        f"frozen {len(jobs)} audit-corrected calls; "
        f"assignment={plan['assignment_sha256'][:12]}, manifest={plan['job_manifest_sha256'][:12]}"
    )
    if not args.freeze_only:
        collect(jobs, CACHE_PATH, args.max_workers, args.max_new_calls)


if __name__ == "__main__":
    main()
