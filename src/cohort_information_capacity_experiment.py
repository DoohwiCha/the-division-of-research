"""Freeze and collect the Information-Capacity Curve experiment.

The experiment holds the team-level information union fixed at the same 32
point-in-time fields while changing two features:

* team size K, from two to seven model families; and
* packet architecture, either domain-mixed or economically coherent.

Each model receives one packet.  Within a date, the K packets contain every
field exactly once and use the same packet-size profile in both architectures.
The prompt does not identify an analyst as a specialist and does not reveal
whether a packet is mixed or coherent.  Existing full-packet responses provide
the shared-information control and are not recollected.

Freeze and inspect the complete design without API calls:
    PYTHONPATH=src python3 src/cohort_information_capacity_experiment.py --freeze-only

Collect the pre-specified pilot or held-out sample:
    OPENROUTER_API_KEY=... PYTHONPATH=src python3 \
      src/cohort_information_capacity_experiment.py --sample pilot
    OPENROUTER_API_KEY=... PYTHONPATH=src python3 \
      src/cohort_information_capacity_experiment.py --sample held-out

Collection is resumable.  Once a cache exists, the script refuses a changed
assignment or prompt manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from pathlib import Path

import numpy as np

from _paths import DATA, RESULTS
from cohort_information_diversity import (
    PILOT_MODELS,
    TEMPERATURE,
    _atomic_json,
    _credit_exhausted,
    _job_key,
    _sha256,
    _strict_complete,
    call_openrouter,
)
from pit_correction import record as pit_correction_record, require_enabled as require_pit_correction
from cohort_information_diversity_random_placebo import (
    FIELD_BY_ID,
    _format_value,
    _raw_value,
    load_packets,
)


PLAN_PATH = Path(RESULTS) / "information_capacity_experiment_plan.json"
CACHE_PATH = Path(DATA) / "cohort_agents_information_capacity.json"

PLAN_SCHEMA_VERSION = "information-capacity-experiment-plan-v1"
PROMPT_SPEC_VERSION = "information-capacity-neutral-packet-v1"
RANDOM_SEED = 2026071602
TEAM_SIZES = (2, 3, 4, 5, 6, 7)
ARCHITECTURES = ("mixed", "coherent")
DEFAULT_MAX_WORKERS = int(os.environ.get("INFORMATION_CAPACITY_MAX_WORKERS", "8"))


FINE_GROUPS = {
    "market_trend": (
        "market.ret_7d",
        "market.ret_30d",
        "market.ret_90d",
        "market.drawdown_90d",
    ),
    "market_liquidity": (
        "market.vol_30d",
        "market.adv_bn",
    ),
    "valuation": (
        "fundamentals.pe",
        "fundamentals.price_to_sales",
        "fundamentals.price_to_book",
    ),
    "quality": (
        "fundamentals.roa",
        "fundamentals.debt_to_assets",
        "fundamentals.cash_to_assets",
        "fundamentals.revenue_growth",
        "fundamentals.accounting_age",
    ),
    "events": (
        "events.eight_k_30d",
        "events.eight_k_90d",
        "events.earnings_release_90d",
        "events.material_event_90d",
        "events.days_since_latest_8k",
        "events.days_since_periodic_filing",
        "events.latest_8k_items",
    ),
    "asset_risk": (
        "risk.market_beta",
        "risk.downside_beta",
        "risk.market_corr",
        "risk.idio_vol",
        "risk.max_drawdown_1y",
    ),
    "macro": (
        "risk.ten_year_yield",
        "risk.three_month_yield",
        "risk.yield_curve_spread",
        "risk.vix",
        "risk.equity_market_return_3m",
        "risk.high_yield_return_3m",
    ),
}

FINE_GROUP_BY_FIELD = {
    field_id: group for group, fields in FINE_GROUPS.items() for field_id in fields
}


def _merge(*groups: str) -> list[str]:
    return [field_id for group in groups for field_id in FINE_GROUPS[group]]


COHERENT_GROUPS = {
    2: (
        ("company_and_filings", _merge("valuation", "quality", "events")),
        ("market_and_risk", _merge("market_trend", "market_liquidity", "asset_risk", "macro")),
    ),
    3: (
        ("market_and_catalysts", _merge("market_trend", "market_liquidity", "events")),
        ("company_fundamentals", _merge("valuation", "quality")),
        ("risk_and_macro", _merge("asset_risk", "macro")),
    ),
    4: (
        ("market", _merge("market_trend", "market_liquidity")),
        ("fundamentals", _merge("valuation", "quality")),
        ("events", _merge("events")),
        ("risk_and_macro", _merge("asset_risk", "macro")),
    ),
    5: (
        ("market", _merge("market_trend", "market_liquidity")),
        ("fundamentals", _merge("valuation", "quality")),
        ("events", _merge("events")),
        ("asset_risk", _merge("asset_risk")),
        ("macro", _merge("macro")),
    ),
    6: (
        ("market", _merge("market_trend", "market_liquidity")),
        ("valuation", _merge("valuation")),
        ("quality", _merge("quality")),
        ("events", _merge("events")),
        ("asset_risk", _merge("asset_risk")),
        ("macro", _merge("macro")),
    ),
    7: tuple((group, list(fields)) for group, fields in FINE_GROUPS.items()),
}

# These mixed partitions were frozen before the first capacity-curve collection.
# Keep the field identities fixed when source values or their rendered widths change.
FROZEN_MIXED_GROUPS = {
    2: (
        (
            "events.latest_8k_items", "risk.vix", "fundamentals.accounting_age",
            "risk.ten_year_yield", "market.vol_30d", "risk.idio_vol",
            "fundamentals.roa", "fundamentals.price_to_book",
            "risk.high_yield_return_3m", "fundamentals.cash_to_assets",
            "events.material_event_90d", "events.eight_k_90d", "market.ret_30d",
            "market.ret_90d", "risk.max_drawdown_1y",
        ),
        (
            "market.drawdown_90d", "events.eight_k_30d", "risk.market_beta",
            "events.earnings_release_90d", "events.days_since_latest_8k",
            "risk.equity_market_return_3m", "risk.yield_curve_spread",
            "fundamentals.price_to_sales", "risk.three_month_yield",
            "events.days_since_periodic_filing", "fundamentals.debt_to_assets",
            "fundamentals.pe", "risk.downside_beta", "market.ret_7d",
            "market.adv_bn", "risk.market_corr", "fundamentals.revenue_growth",
        ),
    ),
    3: (
        (
            "risk.idio_vol", "fundamentals.revenue_growth", "market.vol_30d",
            "risk.max_drawdown_1y", "market.ret_90d", "risk.three_month_yield",
            "events.days_since_latest_8k", "market.ret_7d",
            "risk.yield_curve_spread", "events.latest_8k_items",
            "fundamentals.price_to_book", "events.eight_k_30d",
            "fundamentals.cash_to_assets",
        ),
        (
            "fundamentals.pe", "risk.vix", "risk.equity_market_return_3m",
            "events.earnings_release_90d", "risk.market_corr",
            "events.material_event_90d", "market.drawdown_90d",
            "fundamentals.debt_to_assets",
        ),
        (
            "fundamentals.price_to_sales", "risk.downside_beta", "market.adv_bn",
            "risk.high_yield_return_3m", "market.ret_30d", "fundamentals.roa",
            "risk.market_beta", "risk.ten_year_yield", "events.eight_k_90d",
            "events.days_since_periodic_filing", "fundamentals.accounting_age",
        ),
    ),
    4: (
        (
            "events.latest_8k_items", "risk.equity_market_return_3m",
            "risk.ten_year_yield", "market.ret_90d", "risk.idio_vol",
            "fundamentals.cash_to_assets",
        ),
        (
            "fundamentals.price_to_sales", "market.adv_bn",
            "risk.three_month_yield", "risk.max_drawdown_1y",
            "market.drawdown_90d", "fundamentals.debt_to_assets",
            "events.days_since_latest_8k", "events.eight_k_90d",
        ),
        (
            "market.ret_30d", "risk.downside_beta", "events.earnings_release_90d",
            "fundamentals.revenue_growth", "risk.high_yield_return_3m",
            "fundamentals.price_to_book", "events.eight_k_30d",
        ),
        (
            "fundamentals.accounting_age", "risk.market_corr",
            "events.days_since_periodic_filing", "market.ret_7d",
            "events.material_event_90d", "fundamentals.pe", "risk.vix",
            "risk.market_beta", "risk.yield_curve_spread", "fundamentals.roa",
            "market.vol_30d",
        ),
    ),
    5: (
        (
            "fundamentals.revenue_growth", "risk.max_drawdown_1y",
            "fundamentals.price_to_sales", "risk.yield_curve_spread",
            "events.material_event_90d", "market.drawdown_90d",
        ),
        (
            "fundamentals.cash_to_assets", "market.ret_30d", "risk.market_beta",
            "market.vol_30d", "risk.equity_market_return_3m",
            "events.eight_k_90d", "risk.high_yield_return_3m",
            "events.earnings_release_90d",
        ),
        (
            "risk.market_corr", "fundamentals.debt_to_assets",
            "events.latest_8k_items", "risk.ten_year_yield",
            "fundamentals.accounting_age", "events.days_since_latest_8k",
            "market.ret_90d",
        ),
        (
            "events.days_since_periodic_filing", "fundamentals.pe",
            "market.ret_7d", "risk.idio_vol", "risk.three_month_yield",
        ),
        (
            "market.adv_bn", "fundamentals.price_to_book", "events.eight_k_30d",
            "fundamentals.roa", "risk.downside_beta", "risk.vix",
        ),
    ),
    6: (
        (
            "market.adv_bn", "risk.max_drawdown_1y", "risk.ten_year_yield",
            "fundamentals.roa", "market.ret_90d", "events.material_event_90d",
        ),
        (
            "fundamentals.price_to_book", "risk.equity_market_return_3m",
            "events.earnings_release_90d",
        ),
        (
            "fundamentals.debt_to_assets", "market.drawdown_90d", "risk.vix",
            "risk.idio_vol", "events.eight_k_30d",
        ),
        (
            "fundamentals.accounting_age", "risk.market_corr",
            "fundamentals.price_to_sales", "risk.yield_curve_spread",
            "events.days_since_periodic_filing", "events.latest_8k_items",
            "market.ret_7d",
        ),
        (
            "market.vol_30d", "risk.three_month_yield",
            "fundamentals.cash_to_assets", "events.days_since_latest_8k",
            "fundamentals.pe",
        ),
        (
            "fundamentals.revenue_growth", "events.eight_k_90d",
            "market.ret_30d", "risk.market_beta", "risk.downside_beta",
            "risk.high_yield_return_3m",
        ),
    ),
    7: (
        (
            "events.latest_8k_items", "risk.downside_beta",
            "fundamentals.price_to_sales", "risk.high_yield_return_3m",
        ),
        ("fundamentals.revenue_growth", "risk.ten_year_yield"),
        (
            "fundamentals.cash_to_assets", "events.eight_k_30d",
            "risk.market_corr",
        ),
        (
            "risk.max_drawdown_1y", "market.ret_7d",
            "events.earnings_release_90d", "risk.vix", "market.adv_bn",
        ),
        (
            "fundamentals.accounting_age", "risk.market_beta",
            "market.drawdown_90d", "events.days_since_periodic_filing",
            "fundamentals.price_to_book", "risk.equity_market_return_3m",
            "market.vol_30d",
        ),
        (
            "fundamentals.pe", "market.ret_30d", "risk.yield_curve_spread",
            "events.days_since_latest_8k", "fundamentals.debt_to_assets",
        ),
        (
            "risk.idio_vol", "events.eight_k_90d", "market.ret_90d",
            "fundamentals.roa", "events.material_event_90d",
            "risk.three_month_yield",
        ),
    ),
}
FROZEN_MIXED_GROUPS_SHA256 = (
    "9e8fe8a53ef4657455e727b0482f740673ec6907f245b7e6d9a5cd7645238850"
)


def _validate_field_ontology() -> None:
    expected = set(FIELD_BY_ID)
    observed = set(FINE_GROUP_BY_FIELD)
    if observed != expected:
        raise AssertionError(
            f"fine-domain ontology mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    for k, groups in COHERENT_GROUPS.items():
        flattened = [field_id for _, fields in groups for field_id in fields]
        if len(groups) != k or len(flattened) != len(set(flattened)) or set(flattened) != expected:
            raise AssertionError(f"invalid coherent K={k} partition")
    frozen_payload = json.dumps(
        FROZEN_MIXED_GROUPS, sort_keys=True, separators=(",", ":")
    )
    if _sha256(frozen_payload) != FROZEN_MIXED_GROUPS_SHA256:
        raise AssertionError("frozen mixed-partition fingerprint changed")


def _nmi(groups: list[list[str]], label_by_field: dict[str, str]) -> float:
    observations = [
        (seat, label_by_field[field_id])
        for seat, fields in enumerate(groups)
        for field_id in fields
    ]
    n = len(observations)
    seat_counts: dict[int, int] = {}
    label_counts: dict[str, int] = {}
    joint_counts: dict[tuple[int, str], int] = {}
    for seat, label in observations:
        seat_counts[seat] = seat_counts.get(seat, 0) + 1
        label_counts[label] = label_counts.get(label, 0) + 1
        joint_counts[seat, label] = joint_counts.get((seat, label), 0) + 1
    mutual_information = sum(
        count / n * math.log(count * n / (seat_counts[seat] * label_counts[label]))
        for (seat, label), count in joint_counts.items()
    )
    seat_entropy = -sum(count / n * math.log(count / n) for count in seat_counts.values())
    label_entropy = -sum(count / n * math.log(count / n) for count in label_counts.values())
    if seat_entropy <= 0 or label_entropy <= 0:
        return 0.0
    return float(mutual_information / math.sqrt(seat_entropy * label_entropy))


def _broad_labels() -> dict[str, str]:
    return {field_id: field_id.split(".", 1)[0] for field_id in FIELD_BY_ID}


def _mixed_partition(snapshots: list[dict], k: int) -> list[list[str]]:
    """Return the original frozen partition, independent of rendered values."""
    del snapshots
    groups = [list(fields) for fields in FROZEN_MIXED_GROUPS[k]]
    expected_sizes = [len(fields) for _, fields in COHERENT_GROUPS[k]]
    flattened = [field_id for group in groups for field_id in group]
    if (
        [len(group) for group in groups] != expected_sizes
        or len(flattened) != len(set(flattened))
        or set(flattened) != set(FIELD_BY_ID)
    ):
        raise AssertionError(f"invalid frozen mixed K={k} partition")
    return groups


def _is_macro(field_id: str) -> bool:
    _, field, _, kind = FIELD_BY_ID[field_id]
    return kind.startswith("macro_") or field in {
        "ten_year_yield", "three_month_yield", "yield_curve_spread", "vix"
    }


def _packet_text(snapshot: dict, field_ids: list[str]) -> str:
    asset_fields = [field_id for field_id in field_ids if not _is_macro(field_id)]
    macro_fields = [field_id for field_id in field_ids if _is_macro(field_id)]
    lines = ["RESEARCH PACKET"]
    if asset_fields:
        headers = [FIELD_BY_ID[field_id][2] for field_id in asset_fields]
        lines.append("ticker | " + " | ".join(headers))
        for symbol in snapshot["symbols"]:
            values = []
            for field_id in asset_fields:
                family, field, _, kind = FIELD_BY_ID[field_id]
                values.append(_format_value(_raw_value(snapshot, family, field, symbol), kind))
            lines.append(symbol + " | " + " | ".join(values))
    if macro_fields:
        values = []
        for field_id in macro_fields:
            family, field, label, kind = FIELD_BY_ID[field_id]
            values.append(
                f"{label}={_format_value(_raw_value(snapshot, family, field, None), kind)}"
            )
        lines.append("market context | " + " | ".join(values))
    return "\n".join(lines)


def _build_prompt(snapshot: dict, packet_text: str) -> str:
    example = snapshot["symbols"][0]
    return (
        "You are an institutional U.S. equity analyst. Construct a one-month active long/short book "
        "that seeks the best expected risk-adjusted return from the supplied evidence. Do not create "
        "different positions merely to appear diversified.\n\n"
        "You receive one packet from a research workflow. Its field composition was fixed before model "
        "collection. Use only the supplied packet. Do not infer missing fields, import current facts, or "
        "use outside knowledge. The calendar date is intentionally hidden.\n\n"
        "COMMON UNIVERSE\n"
        f"{snapshot['common_core']}\n\n"
        "PERMITTED INFORMATION\n"
        f"{packet_text}\n\n"
        "Give every listed ticker a signed conviction score from -1 to +1, with both positive and "
        "negative scores. Positive means long and negative means short. These are research scores, not "
        "final capital weights. The fixed evaluation pipeline subtracts their mean and rescales absolute "
        "exposure to 1, imposing the same dollar-neutral book construction on every response.\n\n"
        "Reply with one JSON object and no markdown. Use this schema: "
        f'{{"weights":[{{"symbol":"{example}","weight":0.10}},...],'
        '"thesis":"at most 35 words","primary_signals":["signal 1","signal 2"]}}.'
    )


def make_design(packet_rows: list[dict]) -> list[dict]:
    design = []
    broad = _broad_labels()
    snapshots = [row["snapshot"] for row in packet_rows]
    mixed_by_k = {k: _mixed_partition(snapshots, k) for k in TEAM_SIZES}
    for item in packet_rows:
        snapshot = item["snapshot"]
        for k in TEAM_SIZES:
            coherent_groups = [list(fields) for _, fields in COHERENT_GROUPS[k]]
            mixed_groups = mixed_by_k[k]
            for architecture, groups in (
                ("mixed", mixed_groups),
                ("coherent", coherent_groups),
            ):
                packets = []
                for index, field_ids in enumerate(groups, start=1):
                    text = _packet_text(snapshot, field_ids)
                    packets.append({
                        "packet_index": index,
                        "condition": f"capacity_{architecture}_k{k}_p{index}",
                        "field_ids": field_ids,
                        "n_fields": len(field_ids),
                        "fine_domains": sorted({FINE_GROUP_BY_FIELD[field_id] for field_id in field_ids}),
                        "broad_domains": sorted({broad[field_id] for field_id in field_ids}),
                        "rendered_characters": len(text),
                        "packet_sha256": _sha256(text),
                    })
                flattened = [field_id for row in packets for field_id in row["field_ids"]]
                if len(flattened) != 32 or len(set(flattened)) != 32 or set(flattened) != set(FIELD_BY_ID):
                    raise AssertionError(
                        f"{snapshot['date']} {architecture} K={k} does not preserve the field union"
                    )
                design.append({
                    "sample": item["sample"],
                    "snapshot_id": snapshot["snapshot_id"],
                    "date": snapshot["date"],
                    "team_size": k,
                    "architecture": architecture,
                    "fine_domain_nmi": _nmi(groups, FINE_GROUP_BY_FIELD),
                    "broad_domain_nmi": _nmi(groups, broad),
                    "packets": packets,
                })
    return design


def make_jobs(packet_rows: list[dict], design: list[dict]) -> list[dict]:
    snapshots = {row["snapshot"]["date"]: row for row in packet_rows}
    jobs = []
    for assignment in design:
        item = snapshots[assignment["date"]]
        snapshot = item["snapshot"]
        for packet in assignment["packets"]:
            text = _packet_text(snapshot, packet["field_ids"])
            prompt = _build_prompt(snapshot, text)
            for vendor, model in PILOT_MODELS:
                jobs.append({
                    "experiment": "information_capacity_curve",
                    "sample": item["sample"],
                    "snapshot_id": snapshot["snapshot_id"],
                    "date": snapshot["date"],
                    "return_regime": snapshot["return_regime"],
                    "volatility_regime": snapshot["volatility_regime"],
                    "condition": packet["condition"],
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
                    "packet_sha256": _sha256(text),
                    "require_complete_symbols": True,
                })
    return jobs


def _manifest(jobs: list[dict]) -> list[dict]:
    return [
        {
            key: job[key] for key in (
                "sample", "snapshot_id", "architecture", "team_size", "packet_index",
                "condition", "vendor", "requested_model", "prompt_sha256", "packet_sha256",
            )
        }
        for job in jobs
    ]


def write_plan(packet_rows: list[dict], design: list[dict], jobs: list[dict]) -> dict:
    assignment_payload = json.dumps(design, sort_keys=True, separators=(",", ":"))
    manifest_payload = json.dumps(_manifest(jobs), sort_keys=True, separators=(",", ":"))
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "frozen_before_collection": True,
        "random_seed": RANDOM_SEED,
        "assignment_sha256": _sha256(assignment_payload),
        "job_manifest_sha256": _sha256(manifest_payload),
        "research_question": (
            "When the team's information union is fixed, how does independent trading capacity scale "
            "with roster size and the economic organization of information?"
        ),
        "design": {
            "n_dates": len(packet_rows),
            "pilot_dates": sum(row["sample"] == "pilot" for row in packet_rows),
            "heldout_dates": sum(row["sample"] == "held-out" for row in packet_rows),
            "team_sizes": list(TEAM_SIZES),
            "architectures": list(ARCHITECTURES),
            "n_models": len(PILOT_MODELS),
            "models": [{"vendor": vendor, "model": model} for vendor, model in PILOT_MODELS],
            "full_packet_fields": len(FIELD_BY_ID),
            "team_union_equals_full_packet": True,
            "each_field_appears_once_per_partitioned_team": True,
            "packet_size_profile_matched_between_architectures": True,
            "architecture_and_role_labels_hidden_from_prompt": True,
            "temperature": TEMPERATURE,
            "prompt_spec_version": PROMPT_SPEC_VERSION,
            "frozen_mixed_field_assignment_sha256": FROZEN_MIXED_GROUPS_SHA256,
            "n_new_calls": len(jobs),
            "new_calls_by_sample": {
                sample: sum(job["sample"] == sample for job in jobs)
                for sample in ("pilot", "held-out")
            },
            "shared_full_packet_control": (
                "reused from the frozen 34-date information-diversity panel; no new calls"
            ),
        },
        "pre_specified_tests": {
            "coherent_minus_mixed_N_eff": "positive at matched K",
            "mixed_minus_shared_N_eff": "positive at matched K",
            "coherent_minus_mixed_same_side_overlap": "negative at matched K",
            "coherent_minus_mixed_order_concentration": "negative at matched K",
            "shared_marginal_capacity": "declines with K",
            "coherent_marginal_capacity": "exceeds shared marginal capacity",
            "forward_return": "descriptive; no directional return hypothesis",
        },
        "inference_unit": (
            "date; model subsets and packet assignments are averaged within date before inference"
        ),
        "fine_domain_ontology": FINE_GROUPS,
        "assignments": design,
        "pit_correction": pit_correction_record(),
    }
    if CACHE_PATH.exists() and PLAN_PATH.exists():
        previous = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        if (
            previous.get("assignment_sha256") != plan["assignment_sha256"]
            or previous.get("job_manifest_sha256") != plan["job_manifest_sha256"]
        ):
            require_pit_correction(
                "refusing to change the frozen capacity plan after collection began"
            )
            plan["supersedes"] = {
                "assignment_sha256": previous.get("assignment_sha256"),
                "job_manifest_sha256": previous.get("job_manifest_sha256"),
            }
    _atomic_json(PLAN_PATH, plan)
    return plan


def _load_records() -> dict[tuple, dict]:
    if not CACHE_PATH.exists():
        return {}
    rows = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {_job_key(row): row for row in rows}


EXTRA_RESULT_KEYS = (
    "sample", "architecture", "team_size", "packet_index", "fine_domain_nmi",
    "broad_domain_nmi", "field_ids",
)


def _call(job: dict) -> dict:
    row = call_openrouter(job)
    row.update({key: job[key] for key in EXTRA_RESULT_KEYS})
    return row


def collect(
    jobs: list[dict], sample: str, max_workers: int, max_new_calls: int | None
) -> None:
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
        f"information-capacity: {len(jobs)} total planned, {len(eligible)} eligible, "
        f"{len(records)} cached, {len(pending)} pending"
    )
    if not pending:
        return
    completed = failures = 0
    credit_exhausted = False
    save_interval = max_workers * 5
    next_save = save_interval
    next_report = 25
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
                if completed >= next_report or completed == len(pending):
                    print(
                        f"  {completed}/{len(pending)} completed ({failures} terminal failures)",
                        flush=True,
                    )
                    while next_report <= completed:
                        next_report += 25
            if completed >= next_save or completed == len(pending) or credit_exhausted:
                _atomic_json(CACHE_PATH, sorted(records.values(), key=_job_key))
                while next_save <= completed:
                    next_save += save_interval
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
    _validate_field_ontology()
    packet_rows = load_packets()
    design = make_design(packet_rows)
    jobs = make_jobs(packet_rows, design)
    plan = write_plan(packet_rows, design, jobs)
    print(
        f"frozen {len(design)} date-K-architecture assignments and {len(jobs)} calls; "
        f"assignment={plan['assignment_sha256'][:12]}, manifest={plan['job_manifest_sha256'][:12]}"
    )
    if args.freeze_only:
        return
    collect(jobs, args.sample, args.max_workers, args.max_new_calls)


if __name__ == "__main__":
    main()
