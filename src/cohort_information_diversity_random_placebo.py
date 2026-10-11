"""Collect and analyze the random information-partition placebo.

The existing information-assignment experiment compares a shared full packet
with four domain-structured packets (market, fundamentals, filings, and risk).
This placebo asks whether *any* four-way split would create the same apparent
independence.  For every date, the 32 displayed fields in the full packet are
randomly divided into four balanced, domain-mixed packets.  Their union is
exactly the full packet and every packet contains eight fields drawn from all
four source families.  The partition plan is deterministic and written before
any model call.

Dry run and freeze the assignment plan:
    PYTHONPATH=src python3 src/cohort_information_diversity_random_placebo.py --dry-run

Live collection:
    OPENROUTER_API_KEY=... PYTHONPATH=src \
      python3 src/cohort_information_diversity_random_placebo.py

Offline analysis:
    PYTHONPATH=src python3 \
      src/cohort_information_diversity_random_placebo.py --analyze-only
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA, RESULTS
from cohort_information_diversity import (
    BOOTSTRAP_DRAWS,
    CHANNELS,
    PILOT_MODELS,
    TEMPERATURE,
    _atomic_json,
    _credit_exhausted,
    _job_key,
    _matrix,
    _roster_metrics,
    _sha256,
    _sign_flip_p,
    _strict_complete,
    call_openrouter,
)
from pit_correction import record as pit_correction_record, require_enabled as require_pit_correction
from cohort_information_diversity_scaling import IMPACT_DELTA, _coorder_index


PILOT_PACKET_PATH = Path(DATA) / "information_diversity_packets.json"
HELDOUT_PACKET_PATH = Path(DATA) / "information_diversity_confirmatory_packets.json"
PILOT_STRUCTURED_CACHE = Path(DATA) / "cohort_agents_information_diversity.json"
HELDOUT_STRUCTURED_CACHE = Path(DATA) / "cohort_agents_information_diversity_confirmatory.json"
RANDOM_CACHE_PATH = Path(DATA) / "cohort_agents_information_diversity_random_placebo.json"
PLAN_PATH = Path(RESULTS) / "information_diversity_random_placebo_plan.json"
RESULT_PATH = Path(RESULTS) / "information_diversity_random_placebo_results.json"
SUMMARY_PATH = Path(RESULTS) / "information_diversity_random_placebo_summary.csv"

PROMPT_SPEC_VERSION = "information-diversity-random-partition-placebo-v1"
PLAN_SCHEMA_VERSION = "information-diversity-random-partition-plan-v1"
RESULT_SCHEMA_VERSION = "information-diversity-random-partition-results-v1"
RANDOM_SEED = 2026071601
PACKET_LABELS = ("A", "B", "C", "D")
CONDITIONS = tuple(f"random_{label}" for label in PACKET_LABELS)
DEFAULT_MAX_WORKERS = int(os.environ.get("INFORMATION_DIVERSITY_MAX_WORKERS", "5"))


# These are exactly the fields displayed in the four full-packet tables.  The
# yield-curve spread and accounting age are displayed derived fields, so they
# remain separate atoms in the placebo.
FIELD_SPECS = (
    ("market", "ret_7d", "ret_7d%", "percent"),
    ("market", "ret_30d", "ret_30d%", "percent"),
    ("market", "ret_90d", "ret_90d%", "percent"),
    ("market", "vol_30d", "vol_30d%", "percent"),
    ("market", "drawdown_90d", "dd_90d%", "percent"),
    ("market", "adv_bn", "ADV_$bn", "number"),
    ("fundamentals", "pe", "P/E", "number"),
    ("fundamentals", "price_to_sales", "P/S", "number"),
    ("fundamentals", "price_to_book", "P/B", "number"),
    ("fundamentals", "roa", "ROA%", "percent"),
    ("fundamentals", "debt_to_assets", "debt/A%", "percent"),
    ("fundamentals", "cash_to_assets", "cash/A%", "percent"),
    ("fundamentals", "revenue_growth", "rev_g%", "percent"),
    ("fundamentals", "accounting_age", "age_d", "integer"),
    ("events", "eight_k_30d", "8K_30d", "integer"),
    ("events", "eight_k_90d", "8K_90d", "integer"),
    ("events", "earnings_release_90d", "earn_90d", "integer"),
    ("events", "material_event_90d", "material", "integer"),
    ("events", "days_since_latest_8k", "last_8K_d", "integer"),
    ("events", "days_since_periodic_filing", "periodic_d", "integer"),
    ("events", "latest_8k_items", "latest_event", "text"),
    ("risk", "market_beta", "beta", "number"),
    ("risk", "downside_beta", "down_beta", "number"),
    ("risk", "market_corr", "mkt_corr", "number"),
    ("risk", "idio_vol", "idio_vol%", "percent"),
    ("risk", "max_drawdown_1y", "max_dd%", "percent"),
    ("risk", "ten_year_yield", "10y%", "macro_number"),
    ("risk", "three_month_yield", "3m%", "macro_number"),
    ("risk", "yield_curve_spread", "10y-3m_pp", "macro_number"),
    ("risk", "vix", "VIX", "macro_number"),
    ("risk", "equity_market_return_3m", "SPY_3m%", "macro_percent"),
    ("risk", "high_yield_return_3m", "HYG_3m%", "macro_percent"),
)
FIELD_BY_ID = {f"{family}.{field}": (family, field, label, kind)
               for family, field, label, kind in FIELD_SPECS}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_packets() -> list[dict]:
    packets = []
    for sample, path, expected in (
        ("pilot", PILOT_PACKET_PATH, 12),
        ("held-out", HELDOUT_PACKET_PATH, 22),
    ):
        packet = _load(path)
        if not packet.get("design_frozen_before_llm_collection"):
            raise ValueError(f"{path.name} is not marked frozen")
        if len(packet.get("snapshots", [])) != expected:
            raise ValueError(f"{path.name}: expected {expected} snapshots")
        for snapshot in packet["snapshots"]:
            packets.append({"sample": sample, "source_path": path, "snapshot": snapshot})
    dates = [row["snapshot"]["date"] for row in packets]
    if len(dates) != 34 or len(set(dates)) != 34:
        raise ValueError("random placebo requires 34 unique pilot and held-out dates")
    return sorted(packets, key=lambda row: row["snapshot"]["date"])


def _raw_value(snapshot: dict, family: str, field: str, symbol: str | None) -> object:
    if field == "accounting_age":
        row = snapshot["features"]["fundamentals"][str(symbol)]
        filed = row.get("accounting_filed")
        if not filed:
            return None
        return (datetime.fromisoformat(snapshot["date"]) - datetime.fromisoformat(filed)).days
    if field == "yield_curve_spread":
        macro = snapshot["features"]["macro"]
        left, right = macro.get("ten_year_yield"), macro.get("three_month_yield")
        return None if left is None or right is None else float(left) - float(right)
    if field in snapshot["features"]["macro"]:
        return snapshot["features"]["macro"].get(field)
    return snapshot["features"][family][str(symbol)].get(field)


def _format_value(value: object, kind: str) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if kind == "text":
        text = str(value).strip()
        return text if text else "none"
    if kind == "integer":
        return str(int(round(float(value))))
    if kind in {"percent", "macro_percent"}:
        return f"{100.0 * float(value):.2f}"
    return f"{float(value):.2f}"


def _field_values(snapshot: dict, field_id: str) -> list[str]:
    family, field, _, kind = FIELD_BY_ID[field_id]
    if kind.startswith("macro_") or field in {
        "ten_year_yield", "three_month_yield", "yield_curve_spread", "vix"
    }:
        return [_format_value(_raw_value(snapshot, family, field, None), kind)]
    return [
        _format_value(_raw_value(snapshot, family, field, symbol), kind)
        for symbol in snapshot["symbols"]
    ]


def _field_cost(snapshot: dict, field_id: str) -> int:
    _, _, label, _ = FIELD_BY_ID[field_id]
    return len(label) + sum(len(value) + 1 for value in _field_values(snapshot, field_id))


def _partition(snapshot: dict) -> dict[str, list[str]]:
    """Find a reproducible random, balanced, domain-mixed 8/8/8/8 split."""
    seed_text = f"{RANDOM_SEED}:{snapshot['snapshot_id']}:{snapshot['date']}"
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    field_ids = sorted(FIELD_BY_ID)
    costs = {field_id: _field_cost(snapshot, field_id) for field_id in field_ids}
    best: tuple[float, list[str]] | None = None
    for _ in range(40_000):
        shuffled = list(rng.permutation(field_ids))
        groups = [shuffled[index:index + 8] for index in range(0, 32, 8)]
        family_counts = [
            {FIELD_BY_ID[field_id][0] for field_id in group} for group in groups
        ]
        if any(families != {"market", "fundamentals", "events", "risk"}
               for families in family_counts):
            continue
        loads = np.asarray([sum(costs[field_id] for field_id in group) for group in groups])
        imbalance = float((loads.max() - loads.min()) / loads.mean())
        # The first feasible draw remains random; the search only chooses the
        # most size-balanced assignment among pre-frozen random candidates.
        if best is None or imbalance < best[0]:
            best = (imbalance, shuffled)
    if best is None:
        raise RuntimeError(f"could not construct balanced placebo for {snapshot['date']}")
    groups = [best[1][index:index + 8] for index in range(0, 32, 8)]
    partition = {label: group for label, group in zip(PACKET_LABELS, groups)}
    flattened = [field_id for group in partition.values() for field_id in group]
    if len(flattened) != 32 or set(flattened) != set(FIELD_BY_ID):
        raise AssertionError("random partition does not preserve the full field union")
    return partition


def _packet_text(snapshot: dict, label: str, field_ids: list[str]) -> str:
    asset_fields = [field_id for field_id in field_ids
                    if not FIELD_BY_ID[field_id][3].startswith("macro_")
                    and FIELD_BY_ID[field_id][1] not in {
                        "ten_year_yield", "three_month_yield", "yield_curve_spread", "vix"
                    }]
    macro_fields = [field_id for field_id in field_ids if field_id not in asset_fields]
    lines = [
        f"RANDOMLY PARTITIONED RESEARCH PACKET {label}",
        "The packet label has no economic role. Fields were assigned before model collection.",
    ]
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
            family, field, label_text, kind = FIELD_BY_ID[field_id]
            value = _format_value(_raw_value(snapshot, family, field, None), kind)
            values.append(f"{label_text}={value}")
        lines.append("market context | " + " | ".join(values))
    return "\n".join(lines)


def build_prompt(snapshot: dict, label: str, packet_text: str) -> str:
    symbols = snapshot["symbols"]
    example = symbols[0]
    return (
        "You are an institutional U.S. equity analyst. Construct a one-month active long/short book "
        "that seeks the best expected risk-adjusted return from the supplied evidence. Do not create "
        "different positions merely to appear diversified.\n\n"
        f"You receive randomly partitioned research packet {label}. Across the four analysts, packets "
        f"A-D contain exactly the same displayed fields as the desk's full research packet, with no "
        "duplication. Each packet mixes market, fundamental, filing, and risk fields. Packet labels "
        "carry no specialist role. The snapshot identifier is {snapshot['snapshot_id']}; its calendar "
        "date is intentionally hidden. Use only the supplied packet. Do not import current facts, "
        "outside knowledge, or unavailable fields.\n\n"
        "COMMON UNIVERSE\n"
        f"{snapshot['common_core']}\n\n"
        "PERMITTED INFORMATION\n"
        f"{packet_text}\n\n"
        "Apply the same mandate in every condition. Give every listed ticker a signed conviction score "
        "from -1 to +1, with both positive and negative scores. Positive means long and negative means "
        "short. These are research scores, not final capital weights: the fixed evaluation pipeline "
        "subtracts their mean and rescales absolute exposure to 1, imposing the same dollar-neutral book "
        "construction on every response.\n\n"
        "Reply with one JSON object and no markdown. Use this schema: "
        f'{{"weights":[{{"symbol":"{example}","weight":0.10}},...],'
        '"thesis":"at most 35 words","primary_signals":["signal 1","signal 2"]}}.'
    )


def make_design(packets: list[dict]) -> list[dict]:
    design = []
    for item in packets:
        snapshot = item["snapshot"]
        partition = _partition(snapshot)
        packet_rows = []
        for label in PACKET_LABELS:
            field_ids = partition[label]
            packet_text = _packet_text(snapshot, label, field_ids)
            packet_rows.append({
                "label": label,
                "condition": f"random_{label}",
                "field_ids": field_ids,
                "source_families": sorted({FIELD_BY_ID[field_id][0] for field_id in field_ids}),
                "n_fields": len(field_ids),
                "rendered_characters": len(packet_text),
                "packet_sha256": _sha256(packet_text),
            })
        design.append({
            "sample": item["sample"],
            "snapshot_id": snapshot["snapshot_id"],
            "date": snapshot["date"],
            "packets": packet_rows,
        })
    return design


def make_jobs(packets: list[dict], design: list[dict]) -> list[dict]:
    design_by_date = {row["date"]: row for row in design}
    jobs = []
    for item in packets:
        snapshot = item["snapshot"]
        plan = design_by_date[snapshot["date"]]
        for packet_row in plan["packets"]:
            label = packet_row["label"]
            text = _packet_text(snapshot, label, packet_row["field_ids"])
            prompt = build_prompt(snapshot, label, text)
            for vendor, model in PILOT_MODELS:
                jobs.append({
                    "experiment": "information_diversity_random_partition_placebo",
                    "sample": item["sample"],
                    "snapshot_id": snapshot["snapshot_id"],
                    "date": snapshot["date"],
                    "return_regime": snapshot["return_regime"],
                    "volatility_regime": snapshot["volatility_regime"],
                    "condition": f"random_{label}",
                    "replicate": 0,
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


def write_plan(packets: list[dict], design: list[dict], jobs: list[dict]) -> dict:
    assignment_payload = json.dumps(design, sort_keys=True, separators=(",", ":"))
    job_manifest = [
        {
            "snapshot_id": job["snapshot_id"],
            "condition": job["condition"],
            "vendor": job["vendor"],
            "requested_model": job["requested_model"],
            "prompt_sha256": job["prompt_sha256"],
            "packet_sha256": job["packet_sha256"],
        }
        for job in jobs
    ]
    job_payload = json.dumps(job_manifest, sort_keys=True, separators=(",", ":"))
    char_counts = [packet["rendered_characters"]
                   for row in design for packet in row["packets"]]
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "frozen_before_collection": True,
        "random_seed": RANDOM_SEED,
        "randomization_unit": "date",
        "assignment_sha256": _sha256(assignment_payload),
        "job_manifest_sha256": _sha256(job_payload),
        "hypothesis": (
            "domain-structured assignment creates more independent books and less same-side "
            "concentration than an equally sized arbitrary partition of the same full packet"
        ),
        "design": {
            "n_dates": len(packets),
            "pilot_dates": sum(row["sample"] == "pilot" for row in packets),
            "heldout_dates": sum(row["sample"] == "held-out" for row in packets),
            "n_models": len(PILOT_MODELS),
            "n_packets_per_date": 4,
            "fields_per_packet": 8,
            "full_packet_fields": len(FIELD_BY_ID),
            "all_four_source_families_per_packet": True,
            "team_union_equals_full_packet": True,
            "temperature": TEMPERATURE,
            "models": [{"vendor": vendor, "model": model} for vendor, model in PILOT_MODELS],
            "prompt_spec_version": PROMPT_SPEC_VERSION,
            "n_calls": len(jobs),
            "rendered_packet_characters": {
                "minimum": min(char_counts),
                "mean": float(np.mean(char_counts)),
                "maximum": max(char_counts),
            },
        },
        "directional_tests": {
            "structured_minus_random_N_eff": "positive",
            "structured_minus_random_same_side_overlap": "negative",
            "structured_minus_random_surviving_gross": "negative",
            "structured_minus_random_convex_coorder_adv_weighted": "negative",
            "forward_return_mean": "descriptive; no directional hypothesis",
            "full_consensus_alignment": "descriptive retention check",
        },
        "assignments": design,
        "pit_correction": pit_correction_record(),
    }
    if RANDOM_CACHE_PATH.exists() and PLAN_PATH.exists():
        previous = _load(PLAN_PATH)
        if (
            previous.get("assignment_sha256") != plan["assignment_sha256"]
            or previous.get("job_manifest_sha256") != plan["job_manifest_sha256"]
        ):
            require_pit_correction("refusing to change a randomization plan after collection began")
            plan["supersedes"] = {
                "assignment_sha256": previous.get("assignment_sha256"),
                "job_manifest_sha256": previous.get("job_manifest_sha256"),
            }
    _atomic_json(PLAN_PATH, plan)
    return plan


def _load_records() -> dict[tuple, dict]:
    if not RANDOM_CACHE_PATH.exists():
        return {}
    return {_job_key(row): row for row in _load(RANDOM_CACHE_PATH)}


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
        f"random-partition placebo: {len(jobs)} planned, {len(records)} cached, "
        f"{len(pending)} pending"
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
        _atomic_json(RANDOM_CACHE_PATH, sorted(records.values(), key=_job_key))
        if completed % 10 == 0 or completed == len(pending) or stop_for_credit:
            print(f"  {completed}/{len(pending)} completed ({failures} failures)", flush=True)
        if stop_for_credit:
            print("  stopping collection: OpenRouter credit exhausted", flush=True)
            break


def _strict_rows(path: Path) -> list[dict]:
    rows = _load(path)
    bad = [row for row in rows if not _strict_complete(row)]
    if bad:
        raise ValueError(f"{path.name} contains {len(bad)} incomplete rows")
    return rows


def _alignment(frame: pd.DataFrame, full: pd.DataFrame) -> float:
    left = frame.mean(axis=1).to_numpy(dtype=float)
    right = full.mean(axis=1).to_numpy(dtype=float)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        raise ValueError("cannot align a zero-variance consensus book")
    return float(np.corrcoef(left, right)[0, 1])


def _metrics(
    frame: pd.DataFrame,
    full: pd.DataFrame,
    forward: dict[str, float],
    adv: pd.Series,
    nulls: dict[int, dict],
) -> dict:
    output = _roster_metrics(frame, forward, nulls)
    output["convex_coorder_adv_weighted"] = _coorder_index(frame, adv, IMPACT_DELTA)
    output["full_consensus_alignment"] = _alignment(frame, full)
    return output


def _mean(rows: list[dict]) -> dict:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


ANALYSIS_METRICS = (
    "N_eff",
    "N_eff_over_null",
    "same_side_overlap",
    "surviving_gross",
    "convex_coorder_adv_weighted",
    "full_consensus_alignment",
    "forward_return_mean",
)


def _paired(rows: list[dict], treatment: str, control: str, metric: str, seed: int) -> dict:
    treatment_values = np.asarray([row[treatment][metric] for row in rows], dtype=float)
    control_values = np.asarray([row[control][metric] for row in rows], dtype=float)
    differences = treatment_values - control_values
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([
        rng.choice(differences, size=len(differences), replace=True).mean()
        for _ in range(BOOTSTRAP_DRAWS)
    ])
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


def analyze(packets: list[dict], jobs: list[dict], plan: dict) -> dict:
    records = list(_load_records().values())
    current_jobs = {_job_key(job): job for job in jobs}
    stale = [
        _job_key(row) for row in records
        if _job_key(row) not in current_jobs
        or any(row.get(key) != current_jobs[_job_key(row)].get(key)
               for key in ("prompt_sha256", "packet_sha256", "requested_model"))
    ]
    if stale:
        raise ValueError(f"random-placebo cache has {len(stale)} stale records")
    usable = [row for row in records if _strict_complete(row)]
    expected = len(packets) * len(PILOT_MODELS) * len(PACKET_LABELS)
    if len(usable) != expected:
        raise ValueError(f"analysis requires {expected} usable records; found {len(usable)}")

    structured = [
        *_strict_rows(PILOT_STRUCTURED_CACHE),
        *_strict_rows(HELDOUT_STRUCTURED_CACHE),
    ]
    structured_by_date = {}
    for row in structured:
        structured_by_date.setdefault(row["date"], []).append(row)
    random_by_date = {}
    for row in usable:
        random_by_date.setdefault(row["date"], []).append(row)

    vendors = [vendor for vendor, _ in PILOT_MODELS]
    vendor_combinations = list(itertools.combinations(vendors, 4))
    assignments = list(itertools.permutations(PACKET_LABELS))
    channel_assignments = list(itertools.permutations(CHANNELS))
    nulls: dict[int, dict] = {}
    date_rows = []
    provider_date_rows = []

    for item in packets:
        snapshot = item["snapshot"]
        date = snapshot["date"]
        random_rows = random_by_date[date]
        structured_rows = structured_by_date[date]
        symbols = snapshot["symbols"]
        forward = snapshot["forward_return_21d"]
        adv = pd.Series({
            symbol: snapshot["features"]["market"][symbol]["adv_bn"]
            for symbol in symbols
        })

        same_random, same_structured = [], []
        for vendor in vendors:
            full_rows = [row for row in structured_rows
                         if row["condition"] == "rich" and row["vendor"] == vendor
                         and int(row.get("replicate", 0)) == 0]
            full = _matrix(full_rows, symbols, ("vendor",))
            random_selected = [row for row in random_rows if row["vendor"] == vendor]
            structured_selected = [row for row in structured_rows
                                   if row["vendor"] == vendor and row["condition"] in CHANNELS]
            random_metrics = _metrics(
                _matrix(random_selected, symbols, ("condition",)), full, forward, adv, nulls
            )
            structured_metrics = _metrics(
                _matrix(structured_selected, symbols, ("condition",)), full, forward, adv, nulls
            )
            same_random.append(random_metrics)
            same_structured.append(structured_metrics)
            provider_date_rows.append({
                "sample": item["sample"],
                "snapshot_id": snapshot["snapshot_id"],
                "date": date,
                "vendor": vendor,
                "random_same_model_m4": random_metrics,
                "structured_same_model_m4": structured_metrics,
            })

        shared_models, random_models, structured_models = [], [], []
        for combination in vendor_combinations:
            selected_vendors = set(combination)
            full_rows = [row for row in structured_rows
                         if row["condition"] == "rich" and row["vendor"] in selected_vendors
                         and int(row.get("replicate", 0)) == 0]
            full = _matrix(full_rows, symbols, ("vendor",))
            shared_models.append(_metrics(full, full, forward, adv, nulls))
            for assignment in assignments:
                selected = []
                for vendor, label in zip(combination, assignment):
                    matches = [row for row in random_rows
                               if row["vendor"] == vendor
                               and row["condition"] == f"random_{label}"]
                    if len(matches) != 1:
                        raise ValueError(f"missing random assignment {date} {vendor} {label}")
                    selected.append(matches[0])
                random_models.append(_metrics(
                    _matrix(selected, symbols, ("vendor",)), full, forward, adv, nulls
                ))
            for assignment in channel_assignments:
                selected = []
                for vendor, channel in zip(combination, assignment):
                    matches = [row for row in structured_rows
                               if row["vendor"] == vendor and row["condition"] == channel]
                    if len(matches) != 1:
                        raise ValueError(f"missing structured assignment {date} {vendor} {channel}")
                    selected.append(matches[0])
                structured_models.append(_metrics(
                    _matrix(selected, symbols, ("vendor",)), full, forward, adv, nulls
                ))

        date_rows.append({
            "sample": item["sample"],
            "snapshot_id": snapshot["snapshot_id"],
            "date": date,
            "return_regime": snapshot["return_regime"],
            "volatility_regime": snapshot["volatility_regime"],
            "shared_full_models_m4": _mean(shared_models),
            "random_same_model_m4": _mean(same_random),
            "structured_same_model_m4": _mean(same_structured),
            "random_models_m4": _mean(random_models),
            "structured_models_m4": _mean(structured_models),
        })

    comparison_specs = {
        "structured_vs_random_same_model": (
            "structured_same_model_m4", "random_same_model_m4"
        ),
        "structured_vs_random_models": ("structured_models_m4", "random_models_m4"),
        "random_vs_shared_models": ("random_models_m4", "shared_full_models_m4"),
        "structured_vs_shared_models": ("structured_models_m4", "shared_full_models_m4"),
    }
    comparisons = {}
    seed = 2026071680
    for sample in ("pilot", "held-out", "pooled"):
        sample_rows = date_rows if sample == "pooled" else [
            row for row in date_rows if row["sample"] == sample
        ]
        comparisons[sample] = {}
        for name, (treatment, control) in comparison_specs.items():
            comparisons[sample][name] = {}
            for metric in ANALYSIS_METRICS:
                comparisons[sample][name][metric] = _paired(
                    sample_rows, treatment, control, metric, seed
                )
                seed += 1

    provider_comparisons = {}
    for vendor in vendors:
        vendor_rows = [row for row in provider_date_rows if row["vendor"] == vendor]
        provider_comparisons[vendor] = {}
        for metric in ANALYSIS_METRICS:
            provider_comparisons[vendor][metric] = _paired(
                vendor_rows,
                "structured_same_model_m4",
                "random_same_model_m4",
                metric,
                seed,
            )
            seed += 1

    usage = [row.get("usage", {}) for row in usable]
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "analysis_status": "complete",
        "design": plan,
        "collection": {
            "n_records": len(records),
            "n_parse_ok": len(usable),
            "n_parse_failures": len(records) - len(usable),
            "prompt_tokens": int(sum(row.get("prompt_tokens", 0) for row in usage)),
            "completion_tokens": int(sum(row.get("completion_tokens", 0) for row in usage)),
            "reported_cost": float(sum(row.get("cost", 0.0) for row in usage)),
        },
        "nulls": {str(key): value for key, value in nulls.items()},
        "per_date": date_rows,
        "per_provider_date": provider_date_rows,
        "per_provider_pooled_comparisons": provider_comparisons,
        "paired_date_comparisons": comparisons,
    }
    _atomic_json(RESULT_PATH, result)

    summary_rows = []
    for sample, sample_comparisons in comparisons.items():
        for comparison, metrics in sample_comparisons.items():
            for metric, row in metrics.items():
                summary_rows.append({
                    "sample": sample,
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
    collection = result["collection"]
    print("\nRandom information-partition placebo")
    print(
        f"  usable books={collection['n_parse_ok']}/{collection['n_records']}; "
        f"reported cost=${collection['reported_cost']:.2f}"
    )
    pooled = result["paired_date_comparisons"]["pooled"]
    for comparison in (
        "structured_vs_random_same_model",
        "structured_vs_random_models",
        "random_vs_shared_models",
    ):
        print(f"  {comparison}")
        for metric in (
            "N_eff", "same_side_overlap", "convex_coorder_adv_weighted",
            "full_consensus_alignment", "forward_return_mean",
        ):
            row = pooled[comparison][metric]
            print(
                f"    {metric:32s} {row['treatment_mean']:.4f} vs "
                f"{row['control_mean']:.4f}; delta={row['mean_difference']:+.4f} "
                f"[{row['bootstrap_95'][0]:+.4f},{row['bootstrap_95'][1]:+.4f}]"
            )
    provider = result["per_provider_pooled_comparisons"]
    print("  provider-level structured-minus-random checks")
    for vendor in sorted(provider):
        neff = provider[vendor]["N_eff"]
        overlap = provider[vendor]["same_side_overlap"]
        print(
            f"    {vendor:10s}: delta N_eff={neff['mean_difference']:+.3f}; "
            f"delta overlap={overlap['mean_difference']:+.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-new-calls", type=int)
    args = parser.parse_args()

    packets = load_packets()
    design = make_design(packets)
    jobs = make_jobs(packets, design)
    plan = write_plan(packets, design, jobs)
    if args.dry_run:
        char_counts = [packet["rendered_characters"]
                       for row in design for packet in row["packets"]]
        print(json.dumps({
            "plan": str(PLAN_PATH),
            "assignment_sha256": plan["assignment_sha256"],
            "dates": len(packets),
            "calls": len(jobs),
            "packet_characters": {
                "min": min(char_counts),
                "mean": float(np.mean(char_counts)),
                "max": max(char_counts),
            },
            "prompt_characters": {
                "min": min(len(job["prompt"]) for job in jobs),
                "mean": float(np.mean([len(job["prompt"]) for job in jobs])),
                "max": max(len(job["prompt"]) for job in jobs),
            },
        }, indent=2))
        return
    if not args.analyze_only:
        collect(jobs, args.max_workers, args.max_new_calls)
    if args.collect_only:
        return
    result = analyze(packets, jobs, plan)
    print_result(result)


if __name__ == "__main__":
    main()
