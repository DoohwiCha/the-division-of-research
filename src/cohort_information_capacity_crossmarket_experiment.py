"""Freeze and collect the harmonized Europe/crypto capacity experiment.

Each market uses the same 32 field definitions, dates, seven model routes,
team sizes, prompt, parser, and assignment rules.  A full-information response
is collected once per model and date, then reused for every shared-access team
size.  Partitioned teams receive every field exactly once across their seats.

Examples:
    PYTHONPATH=src python3 src/cohort_information_capacity_crossmarket_experiment.py \
      --market europe --freeze-only
    OPENROUTER_API_KEY=... PYTHONPATH=src python3 \
      src/cohort_information_capacity_crossmarket_experiment.py --market europe
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
from cohort_information_capacity_crossmarket_data import FIELD_SPECS
from cohort_information_diversity import (
    PILOT_MODELS,
    TEMPERATURE,
    _atomic_json,
    _credit_exhausted,
    _sha256,
    _strict_complete,
    call_openrouter,
)
from pit_correction import record as pit_correction_record, require_enabled as require_pit_correction


PLAN_SCHEMA_VERSION = "information-capacity-crossmarket-plan-v1"
PROMPT_SPEC_VERSION = "information-capacity-crossmarket-neutral-v1"
RANDOM_SEED = 2026071701
TEAM_SIZES = (2, 3, 4, 5, 6, 7)
ARCHITECTURES = ("mixed", "coherent")
DEFAULT_MAX_WORKERS = int(os.environ.get("INFORMATION_CAPACITY_MAX_WORKERS", "8"))

FIELD_BY_ID = {
    f"{group}.{field}": (group, field, label, kind)
    for group, field, label, kind in FIELD_SPECS
}
FINE_GROUPS = {
    group: tuple(
        field_id for field_id, spec in FIELD_BY_ID.items() if spec[0] == group
    )
    for group in dict.fromkeys(spec[0] for spec in FIELD_SPECS)
}
FINE_GROUP_BY_FIELD = {
    field_id: group for group, fields in FINE_GROUPS.items() for field_id in fields
}


def _merge(*groups: str) -> list[str]:
    return [field_id for group in groups for field_id in FINE_GROUPS[group]]


COHERENT_GROUPS = {
    2: (
        ("return_signals", _merge("trend", "reversal", "relative")),
        ("risk_and_liquidity", _merge("volatility", "tail", "systematic", "liquidity")),
    ),
    3: (
        ("return_signals", _merge("trend", "reversal", "relative")),
        ("risk", _merge("volatility", "tail", "systematic")),
        ("liquidity", _merge("liquidity")),
    ),
    4: (
        ("return_signals", _merge("trend", "reversal", "relative")),
        ("distribution_risk", _merge("volatility", "tail")),
        ("systematic", _merge("systematic")),
        ("liquidity", _merge("liquidity")),
    ),
    5: (
        ("return_signals", _merge("trend", "reversal")),
        ("distribution_risk", _merge("volatility", "tail")),
        ("systematic", _merge("systematic")),
        ("liquidity", _merge("liquidity")),
        ("relative", _merge("relative")),
    ),
    6: (
        ("trend", _merge("trend")),
        ("reversal", _merge("reversal")),
        ("distribution_risk", _merge("volatility", "tail")),
        ("systematic", _merge("systematic")),
        ("liquidity", _merge("liquidity")),
        ("relative", _merge("relative")),
    ),
    7: tuple((group, list(fields)) for group, fields in FINE_GROUPS.items()),
}


def paths(market: str) -> tuple[Path, Path, Path]:
    return (
        Path(DATA) / f"information_capacity_{market}_packets.json",
        Path(RESULTS) / f"information_capacity_{market}_experiment_plan.json",
        Path(DATA) / f"cohort_agents_information_capacity_{market}.json",
    )


def _load_packet(market: str) -> dict:
    packet_path, _, _ = paths(market)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("market") != market or not packet.get("design_frozen_before_llm_collection"):
        raise ValueError(f"invalid frozen packet for {market}")
    if len(packet.get("snapshots", [])) != 12:
        raise ValueError(f"{market}: expected 12 snapshots")
    return packet


def _validate_ontology() -> None:
    expected = set(FIELD_BY_ID)
    if len(expected) != 32 or set(FINE_GROUP_BY_FIELD) != expected:
        raise AssertionError("cross-market field ontology must contain 32 unique fields")
    for k, groups in COHERENT_GROUPS.items():
        flat = [field_id for _, fields in groups for field_id in fields]
        if len(groups) != k or len(flat) != len(set(flat)) or set(flat) != expected:
            raise AssertionError(f"invalid coherent K={k} partition")


def _nmi(groups: list[list[str]]) -> float:
    observations = [
        (seat, FINE_GROUP_BY_FIELD[field_id])
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
    mi = sum(
        count / n * math.log(count * n / (seat_counts[seat] * label_counts[label]))
        for (seat, label), count in joint_counts.items()
    )
    hs = -sum(count / n * math.log(count / n) for count in seat_counts.values())
    hl = -sum(count / n * math.log(count / n) for count in label_counts.values())
    return 0.0 if hs <= 0 or hl <= 0 else float(mi / math.sqrt(hs * hl))


def _format(value: float, kind: str) -> str:
    if kind == "percent":
        return f"{100.0 * float(value):.2f}"
    if kind == "rank":
        return f"{100.0 * float(value):.1f}"
    return f"{float(value):.3f}"


def _packet_text(snapshot: dict, field_ids: list[str]) -> str:
    labels = [FIELD_BY_ID[field_id][2] for field_id in field_ids]
    lines = ["RESEARCH PACKET", "ticker | " + " | ".join(labels)]
    for symbol in snapshot["symbols"]:
        values = [
            _format(snapshot["features"][symbol][field_id], FIELD_BY_ID[field_id][3])
            for field_id in field_ids
        ]
        lines.append(symbol + " | " + " | ".join(values))
    return "\n".join(lines)


def _field_cost(snapshot: dict, field_id: str) -> int:
    _, _, label, kind = FIELD_BY_ID[field_id]
    return len(label) + sum(
        len(_format(snapshot["features"][symbol][field_id], kind)) + 1
        for symbol in snapshot["symbols"]
    )


def _mixed_partition(market: str, snapshots: list[dict], k: int) -> list[list[str]]:
    coherent = [list(fields) for _, fields in COHERENT_GROUPS[k]]
    sizes = [len(fields) for fields in coherent]
    target = np.asarray([
        np.mean([sum(_field_cost(snapshot, field_id) for field_id in fields) for snapshot in snapshots])
        for fields in coherent
    ])
    costs = {
        field_id: float(np.mean([_field_cost(snapshot, field_id) for snapshot in snapshots]))
        for field_id in FIELD_BY_ID
    }
    seed_text = f"{RANDOM_SEED}:{market}:K{k}"
    rng = np.random.default_rng(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16))
    fields = np.asarray(sorted(FIELD_BY_ID), dtype=object)
    best: tuple[float, list[list[str]]] | None = None
    for _ in range(20_000):
        shuffled = [str(value) for value in rng.permutation(fields)]
        groups, cursor = [], 0
        for size in sizes:
            groups.append(shuffled[cursor:cursor + size])
            cursor += size
        loads = np.asarray([sum(costs[field_id] for field_id in group) for group in groups])
        load_error = float(np.mean(np.abs(loads - target) / np.maximum(target, 1.0)))
        objective = _nmi(groups) + 0.10 * load_error
        if best is None or objective < best[0]:
            best = (objective, groups)
    if best is None:
        raise RuntimeError(f"failed mixed partition for {market} K={k}")
    return best[1]


def _build_prompt(snapshot: dict, packet_text: str) -> str:
    market_phrase = (
        {
            "us": "U.S. large-cap equity",
            "europe": "European large-cap equity",
            "crypto": "crypto-perpetual",
        }[snapshot["market"]]
    )
    example = snapshot["symbols"][0]
    return (
        f"You are an institutional {market_phrase} analyst. Construct a one-month active long/short "
        "book that seeks the best expected risk-adjusted return from the supplied evidence. Do not "
        "create different positions merely to appear diversified.\n\n"
        "You receive one packet from a research workflow. Use only this packet. Do not infer missing "
        "fields, import current facts, or use outside knowledge. The calendar date, packet architecture, "
        "and any research-role label are intentionally hidden.\n\n"
        "COMMON UNIVERSE\n"
        f"{snapshot['common_core']}\n\n"
        "PERMITTED INFORMATION\n"
        f"{packet_text}\n\n"
        "Give every listed ticker a signed conviction score from -1 to +1, with both positive and "
        "negative scores. Positive means long and negative means short. These are research scores. "
        "The fixed evaluation pipeline subtracts their mean and rescales absolute exposure to 1.\n\n"
        "Reply with one JSON object and no markdown. Use this schema: "
        f'{{"weights":[{{"symbol":"{example}","weight":0.10}},...],'
        '"thesis":"at most 35 words","primary_signals":["signal 1","signal 2"]}}.'
    )


def make_design(market: str, snapshots: list[dict]) -> list[dict]:
    mixed_by_k = {k: _mixed_partition(market, snapshots, k) for k in TEAM_SIZES}
    design = []
    for snapshot in snapshots:
        sample = "pilot" if snapshot["date"] in {row["date"] for row in snapshots[:4]} else "held-out"
        for k in TEAM_SIZES:
            for architecture, groups in (
                ("mixed", mixed_by_k[k]),
                ("coherent", [list(fields) for _, fields in COHERENT_GROUPS[k]]),
            ):
                packets = []
                for index, field_ids in enumerate(groups, start=1):
                    text = _packet_text(snapshot, field_ids)
                    packets.append({
                        "packet_index": index,
                        "condition": f"{architecture}_k{k}_p{index}",
                        "field_ids": field_ids,
                        "n_fields": len(field_ids),
                        "fine_domain_nmi": _nmi(groups),
                        "packet_sha256": _sha256(text),
                        "rendered_characters": len(text),
                    })
                flat = [field_id for packet in packets for field_id in packet["field_ids"]]
                if len(flat) != 32 or len(set(flat)) != 32 or set(flat) != set(FIELD_BY_ID):
                    raise AssertionError(f"{market} {snapshot['date']} {architecture} K={k}")
                design.append({
                    "sample": sample,
                    "snapshot_id": snapshot["snapshot_id"],
                    "date": snapshot["date"],
                    "architecture": architecture,
                    "team_size": k,
                    "fine_domain_nmi": _nmi(groups),
                    "packets": packets,
                })
    return design


def make_jobs(market: str, snapshots: list[dict], design: list[dict]) -> list[dict]:
    lookup = {snapshot["date"]: snapshot for snapshot in snapshots}
    jobs = []
    for snapshot in snapshots:
        sample = "pilot" if snapshot in snapshots[:4] else "held-out"
        text = _packet_text(snapshot, list(FIELD_BY_ID))
        prompt = _build_prompt(snapshot, text)
        for vendor, model in PILOT_MODELS:
            jobs.append({
                "experiment": "information_capacity_crossmarket",
                "market": market,
                "sample": sample,
                "snapshot_id": snapshot["snapshot_id"],
                "date": snapshot["date"],
                "return_regime": snapshot["return_regime"],
                "volatility_regime": snapshot["volatility_regime"],
                "condition": "shared_full",
                "replicate": 0,
                "architecture": "shared",
                "team_size": 0,
                "packet_index": 0,
                "fine_domain_nmi": 0.0,
                "field_ids": list(FIELD_BY_ID),
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
    for assignment in design:
        snapshot = lookup[assignment["date"]]
        for packet in assignment["packets"]:
            text = _packet_text(snapshot, packet["field_ids"])
            prompt = _build_prompt(snapshot, text)
            for vendor, model in PILOT_MODELS:
                jobs.append({
                    "experiment": "information_capacity_crossmarket",
                    "market": market,
                    "sample": assignment["sample"],
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


def _key(row: dict) -> tuple:
    return (
        row["market"], row["snapshot_id"], row["condition"],
        int(row.get("replicate", 0)), row["vendor"],
    )


def usable_response(row: dict) -> bool:
    """Require a complete response and a nonzero demeaned active book."""
    if not _strict_complete(row):
        return False
    values = np.asarray(
        [float(item["weight"]) for item in row.get("weights", [])], dtype=float
    )
    if not len(values):
        return False
    return float(np.abs(values - values.mean()).sum()) > 1e-12


def _manifest(jobs: list[dict]) -> list[dict]:
    keys = (
        "market", "sample", "snapshot_id", "architecture", "team_size",
        "packet_index", "condition", "vendor", "requested_model", "prompt_sha256",
        "packet_sha256",
    )
    return [{key: job[key] for key in keys} for job in jobs]


def write_plan(market: str, design: list[dict], jobs: list[dict]) -> dict:
    _, plan_path, cache_path = paths(market)
    assignments = json.dumps(design, sort_keys=True, separators=(",", ":"))
    manifest = json.dumps(_manifest(jobs), sort_keys=True, separators=(",", ":"))
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "frozen_before_collection": True,
        "market": market,
        "random_seed": RANDOM_SEED,
        "assignment_sha256": _sha256(assignments),
        "job_manifest_sha256": _sha256(manifest),
        "research_question": (
            "With a fixed information union, does separating access create independent books, "
            "and does economic organization add capacity beyond separation?"
        ),
        "design": {
            "n_dates": 12,
            "pilot_dates": 4,
            "heldout_dates": 8,
            "team_sizes": list(TEAM_SIZES),
            "architectures": ["shared", *ARCHITECTURES],
            "n_models": len(PILOT_MODELS),
            "models": [{"vendor": vendor, "model": model} for vendor, model in PILOT_MODELS],
            "n_fields": len(FIELD_BY_ID),
            "same_field_union_across_markets": True,
            "each_field_once_per_partitioned_team": True,
            "packet_size_profile_matched": True,
            "architecture_and_role_labels_hidden": True,
            "temperature": TEMPERATURE,
            "prompt_spec_version": PROMPT_SPEC_VERSION,
            "n_calls": len(jobs),
        },
        "pre_specified_tests": {
            "mixed_minus_shared_N_eff": "positive",
            "coherent_minus_shared_N_eff": "positive",
            "coherent_minus_mixed_N_eff": "positive",
            "mixed_minus_shared_overlap": "negative",
            "coherent_minus_shared_overlap": "negative",
        },
        "inference_unit": "date; all subsets and packet-to-model assignments average within date",
        "fine_domain_ontology": FINE_GROUPS,
        "assignments": design,
        "pit_correction": pit_correction_record(),
    }
    if cache_path.exists() and plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        changed = False
        for key in ("assignment_sha256", "job_manifest_sha256"):
            if previous.get(key) != plan[key]:
                changed = True
        if changed:
            require_pit_correction(f"refusing to change frozen {market} plan after collection")
            plan["supersedes"] = {
                key: previous.get(key)
                for key in ("assignment_sha256", "job_manifest_sha256")
            }
    _atomic_json(plan_path, plan)
    return plan


EXTRA_KEYS = (
    "market", "sample", "architecture", "team_size", "packet_index",
    "fine_domain_nmi", "field_ids",
)


def _call(job: dict) -> dict:
    row = call_openrouter(job)
    row.update({key: job[key] for key in EXTRA_KEYS})
    return row


def collect(
    market: str,
    jobs: list[dict],
    sample: str,
    max_workers: int,
    max_new_calls: int | None,
) -> None:
    _, _, cache_path = paths(market)
    existing = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else []
    records = {_key(row): row for row in existing}
    eligible = [job for job in jobs if sample == "all" or job["sample"] == sample]
    pending = []
    for job in eligible:
        row = records.get(_key(job), {})
        current = (
            usable_response(row)
            and row.get("prompt_sha256") == job["prompt_sha256"]
            and row.get("packet_sha256") == job["packet_sha256"]
            and row.get("requested_model") == job["requested_model"]
        )
        if not current:
            pending.append(job)
    if max_new_calls is not None:
        pending = pending[:max_new_calls]
    print(
        f"{market}: {len(jobs)} planned, {len(eligible)} eligible, "
        f"{len(records)} cached, {len(pending)} pending"
    )
    if not pending:
        return
    completed = failures = 0
    save_interval = max_workers * 5
    next_save = save_interval
    next_report = 40
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        job_iter = iter(pending)
        in_flight = {
            executor.submit(_call, job): job
            for job in islice(job_iter, max_workers)
        }
        stop_credit = False
        while in_flight:
            finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                in_flight.pop(future)
                row = future.result()
                records[_key(row)] = row
                completed += 1
                failures += int(not usable_response(row))
                stop_credit |= _credit_exhausted(row)
            if completed >= next_save or completed == len(pending) or stop_credit:
                _atomic_json(cache_path, sorted(records.values(), key=_key))
                while next_save <= completed:
                    next_save += save_interval
            if completed >= next_report or completed == len(pending):
                print(f"{market}: completed {completed}/{len(pending)}, failures={failures}")
                while next_report <= completed:
                    next_report += 40
            if stop_credit:
                for future in in_flight:
                    future.cancel()
                print(
                    f"{market}: stopping at the OpenRouter credit/key limit; "
                    "collection remains resumable"
                )
                break
            for _ in finished:
                try:
                    job = next(job_iter)
                except StopIteration:
                    break
                in_flight[executor.submit(_call, job)] = job
    _atomic_json(cache_path, sorted(records.values(), key=_key))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=("us", "europe", "crypto"), required=True)
    parser.add_argument("--sample", choices=("pilot", "held-out", "all"), default="all")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-new-calls", type=int)
    args = parser.parse_args()

    _validate_ontology()
    packet = _load_packet(args.market)
    snapshots = packet["snapshots"]
    design = make_design(args.market, snapshots)
    jobs = make_jobs(args.market, snapshots, design)
    plan = write_plan(args.market, design, jobs)
    print(
        f"{args.market}: frozen {plan['design']['n_calls']} calls; "
        f"assignment={plan['assignment_sha256'][:12]}"
    )
    if not args.freeze_only:
        collect(args.market, jobs, args.sample, args.max_workers, args.max_new_calls)


if __name__ == "__main__":
    main()
