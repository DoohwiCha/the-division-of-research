"""Identity-masked and leave-one-channel-out validation for the pilot.

The masked arm removes ticker and company identity while retaining sector and
the numerical/source packet.  It collects one shared-rich book and four
source-specific books per model and date.  The offline validation also checks
that the visible-identity result survives removal of any one information
channel.

Live collection:
    OPENROUTER_API_KEY=... PYTHONPATH=src \
      python3 src/cohort_information_diversity_validation.py

Offline analysis:
    PYTHONPATH=src python3 src/cohort_information_diversity_validation.py --analyze-only
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
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
    _roster_metrics,
    _sha256,
    _strict_complete,
    _summarize,
    call_openrouter,
    load_packets,
)
from cohort_information_diversity_data import COMPANIES
from pit_correction import record as pit_correction_record


CACHE_PATH = Path(DATA) / "cohort_agents_information_diversity_masked.json"
PLAN_PATH = Path(RESULTS) / "information_diversity_masked_plan.json"
RESULT_PATH = Path(RESULTS) / "information_diversity_validation.json"
PROMPT_SPEC_VERSION = "information-diversity-identity-mask-v1"
DEFAULT_MAX_WORKERS = int(os.environ.get("INFORMATION_DIVERSITY_MAX_WORKERS", "5"))


def _aliases(symbols: list[str]) -> tuple[list[str], dict[str, str], dict[str, str]]:
    original_to_alias = {symbol: f"ASSET_{index:02d}" for index, symbol in enumerate(symbols, start=1)}
    alias_to_original = {value: key for key, value in original_to_alias.items()}
    return list(original_to_alias.values()), original_to_alias, alias_to_original


def _masked_text(text: str, original_to_alias: dict[str, str]) -> str:
    result = text
    for symbol in sorted(original_to_alias, key=len, reverse=True):
        result = re.sub(rf"\b{re.escape(symbol)}\b", original_to_alias[symbol], result)
    return result


def _masked_core(symbols: list[str], original_to_alias: dict[str, str]) -> str:
    rows = [f"{'asset':<12}{'sector':<26}"]
    for symbol in symbols:
        rows.append(f"{original_to_alias[symbol]:<12}{COMPANIES[symbol][1]:<26}")
    return "\n".join(rows)


def build_prompt(snapshot: dict, condition: str) -> tuple[str, list[str]]:
    aliases, original_to_alias, _ = _aliases(snapshot["symbols"])
    source_note = {
        "rich": "You receive the union of the desk's four research packets.",
        "market": "You receive only the market-behavior and liquidity research packet.",
        "fundamentals": "You receive only the point-in-time fundamentals and valuation research packet.",
        "events": "You receive only the recent public-filing event research packet.",
        "risk": "You receive only the risk and macro-context research packet.",
    }[condition]
    if condition == "rich":
        packet_text = snapshot["tables"]["rich"]
    else:
        label = {
            "market": "MARKET BEHAVIOR AND LIQUIDITY",
            "fundamentals": "POINT-IN-TIME FUNDAMENTALS AND VALUATION",
            "events": "RECENT SEC FILING EVENTS",
            "risk": "RISK AND MACRO CONTEXT",
        }[condition]
        packet_text = label + "\n" + snapshot["tables"][condition]
    packet_text = _masked_text(packet_text, original_to_alias)
    prompt = (
        "You are an institutional U.S. equity analyst. Construct a one-month active long/short book "
        "that seeks the best expected risk-adjusted return from the supplied evidence. Do not create "
        "different positions merely to appear diversified.\n\n"
        f"{source_note} Company and ticker identities are hidden. The snapshot identifier is "
        f"{snapshot['snapshot_id']}; its calendar date is intentionally hidden. Use only the supplied packet. "
        "Do not infer the hidden identities or import outside knowledge.\n\n"
        "COMMON UNIVERSE\n"
        f"{_masked_core(snapshot['symbols'], original_to_alias)}\n\n"
        "PERMITTED INFORMATION\n"
        f"{packet_text}\n\n"
        "Apply the same mandate in every condition. Give every listed asset a signed conviction score from "
        "-1 to +1, with both positive and negative scores. Positive means long and negative means short. These "
        "are research scores, not final capital weights: the fixed evaluation pipeline subtracts their mean and "
        "rescales absolute exposure to 1, imposing the same dollar-neutral book construction on every response.\n\n"
        "Reply with one JSON object and no markdown. Use this schema: "
        f'{{"weights":[{{"symbol":"{aliases[0]}","weight":0.10}},...],'
        '"thesis":"at most 35 words","primary_signals":["signal 1","signal 2"]}}.'
    )
    return prompt, aliases


def make_jobs(packet: dict) -> list[dict]:
    jobs = []
    for snapshot in packet["snapshots"]:
        for condition in ("rich", *CHANNELS):
            prompt, aliases = build_prompt(snapshot, condition)
            for vendor, model in PILOT_MODELS:
                jobs.append({
                    "experiment": "information_diversity_masked",
                    "snapshot_id": snapshot["snapshot_id"],
                    "date": snapshot["date"],
                    "return_regime": snapshot["return_regime"],
                    "volatility_regime": snapshot["volatility_regime"],
                    "condition": condition,
                    "replicate": 0,
                    "temperature": TEMPERATURE,
                    "vendor": vendor,
                    "requested_model": model,
                    "symbols": aliases,
                    "prompt": prompt,
                    "prompt_spec_version": PROMPT_SPEC_VERSION,
                    "prompt_sha256": _sha256(prompt),
                    "packet_sha256": _sha256(prompt.split("PERMITTED INFORMATION\n", 1)[1]),
                    "require_complete_symbols": True,
                })
    return jobs


def _load_records(path: Path = CACHE_PATH) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {_job_key(row): row for row in rows}


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
    print(f"identity-masked: {len(jobs)} planned, {len(records)} cached, {len(pending)} pending")
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


def _unmask_records(records: list[dict], symbols: list[str]) -> list[dict]:
    aliases, _, alias_to_original = _aliases(symbols)
    if len(aliases) != len(symbols):
        raise ValueError("alias map is not one-to-one")
    out = []
    for row in records:
        copy = dict(row)
        copy["weights"] = [
            {"symbol": alias_to_original[item["symbol"]], "weight": item["weight"]}
            for item in row["weights"] if item["symbol"] in alias_to_original
        ]
        out.append(copy)
    return out


def _mean_metrics(rows: list[dict]) -> dict:
    summary = _summarize(rows, METRICS)
    return {key: summary[key]["mean"] for key in METRICS}


def analyze(packet: dict) -> dict:
    masked_records = list(_load_records().values())
    expected = len(packet["snapshots"]) * len(PILOT_MODELS) * (1 + len(CHANNELS))
    if len(masked_records) != expected or not all(_strict_complete(row) for row in masked_records):
        raise ValueError(f"masked analysis requires {expected} usable records")
    visible_path = Path(DATA) / "cohort_agents_information_diversity.json"
    visible_records = json.loads(visible_path.read_text(encoding="utf-8"))
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    vendor_combinations_m4 = list(itertools.combinations(vendors, 4))
    vendor_combinations_m3 = list(itertools.combinations(vendors, 3))
    channel_permutations = list(itertools.permutations(CHANNELS))
    nulls: dict[int, dict] = {}
    masked_date_rows = []
    loo_date_rows = []
    channel_date_rows = []

    for snapshot in packet["snapshots"]:
        sid = snapshot["snapshot_id"]
        symbols = snapshot["symbols"]
        forward = snapshot["forward_return_21d"]
        masked = _unmask_records(
            [row for row in masked_records if row["snapshot_id"] == sid], symbols
        )
        visible = [
            row for row in visible_records
            if row["snapshot_id"] == sid and _strict_complete(row)
        ]

        information_rows = []
        for vendor in vendors:
            rows = [row for row in masked if row["vendor"] == vendor and row["condition"] in CHANNELS]
            information_rows.append(_roster_metrics(
                _matrix(rows, symbols, ("condition",)), forward, nulls
            ))
        model_rows = []
        combined_rows = []
        for combination in vendor_combinations_m4:
            chosen = set(combination)
            rows = [
                row for row in masked
                if row["condition"] == "rich" and row["vendor"] in chosen
            ]
            model_rows.append(_roster_metrics(_matrix(rows, symbols, ("vendor",)), forward, nulls))
            for assignment in channel_permutations:
                assigned = [
                    next(
                        row for row in masked
                        if row["vendor"] == vendor and row["condition"] == channel
                    )
                    for vendor, channel in zip(combination, assignment)
                ]
                combined_rows.append(_roster_metrics(
                    _matrix(assigned, symbols, ("vendor",)), forward, nulls
                ))
        masked_date_rows.append({
            "snapshot_id": sid,
            "date": snapshot["date"],
            "masked_information_same_model_m4": _mean_metrics(information_rows),
            "masked_models_shared_rich_m4": _mean_metrics(model_rows),
            "masked_models_specialized_m4": _mean_metrics(combined_rows),
        })

        model_m3 = []
        for combination in vendor_combinations_m3:
            rows = [
                row for row in visible
                if row["condition"] == "rich" and int(row["replicate"]) == 0
                and row["vendor"] in set(combination)
            ]
            model_m3.append(_roster_metrics(_matrix(rows, symbols, ("vendor",)), forward, nulls))
        loo_row = {
            "snapshot_id": sid,
            "date": snapshot["date"],
            "visible_models_shared_rich_m3": _mean_metrics(model_m3),
        }
        for omitted in CHANNELS:
            rows_by_vendor = []
            kept = set(CHANNELS) - {omitted}
            for vendor in vendors:
                rows = [
                    row for row in visible
                    if row["vendor"] == vendor and row["condition"] in kept
                ]
                rows_by_vendor.append(_roster_metrics(
                    _matrix(rows, symbols, ("condition",)), forward, nulls
                ))
            loo_row[f"visible_information_without_{omitted}_m3"] = _mean_metrics(rows_by_vendor)
        loo_date_rows.append(loo_row)

        channel_row = {"snapshot_id": sid, "date": snapshot["date"]}
        for channel in CHANNELS:
            rows = [row for row in visible if row["condition"] == channel]
            channel_row[channel] = _roster_metrics(
                _matrix(rows, symbols, ("vendor",)), forward, nulls
            )
        channel_date_rows.append(channel_row)

    comparisons: dict[str, dict] = {}
    seed = 2026071580
    masked_specs = {
        "masked_information_vs_models": (
            "masked_information_same_model_m4", "masked_models_shared_rich_m4"
        ),
        "masked_specialized_models_vs_shared_models": (
            "masked_models_specialized_m4", "masked_models_shared_rich_m4"
        ),
    }
    for name, (treatment, control) in masked_specs.items():
        comparisons[name] = {}
        for metric in METRICS:
            comparisons[name][metric] = _paired_effect(
                masked_date_rows, treatment, control, metric, seed
            )
            seed += 1
    for omitted in CHANNELS:
        name = f"visible_without_{omitted}_vs_models_m3"
        comparisons[name] = {}
        treatment = f"visible_information_without_{omitted}_m3"
        control = "visible_models_shared_rich_m3"
        for metric in METRICS:
            comparisons[name][metric] = _paired_effect(
                loo_date_rows, treatment, control, metric, seed
            )
            seed += 1

    channel_summary = {
        channel: _summarize([row[channel] for row in channel_date_rows], METRICS)
        for channel in CHANNELS
    }
    usage = [row.get("usage", {}) for row in masked_records]
    result = {
        "schema_version": "information-diversity-validation-v1",
        "design": {
            "masked_calls": expected,
            "masking": "ticker and company removed; stable ASSET_01..ASSET_14 aliases; sector retained",
            "masked_conditions": ["rich", *CHANNELS],
            "leave_one_channel_out": "visible-identity M=3 information rosters compared with all M=3 rich-packet model rosters",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "analysis_unit": "date",
        },
        "collection": {
            "n_records": len(masked_records),
            "n_parse_ok": sum(_strict_complete(row) for row in masked_records),
            "prompt_tokens": int(sum(row.get("prompt_tokens", 0) for row in usage)),
            "completion_tokens": int(sum(row.get("completion_tokens", 0) for row in usage)),
            "reported_cost": float(sum(row.get("cost", 0.0) for row in usage)),
        },
        "per_date_masked": masked_date_rows,
        "per_date_leave_one_channel_out": loo_date_rows,
        "visible_channel_cross_model": channel_summary,
        "comparisons": comparisons,
    }
    _atomic_json(RESULT_PATH, result)
    return result


def print_result(result: dict) -> None:
    print("\nInformation-diversity validation")
    for name in ("masked_information_vs_models", "masked_specialized_models_vs_shared_models"):
        neff = result["comparisons"][name]["N_eff"]
        overlap = result["comparisons"][name]["same_side_overlap"]
        print(
            f"  {name}: N_eff {neff['treatment_mean']:.3f} vs {neff['control_mean']:.3f} "
            f"(delta {neff['mean_difference']:+.3f}); O {overlap['treatment_mean']:.3f} "
            f"vs {overlap['control_mean']:.3f} (delta {overlap['mean_difference']:+.3f})"
        )
    for omitted in CHANNELS:
        row = result["comparisons"][f"visible_without_{omitted}_vs_models_m3"]["N_eff"]
        print(
            f"  omit {omitted:12s}: N_eff {row['treatment_mean']:.3f} vs "
            f"models {row['control_mean']:.3f}; higher on {row['dates_treatment_higher']}/12 dates"
        )


def write_plan(packet: dict, jobs: list[dict]) -> None:
    plan = {
        "schema_version": "information-diversity-masked-plan-v1",
        "frozen_before_collection": True,
        "frozen_before_masked_collection": True,
        "prompt_spec_version": PROMPT_SPEC_VERSION,
        "n_calls": len(jobs),
        "n_dates": len(packet["snapshots"]),
        "models": [{"vendor": vendor, "model": model} for vendor, model in PILOT_MODELS],
        "conditions": ["rich", *CHANNELS],
        "masking": "company and ticker identity removed; sector and packet values retained",
        "complete_symbol_rows_required": True,
        "pit_correction": pit_correction_record(),
    }
    _atomic_json(PLAN_PATH, plan)


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
    write_plan(packet, jobs)
    if args.dry_run:
        print(f"planned masked calls: {len(jobs)}")
        print(
            f"prompt characters: min={min(len(job['prompt']) for job in jobs):,}, "
            f"max={max(len(job['prompt']) for job in jobs):,}"
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
