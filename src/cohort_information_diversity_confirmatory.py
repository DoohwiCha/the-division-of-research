"""Run the held-out confirmation of the information-diversity experiment.

The confirmatory sample contains every usable U.S. month-end in the existing
panel that was not selected for the 12-date pilot.  The design is deliberately
narrow: each model receives the common rich packet once and each of the four
research channels once.  The date is the unit of inference.

Dry run:
    PYTHONPATH=src python3 src/cohort_information_diversity_confirmatory.py --dry-run

Live collection:
    OPENROUTER_API_KEY=... PYTHONPATH=src \
      python3 src/cohort_information_diversity_confirmatory.py

Offline analysis:
    PYTHONPATH=src python3 src/cohort_information_diversity_confirmatory.py --analyze-only
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA, RESULTS
from cohort_information_diversity import (
    BOOTSTRAP_DRAWS,
    CHANNELS,
    METRICS,
    PILOT_MODELS,
    PROMPT_SPEC_VERSION,
    TEMPERATURE,
    _atomic_json,
    _credit_exhausted,
    _job_key,
    _matrix,
    _roster_metrics,
    _sha256,
    _sign_flip_p,
    _strict_complete,
    _summarize,
    build_prompt,
    call_openrouter,
)
from pit_correction import record as pit_correction_record


PACKET_PATH = Path(DATA) / "information_diversity_confirmatory_packets.json"
PILOT_PACKET_PATH = Path(DATA) / "information_diversity_packets.json"
CACHE_PATH = Path(DATA) / "cohort_agents_information_diversity_confirmatory.json"
PLAN_PATH = Path(RESULTS) / "information_diversity_confirmatory_plan.json"
RESULT_PATH = Path(RESULTS) / "information_diversity_confirmatory_results.json"
SUMMARY_PATH = Path(RESULTS) / "information_diversity_confirmatory_summary.csv"
PILOT_RESULT_PATH = Path(RESULTS) / "information_diversity_results.json"

DEFAULT_MAX_WORKERS = int(os.environ.get("INFORMATION_DIVERSITY_MAX_WORKERS", "5"))
CONDITIONS = ("rich", *CHANNELS)


def load_packets() -> dict:
    if not PACKET_PATH.exists():
        raise FileNotFoundError(
            f"{PACKET_PATH} is missing; run cohort_information_diversity_data.py --confirmatory"
        )
    packet = json.loads(PACKET_PATH.read_text(encoding="utf-8"))
    if not packet.get("design_frozen_before_llm_collection"):
        raise ValueError("confirmatory design is not marked frozen")
    if len(packet.get("snapshots", [])) != 22:
        raise ValueError("expected 22 held-out snapshots")
    dates = [row["date"] for row in packet["snapshots"]]
    if len(dates) != len(set(dates)):
        raise ValueError("confirmatory dates are not unique")
    if PILOT_PACKET_PATH.exists():
        pilot = json.loads(PILOT_PACKET_PATH.read_text(encoding="utf-8"))
        pilot_dates = {row["date"] for row in pilot["snapshots"]}
        overlap = sorted(pilot_dates.intersection(dates))
        if overlap:
            raise ValueError(f"confirmatory dates overlap with pilot: {overlap}")
    return packet


def _packet_text(snapshot: dict, condition: str) -> str:
    if condition == "rich":
        return snapshot["tables"]["rich"]
    label = {
        "market": "MARKET BEHAVIOR AND LIQUIDITY",
        "fundamentals": "POINT-IN-TIME FUNDAMENTALS AND VALUATION",
        "events": "RECENT SEC FILING EVENTS",
        "risk": "RISK AND MACRO CONTEXT",
    }[condition]
    return label + "\n" + snapshot["tables"][condition]


def make_jobs(packet: dict) -> list[dict]:
    jobs = []
    for snapshot in packet["snapshots"]:
        for condition in CONDITIONS:
            prompt = build_prompt(snapshot, condition)
            for vendor, model in PILOT_MODELS:
                jobs.append({
                    "experiment": "information_diversity_confirmatory",
                    "snapshot_id": snapshot["snapshot_id"],
                    "date": snapshot["date"],
                    "return_regime": snapshot["return_regime"],
                    "volatility_regime": snapshot["volatility_regime"],
                    "condition": condition,
                    "replicate": 0,
                    "temperature": TEMPERATURE,
                    "vendor": vendor,
                    "requested_model": model,
                    "symbols": snapshot["symbols"],
                    "prompt": prompt,
                    "prompt_spec_version": PROMPT_SPEC_VERSION,
                    "prompt_sha256": _sha256(prompt),
                    "packet_sha256": _sha256(_packet_text(snapshot, condition)),
                    "require_complete_symbols": True,
                })
    return jobs


def _load_records() -> dict[tuple, dict]:
    if not CACHE_PATH.exists():
        return {}
    rows = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {_job_key(row): row for row in rows}


def write_plan(packet: dict, jobs: list[dict]) -> dict:
    by_condition = {
        condition: sum(job["condition"] == condition for job in jobs)
        for condition in CONDITIONS
    }
    plan = {
        "schema_version": "information-diversity-confirmatory-plan-v1",
        "frozen_before_collection": True,
        "sample_role": "held-out confirmation; no date overlaps the 12-date pilot",
        "date_selection": packet["design"]["date_selection"],
        "packet_path": str(PACKET_PATH.relative_to(Path(DATA).parent)),
        "packet_sha256": hashlib.sha256(PACKET_PATH.read_bytes()).hexdigest(),
        "prompt_spec_version": PROMPT_SPEC_VERSION,
        "temperature": TEMPERATURE,
        "n_dates": len(packet["snapshots"]),
        "dates": [row["date"] for row in packet["snapshots"]],
        "n_models": len(PILOT_MODELS),
        "models": [{"vendor": vendor, "model": model} for vendor, model in PILOT_MODELS],
        "conditions": list(CONDITIONS),
        "n_calls": len(jobs),
        "calls_by_condition": by_condition,
        "analysis_unit": "date; all model subsets and channel assignments are averaged within date",
        "matched_roster_size": 4,
        "complete_symbol_rows_required": True,
        "directional_hypotheses": {
            "information_vs_models_N_eff": "higher",
            "information_vs_models_same_side_overlap": "lower",
            "specialized_models_vs_shared_models_N_eff": "higher",
            "specialized_models_vs_shared_models_same_side_overlap": "lower",
            "forward_return_mean": "no directional hypothesis; reported as an ex-post quality check",
        },
        "pit_correction": pit_correction_record(),
    }
    _atomic_json(PLAN_PATH, plan)
    return plan


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
        f"information-diversity confirmatory: {len(jobs)} planned, "
        f"{len(records)} cached, {len(pending)} pending"
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
        _atomic_json(CACHE_PATH, sorted(records.values(), key=_job_key))
        if completed % 10 == 0 or completed == len(pending) or stop_for_credit:
            print(
                f"  {completed}/{len(pending)} completed ({failures} failures)",
                flush=True,
            )
        if stop_for_credit:
            print("  stopping collection: OpenRouter credit exhausted", flush=True)
            break


def _mean_metrics(summary: dict) -> dict:
    return {metric: summary[metric]["mean"] for metric in METRICS}


def _paired_effect(rows: list[dict], treatment: str, control: str, metric: str, seed: int) -> dict:
    values = np.asarray(
        [row[treatment][metric] - row[control][metric] for row in rows], dtype=float
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(BOOTSTRAP_DRAWS)
    ])
    return {
        "treatment": treatment,
        "control": control,
        "metric": metric,
        "n_dates": len(values),
        "treatment_mean": float(np.mean([row[treatment][metric] for row in rows])),
        "control_mean": float(np.mean([row[control][metric] for row in rows])),
        "mean_difference": float(values.mean()),
        "bootstrap_95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "dates_treatment_higher": int(np.sum(values > 0)),
        "dates_treatment_lower": int(np.sum(values < 0)),
        "dates_equal": int(np.sum(values == 0)),
        "sign_flip_p_two_sided": _sign_flip_p(values),
        "sign_flip_method": (
            "complete enumeration" if len(values) <= 16 else "100,000-draw Monte Carlo"
        ),
    }


def _comparisons(rows: list[dict], seed: int) -> dict:
    specs = {
        "information_vs_models": ("information_same_model_m4", "models_shared_rich_m4"),
        "specialized_models_vs_shared_models": (
            "models_specialized_m4", "models_shared_rich_m4"
        ),
    }
    output = {}
    for name, (treatment, control) in specs.items():
        output[name] = {}
        for metric in METRICS:
            output[name][metric] = _paired_effect(rows, treatment, control, metric, seed)
            seed += 1
    return output


def analyze(packet: dict) -> dict:
    records = list(_load_records().values())
    current_jobs = {_job_key(job): job for job in make_jobs(packet)}
    stale = []
    for row in records:
        job = current_jobs.get(_job_key(row))
        if not job or any(
            row.get(key) != job.get(key)
            for key in ("prompt_sha256", "packet_sha256", "requested_model")
        ):
            stale.append(_job_key(row))
    if stale:
        raise ValueError(f"cache contains {len(stale)} stale records; rerun collection")
    expected = len(packet["snapshots"]) * len(PILOT_MODELS) * len(CONDITIONS)
    usable = [row for row in records if _strict_complete(row)]
    if len(usable) != expected:
        raise ValueError(f"analysis requires {expected} usable records; found {len(usable)}")

    vendors = [vendor for vendor, _ in PILOT_MODELS]
    vendor_combinations = list(itertools.combinations(vendors, len(CHANNELS)))
    channel_permutations = list(itertools.permutations(CHANNELS))
    nulls: dict[int, dict] = {}
    date_rows = []
    model_date_rows = []

    for snapshot in packet["snapshots"]:
        snapshot_id = snapshot["snapshot_id"]
        date_records = [row for row in usable if row["snapshot_id"] == snapshot_id]
        symbols = snapshot["symbols"]
        forward = snapshot["forward_return_21d"]

        rich_records = [row for row in date_records if row["condition"] == "rich"]
        rich_models_m7 = _roster_metrics(
            _matrix(rich_records, symbols, ("vendor",)), forward, nulls
        )
        channel_models_m7 = {}
        for channel in CHANNELS:
            channel_records = [
                row for row in date_records if row["condition"] == channel
            ]
            channel_models_m7[channel] = _roster_metrics(
                _matrix(channel_records, symbols, ("vendor",)), forward, nulls
            )

        information_rows = []
        for vendor in vendors:
            selected = [
                row for row in date_records
                if row["vendor"] == vendor and row["condition"] in CHANNELS
            ]
            information_metrics = _roster_metrics(
                _matrix(selected, symbols, ("condition",)), forward, nulls
            )
            information_rows.append(information_metrics)
            model_date_rows.append({
                "snapshot_id": snapshot_id,
                "date": snapshot["date"],
                "vendor": vendor,
                "information_same_model_m4": information_metrics,
            })

        model_rows = []
        specialized_rows = []
        for combination in vendor_combinations:
            selected_vendors = set(combination)
            shared = [
                row for row in date_records
                if row["condition"] == "rich" and row["vendor"] in selected_vendors
            ]
            model_rows.append(_roster_metrics(
                _matrix(shared, symbols, ("vendor",)), forward, nulls
            ))
            for assignment in channel_permutations:
                assigned = []
                for vendor, channel in zip(combination, assignment):
                    matches = [
                        row for row in date_records
                        if row["vendor"] == vendor and row["condition"] == channel
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            f"missing assignment {snapshot_id} {vendor} {channel}"
                        )
                    assigned.append(matches[0])
                specialized_rows.append(_roster_metrics(
                    _matrix(assigned, symbols, ("vendor",)), forward, nulls
                ))

        information_summary = _summarize(information_rows, METRICS)
        model_summary = _summarize(model_rows, METRICS)
        specialized_summary = _summarize(specialized_rows, METRICS)
        date_rows.append({
            "snapshot_id": snapshot_id,
            "date": snapshot["date"],
            "return_regime": snapshot["return_regime"],
            "volatility_regime": snapshot["volatility_regime"],
            "rich_models_m7": rich_models_m7,
            "channel_models_m7": channel_models_m7,
            "information_same_model_m4": _mean_metrics(information_summary),
            "models_shared_rich_m4": _mean_metrics(model_summary),
            "models_specialized_m4": _mean_metrics(specialized_summary),
            "within_date_distributions": {
                "information_same_model_m4": information_summary,
                "models_shared_rich_m4": model_summary,
                "models_specialized_m4": specialized_summary,
            },
        })

    held_out_comparisons = _comparisons(date_rows, 2026071560)
    pooled = None
    if PILOT_RESULT_PATH.exists():
        pilot_result = json.loads(PILOT_RESULT_PATH.read_text(encoding="utf-8"))
        pilot_rows = pilot_result["per_date"]
        pilot_dates = {row["date"] for row in pilot_rows}
        held_out_dates = {row["date"] for row in date_rows}
        if pilot_dates.intersection(held_out_dates):
            raise ValueError("pilot and confirmatory result dates overlap")
        pooled_rows = sorted([*pilot_rows, *date_rows], key=lambda row: row["date"])
        pooled = {
            "n_dates": len(pooled_rows),
            "dates": [row["date"] for row in pooled_rows],
            "paired_date_comparisons": _comparisons(pooled_rows, 2026071600),
        }

    usage = [row.get("usage", {}) for row in usable]
    descriptive = {
        "rich_models_m7": _summarize(
            [row["rich_models_m7"] for row in date_rows], METRICS
        ),
        "channel_models_m7": {
            channel: _summarize(
                [row["channel_models_m7"][channel] for row in date_rows], METRICS
            )
            for channel in CHANNELS
        },
        "information_same_model_m4_by_vendor": {
            vendor: _summarize(
                [
                    row["information_same_model_m4"]
                    for row in model_date_rows if row["vendor"] == vendor
                ],
                METRICS,
            )
            for vendor in vendors
        },
    }
    result = {
        "schema_version": "information-diversity-confirmatory-results-v1",
        "design": json.loads(PLAN_PATH.read_text(encoding="utf-8")),
        "collection": {
            "n_records": len(records),
            "n_parse_ok": len(usable),
            "n_parse_failures": len(records) - len(usable),
            "prompt_tokens": int(sum(row.get("prompt_tokens", 0) for row in usage)),
            "completion_tokens": int(sum(row.get("completion_tokens", 0) for row in usage)),
            "reported_cost": float(sum(row.get("cost", 0.0) for row in usage)),
        },
        "nulls": {str(key): value for key, value in nulls.items()},
        "descriptive_checks": descriptive,
        "per_date": date_rows,
        "per_model_date": model_date_rows,
        "held_out_paired_date_comparisons": held_out_comparisons,
        "pooled_with_pilot": pooled,
    }
    _atomic_json(RESULT_PATH, result)

    summary_rows = []
    scopes = [("held_out", held_out_comparisons)]
    if pooled is not None:
        scopes.append(("pooled_34_dates", pooled["paired_date_comparisons"]))
    for scope, comparisons in scopes:
        for comparison, metrics in comparisons.items():
            for metric, row in metrics.items():
                summary_rows.append({
                    "scope": scope,
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
                    "sign_flip_method": row["sign_flip_method"],
                })
    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)
    return result


def print_result(result: dict) -> None:
    collection = result["collection"]
    print("\nInformation-diversity held-out confirmation")
    print(
        f"  usable books={collection['n_parse_ok']}/{collection['n_records']}; "
        f"prompt tokens={collection['prompt_tokens']:,}; "
        f"completion tokens={collection['completion_tokens']:,}"
    )
    selected = (
        ("information_vs_models", "N_eff"),
        ("information_vs_models", "same_side_overlap"),
        ("specialized_models_vs_shared_models", "N_eff"),
        ("specialized_models_vs_shared_models", "same_side_overlap"),
        ("specialized_models_vs_shared_models", "forward_return_mean"),
    )
    for comparison, metric in selected:
        row = result["held_out_paired_date_comparisons"][comparison][metric]
        p_value = row["sign_flip_p_two_sided"]
        p_text = "<0.0001" if p_value < 0.0001 else f"={p_value:.4f}"
        print(
            f"  {comparison:38s} {metric:22s}: "
            f"{row['treatment_mean']:.4f} vs {row['control_mean']:.4f}; "
            f"delta={row['mean_difference']:+.4f} "
            f"[{row['bootstrap_95'][0]:+.4f},{row['bootstrap_95'][1]:+.4f}], "
            f"p{p_text}"
        )


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
    plan = write_plan(packet, jobs)
    if args.dry_run:
        lengths = [len(job["prompt"]) for job in jobs]
        print(f"planned calls: {plan['n_calls']}")
        print(f"calls by condition: {json.dumps(plan['calls_by_condition'], sort_keys=True)}")
        print(
            f"prompt characters: min={min(lengths):,}, "
            f"mean={np.mean(lengths):,.0f}, max={max(lengths):,}"
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
