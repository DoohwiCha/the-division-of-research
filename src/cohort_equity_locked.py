"""Collect and analyze the locked U.S.-equity repeated-draw and temperature panels.

The collector deliberately does not reuse ``cohort_agents_equity.json``.  That
legacy keystone cache was produced through the crypto prompt builder.  This
experiment uses one equity-specific prompt for all four dates and collects all
three T=0.7 draws in the same run window.

Live collection (resumable):
    OPENROUTER_API_KEY=... PYTHONPATH=src python3 src/cohort_equity_locked.py

Offline analysis:
    PYTHONPATH=src python3 src/cohort_equity_locked.py --analyze-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from _paths import DATA, RESULTS, inp
from cohort_analyze import n_eff
from cohort_crossvendor import OPENROUTER_MODELS, OPENROUTER_URL, PERSONAS, _extract_json, _post
from cohort_crossvendor_hardening import books, matched_null, same_side_overlap
from cohort_equity_multiregime import EQUITY_PROMPT_SPEC_VERSION, build_equity_prompt


REPEATED_TEMPERATURE = 0.7
REPEATED_DRAWS = 3
GRID_TEMPERATURES = (0.2, 1.0)
DEFAULT_MAX_WORKERS = int(os.environ.get("EQUITY_LOCKED_MAX_WORKERS", "5"))
REPEATED_CACHE = Path(DATA) / "cohort_agents_equity_locked.json"
TEMPERATURE_CACHE = Path(DATA) / "cohort_agents_equity_temperature_grid.json"
RESULT_PATH = Path(RESULTS) / "cohort_equity_locked.json"
TEMPERATURE_RESULT_PATH = Path(RESULTS) / "cohort_equity_temperature_grid.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def load_snapshots() -> list[dict]:
    current = json.loads((Path(DATA) / "equity_snapshot_2026-05-31.json").read_text(encoding="utf-8"))
    past = json.loads(Path(inp("equity_snapshots_multi.json")).read_text(encoding="utf-8"))
    snapshots = [current, *past]
    dates = [row["date"] for row in snapshots]
    if len(snapshots) != 4 or len(set(dates)) != 4:
        raise ValueError(f"expected four distinct equity snapshots, got {dates}")
    universes = {tuple(row["syms"]) for row in snapshots}
    if len(universes) != 1 or len(next(iter(universes))) != 14:
        raise ValueError("locked equity snapshots must share one 14-stock universe")
    return snapshots


def _job_key(row: dict) -> tuple:
    return (
        row["experiment"],
        row["date"],
        int(row["replicate"]),
        float(row["temperature"]),
        row["vendor"],
        row["persona"],
    )


def _load_records(path: Path) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {_job_key(row): row for row in rows}


def _usage(response: dict) -> dict:
    raw = response.get("usage") or {}
    keep = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {key: raw[key] for key in keep if raw.get(key) is not None}


def _clean_weights(obj: dict, syms: list[str]) -> tuple[list[dict], dict]:
    raw = obj.get("weights")
    if not isinstance(raw, list):
        raise ValueError("weights is not a list")
    allowed = set(syms)
    values: dict[str, float] = {}
    unknown: list[str] = []
    duplicate: list[str] = []
    for item in raw:
        if not isinstance(item, dict) or "symbol" not in item or "weight" not in item:
            raise ValueError("malformed weight row")
        symbol = str(item["symbol"]).strip().upper()
        if symbol not in allowed:
            unknown.append(symbol)
            continue
        weight = float(item["weight"])
        if not math.isfinite(weight):
            raise ValueError(f"non-finite weight for {symbol}")
        if symbol in values:
            duplicate.append(symbol)
        values[symbol] = weight
    if len(values) < 2 or not any(abs(value) > 1e-12 for value in values.values()):
        raise ValueError("fewer than two usable, nonzero symbols")
    missing = [symbol for symbol in syms if symbol not in values]
    cleaned = [{"symbol": symbol, "weight": float(values.get(symbol, 0.0))} for symbol in syms]
    diagnostics = {
        "n_symbols_returned": len(values),
        "missing_symbols_zero_filled": missing,
        "unknown_symbols_dropped": sorted(set(unknown)),
        "duplicate_symbols_last_value_used": sorted(set(duplicate)),
    }
    return cleaned, diagnostics


def call_openrouter(job: dict) -> dict:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    prompt = job.get("prompt") or build_equity_prompt(
        job["persona"], job["syms"], job["table"], job.get("market_name", "U.S.")
    )
    payload = {
        "model": job["requested_model"],
        "temperature": job["temperature"],
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://icaif-cohort.local",
        "X-Title": "cohort-equity-locked",
    }
    last_error = "unknown error"
    for attempt in range(5):
        try:
            response = _post(OPENROUTER_URL, payload, headers)
            text = response["choices"][0]["message"]["content"]
            obj = _extract_json(text)
            weights, diagnostics = _clean_weights(obj, job["syms"])
            return {
                **{key: job[key] for key in (
                    "experiment", "date", "regime", "replicate", "temperature", "vendor",
                    "requested_model", "persona", "prompt_spec_version", "prompt_sha256",
                    "snapshot_sha256",
                )},
                "model": job["requested_model"],
                "response_model": response.get("model"),
                "collected_at_utc": _utc_now(),
                "weights": weights,
                "thesis": obj.get("thesis", ""),
                "parse_ok": True,
                "usage": _usage(response),
                **diagnostics,
            }
        except Exception as exc:  # pragma: no cover - network-dependent
            last_error = f"{type(exc).__name__}: {exc}"[:300]
            if attempt < 4:
                time.sleep((2 ** attempt) + random.random())
    return {
        **{key: job[key] for key in (
            "experiment", "date", "regime", "replicate", "temperature", "vendor",
            "requested_model", "persona", "prompt_spec_version", "prompt_sha256",
            "snapshot_sha256",
        )},
        "model": job["requested_model"],
        "collected_at_utc": _utc_now(),
        "parse_ok": False,
        "error": last_error,
    }


def _jobs(snapshots: list[dict], experiment: str) -> list[dict]:
    if experiment == "repeated_draws":
        plan = [(snapshot, replicate, REPEATED_TEMPERATURE) for snapshot in snapshots for replicate in range(REPEATED_DRAWS)]
    elif experiment == "temperature_grid":
        current = next(row for row in snapshots if row["date"] == "2026-05-31")
        plan = [(current, 0, temperature) for temperature in GRID_TEMPERATURES]
    else:
        raise ValueError(experiment)
    jobs = []
    for snapshot, replicate, temperature in plan:
        for vendor, model in OPENROUTER_MODELS:
            for persona in PERSONAS:
                prompt = build_equity_prompt(persona, snapshot["syms"], snapshot["snapshot"])
                jobs.append({
                    "experiment": experiment,
                    "date": snapshot["date"],
                    "regime": snapshot["regime"],
                    "replicate": replicate,
                    "temperature": float(temperature),
                    "vendor": vendor,
                    "requested_model": model,
                    "persona": persona,
                    "syms": snapshot["syms"],
                    "table": snapshot["snapshot"],
                    "prompt_spec_version": EQUITY_PROMPT_SPEC_VERSION,
                    "prompt_sha256": _sha256(prompt),
                    "snapshot_sha256": _sha256(snapshot["snapshot"]),
                })
    return jobs


def collect(jobs: list[dict], cache_path: Path, max_workers: int, max_new_calls: int | None = None) -> None:
    records = _load_records(cache_path)
    pending = [job for job in jobs if not records.get(_job_key(job), {}).get("parse_ok")]
    if max_new_calls is not None:
        pending = pending[:max_new_calls]
    print(f"{jobs[0]['experiment']}: {len(jobs)} planned, {len(records)} cached, {len(pending)} pending")
    if not pending:
        return
    done = failures = 0
    save_every = max(1, int(os.environ.get("COHORT_SAVE_EVERY", "1")))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(call_openrouter, job) for job in pending]
        for future in as_completed(futures):
            row = future.result()
            records[_job_key(row)] = row
            done += 1
            failures += int(not row.get("parse_ok"))
            if done % save_every == 0 or done == len(pending):
                _atomic_json(cache_path, sorted(records.values(), key=_job_key))
            if done % 10 == 0 or done == len(pending):
                print(f"  {done}/{len(pending)} completed in this run ({failures} failures)", flush=True)


def _same_persona_cross_vendor(C: np.ndarray, agents: list[dict]) -> float:
    values = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if agents[i]["persona"] == agents[j]["persona"] and agents[i]["vendor"] != agents[j]["vendor"]:
                values.append(float(C[i, j]))
    return float(np.mean(values)) if values else float("nan")


def _panel_metrics(
    agents: list[dict], syms: list[str], null_cache: dict[int, dict], expected_m: int | None = None
) -> dict:
    clean = [row for row in agents if row.get("parse_ok") and row.get("weights")]
    if len(clean) < 4:
        return {"M": len(clean), "complete": False}
    W = books(clean, syms)
    W = W.loc[:, W.std() > 1e-12]
    keep = set(W.columns)
    clean = [row for row in clean if f"{row['persona']}:{row['vendor']}" in keep]
    C = np.corrcoef(W.to_numpy(dtype=float).T)
    M = W.shape[1]
    if M not in null_cache:
        null_cache[M] = matched_null(M, syms)
    iu = np.triu_indices(M, 1)
    if expected_m is None:
        expected_m = len(OPENROUTER_MODELS) * len(PERSONAS)
    return {
        "M": int(M),
        "complete": M == expected_m,
        "N_eff": float(n_eff(C)),
        "mean_abs_corr": float(np.abs(C[iu]).mean()),
        "same_side_overlap": float(same_side_overlap(W)),
        "same_persona_cross_vendor_corr": _same_persona_cross_vendor(C, clean),
        "null": null_cache[M],
        "below_null_p2_5": bool(n_eff(C) < null_cache[M]["p2_5"]),
    }


def _summarize_rows(rows: list[dict]) -> dict:
    complete = [row for row in rows if row.get("complete")]
    summary: dict[str, object] = {
        "n_panels": len(rows),
        "n_complete_panels": len(complete),
        "n_below_null_p2_5": sum(bool(row.get("below_null_p2_5")) for row in complete),
    }
    for key in ("N_eff", "mean_abs_corr", "same_side_overlap", "same_persona_cross_vendor_corr"):
        values = np.asarray(
            [row[key] for row in complete if key in row and np.isfinite(float(row[key]))], dtype=float
        )
        if values.size:
            summary[key] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return summary


def analyze_repeated(snapshots: list[dict]) -> dict:
    records = list(_load_records(REPEATED_CACHE).values())
    null_cache: dict[int, dict] = {}
    panels = []
    for snapshot in snapshots:
        for replicate in range(REPEATED_DRAWS):
            agents = [row for row in records if row["date"] == snapshot["date"] and int(row["replicate"]) == replicate]
            metrics = _panel_metrics(agents, snapshot["syms"], null_cache)
            panels.append({"date": snapshot["date"], "regime": snapshot["regime"], "replicate": replicate, **metrics})
    by_date = {}
    for snapshot in snapshots:
        by_date[snapshot["date"]] = _summarize_rows([row for row in panels if row["date"] == snapshot["date"]])
        by_date[snapshot["date"]]["regime"] = snapshot["regime"]
    usage_rows = [row.get("usage", {}) for row in records if row.get("parse_ok")]
    result = {
        "schema_version": "cohort-equity-locked-v1",
        "prompt_spec_version": EQUITY_PROMPT_SPEC_VERSION,
        "design": {
            "dates": [row["date"] for row in snapshots],
            "replicates_per_date": REPEATED_DRAWS,
            "temperature": REPEATED_TEMPERATURE,
            "vendors": [vendor for vendor, _ in OPENROUTER_MODELS],
            "models": [model for _, model in OPENROUTER_MODELS],
            "personas": list(PERSONAS),
            "assets_per_panel": 14,
        },
        "collection": {
            "n_records": len(records),
            "n_parse_ok": sum(bool(row.get("parse_ok")) for row in records),
            "n_parse_failures": sum(not row.get("parse_ok") for row in records),
            "prompt_tokens": int(sum(row.get("prompt_tokens", 0) for row in usage_rows)),
            "completion_tokens": int(sum(row.get("completion_tokens", 0) for row in usage_rows)),
        },
        "panels": panels,
        "by_date": by_date,
        "overall": _summarize_rows(panels),
    }
    _atomic_json(RESULT_PATH, result)
    return result


def analyze_temperature(snapshots: list[dict]) -> dict:
    current = next(row for row in snapshots if row["date"] == "2026-05-31")
    repeated = list(_load_records(REPEATED_CACHE).values())
    grid = list(_load_records(TEMPERATURE_CACHE).values())
    null_cache: dict[int, dict] = {}
    rows = []
    for temperature in (0.2, 0.7, 1.0):
        source = repeated if temperature == 0.7 else grid
        agents = [
            row for row in source
            if row["date"] == current["date"] and int(row["replicate"]) == 0
            and math.isclose(float(row["temperature"]), temperature)
        ]
        rows.append({"temperature": temperature, **_panel_metrics(agents, current["syms"], null_cache)})
    complete = [row for row in rows if row.get("complete")]
    result = {
        "schema_version": "cohort-equity-temperature-grid-v1",
        "prompt_spec_version": EQUITY_PROMPT_SPEC_VERSION,
        "date": current["date"],
        "baseline_replicate": 0,
        "rows": rows,
        "summary": {
            "n_complete_temperatures": len(complete),
            "N_eff_range": [float(min(row["N_eff"] for row in complete)), float(max(row["N_eff"] for row in complete))] if complete else [],
            "same_side_overlap_range": [float(min(row["same_side_overlap"] for row in complete)), float(max(row["same_side_overlap"] for row in complete))] if complete else [],
        },
    }
    _atomic_json(TEMPERATURE_RESULT_PATH, result)
    return result


def _print_results(repeated: dict, temperature: dict) -> None:
    overall = repeated["overall"]
    print("\nLocked U.S.-equity repeated-draw audit")
    print(
        f"  complete panels: {overall['n_complete_panels']}/{overall['n_panels']}; "
        f"below matched-null p2.5: {overall['n_below_null_p2_5']}/{overall['n_complete_panels']}"
    )
    if "N_eff" in overall:
        print(
            f"  N_eff={overall['N_eff']['mean']:.3f} +/- {overall['N_eff']['sd']:.3f}; "
            f"O={overall['same_side_overlap']['mean']:.3f} +/- {overall['same_side_overlap']['sd']:.3f}"
        )
    for date, row in repeated["by_date"].items():
        if "N_eff" in row:
            print(
                f"  {date} ({row['regime']}): N_eff={row['N_eff']['mean']:.3f} +/- {row['N_eff']['sd']:.3f}; "
                f"O={row['same_side_overlap']['mean']:.3f}"
            )
    print("\nLocked U.S.-equity temperature grid")
    for row in temperature["rows"]:
        if row.get("complete"):
            print(f"  T={row['temperature']:.1f}: N_eff={row['N_eff']:.3f}; O={row['same_side_overlap']:.3f}")
        else:
            print(f"  T={row['temperature']:.1f}: incomplete M={row.get('M', 0)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-new-calls", type=int)
    parser.add_argument("--skip-repeated", action="store_true")
    parser.add_argument("--skip-temperature", action="store_true")
    args = parser.parse_args()

    snapshots = load_snapshots()
    repeated_jobs = _jobs(snapshots, "repeated_draws")
    temperature_jobs = _jobs(snapshots, "temperature_grid")
    if args.dry_run:
        print(f"repeated draws: {len(repeated_jobs)} calls")
        print(f"temperature grid additions: {len(temperature_jobs)} calls")
        return
    if not args.analyze_only and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required for collection")
    if not args.analyze_only:
        if not args.skip_repeated:
            collect(repeated_jobs, REPEATED_CACHE, args.max_workers, args.max_new_calls)
        if not args.skip_temperature:
            collect(temperature_jobs, TEMPERATURE_CACHE, args.max_workers, args.max_new_calls)
    repeated = analyze_repeated(snapshots)
    temperature = analyze_temperature(snapshots)
    _print_results(repeated, temperature)


if __name__ == "__main__":
    main()
