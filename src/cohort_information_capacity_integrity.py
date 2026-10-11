"""Collect a clean K=7 integrity rerun of the information-capacity experiment.

The rerun addresses four audit findings at once:

* complete SEC submissions history in the input packets;
* one prompt template for shared, mixed, and coherent access;
* identical-request retries with every attempt recorded; and
* generation IDs, routing metadata, model-catalog metadata, and seeds where supported.

The original caches remain untouched.  The main rerun has 3,570 calls
(34 dates x 7 models x [1 shared + 7 mixed + 7 coherent packets]).  A separate
238-call sensitivity masks company identity for the macro-only coherent seat.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from _paths import DATA, RESULTS
from cohort_crossvendor import OPENROUTER_URL, _extract_json, _post
from cohort_equity_locked import _clean_weights
from cohort_information_capacity_experiment import (
    COHERENT_GROUPS,
    FIELD_BY_ID,
    FINE_GROUPS,
    PILOT_MODELS,
    TEMPERATURE,
    _mixed_partition,
    _packet_text,
)
from cohort_information_diversity import _atomic_json, _credit_exhausted, _usage
from cohort_information_capacity_integrity_data import (
    CONFIRMATORY_OUTPUT,
    PILOT_OUTPUT,
)


PLAN_PATH = Path(RESULTS) / "information_capacity_integrity_plan.json"
CATALOG_PATH = Path(RESULTS) / "information_capacity_integrity_model_catalog.json"
CACHE_PATH = Path(DATA) / "cohort_agents_information_capacity_integrity.json"
MASKED_CACHE_PATH = Path(DATA) / "cohort_agents_information_capacity_integrity_macro_masked.json"
PREDECESSOR_PLAN_PATH = Path(RESULTS) / "information_capacity_experiment_plan.json"

SCHEMA_VERSION = "information-capacity-integrity-plan-v1"
PROMPT_SPEC_VERSION = "information-capacity-neutral-packet-v3-unified-object"
REQUEST_SPEC_VERSION = "openrouter-json-schema-audited-v4-portable-object"
INTEGRITY_CORRECTION_ID = "2026-07-21-sec-history-unified-prompt-v1"
TEAM_SIZE = 7
BASE_SEED = 2026072100
DEFAULT_MAX_WORKERS = int(os.environ.get("INFORMATION_CAPACITY_MAX_WORKERS", "16"))
MAX_ATTEMPTS = 5


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_packets() -> list[dict]:
    rows = []
    for sample, path in (("development", PILOT_OUTPUT), ("complement", CONFIRMATORY_OUTPUT)):
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; build integrity packets first")
        packet = json.loads(path.read_text(encoding="utf-8"))
        if packet.get("schema_version") != "information-diversity-packets-v3-sec-history-corrected":
            raise ValueError(f"unexpected packet schema in {path}: {packet.get('schema_version')}")
        rows.extend({"sample": sample, "snapshot": row} for row in packet["snapshots"])
    if len(rows) != 34 or len({row["snapshot"]["date"] for row in rows}) != 34:
        raise ValueError("integrity rerun requires 34 unique U.S. month-ends")
    return sorted(rows, key=lambda row: row["snapshot"]["date"])


def _model_catalog() -> dict:
    if CATALOG_PATH.exists():
        return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Accept": "application/json", "User-Agent": "Information-access integrity rerun"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    by_id = {row["id"]: row for row in payload["data"]}
    models = []
    for vendor, model in PILOT_MODELS:
        row = by_id.get(model)
        if row is None:
            raise ValueError(f"OpenRouter catalog does not contain {model}")
        models.append({
            "vendor": vendor,
            "requested_model": model,
            "canonical_slug": row.get("canonical_slug"),
            "created": row.get("created"),
            "supported_parameters": sorted(row.get("supported_parameters") or []),
            "pricing": row.get("pricing"),
        })
    result = {
        "schema_version": "openrouter-model-catalog-snapshot-v1",
        "retrieved_at_utc": _utc_now(),
        "source": "https://openrouter.ai/api/v1/models",
        "models": models,
    }
    _atomic_json(CATALOG_PATH, result)
    return result


def _common_core_masked(symbols: list[str]) -> tuple[str, dict[str, str]]:
    mapping = {f"ASSET_{index:02d}": symbol for index, symbol in enumerate(symbols, start=1)}
    text = "ticker\n" + "\n".join(mapping)
    return text, mapping


def _response_format(symbols: list[str]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "active_book",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "weights": {
                        "type": "object",
                        "properties": {
                            symbol: {"type": "number"}
                            for symbol in symbols
                        },
                        "required": symbols,
                        "additionalProperties": False,
                    },
                    "thesis": {"type": "string"},
                    "primary_signals": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["weights", "thesis", "primary_signals"],
                "additionalProperties": False,
            },
        },
    }


def _build_integrity_prompt(snapshot: dict, packet_text: str) -> str:
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
        "Reply with one JSON object and no markdown. The weights field must be an object containing every "
        "listed ticker exactly once. Use this schema: "
        f'{{"weights":{{"{example}":0.10,"...":0.00}},'
        '"thesis":"at most 35 words","primary_signals":["signal 1","signal 2"]}}.'
    )


def _request_seed(job: dict) -> int:
    identity = "|".join(str(job[key]) for key in (
        "date", "architecture", "team_size", "packet_index", "vendor",
    ))
    return BASE_SEED + int(_sha256_text(identity)[:7], 16) % 100_000_000


def _base_job(
    *,
    item: dict,
    architecture: str,
    packet_index: int,
    field_ids: list[str],
    vendor: str,
    model: str,
    catalog_by_model: dict[str, dict],
    masked_macro: bool = False,
) -> dict:
    snapshot = item["snapshot"]
    prompt_snapshot = snapshot
    output_symbol_map = None
    if masked_macro:
        core, output_symbol_map = _common_core_masked(snapshot["symbols"])
        prompt_snapshot = {**snapshot, "symbols": list(output_symbol_map), "common_core": core}
    packet_text = _packet_text(snapshot, field_ids)
    prompt = _build_integrity_prompt(prompt_snapshot, packet_text)
    symbols = prompt_snapshot["symbols"]
    response_format = _response_format(symbols)
    supported = set(catalog_by_model[model].get("supported_parameters") or [])
    if not {"response_format", "structured_outputs"}.intersection(supported):
        raise ValueError(f"{model} does not advertise structured output support")
    job = {
        "experiment": "information_capacity_integrity_rerun",
        "integrity_correction_id": INTEGRITY_CORRECTION_ID,
        "sample": item["sample"],
        "snapshot_id": snapshot["snapshot_id"],
        "date": snapshot["date"],
        "return_regime": snapshot["return_regime"],
        "volatility_regime": snapshot["volatility_regime"],
        "condition": (
            "coherent_k7_macro_masked"
            if masked_macro
            else ("shared_full" if architecture == "shared" else f"{architecture}_k7_p{packet_index}")
        ),
        "architecture": "coherent_masked_macro" if masked_macro else architecture,
        "team_size": TEAM_SIZE,
        "packet_index": packet_index,
        "field_ids": field_ids,
        "replicate": 0,
        "temperature": TEMPERATURE,
        "vendor": vendor,
        "requested_model": model,
        "canonical_model_slug": catalog_by_model[model].get("canonical_slug"),
        "symbols": symbols,
        "original_symbols": snapshot["symbols"],
        "output_symbol_map": output_symbol_map,
        "prompt": prompt,
        "prompt_spec_version": PROMPT_SPEC_VERSION,
        "prompt_sha256": _sha256_text(prompt),
        "packet_sha256": _sha256_text(packet_text),
        "response_format": response_format,
        "response_format_sha256": _sha256_text(_stable_json(response_format)),
        "request_spec_version": REQUEST_SPEC_VERSION,
        "request_seed": _request_seed({
            "date": snapshot["date"],
            "architecture": "coherent_masked_macro" if masked_macro else architecture,
            "team_size": TEAM_SIZE,
            "packet_index": packet_index,
            "vendor": vendor,
        }),
        "seed_supported": "seed" in supported,
        "require_complete_symbols": True,
    }
    payload = _payload(job)
    job["request_payload_sha256"] = _sha256_text(_stable_json(payload))
    return job


def _payload(job: dict) -> dict:
    payload: dict[str, Any] = {
        "model": job["requested_model"],
        "temperature": job["temperature"],
        "max_tokens": 1400,
        "messages": [{"role": "user", "content": job["prompt"]}],
        "response_format": job["response_format"],
        "provider": {"require_parameters": True},
    }
    if job.get("seed_supported"):
        payload["seed"] = job["request_seed"]
    return payload


def make_jobs(packet_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    catalog = _model_catalog()
    catalog_by_model = {row["requested_model"]: row for row in catalog["models"]}
    snapshots = [row["snapshot"] for row in packet_rows]
    mixed_fields = _mixed_partition(snapshots, TEAM_SIZE)
    coherent_fields = [list(fields) for _, fields in COHERENT_GROUPS[TEAM_SIZE]]
    all_fields = list(FIELD_BY_ID)
    jobs = []
    masked_jobs = []
    assignments = []
    for item in packet_rows:
        date_assignment = {
            "sample": item["sample"],
            "date": item["snapshot"]["date"],
            "shared_fields": all_fields,
            "mixed_fields": mixed_fields,
            "coherent_fields": coherent_fields,
        }
        assignments.append(date_assignment)
        for vendor, model in PILOT_MODELS:
            jobs.append(_base_job(
                item=item,
                architecture="shared",
                packet_index=0,
                field_ids=all_fields,
                vendor=vendor,
                model=model,
                catalog_by_model=catalog_by_model,
            ))
            for architecture, groups in (("mixed", mixed_fields), ("coherent", coherent_fields)):
                for packet_index, field_ids in enumerate(groups, start=1):
                    jobs.append(_base_job(
                        item=item,
                        architecture=architecture,
                        packet_index=packet_index,
                        field_ids=field_ids,
                        vendor=vendor,
                        model=model,
                        catalog_by_model=catalog_by_model,
                    ))
            masked_jobs.append(_base_job(
                item=item,
                architecture="coherent",
                packet_index=TEAM_SIZE,
                field_ids=list(FINE_GROUPS["macro"]),
                vendor=vendor,
                model=model,
                catalog_by_model=catalog_by_model,
                masked_macro=True,
            ))
    return jobs, masked_jobs, assignments


def _job_key(row: dict) -> tuple:
    return (
        row["date"], row["architecture"], int(row["team_size"]),
        int(row["packet_index"]), row["vendor"],
    )


def _manifest(jobs: list[dict]) -> list[dict]:
    keys = (
        "sample", "snapshot_id", "date", "architecture", "team_size", "packet_index",
        "condition", "vendor", "requested_model", "canonical_model_slug", "field_ids",
        "prompt_sha256", "packet_sha256", "response_format_sha256",
        "request_payload_sha256", "request_seed", "seed_supported",
    )
    return [{key: job[key] for key in keys} for job in jobs]


def write_plan(jobs: list[dict], masked_jobs: list[dict], assignments: list[dict]) -> dict:
    catalog = _model_catalog()
    predecessor = json.loads(PREDECESSOR_PLAN_PATH.read_text(encoding="utf-8"))
    predecessor_record = {
        "path": str(PREDECESSOR_PLAN_PATH.relative_to(Path(DATA).parent)),
        "file_sha256": _sha256_file(PREDECESSOR_PLAN_PATH),
        "assignment_sha256": predecessor.get("assignment_sha256"),
        "job_manifest_sha256": predecessor.get("job_manifest_sha256"),
    }
    manifest = _manifest(jobs)
    masked_manifest = _manifest(masked_jobs)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "frozen_before_integrity_collection": True,
        "integrity_correction_id": INTEGRITY_CORRECTION_ID,
        "corrections": [
            "complete SEC submissions archive pagination",
            "one prompt template for shared, mixed, and coherent access",
            "strict JSON schema with identical-request retries",
            "attempt-level generation metadata and model catalog snapshot",
            "ticker-masked macro-seat sensitivity",
        ],
        "sample_labels": {
            "development": "12 dates used in earlier design development",
            "complement": "22 complementary dates; not claimed as untouched held-out data",
        },
        "design": {
            "n_dates": 34,
            "team_size": TEAM_SIZE,
            "models": [{"vendor": vendor, "model": model} for vendor, model in PILOT_MODELS],
            "architectures": ["shared", "mixed", "coherent"],
            "temperature": TEMPERATURE,
            "prompt_spec_version": PROMPT_SPEC_VERSION,
            "request_spec_version": REQUEST_SPEC_VERSION,
            "retry_policy": "up to five byte-identical request payloads; no augmented prompt",
            "seed_rule": "stable job-specific integer; sent only when model advertises seed support",
            "main_calls": len(jobs),
            "masked_macro_calls": len(masked_jobs),
        },
        "packet_files": [
            {"path": str(path.relative_to(Path(DATA).parent)), "sha256": _sha256_file(path)}
            for path in (PILOT_OUTPUT, CONFIRMATORY_OUTPUT)
        ],
        "model_catalog": {
            "path": str(CATALOG_PATH.relative_to(Path(DATA).parent)),
            "sha256": _sha256_file(CATALOG_PATH),
            "retrieved_at_utc": catalog["retrieved_at_utc"],
        },
        "assignment_sha256": _sha256_text(_stable_json(assignments)),
        "job_manifest_sha256": _sha256_text(_stable_json(manifest)),
        "masked_job_manifest_sha256": _sha256_text(_stable_json(masked_manifest)),
        "lineage": [predecessor_record],
        "assignments": assignments,
        "job_manifest": manifest,
        "masked_job_manifest": masked_manifest,
    }
    if PLAN_PATH.exists():
        previous = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        existing = previous.get("lineage", [])
        same_design = all(
            previous.get(key) == plan.get(key)
            for key in (
                "assignment_sha256",
                "job_manifest_sha256",
                "masked_job_manifest_sha256",
            )
        )
        if same_design:
            plan["lineage"] = existing or [predecessor_record]
        else:
            previous_record = {
                "path": str(PLAN_PATH.relative_to(Path(DATA).parent)),
                "file_sha256": _sha256_file(PLAN_PATH),
                "assignment_sha256": previous.get("assignment_sha256"),
                "job_manifest_sha256": previous.get("job_manifest_sha256"),
                "masked_job_manifest_sha256": previous.get("masked_job_manifest_sha256"),
            }
            plan["lineage"] = [*existing, previous_record]
    _atomic_json(PLAN_PATH, plan)
    return plan


def _attempt_record(response: dict, *, index: int, text: str | None = None) -> dict:
    choice = (response.get("choices") or [{}])[0]
    return {
        "attempt": index,
        "completed_at_utc": _utc_now(),
        "response_id": response.get("id"),
        "response_model": response.get("model"),
        "response_provider": response.get("provider"),
        "response_created": response.get("created"),
        "system_fingerprint": response.get("system_fingerprint"),
        "finish_reason": choice.get("finish_reason"),
        "native_finish_reason": choice.get("native_finish_reason"),
        "raw_response_sha256": _sha256_text(text) if text is not None else None,
    }


def call_openrouter(job: dict) -> dict:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://icaif-cohort.local",
        "X-Title": "information-access-integrity",
    }
    payload = _payload(job)
    if _sha256_text(_stable_json(payload)) != job["request_payload_sha256"]:
        raise AssertionError("request payload changed after plan freeze")
    attempts = []
    last_error = "unknown error"
    for attempt_index in range(1, MAX_ATTEMPTS + 1):
        response = None
        response_text = None
        try:
            response = _post(OPENROUTER_URL, payload, headers)
            response_text = response["choices"][0]["message"]["content"]
            obj = _extract_json(response_text)
            if isinstance(obj.get("weights"), dict):
                obj = {
                    **obj,
                    "weights": [
                        {"symbol": symbol, "weight": weight}
                        for symbol, weight in obj["weights"].items()
                    ],
                }
            weights, diagnostics = _clean_weights(obj, job["symbols"])
            format_issues = any(diagnostics.get(key) for key in (
                "missing_symbols_zero_filled",
                "duplicate_symbols_last_value_used",
                "unknown_symbols_dropped",
            ))
            raw_values = np.asarray([row["weight"] for row in weights], dtype=float)
            if format_issues:
                raise ValueError(f"strict book constraint failed: {diagnostics}")
            if not math.isfinite(float(np.abs(raw_values).sum())):
                raise ValueError("non-finite raw book")
            mandate_metadata = {}
            if job.get("portfolio_mandate") == "long_only":
                raw_sum = float(raw_values.sum())
                if float(raw_values.min()) < -1e-10 or raw_sum <= 0.0:
                    raise ValueError("response is not a positive long-only portfolio")
                if float(raw_values.max()) > 1.0 + 1e-10:
                    raise ValueError("long-only weight exceeds 1")
                normalized = np.maximum(raw_values, 0.0) / raw_sum
                benchmark = 1.0 / len(normalized)
                active_share = 0.5 * float(np.abs(normalized - benchmark).sum())
                if active_share <= 1e-6:
                    raise ValueError("response reproduces the equal-weight benchmark")
                weights = [
                    {"symbol": symbol, "weight": float(value)}
                    for symbol, value in zip(job["symbols"], normalized)
                ]
                mandate_metadata = {
                    "portfolio_mandate": "long_only",
                    "long_only_valid": True,
                    "raw_weight_sum": raw_sum,
                    "normalized_weight_sum": float(normalized.sum()),
                    "normalization_l1_adjustment": float(
                        np.abs(normalized - raw_values).sum()
                    ),
                    "active_share": active_share,
                    "max_portfolio_weight": float(normalized.max()),
                }
            else:
                if float(np.abs(raw_values).max()) > 1.0 + 1e-12:
                    raise ValueError("conviction score falls outside [-1, 1]")
                if float(np.abs(raw_values - raw_values.mean()).sum()) <= 1e-12:
                    raise ValueError("response does not express a nonzero active book")
            attempt_row = _attempt_record(
                response, index=attempt_index, text=response_text
            )
            attempt_row.update({"parse_ok": True, "diagnostics": diagnostics})
            attempts.append(attempt_row)

            output_weights = weights
            masked_weights = None
            if job.get("output_symbol_map"):
                masked_weights = weights
                output_weights = [
                    {
                        "symbol": job["output_symbol_map"][row["symbol"]],
                        "weight": row["weight"],
                    }
                    for row in weights
                ]
            choice = response["choices"][0]
            return {
                **{key: job[key] for key in (
                    "experiment", "integrity_correction_id", "sample", "snapshot_id", "date",
                    "return_regime", "volatility_regime", "condition", "architecture",
                    "team_size", "packet_index", "field_ids", "replicate", "temperature",
                    "vendor", "requested_model", "canonical_model_slug", "prompt_spec_version",
                    "prompt_sha256", "packet_sha256", "response_format_sha256",
                    "request_spec_version", "request_payload_sha256", "request_seed",
                    "seed_supported",
                )},
                "model": job["requested_model"],
                "response_id": response.get("id"),
                "response_model": response.get("model"),
                "response_provider": response.get("provider"),
                "response_created": response.get("created"),
                "system_fingerprint": response.get("system_fingerprint"),
                "finish_reason": choice.get("finish_reason"),
                "native_finish_reason": choice.get("native_finish_reason"),
                "collected_at_utc": _utc_now(),
                "weights": output_weights,
                "masked_weights": masked_weights,
                "thesis": obj.get("thesis", ""),
                "primary_signals": obj.get("primary_signals", []),
                "parse_ok": True,
                "format_retry_count": attempt_index - 1,
                "request_attempts": attempts,
                "raw_net": float(raw_values.sum()),
                "raw_gross": float(np.abs(raw_values).sum()),
                "raw_max_abs_weight": float(np.abs(raw_values).max()),
                "usage": _usage(response),
                **mandate_metadata,
                **diagnostics,
            }
        except Exception as exc:  # pragma: no cover - network-dependent
            last_error = f"{type(exc).__name__}: {exc}"[:800]
            if response is not None:
                attempt_row = _attempt_record(
                    response, index=attempt_index, text=response_text
                )
                attempt_row.update({
                    "parse_ok": False,
                    "error": last_error,
                    "raw_response_text": response_text,
                })
            else:
                attempt_row = {
                    "attempt": attempt_index,
                    "completed_at_utc": _utc_now(),
                    "parse_ok": False,
                    "error": last_error,
                }
            attempts.append(attempt_row)
            if _credit_exhausted(exc):
                break
            if attempt_index < MAX_ATTEMPTS:
                time.sleep((2 ** (attempt_index - 1)) + random.random())
    return {
        **{key: job[key] for key in (
            "experiment", "integrity_correction_id", "sample", "snapshot_id", "date",
            "return_regime", "volatility_regime", "condition", "architecture",
            "team_size", "packet_index", "field_ids", "replicate", "temperature",
            "vendor", "requested_model", "canonical_model_slug", "prompt_spec_version",
            "prompt_sha256", "packet_sha256", "response_format_sha256",
            "request_spec_version", "request_payload_sha256", "request_seed", "seed_supported",
        )},
        "model": job["requested_model"],
        "collected_at_utc": _utc_now(),
        "parse_ok": False,
        "format_retry_count": max(0, len(attempts) - 1),
        "request_attempts": attempts,
        "error": last_error,
    }


def _complete(row: dict, job: dict) -> bool:
    if not row.get("parse_ok") or not row.get("response_id"):
        return False
    if any(row.get(key) != job.get(key) for key in (
        "integrity_correction_id", "prompt_sha256", "packet_sha256",
        "response_format_sha256", "request_payload_sha256", "requested_model",
    )):
        return False
    weights = row.get("weights") or []
    if len(weights) != len(job["original_symbols"]):
        return False
    if job.get("portfolio_mandate") == "long_only":
        return bool(
            row.get("portfolio_mandate") == "long_only"
            and row.get("long_only_valid")
            and abs(float(row.get("normalized_weight_sum", 0.0)) - 1.0) <= 1e-10
        )
    return True


def collect(jobs: list[dict], cache_path: Path, max_workers: int, max_new_calls: int | None) -> None:
    rows = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else []
    records = {_job_key(row): row for row in rows}
    pending = [job for job in jobs if not _complete(records.get(_job_key(job), {}), job)]
    if max_new_calls is not None:
        pending = pending[:max_new_calls]
    print(f"integrity collection {cache_path.name}: {len(jobs)} planned, {len(pending)} pending")
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
                    print(f"  {completed}/{len(pending)} completed ({failures} failures)", flush=True)
            if completed % max(25, max_workers * 5) < len(finished) or completed == len(pending):
                _atomic_json(cache_path, sorted(records.values(), key=_job_key))
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
    _atomic_json(cache_path, sorted(records.values(), key=_job_key))
    if stop_for_credit:
        raise RuntimeError("OpenRouter credit or key limit reached; cache is resumable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--masked-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-new-calls", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet_rows = _load_packets()
    jobs, masked_jobs, assignments = make_jobs(packet_rows)
    plan = write_plan(jobs, masked_jobs, assignments)
    print(
        f"frozen {len(jobs)} main and {len(masked_jobs)} masked-macro calls; "
        f"manifest={plan['job_manifest_sha256'][:12]}"
    )
    if args.freeze_only:
        return
    if not args.masked_only:
        collect(jobs, CACHE_PATH, args.max_workers, args.max_new_calls)
    collect(masked_jobs, MASKED_CACHE_PATH, args.max_workers, args.max_new_calls)


if __name__ == "__main__":
    main()
