"""Validate point-in-time inputs and prompt/cache lineage for ICAIF-v4."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from _paths import DATA, RESULTS
import cohort_information_capacity_crossmarket_experiment as crossmarket
import cohort_information_capacity_integrity as integrity
from cohort_information_capacity_audited_architecture import (
    CACHE_PATH as ENDPOINT_CACHE,
    make_jobs as make_endpoint_jobs,
)
from cohort_information_capacity_audited_curve import (
    CACHE_PATH as CURVE_CACHE,
    make_jobs as make_curve_jobs,
)
from cohort_information_capacity_audited_longonly import (
    CACHE_PATH as LONGONLY_CACHE,
    make_jobs as make_longonly_jobs,
)
from pit_correction import PIT_CORRECTION_ID


OUTPUT_PATH = Path(RESULTS) / "icaif_v4_pit_validation.json"
PRIMARY_PACKET_PATHS = (
    Path(DATA) / "information_diversity_integrity_packets.json",
    Path(DATA) / "information_diversity_integrity_confirmatory_packets.json",
)
CROSSMARKET_PACKET_PATHS = {
    market: Path(DATA) / f"information_capacity_{market}_packets.json"
    for market in ("us", "europe", "crypto")
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "data":
        return Path(DATA).parent / path
    return Path(DATA) / path


def _primary_packet_checks() -> dict:
    issues: list[str] = []
    snapshots = []
    schemas = set()
    for path in PRIMARY_PACKET_PATHS:
        packet = _load(path)
        schemas.add(packet.get("schema_version"))
        snapshots.extend(packet.get("snapshots", []))
        sources = packet.get("sources", {})
        for label in ("return_prices", "valuation_closes", "corporate_actions", "volumes"):
            source = sources.get(label, {})
            value = source.get("path") or source.get("file")
            if not value:
                issues.append(f"{path.name}: missing source record {label}")
                continue
            source_path = _source_path(value)
            if not source_path.exists():
                issues.append(f"{path.name}: missing source file {source_path}")
            elif source.get("sha256") != _sha256(source_path):
                issues.append(f"{path.name}: changed source hash for {label}")
        sec = sources.get("sec", {})
        for symbol, record in sec.items():
            if int(record.get("complete_submission_rows", 0)) <= 0:
                issues.append(f"{path.name}: {symbol} lacks complete SEC history")
            for history in record.get("submission_history_files", []):
                if not history.get("sha256") or int(history.get("filing_count", 0)) <= 0:
                    issues.append(f"{path.name}: incomplete SEC history metadata for {symbol}")

    dates = [row.get("date") for row in snapshots]
    if len(dates) != 34 or len(set(dates)) != 34:
        issues.append(f"expected 34 unique U.S. dates, found {len(dates)}/{len(set(dates))}")
    expected_schema = {"information-diversity-packets-v3-sec-history-corrected"}
    if schemas != expected_schema:
        issues.append(f"unexpected primary packet schemas: {sorted(schemas)}")

    filing_ages = []
    for snapshot in snapshots:
        cutoff = pd.Timestamp(snapshot["date"])
        if len(snapshot.get("symbols", [])) != 14:
            issues.append(f"{snapshot.get('date')}: primary universe is not 14 assets")
        for symbol, values in snapshot.get("features", {}).get("fundamentals", {}).items():
            for field in ("accounting_filed", "accounting_period_end", "shares_filed", "shares_period_end"):
                value = values.get(field)
                if value and pd.Timestamp(value) > cutoff:
                    issues.append(f"{snapshot['date']} {symbol}: future {field}={value}")
            if values.get("shares_filed"):
                filing_ages.append((cutoff - pd.Timestamp(values["shares_filed"])).days)
            split = float(values.get("shares_split_adjustment") or 0.0)
            as_filed = values.get("shares_as_filed")
            on_basis = values.get("shares_on_price_basis")
            if split <= 0 or as_filed is None or on_basis is None:
                issues.append(f"{snapshot['date']} {symbol}: incomplete share-basis record")
            elif not math.isclose(float(as_filed) * split, float(on_basis), rel_tol=1e-12, abs_tol=1e-3):
                issues.append(f"{snapshot['date']} {symbol}: share-basis arithmetic mismatch")
        for symbol, values in snapshot.get("features", {}).get("events", {}).items():
            for field in ("eight_k_30d", "eight_k_90d", "earnings_release_90d", "material_event_90d",
                          "days_since_periodic_filing"):
                value = values.get(field)
                if value is None or float(value) < 0:
                    issues.append(f"{snapshot['date']} {symbol}: invalid event field {field}")
            days_since_8k = values.get("days_since_latest_8k")
            no_recent_8k = (
                values.get("eight_k_90d") == 0
                and values.get("latest_8k_items") == "none in prior 90d"
            )
            if days_since_8k is None:
                if not no_recent_8k:
                    issues.append(f"{snapshot['date']} {symbol}: unexplained missing days_since_latest_8k")
            elif float(days_since_8k) < 0:
                issues.append(f"{snapshot['date']} {symbol}: invalid event field days_since_latest_8k")

    return {
        "schema_versions": sorted(schemas),
        "n_snapshots": len(snapshots),
        "n_unique_dates": len(set(dates)),
        "maximum_share_filing_age_days": max(filing_ages) if filing_ages else None,
        "issues": issues,
        "passed": not issues,
    }


def _crossmarket_packet_checks() -> dict:
    issues: list[str] = []
    date_sets = {}
    field_sets = {}
    for market, path in CROSSMARKET_PACKET_PATHS.items():
        packet = _load(path)
        if packet.get("schema_version") != "information-capacity-crossmarket-packets-v2-pit-corrected":
            issues.append(f"{market}: unexpected packet schema")
        snapshots = packet.get("snapshots", [])
        date_sets[market] = tuple(row["date"] for row in snapshots)
        field_sets[market] = tuple(spec["id"] for spec in packet["design"]["field_specs"])
        if len(snapshots) != 12:
            issues.append(f"{market}: expected 12 snapshots, found {len(snapshots)}")
        for snapshot in snapshots:
            if pd.Timestamp(snapshot["observed_through"]) > pd.Timestamp(snapshot["date"]):
                issues.append(f"{market} {snapshot['date']}: future observation cutoff")
        for label, source in packet.get("sources", {}).items():
            value = source.get("path") or source.get("file")
            if not value:
                issues.append(f"{market}: missing source record {label}")
                continue
            source_path = _source_path(value)
            if not source_path.exists() or source.get("sha256") != _sha256(source_path):
                issues.append(f"{market}: missing or changed source {label}")
    if len(set(date_sets.values())) != 1:
        issues.append("cross-market dates are not identical")
    if len(set(field_sets.values())) != 1:
        issues.append("cross-market field definitions are not identical")
    return {
        "dates_per_market": {market: len(values) for market, values in date_sets.items()},
        "fields_per_market": {market: len(values) for market, values in field_sets.items()},
        "issues": issues,
        "passed": not issues,
    }


def _canonical_cache_check(label: str, jobs: list[dict], cache_path: Path) -> dict:
    rows = _load(cache_path)
    expected = {integrity._job_key(job): job for job in jobs}
    relevant = {
        integrity._job_key(row): row
        for row in rows if integrity._job_key(row) in expected
    }
    issues: list[str] = []
    if len(expected) != len(jobs):
        issues.append("duplicate planned job key")
    if set(relevant) != set(expected):
        issues.append(f"cache keys differ: expected {len(expected)}, found {len(relevant)}")
    response_ids = []
    for key, job in expected.items():
        prompt = str(job.get("prompt", ""))
        if str(job.get("date")) in prompt or "forward_return_21d" in prompt:
            issues.append(f"{key}: prompt exposes date or forward return")
        row = relevant.get(key)
        if row is None:
            continue
        if not integrity._complete(row, job):
            issues.append(f"{key}: incomplete or stale response")
            continue
        for field in ("prompt_sha256", "packet_sha256", "response_format_sha256",
                      "request_payload_sha256", "request_seed", "requested_model"):
            if row.get(field) != job.get(field):
                issues.append(f"{key}: changed {field}")
        if row.get("integrity_correction_id") != integrity.INTEGRITY_CORRECTION_ID:
            issues.append(f"{key}: stale correction wave")
        if not row.get("response_id") or not row.get("response_provider"):
            issues.append(f"{key}: missing generation or routing metadata")
        else:
            response_ids.append(row["response_id"])
        attempts = row.get("request_attempts", [])
        if int(row.get("format_retry_count", 0)) != max(0, len(attempts) - 1):
            issues.append(f"{key}: attempt history mismatch")
    if len(response_ids) != len(set(response_ids)):
        issues.append("generation IDs are duplicated")
    return {
        "label": label,
        "planned": len(jobs),
        "matched": len(relevant),
        "unique_generation_ids": len(set(response_ids)),
        "issues": issues[:100],
        "passed": not issues,
    }


def _crossmarket_cache_checks() -> list[dict]:
    output = []
    for market in ("us", "europe", "crypto"):
        packet = crossmarket._load_packet(market)
        design = crossmarket.make_design(market, packet["snapshots"])
        jobs = crossmarket.make_jobs(market, packet["snapshots"], design)
        _, _, cache_path = crossmarket.paths(market)
        rows = _load(cache_path)
        expected = {crossmarket._key(job): job for job in jobs}
        observed = {crossmarket._key(row): row for row in rows}
        issues = []
        if len(rows) != len(observed) or set(observed) != set(expected):
            issues.append("cache does not match the frozen job set")
        for key, job in expected.items():
            prompt = str(job.get("prompt", ""))
            row = observed.get(key)
            if str(job.get("date")) in prompt or "forward_return_21d" in prompt:
                issues.append(f"{key}: prompt exposes date or forward return")
            if row is None:
                continue
            if not crossmarket.usable_response(row):
                issues.append(f"{key}: unusable response")
            for field in ("prompt_sha256", "packet_sha256", "requested_model"):
                if row.get(field) != job.get(field):
                    issues.append(f"{key}: changed {field}")
            if row.get("pit_correction_id") != PIT_CORRECTION_ID:
                issues.append(f"{key}: stale PIT correction")
        output.append({
            "label": f"cross-market {market}",
            "planned": len(jobs),
            "matched": len(observed),
            "issues": issues[:100],
            "passed": not issues,
        })
    return output


def _prompt_cache_checks() -> list[dict]:
    packet_rows = integrity._load_packets()
    all_integrity_jobs, _masked, _assignment = integrity.make_jobs(packet_rows)
    shared_jobs = [job for job in all_integrity_jobs if job["architecture"] == "shared"]
    endpoint_jobs, _ = make_endpoint_jobs()
    curve_jobs, _ = make_curve_jobs()
    longonly_jobs, _ = make_longonly_jobs()
    return [
        _canonical_cache_check("audited shared", shared_jobs, integrity.CACHE_PATH),
        _canonical_cache_check("audited K=7 split", endpoint_jobs, ENDPOINT_CACHE),
        _canonical_cache_check("audited K=2..6 curve", curve_jobs, CURVE_CACHE),
        _canonical_cache_check("audited long-only", longonly_jobs, LONGONLY_CACHE),
        *_crossmarket_cache_checks(),
    ]


def _window_checks() -> dict:
    result = _load(Path(RESULTS) / "information_capacity_audited_longonly_results.json")
    windows = result["operating_performance"]["evaluation_windows"]
    issues = []
    for window in windows:
        if window["start_session"] <= window["decision_date"]:
            issues.append(f"same-close entry: {window}")
        if window["end_session"] < window["start_session"]:
            issues.append(f"negative evaluation window: {window}")
    for left, right in zip(windows, windows[1:]):
        if left["end_session"] >= right["start_session"]:
            issues.append(f"overlapping windows: {left} / {right}")
    return {
        "n_windows": len(windows),
        "first": windows[0] if windows else None,
        "last": windows[-1] if windows else None,
        "issues": issues,
        "passed": len(windows) == 34 and not issues,
    }


def main() -> None:
    primary = _primary_packet_checks()
    crossmarket_packets = _crossmarket_packet_checks()
    caches = _prompt_cache_checks()
    windows = _window_checks()
    result = {
        "schema_version": "icaif-v4-post-audit-pit-validation-v2",
        "integrity_correction_id": integrity.INTEGRITY_CORRECTION_ID,
        "primary_packet_checks": primary,
        "crossmarket_packet_checks": crossmarket_packets,
        "prompt_cache_checks": caches,
        "prompt_records_checked": sum(row["planned"] for row in caches),
        "evaluation_window_checks": windows,
        "scope": (
            "historical packets are point-in-time; causal access contrasts are conditional "
            "on one current model-vintage window and fixed universes"
        ),
    }
    result["passed"] = (
        primary["passed"]
        and crossmarket_packets["passed"]
        and windows["passed"]
        and all(row["passed"] for row in caches)
    )
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        failed = [row["label"] for row in caches if not row["passed"]]
        raise SystemExit(
            f"PIT validation failed: primary={primary['passed']} "
            f"crossmarket={crossmarket_packets['passed']} windows={windows['passed']} "
            f"caches={failed}"
        )
    print(
        f"PIT validation PASS: {primary['n_snapshots']} primary snapshots, "
        f"{result['prompt_records_checked']} prompt records, {windows['n_windows']} windows"
    )


if __name__ == "__main__":
    main()
