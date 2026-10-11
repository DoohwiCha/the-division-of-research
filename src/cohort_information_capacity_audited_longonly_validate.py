"""Validate the audit-corrected K=7 long-only collection and result artifact."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from _paths import RESULTS
from cohort_information_capacity_audited_longonly import (
    CACHE_PATH,
    PLAN_PATH,
    _manifest,
    make_jobs,
)
from cohort_information_capacity_audited_longonly_analysis import RESULT_PATH
from cohort_information_capacity_integrity import _load_packets, _sha256_text, _stable_json
from cohort_information_diversity import PILOT_MODELS, _atomic_json
from cohort_information_diversity_scaling import IMPACT_DELTA


VALIDATION_PATH = Path(RESULTS) / "information_capacity_audited_longonly_validation.json"
SPOT_DATES = ("2023-07-31", "2024-11-30", "2026-04-30")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _portfolio(row: dict, symbols: list[str]) -> np.ndarray:
    values = {item["symbol"]: float(item["weight"]) for item in row["weights"]}
    portfolio = np.asarray([values[symbol] for symbol in symbols], dtype=float)
    return portfolio / portfolio.sum()


def _active(portfolio: np.ndarray) -> np.ndarray:
    values = portfolio - 1.0 / len(portfolio)
    return values / np.abs(values).sum()


def _raw_metrics(rows: list[dict], symbols: list[str], adv: np.ndarray) -> dict:
    holdings = np.column_stack([_portfolio(row, symbols) for row in rows])
    matrix = np.column_stack([
        _active(holdings[:, index]) for index in range(holdings.shape[1])
    ])
    corr = np.corrcoef(matrix.T)
    neff = float(matrix.shape[1] ** 2 / np.square(corr).sum())
    overlaps = []
    holding_overlaps = []
    for left, right in itertools.combinations(range(matrix.shape[1]), 2):
        x, y = matrix[:, left], matrix[:, right]
        same = np.sign(x) == np.sign(y)
        overlaps.append(float(np.minimum(np.abs(x), np.abs(y))[same].sum()))
        holding_overlaps.append(
            float(np.minimum(holdings[:, left], holdings[:, right]).sum())
        )
    scale = np.power(adv / np.median(adv), IMPACT_DELTA)
    aggregate = matrix.sum(axis=1)
    numerator = np.sum(np.power(np.abs(aggregate), 1.0 + IMPACT_DELTA) / scale)
    denominator = np.sum(
        np.power(np.abs(matrix), 1.0 + IMPACT_DELTA) / scale[:, None]
    )
    team = holdings.mean(axis=1)
    return {
        "N_eff": neff,
        "same_side_overlap": float(np.mean(overlaps)),
        "holdings_overlap": float(np.mean(holding_overlaps)),
        "convex_order_concentration": float(numerator / denominator),
        "team_hhi": float(np.square(team).sum()),
    }


def _spot_recompute(rows: list[dict], result: dict) -> list[dict]:
    packets = {row["snapshot"]["date"]: row["snapshot"] for row in _load_packets()}
    result_by_date = {row["date"]: row for row in result["per_date"]}
    lookup = {
        (row["date"], row["vendor"], row["architecture"], int(row["packet_index"])): row
        for row in rows
    }
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    assignments = list(itertools.permutations(range(1, 8)))
    output = []
    for date in SPOT_DATES:
        snapshot = packets[date]
        symbols = snapshot["symbols"]
        adv = np.asarray([snapshot["features"]["market"][symbol]["adv_bn"] for symbol in symbols])
        configurations = {
            "shared": [[lookup[date, vendor, "shared", 0] for vendor in vendors]],
            "mixed": [
                [
                    lookup[date, vendor, "mixed_audited", packet]
                    for vendor, packet in zip(vendors, assignment)
                ]
                for assignment in assignments
            ],
            "coherent": [
                [
                    lookup[date, vendor, "coherent_audited", packet]
                    for vendor, packet in zip(vendors, assignment)
                ]
                for assignment in assignments
            ],
        }
        for design, groups in configurations.items():
            metrics = [_raw_metrics(group, symbols, adv) for group in groups]
            recomputed = {
                metric: float(np.mean([row[metric] for row in metrics]))
                for metric in metrics[0]
            }
            stored = result_by_date[date][design]
            max_error = max(abs(recomputed[key] - stored[key]) for key in recomputed)
            if max_error > 1e-11:
                raise AssertionError(f"spot recomputation failed: {date} {design} {max_error}")
            output.append({"date": date, "design": design, "max_abs_error": max_error})
    return output


def main() -> None:
    failures = []
    jobs, _assignment = make_jobs()
    plan = _load(PLAN_PATH)
    rows = _load(CACHE_PATH)
    result = _load(RESULT_PATH)
    if "predecessor_K7_levels" in result:
        failures.append("result exposes a superseded numeric panel")
    manifest_hash = _sha256_text(_stable_json(_manifest(jobs)))
    if manifest_hash != plan["job_manifest_sha256"]:
        failures.append("job manifest hash mismatch")
    if len(rows) != 3570:
        failures.append(f"row count is {len(rows)}, expected 3570")
    ids = [row.get("response_id") for row in rows]
    if len(set(ids)) != len(rows) or any(value is None for value in ids):
        failures.append("generation IDs are missing or duplicated")
    for row in rows:
        weights = np.asarray([float(item["weight"]) for item in row.get("weights", [])])
        if (
            not row.get("parse_ok")
            or not row.get("long_only_valid")
            or len(weights) != 14
            or float(weights.min(initial=0.0)) < -1e-10
            or not np.isclose(weights.sum(), 1.0, atol=1e-10)
        ):
            failures.append(f"invalid portfolio: {row.get('date')} {row.get('condition')} {row.get('vendor')}")
            break
    for design in ("shared", "mixed", "coherent"):
        stored = result["levels"][design]
        for metric in (
            "N_eff", "same_side_overlap", "holdings_overlap",
            "convex_order_concentration", "team_hhi",
        ):
            direct = float(np.mean([row[design][metric] for row in result["per_date"]]))
            if not np.isclose(direct, stored[metric]["mean"], atol=1e-12):
                failures.append(f"aggregate mismatch: {design} {metric}")
    operating = result.get("operating_performance", {})
    operating_start = len(failures)
    if operating.get("comparison_control") != "shared access over identical holding windows":
        failures.append("operating comparison control mismatch")
    if any(key in operating for key in ("performance_benchmark", "spy_benchmark", "spy_benchmark_source")):
        failures.append("external benchmark remains in operating results")
    if operating.get("equal_weight_reference") != "constant 1/14 target weights over the same holding windows":
        failures.append("equal-weight reference definition mismatch")
    if len(operating.get("evaluation_windows", [])) != 34:
        failures.append("operating evaluation-window count mismatch")
    expected_metrics = (
        "annualized_total_return", "annualized_total_volatility", "sharpe_ratio",
        "compound_annual_growth", "maximum_drawdown",
        "mean_monthly_target_turnover", "mean_monthly_rebalance_turnover",
    )
    for design, expected_paths in (
        ("equal_weight", 1), ("shared", 1), ("mixed", 5040), ("coherent", 5040)
    ):
        level = operating.get("levels", {}).get(design, {})
        for metric in expected_metrics:
            item = level.get(metric, {})
            if not np.isfinite(item.get("mean", np.nan)) or len(item.get("assignment_95", [])) != 2:
                failures.append(f"invalid operating metric: {design} {metric}")
        if level.get("n_fixed_assignments") != expected_paths:
            failures.append(f"fixed assignment count mismatch: {design}")
    allocation = operating.get("allocation_decomposition", {})
    allocation_levels = allocation.get("levels", {})
    for design, expected_paths in (("shared", 1), ("mixed", 5040), ("coherent", 5040)):
        level = allocation_levels.get(design, {})
        total = level.get("annualized_team_variance", {}).get("mean", np.nan)
        standalone = level.get("standalone_variance_component", {}).get("mean", np.nan)
        covariance = level.get("cross_seat_covariance_component", {}).get("mean", np.nan)
        correlation = level.get("mean_pairwise_return_correlation", {}).get("mean", np.nan)
        if not np.isclose(total, standalone + covariance, atol=1e-12):
            failures.append(f"allocation variance identity mismatch: {design}")
        if not np.isfinite(correlation) or not -1.0 <= correlation <= 1.0:
            failures.append(f"invalid seat-return correlation: {design}")
        if level.get("n_fixed_assignments") != expected_paths:
            failures.append(f"allocation path count mismatch: {design}")
    shared_total = allocation_levels.get("shared", {}).get(
        "annualized_team_variance", {}
    ).get("mean", np.nan)
    for design in ("mixed", "coherent"):
        comparison = allocation.get("comparisons", {}).get(
            f"{design}_minus_shared", {}
        )
        total_reduction = comparison.get("team_variance_reduction", np.nan)
        component_reduction = (
            comparison.get("standalone_variance_reduction", np.nan)
            + comparison.get("cross_seat_covariance_reduction", np.nan)
        )
        if not np.isclose(total_reduction, component_reduction, atol=1e-12):
            failures.append(f"allocation reduction identity mismatch: {design}")
        if not np.isclose(
            comparison.get("team_variance_reduction_fraction", np.nan),
            total_reduction / shared_total,
            atol=1e-12,
        ):
            failures.append(f"allocation reduction fraction mismatch: {design}")
        share = comparison.get("covariance_share_of_variance_reduction", np.nan)
        if not np.isfinite(share) or not 0.0 <= share <= 1.0:
            failures.append(f"invalid covariance reduction share: {design}")
    operating_control_checked = len(failures) == operating_start
    spot = _spot_recompute(rows, result) if not failures else []
    output = {
        "schema_version": "information-capacity-audited-long-only-validation-v1",
        "passed": not failures,
        "failures": failures,
        "checks": {
            "rows_checked": len(rows),
            "generation_ids_unique": len(set(ids)),
            "dates": len(result.get("per_date", [])),
            "designs": list(result.get("levels", {})),
            "retried_rows": result["collection_diagnostics"]["retried_rows"],
            "shared_access_control_checked": operating_control_checked,
            "spot_recomputation": spot,
        },
    }
    _atomic_json(VALIDATION_PATH, output)
    print(json.dumps(output, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
