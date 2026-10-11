"""Independent integrity checks for the corrected K=1..7 curve."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA, RESULTS
from cohort_information_capacity_audited_analysis import RESULT_PATH as K7_RESULT_PATH
from cohort_information_capacity_audited_curve import CACHE_PATH, PLAN_PATH, TEAM_SIZES
from cohort_information_capacity_audited_curve_analysis import RESULT_PATH, SUMMARY_PATH
from cohort_information_capacity_experiment import _is_macro
from cohort_information_capacity_integrity import INTEGRITY_CORRECTION_ID


OUTPUT_PATH = Path(RESULTS) / "information_capacity_audited_curve_validation.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    plan = _load(PLAN_PATH)
    cache = _load(CACHE_PATH)
    result = _load(RESULT_PATH)
    k7 = _load(K7_RESULT_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    failures = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    expected = 34 * 7 * 2 * sum(TEAM_SIZES)
    manifest = {
        (
            row["date"], int(row["team_size"]), row["architecture"],
            int(row["packet_index"]), row["vendor"],
        ): row
        for row in plan["job_manifest"]
    }
    cache_by_key = {
        (
            row["date"], int(row["team_size"]), row["architecture"],
            int(row["packet_index"]), row["vendor"],
        ): row
        for row in cache
    }
    require(len(cache) == expected, f"cache row count is {len(cache)}, expected {expected}")
    require(len(cache_by_key) == len(cache), "cache has duplicate job keys")
    require(set(cache_by_key) == set(manifest), "cache keys differ from frozen manifest")
    require(all(row.get("parse_ok") for row in cache), "cache contains terminal failures")
    require(all(row.get("response_id") for row in cache), "response ID is missing")
    require(all(row.get("response_provider") for row in cache), "response provider is missing")
    require(len({row["response_id"] for row in cache}) == len(cache), "generation IDs are not unique")
    require(
        all(row.get("integrity_correction_id") == INTEGRITY_CORRECTION_ID for row in cache),
        "cache contains a stale correction wave",
    )
    require(_sha(plan["job_manifest"]) == plan["job_manifest_sha256"], "manifest hash mismatch")
    require(_sha(plan["assignment"]) == plan["assignment_sha256"], "assignment hash mismatch")
    require("held-out" not in json.dumps(plan).lower(), "plan retains a held-out claim")
    require("held-out" not in json.dumps(result).lower(), "result retains a held-out claim")

    for key, row in cache_by_key.items():
        frozen = manifest[key]
        for field in (
            "prompt_sha256", "packet_sha256", "response_format_sha256",
            "request_payload_sha256", "request_seed", "requested_model",
        ):
            require(row.get(field) == frozen.get(field), f"{key}: changed {field}")
        require(
            int(row.get("format_retry_count", 0)) == len(row.get("request_attempts", [])) - 1,
            f"{key}: retry count does not match attempt records",
        )

    for k, designs in plan["assignment"].items():
        for design in ("mixed", "coherent"):
            flat = [field for seat in designs[design] for field in seat["field_ids"]]
            require(len(flat) == 32 and len(set(flat)) == 32, f"K={k} {design}: bad field union")
            require(
                all(not all(_is_macro(field) for field in seat["field_ids"]) for seat in designs[design]),
                f"K={k} {design}: macro-only seat",
            )

    for design in ("shared", "mixed", "coherent"):
        for metric in ("N_eff", "same_side_overlap", "surviving_gross", "convex_order_concentration"):
            curve_value = result["levels"]["7"][design][metric]["mean"]
            endpoint_value = k7["levels"][design][metric]["mean"]
            require(np.isclose(curve_value, endpoint_value, atol=1e-12), f"K7 mismatch: {design} {metric}")

    for k in range(2, 8):
        for design in ("shared", "mixed", "coherent"):
            current = summary[(summary["team_size"] == k) & (summary["design"] == design)]
            require(len(current) == 34, f"K={k} {design}: summary does not have 34 dates")
            for metric in ("N_eff", "same_side_overlap", "surviving_gross", "convex_order_concentration"):
                observed = float(current[metric].mean())
                reported = float(result["levels"][str(k)][design][metric]["mean"])
                require(np.isclose(observed, reported, atol=1e-12), f"K={k} {design} {metric}: mismatch")

    directions = {}
    for k in range(2, 8):
        directions[str(k)] = {}
        for design in ("mixed", "coherent"):
            key = f"{design}_minus_shared"
            comparison = result["paired_comparisons"][str(k)][key]
            directions[str(k)][design] = {
                "N_eff_positive_dates": comparison["N_eff"]["dates_positive"],
                "overlap_negative_dates": comparison["same_side_overlap"]["dates_negative"],
                "order_concentration_negative_dates": comparison["convex_order_concentration"]["dates_negative"],
            }

    validation = {
        "schema_version": "information-capacity-audited-curve-validation-v1",
        "passed": not failures,
        "failures": failures,
        "checks": {
            "rows_checked": len(cache),
            "generation_ids_unique": len({row["response_id"] for row in cache}),
            "frozen_manifest_rows": len(manifest),
            "dates": int(summary["date"].nunique()),
            "team_sizes": sorted(int(value) for value in summary["team_size"].unique()),
            "retry_rows": sum(int(row.get("format_retry_count", 0)) > 0 for row in cache),
            "directional_results": directions,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if failures:
        raise AssertionError("; ".join(failures))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
