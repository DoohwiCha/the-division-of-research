"""Independent integrity checks for the audit-corrected K=7 result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA, RESULTS
from cohort_information_capacity_audited_analysis import RESULT_PATH, SUMMARY_PATH
from cohort_information_capacity_audited_analysis import _contains_ai_narrative
from cohort_information_capacity_audited_architecture import (
    CACHE_PATH as AUDITED_CACHE_PATH,
    MIXED_CANDIDATES,
    PLAN_PATH,
    regenerate_mixed_assignment,
)
from cohort_information_capacity_integrity import (
    CACHE_PATH as INTEGRITY_CACHE_PATH,
    INTEGRITY_CORRECTION_ID,
    MASKED_CACHE_PATH,
)


OUTPUT_PATH = Path(RESULTS) / "information_capacity_audited_architecture_validation.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    plan = _load(PLAN_PATH)
    shared = [
        row for row in _load(INTEGRITY_CACHE_PATH)
        if row.get("architecture") == "shared"
    ]
    audited = _load(AUDITED_CACHE_PATH)
    rows = [*shared, *audited]
    result = _load(RESULT_PATH)
    masked = [row for row in _load(MASKED_CACHE_PATH) if row.get("parse_ok")]
    summary = pd.read_csv(SUMMARY_PATH)
    manifest = {
        (
            row["date"], row["architecture"], int(row["team_size"]),
            int(row["packet_index"]), row["vendor"],
        ): row
        for row in plan["job_manifest"]
    }
    cache_keys = {
        (
            row["date"], row["architecture"], int(row["team_size"]),
            int(row["packet_index"]), row["vendor"],
        )
        for row in audited
    }
    failures = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(len(shared) == 238, f"shared row count is {len(shared)}, expected 238")
    require(len(audited) == 3332, f"audited row count is {len(audited)}, expected 3332")
    require(len(cache_keys) == len(audited), "audited cache has duplicate job keys")
    require(cache_keys == set(manifest), "audited cache keys differ from frozen manifest")
    require(all(row.get("parse_ok") for row in rows), "cache contains terminal failures")
    require(
        all(row.get("integrity_correction_id") == INTEGRITY_CORRECTION_ID for row in rows),
        "cache contains a stale correction wave",
    )
    require(all(row.get("response_id") for row in rows), "response ID is missing")
    require(all(row.get("response_provider") for row in rows), "response provider is missing")
    require(
        len({row["response_id"] for row in rows}) == len(rows),
        "generation IDs are not unique",
    )
    for row in audited:
        key = (
            row["date"], row["architecture"], int(row["team_size"]),
            int(row["packet_index"]), row["vendor"],
        )
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
    require(
        _sha(plan["job_manifest"]) == plan["job_manifest_sha256"],
        "job manifest hash mismatch",
    )
    regenerated_mixed, regenerated_nmi = regenerate_mixed_assignment(
        list(plan["assignment"]["packet_sizes"])
    )
    frozen_mixed = [row["field_ids"] for row in plan["assignment"]["mixed"]]
    require(regenerated_mixed == frozen_mixed,
            "mixed partition does not regenerate from the recorded seed")
    require(np.isclose(
        regenerated_nmi,
        plan["assignment"]["mixed_fine_domain_nmi"],
        atol=1e-15,
    ), "mixed partition NMI does not regenerate")
    require(len(plan.get("lineage", [])) >= 1, "plan lineage is missing")
    require("held-out" not in json.dumps(plan).lower(), "plan retains a held-out claim")
    require("held-out" not in json.dumps(result).lower(), "result retains a held-out claim")
    require("predecessor_K7_levels" not in result, "result exposes a superseded numeric panel")

    for design in ("shared", "mixed", "coherent"):
        design_rows = summary[summary["design"] == design]
        require(len(design_rows) == 34, f"{design}: summary does not have 34 dates")
        for metric in (
            "N_eff", "same_side_overlap", "surviving_gross", "convex_order_concentration"
        ):
            observed = float(design_rows[metric].mean())
            reported = float(result["levels"][design][metric]["mean"])
            require(
                np.isclose(observed, reported, atol=1e-12),
                f"{design} {metric}: CSV mean {observed} != JSON {reported}",
            )
    comparisons = result["paired_comparisons"]
    for treatment, control in (
        ("mixed", "shared"),
        ("coherent", "shared"),
        ("coherent", "mixed"),
    ):
        key = f"{treatment}_minus_{control}"
        for metric, values in comparisons[key].items():
            arithmetic = values["treatment_mean"] - values["control_mean"]
            require(
                np.isclose(arithmetic, values["mean"], atol=1e-12),
                f"{key} {metric}: paired arithmetic mismatch",
            )

    for design in ("mixed", "coherent"):
        key = f"{design}_minus_shared"
        n_eff = comparisons[key]["N_eff"]
        overlap = comparisons[key]["same_side_overlap"]
        concentration = comparisons[key]["convex_order_concentration"]
        require(n_eff["dates_positive"] == 34, f"{key}: N_eff is not positive on all dates")
        require(overlap["dates_negative"] == 34, f"{key}: overlap is not lower on all dates")
        require(
            concentration["dates_negative"] == 34,
            f"{key}: order concentration is not lower on all dates",
        )
        require(n_eff["date_bootstrap_95"][0] > 0, f"{key}: N_eff CI includes zero")
        require(overlap["date_bootstrap_95"][1] < 0, f"{key}: overlap CI includes zero")
        require(
            concentration["date_bootstrap_95"][1] < 0,
            f"{key}: concentration CI includes zero",
        )

    mechanism = result["mechanism_controls"]
    for design in ("mixed", "coherent"):
        broadcast = f"{design}_short_packet_broadcast"
        comparison = mechanism["paired_comparisons"][f"{design}_minus_{broadcast}"]["N_eff"]
        require(comparison["dates_positive"] == 34,
                f"{design}: split access does not beat the short-packet broadcast on every date")
        require(comparison["date_bootstrap_95"][0] > 0,
                f"{design}: short-packet broadcast interval includes zero")
        clean = f"{design}_without_explicit_ai_narratives"
        clean_effect = mechanism["paired_comparisons"][f"{clean}_minus_{design}"]["N_eff"]
        require(abs(clean_effect["mean"]) < 0.01,
                f"{design}: explicit-narrative exclusion changes N_eff materially")

    for label, expected_dates in (("complement_dates", 22),
                                  ("post_latest_route_registration_dates", 8)):
        sample = result["sample_sensitivity"][label]
        require(sample["n_dates"] == expected_dates,
                f"{label}: expected {expected_dates} dates, found {sample['n_dates']}")
        for treatment in ("mixed", "coherent"):
            contrast = sample[f"{treatment}_minus_shared_N_eff"]
            require(contrast["dates_positive"] == expected_dates,
                    f"{label}: {treatment} gain is not positive on every date")
            require(contrast["date_bootstrap_95"][0] > 0,
                    f"{label}: {treatment} interval includes zero")

    within_route = result["within_route_split_diagnostic"]
    for design in ("mixed", "coherent"):
        level = within_route[design]["levels"]["N_eff"]
        require(level["n_dates"] == 34,
                f"{design}: within-route diagnostic does not cover 34 dates")
        observed = np.mean([
            row[f"{design}_within_route"]["N_eff"] for row in result["per_date"]
        ])
        require(np.isclose(level["mean"], observed, atol=1e-12),
                f"{design}: within-route date mean mismatch")

    visible = {
        (row["date"], row["vendor"]): row for row in shared + [
            row for row in _load(INTEGRITY_CACHE_PATH)
            if row.get("architecture") == "coherent"
            and int(row.get("packet_index", 0)) == 7
            and row.get("parse_ok")
        ]
        if row.get("architecture") == "coherent"
    }
    masked_lookup = {(row["date"], row["vendor"]): row for row in masked}
    paired = set(visible).intersection(masked_lookup)
    audit = result["design_audit"]["legacy_macro_ticker_sensitivity"]
    require(len(paired) == audit["successful_matched_pairs"] == 217,
            "ticker-mask audit pair count changed")
    require(sum(_contains_ai_narrative(visible[key]) for key in paired)
            == audit["visible_ticker_explicit_ai_narratives"] == 24,
            "visible-ticker narrative count changed")
    require(sum(_contains_ai_narrative(masked_lookup[key]) for key in paired)
            == audit["masked_ticker_explicit_ai_narratives"] == 0,
            "masked-ticker narrative count changed")

    validation = {
        "schema_version": "information-capacity-audited-architecture-validation-v1",
        "passed": not failures,
        "failures": failures,
        "checks": {
            "rows_checked": len(rows),
            "generation_ids_unique": len({row["response_id"] for row in rows}),
            "frozen_manifest_rows": len(manifest),
            "dates": int(summary["date"].nunique()),
            "designs": sorted(summary["design"].unique()),
            "retry_rows": sum(int(row.get("format_retry_count", 0)) > 0 for row in rows),
            "ticker_mask_pairs": len(paired),
            "short_packet_broadcast_checked": True,
            "explicit_narrative_exclusion_checked": True,
            "complement_dates_checked": 22,
            "post_latest_route_registration_dates_checked": 8,
            "within_route_split_checked": True,
            "mixed_partition_candidates_checked": MIXED_CANDIDATES,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if failures:
        raise AssertionError("; ".join(failures))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
