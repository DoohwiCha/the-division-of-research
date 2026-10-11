"""Validate the repeatability, risk-space, and block-bootstrap add-ons."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from _paths import DATA, RESULTS
from cohort_information_capacity_block_bootstrap import (
    BLOCK_LENGTHS,
    RESULT_PATH as BLOCK_RESULT_PATH,
)
from cohort_information_capacity_audited_longonly_analysis import (
    RESULT_PATH as LONG_ONLY_RESULT_PATH,
)
from cohort_information_capacity_audited_curve_analysis import (
    RESULT_PATH as CURVE_RESULT_PATH,
)
from cohort_information_capacity_integrity import _sha256_file, _sha256_text, _stable_json
from cohort_information_capacity_repeatability import CACHE_PATH, PLAN_PATH
from cohort_information_capacity_repeatability_analysis import (
    RESULT_PATH as REPEAT_RESULT_PATH,
)
from cohort_information_capacity_risk_space import (
    PRIMARY_SHRINKAGE,
    RESULT_PATH as RISK_RESULT_PATH,
    SHRINKAGES,
    TRAILING_DAYS,
)
from cohort_information_diversity import _atomic_json


VALIDATION_PATH = Path(RESULTS) / "information_capacity_robustness_validation.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _key(row: dict) -> tuple:
    return (
        row["date"], row["condition"], row["vendor"],
        float(row["temperature"]), int(row["replicate"]),
    )


def main() -> None:
    checks = []

    def require(name: str, condition: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    plan = _load(PLAN_PATH)
    cache = _load(CACHE_PATH)
    repeat = _load(REPEAT_RESULT_PATH)
    planned = {_key(row): row for row in plan["job_manifest"]}
    observed = {_key(row): row for row in cache}
    require(
        "repeat_plan_manifest_hash",
        plan["job_manifest_sha256"] == _sha256_text(_stable_json(plan["job_manifest"])),
        plan["job_manifest_sha256"],
    )
    require(
        "repeat_plan_assignment_hash",
        plan["assignment_sha256"] == _sha256_text(_stable_json(plan["date_assignments"])),
        plan["assignment_sha256"],
    )
    root = Path(DATA).parent
    require(
        "repeat_source_hashes",
        all(
            _sha256_file(root / record["path"]) == record["sha256"]
            for record in (plan["source_architecture_plan"], plan["model_catalog"])
        ),
        {
            "source_architecture_plan": plan["source_architecture_plan"],
            "model_catalog": plan["model_catalog"],
        },
    )
    require(
        "repeat_cell_keys",
        len(cache) == plan["n_calls"] == 1008 and set(observed) == set(planned),
        {"planned": plan["n_calls"], "observed": len(cache), "unique": len(observed)},
    )
    metadata_fields = (
        "packet_index", "requested_model", "prompt_sha256", "packet_sha256",
        "response_format_sha256", "request_payload_sha256", "request_seed",
    )
    mismatches = []
    for key, row in observed.items():
        for field in metadata_fields:
            if row.get(field) != planned[key].get(field):
                mismatches.append({"key": key, "field": field})
    require("repeat_frozen_metadata", not mismatches, mismatches[:10])
    valid = [row for row in cache if row.get("parse_ok") and row.get("response_id")]
    invalid = [row for row in cache if not (row.get("parse_ok") and row.get("response_id"))]
    require(
        "repeat_response_completeness",
        len(valid) == 1007 and len(invalid) == 1,
        {"valid": len(valid), "invalid": len(invalid)},
    )
    require(
        "repeat_result_provenance",
        repeat["provenance"]["plan_sha256"] == _sha256_file(PLAN_PATH)
        and repeat["provenance"]["cache_sha256"] == _sha256_file(CACHE_PATH),
        repeat["provenance"],
    )
    provider_counts = defaultdict(Counter)
    for row in valid:
        provider_counts[row["vendor"]][row["response_provider"]] += 1
    stable_vendors = sorted(
        vendor for vendor, counts in provider_counts.items()
        if max(counts.values()) / sum(counts.values()) >= 0.95
    )
    require(
        "repeat_stable_backend_selection",
        stable_vendors == sorted(repeat["stable_backend_sensitivity"]["vendors"]),
        {"derived": stable_vendors, "reported": repeat["stable_backend_sensitivity"]["vendors"]},
    )
    for temperature in (0.0, 0.7):
        label = f"temperature_{temperature:.1f}"
        summary = repeat["summary"][label]
        for treatment in ("repeat_mixed_split", "repeat_coherent_split"):
            contrast = summary["contrasts"][f"{treatment}_minus_shared"]
            persistent = contrast["persistent_repeat_average"]
            require(
                f"repeat_persistent_{temperature:.1f}_{treatment}",
                persistent["mean"] > 0
                and persistent["date_bootstrap_95"][0] > 0
                and persistent["dates_positive"] == 6,
                persistent,
            )
            complete = summary["complete_panel_sensitivity"]["contrasts"][
                f"{treatment}_minus_shared"
            ]
            require(
                f"repeat_complete_panel_{temperature:.1f}_{treatment}",
                complete["mean"] > 0 and complete["date_bootstrap_95"][0] > 0,
                complete,
            )
            stable = repeat["stable_backend_sensitivity"]["summary"][label][
                "contrasts"
            ][f"{treatment}_minus_shared"]
            require(
                f"repeat_stable_backend_{temperature:.1f}_{treatment}",
                stable["mean"] > 0
                and stable["date_bootstrap_95"][0] > 0
                and stable["dates_positive"] == 6,
                stable,
            )

    risk = _load(RISK_RESULT_PATH)
    risk_provenance = risk["provenance"]
    require(
        "risk_source_hashes",
        all(
            _sha256_file(root / risk_provenance[path_key])
            == risk_provenance[hash_key]
            for path_key, hash_key in (
                ("price_path", "price_sha256"),
                ("shared_cache_path", "shared_cache_sha256"),
                ("split_cache_path", "split_cache_sha256"),
                ("position_result_path", "position_result_sha256"),
            )
        ),
        risk_provenance,
    )
    require(
        "risk_primary_sample_covariance",
        PRIMARY_SHRINKAGE == 0.0
        and risk["covariance_specification"]["primary_diagonal_shrinkage"] == 0.0,
        risk["covariance_specification"],
    )
    require(
        "risk_pit_window",
        all(
            row["covariance"]["history_end"] <= row["date"]
            and row["covariance"]["n_returns"] == TRAILING_DAYS
            for row in risk["per_date"]
        ),
        {"rows": len(risk["per_date"]), "trailing_days": TRAILING_DAYS},
    )
    require(
        "risk_position_crosscheck",
        max(
            value
            for row in risk["per_date"]
            for value in row["position_crosscheck"].values()
        ) <= 1e-10,
        "maximum absolute error <= 1e-10",
    )
    for shrinkage in SHRINKAGES:
        label = f"shrinkage_{shrinkage:.2f}"
        for treatment in ("mixed", "coherent"):
            contrast = risk["summary"][label]["paired_comparisons"][
                f"{treatment}_minus_shared"
            ]
            require(
                f"risk_{label}_{treatment}_minus_shared",
                contrast["mean"] > 0 and contrast["date_bootstrap_95"][0] > 0,
                contrast,
            )

    block = _load(BLOCK_RESULT_PATH)
    require(
        "block_source_hashes",
        block["provenance"]["risk_result_sha256"] == _sha256_file(RISK_RESULT_PATH)
        and block["provenance"]["curve_result_sha256"] == _sha256_file(CURVE_RESULT_PATH)
        and block["provenance"]["long_only_result_sha256"]
        == _sha256_file(LONG_ONLY_RESULT_PATH),
        block["provenance"],
    )
    for space, comparisons in (
        ("position", block["position_space"]),
        ("risk", block["risk_space"]["paired_comparisons"]),
    ):
        for treatment in ("mixed", "coherent"):
            result = comparisons[f"{treatment}_minus_shared"]["N_eff"]
            for length in BLOCK_LENGTHS:
                estimate = result[f"block_{length}"]
                require(
                    f"block_{space}_{treatment}_{length}",
                    estimate["moving_block_95"][0] > 0
                    and estimate["overlapping_blocks"]
                    and not estimate["circular_blocks"],
                    estimate,
                )
    for team_size in range(2, 8):
        comparisons = block["effective_book_curve"]["paired_comparisons"][
            str(team_size)
        ]
        for treatment in ("mixed", "coherent"):
            estimate = comparisons[f"{treatment}_minus_shared"]["block_3"]
            require(
                f"curve_block_{team_size}_{treatment}",
                estimate["moving_block_95"][0] > 0,
                estimate,
            )
    for treatment in ("mixed", "coherent"):
        estimate = block["long_only"]["paired_comparisons"][
            f"{treatment}_minus_shared"
        ]["team_forward_return"]["block_3"]
        require(
            f"long_only_return_block_{treatment}",
            estimate["moving_block_95"][0] < 0 < estimate["moving_block_95"][1],
            estimate,
        )

    output = {
        "schema_version": "information-capacity-robustness-validation-v1",
        "all_passed": True,
        "n_checks": len(checks),
        "checks": checks,
        "source_hashes": {
            "repeat_plan": _sha256_file(PLAN_PATH),
            "repeat_cache": _sha256_file(CACHE_PATH),
            "repeat_result": _sha256_file(REPEAT_RESULT_PATH),
            "risk_result": _sha256_file(RISK_RESULT_PATH),
            "block_result": _sha256_file(BLOCK_RESULT_PATH),
            "long_only_result": _sha256_file(LONG_ONLY_RESULT_PATH),
        },
    }
    _atomic_json(VALIDATION_PATH, output)
    print(f"validated {len(checks)} checks; wrote {VALIDATION_PATH}")


if __name__ == "__main__":
    main()
