"""Moving-block inference for the monthly access-treatment contrasts."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from _paths import DATA, RESULTS
from cohort_information_capacity_audited_analysis import RESULT_PATH as POSITION_RESULT_PATH
from cohort_information_capacity_audited_curve_analysis import (
    RESULT_PATH as CURVE_RESULT_PATH,
)
from cohort_information_capacity_audited_longonly_analysis import (
    RESULT_PATH as LONG_ONLY_RESULT_PATH,
)
from cohort_information_capacity_risk_space import (
    PRIMARY_SHRINKAGE,
    RESULT_PATH as RISK_RESULT_PATH,
)
from cohort_information_diversity import _atomic_json
from cohort_information_capacity_integrity import _sha256_file


RESULT_PATH = Path(RESULTS) / "information_capacity_block_bootstrap_results.json"
BLOCK_LENGTHS = (3, 6)
PRIMARY_BLOCK_LENGTH = 3
BOOTSTRAP_DRAWS = 100_000


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _moving_block(values: np.ndarray, block_length: int, seed: int) -> dict:
    n = len(values)
    blocks_per_draw = math.ceil(n / block_length)
    rng = np.random.default_rng(seed)
    starts = rng.integers(
        0, n - block_length + 1, size=(BOOTSTRAP_DRAWS, blocks_per_draw)
    )
    offsets = np.arange(block_length)
    indices = starts[:, :, None] + offsets[None, None, :]
    indices = indices.reshape(BOOTSTRAP_DRAWS, -1)[:, :n]
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "moving_block_95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "block_length_months": block_length,
        "overlapping_blocks": True,
        "circular_blocks": False,
        "draws": BOOTSTRAP_DRAWS,
        "n_dates": n,
        "dates_positive": int(np.sum(values > 0)),
        "dates_negative": int(np.sum(values < 0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _all_blocks(values: np.ndarray, seed: int) -> dict:
    return {
        f"block_{length}": _moving_block(values, length, seed + index)
        for index, length in enumerate(BLOCK_LENGTHS)
    }


def main() -> None:
    position = _load(POSITION_RESULT_PATH)
    rows = sorted(position["per_date"], key=lambda row: row["date"])
    if len(rows) != 34:
        raise ValueError(f"expected 34 position dates, found {len(rows)}")

    result = {
        "schema_version": "information-capacity-moving-block-results-v3",
        "provenance": {
            "position_result_path": str(
                POSITION_RESULT_PATH.relative_to(Path(DATA).parent)
            ),
            "position_result_sha256": _sha256_file(POSITION_RESULT_PATH),
            "curve_result_path": str(CURVE_RESULT_PATH.relative_to(Path(DATA).parent)),
            "curve_result_sha256": _sha256_file(CURVE_RESULT_PATH),
            "risk_result_path": str(RISK_RESULT_PATH.relative_to(Path(DATA).parent)),
            "risk_result_sha256": _sha256_file(RISK_RESULT_PATH),
            "long_only_result_path": str(
                LONG_ONLY_RESULT_PATH.relative_to(Path(DATA).parent)
            ),
            "long_only_result_sha256": _sha256_file(LONG_ONLY_RESULT_PATH),
        },
        "date_range": [rows[0]["date"], rows[-1]["date"]],
        "n_dates": len(rows),
        "primary_block_length_months": PRIMARY_BLOCK_LENGTH,
        "sensitivity_block_lengths_months": list(BLOCK_LENGTHS),
        "position_space": {},
        "effective_book_curve": {"levels": {}, "paired_comparisons": {}},
        "risk_space": {},
        "long_only": {"paired_comparisons": {}},
    }
    seed = 2026072350
    result["position_space"]["levels"] = {}
    for design in ("shared", "mixed", "coherent"):
        result["position_space"]["levels"][design] = {}
        for metric in (
            "N_eff", "same_side_overlap", "surviving_gross",
            "convex_order_concentration",
        ):
            values = np.asarray([row[design][metric] for row in rows])
            result["position_space"]["levels"][design][metric] = _all_blocks(
                values, seed
            )
            seed += len(BLOCK_LENGTHS)

    for treatment, control in (
        ("mixed", "shared"),
        ("coherent", "shared"),
        ("coherent", "mixed"),
    ):
        comparison = f"{treatment}_minus_{control}"
        result["position_space"][comparison] = {}
        for metric in (
            "N_eff", "same_side_overlap", "surviving_gross",
            "convex_order_concentration",
        ):
            values = np.asarray([
                row[treatment][metric] - row[control][metric] for row in rows
            ])
            result["position_space"][comparison][metric] = _all_blocks(values, seed)
            seed += len(BLOCK_LENGTHS)

    curve = _load(CURVE_RESULT_PATH)
    curve_rows = curve["per_date"]
    for team_size in range(2, 8):
        selected = sorted(
            [row for row in curve_rows if int(row["team_size"]) == team_size],
            key=lambda row: row["date"],
        )
        if [row["date"] for row in selected] != [row["date"] for row in rows]:
            raise ValueError(f"curve and endpoint dates differ at K={team_size}")
        k_label = str(team_size)
        result["effective_book_curve"]["levels"][k_label] = {}
        for design in ("shared", "mixed", "coherent"):
            values = np.asarray([row[design]["N_eff"] for row in selected])
            result["effective_book_curve"]["levels"][k_label][design] = _all_blocks(
                values, seed
            )
            seed += len(BLOCK_LENGTHS)
        result["effective_book_curve"]["paired_comparisons"][k_label] = {}
        for treatment in ("mixed", "coherent"):
            values = np.asarray([
                row[treatment]["N_eff"] - row["shared"]["N_eff"]
                for row in selected
            ])
            result["effective_book_curve"]["paired_comparisons"][k_label][
                f"{treatment}_minus_shared"
            ] = _all_blocks(values, seed)
            seed += len(BLOCK_LENGTHS)

    risk = _load(RISK_RESULT_PATH)
    risk_rows = sorted(
        [
            row for row in risk["per_date"]
            if float(row["shrinkage"]) == PRIMARY_SHRINKAGE
        ],
        key=lambda row: row["date"],
    )
    if [row["date"] for row in risk_rows] != [row["date"] for row in rows]:
        raise ValueError("risk-space and position-space date panels differ")
    result["risk_space"]["diagonal_shrinkage"] = PRIMARY_SHRINKAGE
    result["risk_space"]["paired_comparisons"] = {}
    for treatment, control in (
        ("mixed", "shared"),
        ("coherent", "shared"),
        ("coherent", "mixed"),
    ):
        comparison = f"{treatment}_minus_{control}"
        values = np.asarray([
            row["risk_space"][treatment] - row["risk_space"][control]
            for row in risk_rows
        ])
        result["risk_space"]["paired_comparisons"][comparison] = {
            "N_eff": _all_blocks(values, seed)
        }
        seed += len(BLOCK_LENGTHS)

    long_only = _load(LONG_ONLY_RESULT_PATH)
    long_only_rows = sorted(long_only["per_date"], key=lambda row: row["date"])
    if [row["date"] for row in long_only_rows] != [row["date"] for row in rows]:
        raise ValueError("long-only and position-space date panels differ")
    long_only_metrics = (
        "N_eff",
        "same_side_overlap",
        "team_effective_holdings",
        "team_top3_weight",
        "team_max_weight",
        "team_forward_return",
    )
    for treatment in ("mixed", "coherent"):
        comparison = f"{treatment}_minus_shared"
        result["long_only"]["paired_comparisons"][comparison] = {}
        for metric in long_only_metrics:
            values = np.asarray([
                row[treatment][metric] - row["shared"][metric]
                for row in long_only_rows
            ])
            result["long_only"]["paired_comparisons"][comparison][metric] = (
                _all_blocks(values, seed)
            )
            seed += len(BLOCK_LENGTHS)

    _atomic_json(RESULT_PATH, result)
    print("Moving-block bootstrap")
    for space, comparisons in (
        ("position", result["position_space"]),
        ("risk", result["risk_space"]["paired_comparisons"]),
    ):
        for comparison in ("mixed_minus_shared", "coherent_minus_shared"):
            values = comparisons[comparison]["N_eff"][f"block_{PRIMARY_BLOCK_LENGTH}"]
            print(
                f"  {space} {comparison}: {values['mean']:+.3f} "
                f"MBB95={values['moving_block_95']}"
            )
    for comparison, metrics in result["long_only"]["paired_comparisons"].items():
        values = metrics["team_forward_return"][f"block_{PRIMARY_BLOCK_LENGTH}"]
        print(
            f"  long_only {comparison}: monthly return={values['mean']:+.4%} "
            f"MBB95={values['moving_block_95']}"
        )
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
