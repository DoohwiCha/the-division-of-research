"""Attribute K=4 roster capacity to Common-Book Beta and analyst seats."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from _paths import DATA, RESULTS
from cohort_information_capacity_analysis import _book, _bootstrap
from cohort_information_capacity_experiment import CACHE_PATH, PLAN_PATH
from cohort_information_capacity_line import _common_beta, _shapley_capacity
from cohort_information_diversity import PILOT_MODELS, _atomic_json, _strict_complete
from cohort_information_diversity_random_placebo import load_packets


PILOT_SHARED_CACHE = Path(DATA) / "cohort_agents_information_diversity.json"
HELDOUT_SHARED_CACHE = Path(DATA) / "cohort_agents_information_diversity_confirmatory.json"
OUTPUT_PATH = Path(RESULTS) / "information_capacity_attribution.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sample(rows: list[dict], sample: str) -> list[dict]:
    return rows if sample == "all" else [row for row in rows if row.get("sample") == sample]


def _inputs(sample: str, allow_partial: bool):
    packet_rows = _sample(load_packets(), sample)
    capacity = _sample(_load(CACHE_PATH), sample)
    capacity = [row for row in capacity if _strict_complete(row)]
    shared = []
    for path, current_sample in (
        (PILOT_SHARED_CACHE, "pilot"),
        (HELDOUT_SHARED_CACHE, "held-out"),
    ):
        if sample not in ("all", current_sample):
            continue
        for row in _load(path):
            if row.get("condition") == "rich" and int(row.get("replicate", 0)) == 0:
                shared.append({**row, "sample": current_sample})
    expected_per_date = 378
    dates = {row["snapshot"]["date"] for row in packet_rows}
    complete_dates = {
        date for date in dates
        if sum(row["date"] == date for row in capacity) == expected_per_date
    }
    if not allow_partial and complete_dates != dates:
        raise ValueError(
            f"capacity collection incomplete: {len(complete_dates)}/{len(dates)} dates complete"
        )
    if not complete_dates:
        raise ValueError("no complete capacity date")
    packet_rows = [row for row in packet_rows if row["snapshot"]["date"] in complete_dates]
    capacity = [row for row in capacity if row["date"] in complete_dates]
    shared = [row for row in shared if row["date"] in complete_dates]
    return packet_rows, capacity, shared


def _regression(beta_squared: np.ndarray, shapley: np.ndarray) -> tuple[float, float]:
    centered = beta_squared - beta_squared.mean()
    denominator = float(np.dot(centered, centered))
    slope = 0.0 if denominator <= 1e-15 else float(
        np.dot(centered, shapley - shapley.mean()) / denominator
    )
    rho = float(spearmanr(beta_squared, shapley).statistic)
    return slope, rho


def _date_design(
    date: str,
    design: str,
    symbols: list[str],
    vendors: list[str],
    capacity_lookup: dict,
    shared_lookup: dict,
) -> dict:
    observations = []
    if design == "shared":
        configurations = [
            [(vendor, None) for vendor in combination]
            for combination in itertools.combinations(vendors, 4)
        ]
    else:
        configurations = [
            list(zip(combination, assignment))
            for combination in itertools.combinations(vendors, 4)
            for assignment in itertools.permutations(range(1, 5))
        ]
    for configuration in configurations:
        if design == "shared":
            matrix = np.column_stack([
                _book(shared_lookup[date, vendor], symbols) for vendor, _ in configuration
            ])
        else:
            matrix = np.column_stack([
                _book(
                    capacity_lookup[date, vendor, f"capacity_{design}_k4_p{packet}"],
                    symbols,
                )
                for vendor, packet in configuration
            ])
        beta = _common_beta(matrix)
        shapley = _shapley_capacity(matrix)
        for index, (vendor, packet) in enumerate(configuration):
            observations.append({
                "vendor": vendor,
                "packet_index": packet,
                "common_book_beta": float(beta[index]),
                "common_book_beta_squared": float(beta[index] ** 2),
                "shapley_capacity": float(shapley[index]),
            })
    beta = np.asarray([row["common_book_beta"] for row in observations])
    beta_squared = np.square(beta)
    shapley = np.asarray([row["shapley_capacity"] for row in observations])
    slope, rho = _regression(beta_squared, shapley)
    order = np.argsort(beta_squared, kind="stable")
    bin_index = np.empty(len(order), dtype=int)
    bin_index[order] = np.floor(5 * np.arange(len(order)) / len(order)).astype(int)
    beta_capacity_bins = [
        {
            "bin": current_bin + 1,
            "n_observations": int(np.sum(bin_index == current_bin)),
            "mean_common_book_beta_squared": float(
                beta_squared[bin_index == current_bin].mean()
            ),
            "mean_shapley_capacity": float(shapley[bin_index == current_bin].mean()),
        }
        for current_bin in range(5)
    ]
    by_vendor = {
        vendor: {
            "mean_common_book_beta": float(np.mean([
                row["common_book_beta"] for row in observations if row["vendor"] == vendor
            ])),
            "mean_shapley_capacity": float(np.mean([
                row["shapley_capacity"] for row in observations if row["vendor"] == vendor
            ])),
        }
        for vendor in vendors
    }
    by_packet = {}
    if design != "shared":
        by_packet = {
            str(packet): {
                "mean_common_book_beta": float(np.mean([
                    row["common_book_beta"] for row in observations
                    if row["packet_index"] == packet
                ])),
                "mean_shapley_capacity": float(np.mean([
                    row["shapley_capacity"] for row in observations
                    if row["packet_index"] == packet
                ])),
            }
            for packet in range(1, 5)
        }
    return {
        "date": date,
        "design": design,
        "n_book_team_observations": len(observations),
        "mean_common_book_beta": float(beta.mean()),
        "mean_common_book_beta_squared": float(beta_squared.mean()),
        "mean_shapley_capacity": float(shapley.mean()),
        "slope_capacity_on_beta_squared": slope,
        "spearman_beta_squared_capacity": rho,
        "negative_shapley_share": float(np.mean(shapley < 0)),
        "beta_capacity_bins": beta_capacity_bins,
        "by_vendor": by_vendor,
        "by_packet": by_packet,
    }


def _summaries(rows: list[dict]) -> dict:
    output = {}
    seed = 2026072000
    for design in ("shared", "mixed", "coherent"):
        current = [row for row in rows if row["design"] == design]
        output[design] = {}
        for metric in (
            "mean_common_book_beta", "mean_common_book_beta_squared",
            "mean_shapley_capacity", "slope_capacity_on_beta_squared",
            "spearman_beta_squared_capacity", "negative_shapley_share",
        ):
            output[design][metric] = _bootstrap(
                np.asarray([row[metric] for row in current]), seed
            )
            seed += 1
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=("pilot", "held-out", "all"), default="all")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet_rows, capacity, shared = _inputs(args.sample, args.allow_partial)
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    capacity_lookup = {
        (row["date"], row["vendor"], row["condition"]): row for row in capacity
    }
    shared_lookup = {(row["date"], row["vendor"]): row for row in shared}
    rows = []
    for item in packet_rows:
        date = item["snapshot"]["date"]
        symbols = item["snapshot"]["symbols"]
        for design in ("shared", "mixed", "coherent"):
            rows.append(
                _date_design(
                    date, design, symbols, vendors, capacity_lookup, shared_lookup
                )
            )
            print(f"  attributed {date} {design}", flush=True)
    result = {
        "schema_version": "information-capacity-attribution-v1",
        "sample": args.sample,
        "status": (
            "partial engineering check; not inferential evidence"
            if args.allow_partial else "complete pre-frozen experiment"
        ),
        "team_size": 4,
        "common_book_beta": (
            "cross-asset correlation between one active book and the mean of the other three"
        ),
        "marginal_capacity": "exact Shapley contribution to team N_eff - 1",
        "inference_unit": "date",
        "per_date": rows,
        "summaries": _summaries(rows),
    }
    suffix = ""
    if args.sample != "all":
        suffix += f"_{args.sample.replace('-', '_')}"
    if args.allow_partial:
        suffix += "_partial"
    output = OUTPUT_PATH.with_name(OUTPUT_PATH.stem + suffix + ".json")
    _atomic_json(output, result)
    for design, summary in result["summaries"].items():
        slope = summary["slope_capacity_on_beta_squared"]
        print(
            f"{design}: beta={summary['mean_common_book_beta']['mean']:.3f}, "
            f"slope={slope['mean']:+.3f}, "
            f"rho={summary['spearman_beta_squared_capacity']['mean']:+.3f}"
        )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
