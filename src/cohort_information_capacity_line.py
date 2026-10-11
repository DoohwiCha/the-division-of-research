"""Test a book-level common-beta and marginal-capacity relation.

For every matched four-seat team in the frozen 34-date experiment, the script
computes two quantities for each seat:

* leave-one-out common-book beta: the correlation between a book and the mean
  of the other three books;
* Shapley marginal capacity: the seat's exact contribution to N_eff - 1.

The date is the inference unit.  Configuration-level observations are first
collapsed into a date-specific slope and Spearman correlation.  The analysis
therefore tests whether the relation is stable across market snapshots rather
than treating overlapping model subsets as independent observations.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _paths import DATA, RESULTS
from cohort_analyze import n_eff
from cohort_information_diversity import CHANNELS, PILOT_MODELS, _atomic_json
from cohort_information_diversity_random_placebo import PACKET_LABELS


PILOT_CACHE = Path(DATA) / "cohort_agents_information_diversity.json"
HELDOUT_CACHE = Path(DATA) / "cohort_agents_information_diversity_confirmatory.json"
RANDOM_CACHE = Path(DATA) / "cohort_agents_information_diversity_random_placebo.json"
OUTPUT_PATH = Path(RESULTS) / "information_capacity_line_preliminary.json"
SUMMARY_PATH = Path(RESULTS) / "information_capacity_line_preliminary.csv"

RANDOM_CONDITIONS = tuple(f"random_{label}" for label in PACKET_LABELS)
BOOTSTRAP_DRAWS = 50_000


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _book(row: dict, symbols: list[str]) -> np.ndarray:
    values = {item["symbol"]: float(item["weight"]) for item in row["weights"]}
    vector = np.asarray([values.get(symbol, 0.0) for symbol in symbols], dtype=float)
    vector = vector - vector.mean()
    gross = np.abs(vector).sum()
    if gross <= 1e-12:
        raise ValueError("zero-gross active book")
    return vector / gross


def _value(matrix: np.ndarray, mask: int) -> float:
    members = [index for index in range(matrix.shape[1]) if mask & (1 << index)]
    if len(members) <= 1:
        return 0.0
    corr = np.corrcoef(matrix[:, members].T)
    return float(n_eff(corr) - 1.0)


def _shapley_capacity(matrix: np.ndarray) -> np.ndarray:
    m = matrix.shape[1]
    values = {mask: _value(matrix, mask) for mask in range(1 << m)}
    factorial = math.factorial
    denominator = factorial(m)
    output = np.zeros(m)
    for player in range(m):
        for mask in range(1 << m):
            if mask & (1 << player):
                continue
            size = int(mask.bit_count())
            weight = factorial(size) * factorial(m - size - 1) / denominator
            output[player] += weight * (
                values[mask | (1 << player)] - values[mask]
            )
    full = values[(1 << m) - 1]
    if not np.isclose(output.sum(), full, atol=1e-10):
        raise AssertionError("Shapley capacity does not sum to N_eff - 1")
    return output


def _common_beta(matrix: np.ndarray) -> np.ndarray:
    output = []
    for index in range(matrix.shape[1]):
        book = matrix[:, index]
        consensus = np.mean(np.delete(matrix, index, axis=1), axis=1)
        if np.std(consensus) <= 1e-12:
            output.append(0.0)
        else:
            output.append(float(np.corrcoef(book, consensus)[0, 1]))
    return np.asarray(output)


def _team_rows(matrix: np.ndarray, design: str, date: str) -> list[dict]:
    beta = _common_beta(matrix)
    shapley = _shapley_capacity(matrix)
    return [
        {
            "date": date,
            "design": design,
            "common_book_beta": float(beta[index]),
            "common_book_beta_squared": float(beta[index] ** 2),
            "shapley_capacity": float(shapley[index]),
        }
        for index in range(matrix.shape[1])
    ]


def _date_statistics(rows: list[dict], design: str, date: str) -> dict:
    current = [row for row in rows if row["design"] == design and row["date"] == date]
    beta = np.asarray([row["common_book_beta"] for row in current])
    beta2 = np.square(beta)
    capacity = np.asarray([row["shapley_capacity"] for row in current])
    centered = beta2 - beta2.mean()
    slope = float(np.dot(centered, capacity - capacity.mean()) / np.dot(centered, centered))
    rho = float(spearmanr(beta, capacity).statistic)
    rho2 = float(spearmanr(beta2, capacity).statistic)
    return {
        "date": date,
        "design": design,
        "n_book_team_observations": len(current),
        "mean_common_book_beta": float(beta.mean()),
        "mean_common_book_beta_squared": float(beta2.mean()),
        "mean_shapley_capacity": float(capacity.mean()),
        "slope_capacity_on_beta_squared": slope,
        "spearman_beta_capacity": rho,
        "spearman_beta_squared_capacity": rho2,
    }


def _bootstrap(values: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "date_bootstrap_95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "dates_negative": int(np.sum(values < 0)),
        "dates_positive": int(np.sum(values > 0)),
        "n_dates": len(values),
    }


def main() -> None:
    records = []
    for path in (PILOT_CACHE, HELDOUT_CACHE, RANDOM_CACHE):
        records.extend(_load(path))
    random_rows = _load(RANDOM_CACHE)
    dates = sorted({row["date"] for row in random_rows})
    symbols_by_date = {
        row["date"]: [item["symbol"] for item in row["weights"]]
        for row in records if row.get("parse_ok")
    }
    lookup = {
        (row["date"], row["vendor"], row["condition"], int(row.get("replicate", 0))): row
        for row in records if row.get("parse_ok")
    }
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    observations: list[dict] = []
    for date in dates:
        symbols = symbols_by_date[date]
        for combination in itertools.combinations(vendors, 4):
            shared = np.column_stack([
                _book(lookup[date, vendor, "rich", 0], symbols) for vendor in combination
            ])
            observations.extend(_team_rows(shared, "shared", date))
            for assignment in itertools.permutations(CHANNELS):
                domain = np.column_stack([
                    _book(lookup[date, vendor, condition, 0], symbols)
                    for vendor, condition in zip(combination, assignment)
                ])
                observations.extend(_team_rows(domain, "domain", date))
            for assignment in itertools.permutations(RANDOM_CONDITIONS):
                random_split = np.column_stack([
                    _book(lookup[date, vendor, condition, 0], symbols)
                    for vendor, condition in zip(combination, assignment)
                ])
                observations.extend(_team_rows(random_split, "random", date))

    date_rows = [
        _date_statistics(observations, design, date)
        for date in dates for design in ("shared", "random", "domain")
    ]
    summaries = {}
    seed = 2026072900
    for design in ("shared", "random", "domain"):
        current = [row for row in date_rows if row["design"] == design]
        summaries[design] = {}
        for metric in (
            "mean_common_book_beta",
            "mean_common_book_beta_squared",
            "mean_shapley_capacity",
            "slope_capacity_on_beta_squared",
            "spearman_beta_capacity",
            "spearman_beta_squared_capacity",
        ):
            summaries[design][metric] = _bootstrap(
                np.asarray([row[metric] for row in current]), seed
            )
            seed += 1

    result = {
        "schema_version": "information-capacity-line-preliminary-v1",
        "analysis_status": "cached-evidence preliminary; no new model calls",
        "common_book_beta": (
            "cross-asset correlation between a seat's active book and the mean active book "
            "of the other three seats"
        ),
        "marginal_capacity": "exact Shapley contribution to team N_eff - 1",
        "inference_unit": "date; overlapping team configurations are collapsed within date",
        "n_team_book_observations": len(observations),
        "date_statistics": date_rows,
        "summaries": summaries,
    }
    _atomic_json(OUTPUT_PATH, result)
    pd.DataFrame(date_rows).to_csv(SUMMARY_PATH, index=False)
    print("Common-book beta and marginal capacity")
    for design, row in summaries.items():
        slope = row["slope_capacity_on_beta_squared"]
        rho = row["spearman_beta_squared_capacity"]
        print(
            f"  {design:6s}: beta={row['mean_common_book_beta']['mean']:.3f}; "
            f"slope={slope['mean']:+.3f} [{slope['date_bootstrap_95'][0]:+.3f},"
            f"{slope['date_bootstrap_95'][1]:+.3f}]; "
            f"rho={rho['mean']:+.3f} [{rho['date_bootstrap_95'][0]:+.3f},"
            f"{rho['date_bootstrap_95'][1]:+.3f}]"
        )
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
