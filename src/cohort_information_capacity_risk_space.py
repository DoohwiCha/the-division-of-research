"""Recompute effective-book capacity in ex-ante risk space.

Position-space correlation treats assets as equally scaled coordinates. This
check instead uses a covariance inner product estimated only from the 252
trading days available at each decision date.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA, RESULTS
from cohort_information_capacity_analysis import _book
from cohort_information_capacity_audited_analysis import (
    RESULT_PATH as POSITION_RESULT_PATH,
)
from cohort_information_capacity_audited_architecture import CACHE_PATH as SPLIT_CACHE_PATH
from cohort_information_capacity_integrity import (
    CACHE_PATH as INTEGRITY_CACHE_PATH,
    INTEGRITY_CORRECTION_ID,
    TEAM_SIZE,
    _load_packets,
    _sha256_file,
)
from cohort_information_diversity import PILOT_MODELS, _atomic_json


PRICE_PATH = Path(DATA) / "jfs_global_prices_united_states.csv"
RESULT_PATH = Path(RESULTS) / "information_capacity_risk_space_results.json"
SUMMARY_PATH = Path(RESULTS) / "information_capacity_risk_space_summary.csv"
TRAILING_DAYS = 252
PRIMARY_SHRINKAGE = 0.0
SHRINKAGES = (PRIMARY_SHRINKAGE, 0.25, 0.5)
NULL_DRAWS = 10_000
BOOTSTRAP_DRAWS = 50_000


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _current(row: dict) -> bool:
    return bool(
        row.get("parse_ok")
        and row.get("response_id")
        and row.get("integrity_correction_id") == INTEGRITY_CORRECTION_ID
        and row.get("request_payload_sha256")
    )


def _covariance(
    prices: pd.DataFrame,
    date: str,
    symbols: list[str],
    shrinkage: float,
) -> tuple[np.ndarray, dict]:
    history = prices.loc[prices.index <= pd.Timestamp(date), symbols].ffill().dropna()
    if len(history) < TRAILING_DAYS + 1:
        raise ValueError(f"insufficient PIT price history at {date}: {len(history)} rows")
    history = history.iloc[-(TRAILING_DAYS + 1):]
    returns = np.log(history / history.shift(1)).dropna().to_numpy(dtype=float)
    sample = np.cov(returns, rowvar=False, ddof=1)
    diagonal = np.diag(np.diag(sample))
    covariance = (1.0 - shrinkage) * sample + shrinkage * diagonal
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues.min() <= 0.0:
        covariance = covariance + np.eye(len(symbols)) * (abs(eigenvalues.min()) + 1e-12)
        eigenvalues = np.linalg.eigvalsh(covariance)
    return covariance, {
        "history_start": history.index[0].strftime("%Y-%m-%d"),
        "history_end": history.index[-1].strftime("%Y-%m-%d"),
        "n_returns": int(len(returns)),
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
        "condition_number": float(eigenvalues.max() / eigenvalues.min()),
    }


def _neff_from_gram(gram: np.ndarray) -> np.ndarray:
    diagonal = np.diagonal(gram, axis1=-2, axis2=-1)
    scale = np.sqrt(diagonal[..., :, None] * diagonal[..., None, :])
    correlation = np.divide(gram, scale, out=np.zeros_like(gram), where=scale > 1e-20)
    return TEAM_SIZE ** 2 / np.square(correlation).sum(axis=(-2, -1))


def _position_neff(matrix: np.ndarray) -> float:
    gram = matrix.T @ matrix
    return float(_neff_from_gram(gram))


def _risk_neff(matrix: np.ndarray, covariance: np.ndarray) -> float:
    gram = matrix.T @ covariance @ matrix
    return float(_neff_from_gram(gram))


def _configuration_means(
    cube: np.ndarray,
    covariance: np.ndarray,
    permutations: np.ndarray,
) -> tuple[float, float]:
    """Average position- and risk-space N_eff over packet assignments.

    cube is route x packet x asset. The route axis remains fixed while each row
    of permutations assigns one packet to each route.
    """
    route_indices = np.arange(TEAM_SIZE)[None, :]
    selected = cube[route_indices, permutations, :]  # assignment x route x asset
    position_gram = np.einsum("bkd,bld->bkl", selected, selected)
    position = _neff_from_gram(position_gram)

    chol = np.linalg.cholesky(covariance)
    transformed_cube = np.einsum("de,vpe->vpd", chol.T, cube)
    transformed = transformed_cube[route_indices, permutations, :]
    risk_gram = np.einsum("bkd,bld->bkl", transformed, transformed)
    risk = _neff_from_gram(risk_gram)
    return float(position.mean()), float(risk.mean())


def _risk_null(
    covariance: np.ndarray,
    n_assets: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    books = rng.standard_normal((NULL_DRAWS, n_assets, TEAM_SIZE))
    books -= books.mean(axis=1, keepdims=True)
    gross = np.abs(books).sum(axis=1, keepdims=True)
    books /= gross
    chol = np.linalg.cholesky(covariance)
    transformed = np.einsum("de,bek->bdk", chol.T, books)
    gram = np.einsum("bdk,bdl->bkl", transformed, transformed)
    values = _neff_from_gram(gram)
    return {
        "mean": float(values.mean()),
        "p025": float(np.quantile(values, 0.025)),
        "p975": float(np.quantile(values, 0.975)),
        "draws": NULL_DRAWS,
    }


def _bootstrap(values: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "date_bootstrap_95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "dates_positive": int(np.sum(values > 0)),
        "dates_negative": int(np.sum(values < 0)),
        "n_dates": int(len(values)),
    }


def main() -> None:
    shared_rows = [
        row for row in _load(INTEGRITY_CACHE_PATH)
        if row.get("architecture") == "shared" and _current(row)
    ]
    split_rows = [row for row in _load(SPLIT_CACHE_PATH) if _current(row)]
    if len(shared_rows) != 34 * TEAM_SIZE:
        raise ValueError(f"expected 238 shared books, found {len(shared_rows)}")
    if len(split_rows) != 34 * 2 * TEAM_SIZE * TEAM_SIZE:
        raise ValueError(f"expected 3332 split books, found {len(split_rows)}")

    shared_lookup = {(row["date"], row["vendor"]): row for row in shared_rows}
    split_lookup = {
        (row["date"], row["vendor"], row["architecture"], int(row["packet_index"])): row
        for row in split_rows
    }
    vendors = [vendor for vendor, _model in PILOT_MODELS]
    prices = pd.read_csv(PRICE_PATH, index_col=0, parse_dates=True).sort_index()
    permutations = np.asarray(list(itertools.permutations(range(TEAM_SIZE))), dtype=int)
    position_results = _load(POSITION_RESULT_PATH)
    expected_position = {
        row["date"]: row for row in position_results["per_date"]
    }

    per_date = []
    for date_index, item in enumerate(_load_packets()):
        snapshot = item["snapshot"]
        date = snapshot["date"]
        symbols = snapshot["symbols"]
        shared_matrix = np.column_stack([
            _book(shared_lookup[date, vendor], symbols) for vendor in vendors
        ])
        cubes = {}
        for architecture in ("mixed_audited", "coherent_audited"):
            cubes[architecture] = np.asarray([
                [
                    _book(split_lookup[date, vendor, architecture, packet], symbols)
                    for packet in range(1, TEAM_SIZE + 1)
                ]
                for vendor in vendors
            ])

        for shrinkage in SHRINKAGES:
            covariance, covariance_diagnostics = _covariance(
                prices, date, symbols, shrinkage
            )
            mixed_position, mixed_risk = _configuration_means(
                cubes["mixed_audited"], covariance, permutations
            )
            coherent_position, coherent_risk = _configuration_means(
                cubes["coherent_audited"], covariance, permutations
            )
            shared_position = _position_neff(shared_matrix)
            shared_risk = _risk_neff(shared_matrix, covariance)

            expected = expected_position[date]
            checks = {
                "shared_abs_error": abs(shared_position - expected["shared"]["N_eff"]),
                "mixed_abs_error": abs(mixed_position - expected["mixed"]["N_eff"]),
                "coherent_abs_error": abs(coherent_position - expected["coherent"]["N_eff"]),
            }
            if max(checks.values()) > 1e-10:
                raise AssertionError(f"position-space cross-check failed at {date}: {checks}")

            null = _risk_null(
                covariance,
                len(symbols),
                seed=2026072290 + 100 * date_index + int(round(shrinkage * 100)),
            )
            per_date.append({
                "date": date,
                "return_regime": snapshot["return_regime"],
                "volatility_regime": snapshot["volatility_regime"],
                "shrinkage": shrinkage,
                "trailing_days": TRAILING_DAYS,
                "covariance": covariance_diagnostics,
                "position_space": {
                    "shared": shared_position,
                    "mixed": mixed_position,
                    "coherent": coherent_position,
                },
                "risk_space": {
                    "shared": shared_risk,
                    "mixed": mixed_risk,
                    "coherent": coherent_risk,
                    "random_book_null": null,
                },
                "position_crosscheck": checks,
            })
        print(f"  risk-space {date}", flush=True)

    summary = {}
    seed = 2026072300
    for shrinkage in SHRINKAGES:
        label = f"shrinkage_{shrinkage:.2f}"
        selected = [row for row in per_date if row["shrinkage"] == shrinkage]
        summary[label] = {
            "trailing_days": TRAILING_DAYS,
            "levels": {},
            "paired_comparisons": {},
        }
        for design in ("shared", "mixed", "coherent"):
            values = np.asarray([row["risk_space"][design] for row in selected])
            summary[label]["levels"][design] = _bootstrap(values, seed)
            seed += 1
        null_values = np.asarray([
            row["risk_space"]["random_book_null"]["mean"] for row in selected
        ])
        summary[label]["levels"]["random_book_null"] = _bootstrap(null_values, seed)
        seed += 1
        for treatment, control in (
            ("mixed", "shared"),
            ("coherent", "shared"),
            ("coherent", "mixed"),
        ):
            differences = np.asarray([
                row["risk_space"][treatment] - row["risk_space"][control]
                for row in selected
            ])
            summary[label]["paired_comparisons"][f"{treatment}_minus_{control}"] = _bootstrap(
                differences, seed
            )
            seed += 1

    flat = []
    for row in per_date:
        flat.append({
            "date": row["date"],
            "return_regime": row["return_regime"],
            "volatility_regime": row["volatility_regime"],
            "shrinkage": row["shrinkage"],
            "shared_position_N_eff": row["position_space"]["shared"],
            "mixed_position_N_eff": row["position_space"]["mixed"],
            "coherent_position_N_eff": row["position_space"]["coherent"],
            "shared_risk_N_eff": row["risk_space"]["shared"],
            "mixed_risk_N_eff": row["risk_space"]["mixed"],
            "coherent_risk_N_eff": row["risk_space"]["coherent"],
            "risk_null_N_eff": row["risk_space"]["random_book_null"]["mean"],
        })

    result = {
        "schema_version": "information-capacity-risk-space-results-v1",
        "provenance": {
            "price_path": str(PRICE_PATH.relative_to(Path(DATA).parent)),
            "price_sha256": _sha256_file(PRICE_PATH),
            "shared_cache_path": str(INTEGRITY_CACHE_PATH.relative_to(Path(DATA).parent)),
            "shared_cache_sha256": _sha256_file(INTEGRITY_CACHE_PATH),
            "split_cache_path": str(SPLIT_CACHE_PATH.relative_to(Path(DATA).parent)),
            "split_cache_sha256": _sha256_file(SPLIT_CACHE_PATH),
            "position_result_path": str(
                POSITION_RESULT_PATH.relative_to(Path(DATA).parent)
            ),
            "position_result_sha256": _sha256_file(POSITION_RESULT_PATH),
        },
        "estimand": (
            "effective directions of unit-gross active books under a PIT covariance inner product"
        ),
        "covariance_specification": {
            "price_path": str(PRICE_PATH.relative_to(Path(DATA).parent)),
            "trailing_days": TRAILING_DAYS,
            "return_type": "daily log return",
            "primary_diagonal_shrinkage": PRIMARY_SHRINKAGE,
            "sensitivity_diagonal_shrinkage": list(SHRINKAGES),
            "future_returns_used": False,
        },
        "n_dates": 34,
        "team_size": TEAM_SIZE,
        "assignment_count_per_split_date": int(len(permutations)),
        "summary": summary,
        "per_date": per_date,
    }
    _atomic_json(RESULT_PATH, result)
    pd.DataFrame(flat).to_csv(SUMMARY_PATH, index=False)

    primary = summary[f"shrinkage_{PRIMARY_SHRINKAGE:.2f}"]
    print("Risk-space result (primary shrinkage)")
    print(
        "  N_eff: "
        f"shared={primary['levels']['shared']['mean']:.3f}, "
        f"mixed={primary['levels']['mixed']['mean']:.3f}, "
        f"coherent={primary['levels']['coherent']['mean']:.3f}, "
        f"null={primary['levels']['random_book_null']['mean']:.3f}"
    )
    for comparison, values in primary["paired_comparisons"].items():
        print(
            f"  {comparison}: {values['mean']:+.3f} "
            f"CI={values['date_bootstrap_95']} positive={values['dates_positive']}/34"
        )
    print(f"wrote {RESULT_PATH} and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
