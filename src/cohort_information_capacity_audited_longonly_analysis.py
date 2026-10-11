"""Analyze the final K=7 long-only implementation experiment."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA, RESULTS
from cohort_crossvendor_hardening import matched_null
from cohort_information_capacity_analysis import METRICS as CAPACITY_METRICS, _metrics
from cohort_information_capacity_audited_longonly import CACHE_PATH, PLAN_PATH
from cohort_information_capacity_integrity import INTEGRITY_CORRECTION_ID, TEAM_SIZE, _load_packets
from cohort_information_capacity_longonly_analysis import (
    RETURN_PRICE_PATH,
    _month_end_forward_returns,
)
from cohort_information_diversity import PILOT_MODELS, _atomic_json, _sign_flip_p


RESULT_PATH = Path(RESULTS) / "information_capacity_audited_longonly_results.json"
SUMMARY_PATH = Path(RESULTS) / "information_capacity_audited_longonly_summary.csv"
BOOTSTRAP_DRAWS = 20_000
DESIGNS = ("shared", "mixed", "coherent")
OPERATING_PORTFOLIOS = ("equal_weight", *DESIGNS)
ALLOCATION_METRICS = (
    "annualized_team_variance",
    "standalone_variance_component",
    "cross_seat_covariance_component",
    "mean_seat_volatility",
    "team_volatility",
    "mean_pairwise_return_correlation",
    "diversification_ratio",
)
HOLDING_METRICS = (
    "holdings_overlap",
    "team_hhi",
    "team_effective_holdings",
    "team_top3_weight",
    "team_max_weight",
    "team_active_share",
    "team_forward_return",
)
ALL_METRICS = CAPACITY_METRICS + HOLDING_METRICS


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _portfolio(row: dict, symbols: list[str]) -> np.ndarray:
    values = {item["symbol"]: float(item["weight"]) for item in row["weights"]}
    portfolio = np.asarray([values[symbol] for symbol in symbols], dtype=float)
    if not np.isfinite(portfolio).all() or float(portfolio.min()) < -1e-10:
        raise ValueError(f"invalid long-only portfolio in {row['date']} {row['vendor']}")
    total = float(portfolio.sum())
    if total <= 0.0:
        raise ValueError("non-positive long-only portfolio")
    return np.maximum(portfolio, 0.0) / total


def _active_book(portfolio: np.ndarray) -> np.ndarray:
    active = portfolio - 1.0 / len(portfolio)
    gross = float(np.abs(active).sum())
    if gross <= 1e-12:
        raise ValueError("uniform response has no active book")
    return active / gross


def _holding_metrics(holdings: np.ndarray, forward: np.ndarray) -> dict:
    pair_overlap = [
        float(np.minimum(holdings[:, left], holdings[:, right]).sum())
        for left, right in itertools.combinations(range(holdings.shape[1]), 2)
    ]
    team = holdings.mean(axis=1)
    hhi = float(np.square(team).sum())
    center = np.full(len(team), 1.0 / len(team))
    return {
        "holdings_overlap": float(np.mean(pair_overlap)),
        "team_hhi": hhi,
        "team_effective_holdings": float(1.0 / hhi),
        "team_top3_weight": float(np.sort(team)[-3:].sum()),
        "team_max_weight": float(team.max()),
        "team_active_share": float(0.5 * np.abs(team - center).sum()),
        "team_forward_return": float(team.dot(forward)),
    }


def _configuration_metrics(
    rows: list[dict],
    symbols: list[str],
    full_consensus: np.ndarray,
    forward: np.ndarray,
    adv: np.ndarray,
    null_mean: float,
) -> dict:
    holdings = np.column_stack([_portfolio(row, symbols) for row in rows])
    active = np.column_stack([
        _active_book(holdings[:, index]) for index in range(holdings.shape[1])
    ])
    return {
        **_metrics(active, full_consensus, forward, adv, null_mean, rows),
        **_holding_metrics(holdings, forward),
    }


def _mean_metrics(rows: list[dict]) -> dict:
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in ALL_METRICS}


def _current(row: dict) -> bool:
    return bool(
        row.get("parse_ok")
        and row.get("long_only_valid")
        and row.get("portfolio_mandate") == "long_only"
        and row.get("integrity_correction_id") == INTEGRITY_CORRECTION_ID
        and row.get("response_id")
        and abs(float(row.get("normalized_weight_sum", 0.0)) - 1.0) <= 1e-10
    )


def _inputs() -> tuple[list[dict], list[dict], list[str]]:
    rows = _load(CACHE_PATH)
    current = [row for row in rows if _current(row)]
    if len(rows) != 34 * 7 * 15 or len(current) != len(rows):
        raise ValueError(f"expected 3570 complete audited long-only rows, found {len(current)}/{len(rows)}")
    if len({row["response_id"] for row in current}) != len(current):
        raise ValueError("response generation IDs are not unique")
    return _load_packets(), current, [vendor for vendor, _ in PILOT_MODELS]


def _date_rows(packet_rows: list[dict], rows: list[dict], vendors: list[str]) -> list[dict]:
    lookup = {
        (row["date"], row["vendor"], row["architecture"], int(row["packet_index"])): row
        for row in rows
    }
    snapshots = sorted(
        (item["snapshot"] for item in packet_rows),
        key=lambda snapshot: snapshot["date"],
    )
    symbols = snapshots[0]["symbols"]
    if any(snapshot["symbols"] != symbols for snapshot in snapshots):
        raise ValueError("long-only snapshots do not share one symbol universe")
    month_end_forward, _ = _month_end_forward_returns(snapshots, symbols)
    forward_by_date = {
        snapshot["date"]: month_end_forward[index]
        for index, snapshot in enumerate(snapshots)
    }
    null_mean = matched_null(
        TEAM_SIZE,
        [f"A{index}" for index in range(14)],
        B=10_000,
        seed=2026071700 + TEAM_SIZE,
    )["mean"]
    assignments = list(itertools.permutations(range(1, TEAM_SIZE + 1)))
    output = []
    for item in packet_rows:
        snapshot = item["snapshot"]
        date = snapshot["date"]
        symbols = snapshot["symbols"]
        forward = np.asarray(forward_by_date[date], dtype=float)
        adv = np.asarray([snapshot["features"]["market"][symbol]["adv_bn"] for symbol in symbols])
        shared_rows = [lookup[date, vendor, "shared", 0] for vendor in vendors]
        shared_holdings = np.column_stack([_portfolio(row, symbols) for row in shared_rows])
        full_consensus = np.mean(
            [_active_book(shared_holdings[:, index]) for index in range(TEAM_SIZE)],
            axis=0,
        )
        date_result = {
            "sample": item["sample"],
            "snapshot_id": snapshot["snapshot_id"],
            "date": date,
            "return_regime": snapshot["return_regime"],
            "volatility_regime": snapshot["volatility_regime"],
            "team_size": TEAM_SIZE,
            "shared": _configuration_metrics(
                shared_rows, symbols, full_consensus, forward, adv, null_mean
            ),
            "configuration_counts": {"shared": 1},
        }
        for label, architecture in (
            ("mixed", "mixed_audited"),
            ("coherent", "coherent_audited"),
        ):
            configurations = []
            for assignment in assignments:
                configuration = [
                    lookup[date, vendor, architecture, packet]
                    for vendor, packet in zip(vendors, assignment)
                ]
                configurations.append(_configuration_metrics(
                    configuration, symbols, full_consensus, forward, adv, null_mean
                ))
            date_result[label] = _mean_metrics(configurations)
            date_result["configuration_counts"][label] = len(configurations)
        output.append(date_result)
        print(f"  analyzed audited long-only {date}", flush=True)
    return output


def _bootstrap(values: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "date_bootstrap_95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "n_dates": len(values),
    }


def _paired(rows: list[dict], treatment: str, control: str, metric: str, seed: int) -> dict:
    values = np.asarray([
        row[treatment][metric] - row[control][metric] for row in rows
    ], dtype=float)
    return {
        **_bootstrap(values, seed),
        "treatment": treatment,
        "control": control,
        "metric": metric,
        "treatment_mean": float(np.mean([row[treatment][metric] for row in rows])),
        "control_mean": float(np.mean([row[control][metric] for row in rows])),
        "dates_positive": int(np.sum(values > 0)),
        "dates_negative": int(np.sum(values < 0)),
        "sign_flip_p_two_sided": _sign_flip_p(values),
    }


def _summaries(date_rows: list[dict]) -> dict:
    seed = 2026072180
    levels = {}
    for design in DESIGNS:
        levels[design] = {}
        for metric in ALL_METRICS:
            levels[design][metric] = _bootstrap(
                np.asarray([row[design][metric] for row in date_rows]), seed
            )
            seed += 1
    comparisons = {}
    for treatment, control in (
        ("mixed", "shared"),
        ("coherent", "shared"),
        ("coherent", "mixed"),
    ):
        label = f"{treatment}_minus_{control}"
        comparisons[label] = {}
        for metric in ALL_METRICS:
            comparisons[label][metric] = _paired(
                date_rows, treatment, control, metric, seed
            )
            seed += 1
    return {"levels": levels, "paired_comparisons": comparisons}


def _path_portfolios(
    packet_rows: list[dict], rows: list[dict], vendors: list[str]
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[dict],
    np.ndarray,
    np.ndarray,
]:
    snapshots = sorted((item["snapshot"] for item in packet_rows), key=lambda row: row["date"])
    dates = [snapshot["date"] for snapshot in snapshots]
    symbols = snapshots[0]["symbols"]
    forward, windows = _month_end_forward_returns(snapshots, symbols)
    lookup = {
        (row["date"], row["vendor"], row["architecture"], int(row["packet_index"])): row
        for row in rows
    }
    assignments = np.asarray(list(itertools.permutations(range(1, 8))), dtype=int) - 1
    vendor_indices = np.arange(7)[None, :]
    shared_paths = []
    mixed_paths = []
    coherent_paths = []
    shared_seat_returns = []
    mixed_seat_returns = []
    coherent_seat_returns = []
    for date_index, date in enumerate(dates):
        shared_holdings = np.asarray([
            _portfolio(lookup[date, vendor, "shared", 0], symbols) for vendor in vendors
        ])
        shared_paths.append(shared_holdings.mean(axis=0))
        shared_seat_returns.append(shared_holdings @ forward[date_index])
        for target, seat_target, architecture in (
            (mixed_paths, mixed_seat_returns, "mixed_audited"),
            (coherent_paths, coherent_seat_returns, "coherent_audited"),
        ):
            cube = np.asarray([
                [
                    _portfolio(lookup[date, vendor, architecture, packet], symbols)
                    for packet in range(1, 8)
                ]
                for vendor in vendors
            ])
            selected = cube[vendor_indices, assignments]
            target.append(selected.mean(axis=1))
            seat_target.append(
                np.einsum("pka,a->pk", selected, forward[date_index])
            )
    risk_free = np.asarray([
        snapshot["features"]["macro"]["three_month_yield"] / 1200.0
        for snapshot in snapshots
    ])
    paths = {
        "equal_weight": np.full((1, len(dates), len(symbols)), 1.0 / len(symbols)),
        "shared": np.asarray(shared_paths)[None, :, :],
        "mixed": np.stack(mixed_paths, axis=1),
        "coherent": np.stack(coherent_paths, axis=1),
    }
    seat_returns = {
        "shared": np.asarray(shared_seat_returns)[None, :, :],
        "mixed": np.stack(mixed_seat_returns, axis=1),
        "coherent": np.stack(coherent_seat_returns, axis=1),
    }
    return paths, seat_returns, windows, forward, risk_free


def _allocation_path_metrics(returns: np.ndarray) -> dict[str, np.ndarray]:
    centered = returns - returns.mean(axis=1, keepdims=True)
    covariance = (
        np.einsum("pti,ptj->pij", centered, centered)
        * 12.0
        / (returns.shape[1] - 1)
    )
    diagonal = np.diagonal(covariance, axis1=1, axis2=2)
    team_size = returns.shape[2]
    standalone = diagonal.sum(axis=1) / team_size**2
    total = covariance.sum(axis=(1, 2)) / team_size**2
    cross = total - standalone
    denominator = np.sqrt(diagonal[:, :, None] * diagonal[:, None, :])
    correlation = covariance / denominator
    upper = np.triu_indices(team_size, 1)
    seat_volatility = np.sqrt(diagonal)
    return {
        "annualized_team_variance": total,
        "standalone_variance_component": standalone,
        "cross_seat_covariance_component": cross,
        "mean_seat_volatility": seat_volatility.mean(axis=1),
        "team_volatility": np.sqrt(total),
        "mean_pairwise_return_correlation": correlation[:, upper[0], upper[1]].mean(axis=1),
        "diversification_ratio": seat_volatility.mean(axis=1) / np.sqrt(total),
    }


def _allocation_decomposition(seat_returns: dict[str, np.ndarray]) -> dict:
    path_values = {
        design: _allocation_path_metrics(seat_returns[design])
        for design in DESIGNS
    }
    levels = {}
    for design in DESIGNS:
        levels[design] = {
            metric: {
                "mean": float(path_values[design][metric].mean()),
                "assignment_95": [
                    float(value)
                    for value in np.quantile(path_values[design][metric], [0.025, 0.975])
                ],
            }
            for metric in ALLOCATION_METRICS
        }
        levels[design]["n_fixed_assignments"] = int(
            len(path_values[design]["annualized_team_variance"])
        )

    comparisons = {}
    shared = levels["shared"]
    for design in ("mixed", "coherent"):
        total_reduction = (
            shared["annualized_team_variance"]["mean"]
            - levels[design]["annualized_team_variance"]["mean"]
        )
        standalone_reduction = (
            shared["standalone_variance_component"]["mean"]
            - levels[design]["standalone_variance_component"]["mean"]
        )
        covariance_reduction = (
            shared["cross_seat_covariance_component"]["mean"]
            - levels[design]["cross_seat_covariance_component"]["mean"]
        )
        comparisons[f"{design}_minus_shared"] = {
            "team_variance_reduction": total_reduction,
            "team_variance_reduction_fraction": (
                total_reduction / shared["annualized_team_variance"]["mean"]
            ),
            "standalone_variance_reduction": standalone_reduction,
            "cross_seat_covariance_reduction": covariance_reduction,
            "covariance_share_of_variance_reduction": (
                covariance_reduction / total_reduction
            ),
            "pairwise_return_correlation_reduction": (
                shared["mean_pairwise_return_correlation"]["mean"]
                - levels[design]["mean_pairwise_return_correlation"]["mean"]
            ),
        }
    return {
        "identity": (
            "Var(equal-weight team return) = sum_i Var(r_i)/K^2 "
            "+ 2 sum_{i<j} Cov(r_i,r_j)/K^2"
        ),
        "return_frequency": "34 nonoverlapping monthly holding windows",
        "annualization_factor": 12,
        "levels": levels,
        "comparisons": comparisons,
    }


def _operating_performance(
    packet_rows: list[dict], rows: list[dict], vendors: list[str]
) -> dict:
    paths, seat_returns, windows, forward, risk_free = _path_portfolios(
        packet_rows, rows, vendors
    )
    levels = {}
    path_values = {}
    for design, portfolios in paths.items():
        total = np.einsum("pta,ta->pt", portfolios, forward)
        target_turnover = 0.5 * np.abs(np.diff(portfolios, axis=1)).sum(axis=2).mean(axis=1)
        drifted = (
            portfolios[:, :-1, :]
            * (1.0 + forward[:-1, :])[None, :, :]
            / (1.0 + total[:, :-1])[:, :, None]
        )
        rebalance_turnover = 0.5 * np.abs(
            portfolios[:, 1:, :] - drifted
        ).sum(axis=2).mean(axis=1)
        metrics = []
        for returns, target_turn, rebalance_turn in zip(
            total, target_turnover, rebalance_turnover
        ):
            volatility = float(returns.std(ddof=1))
            wealth = np.cumprod(1.0 + returns)
            peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
            metrics.append({
                "annualized_total_return": float(returns.mean() * 12.0),
                "annualized_total_volatility": float(volatility * np.sqrt(12.0)),
                "sharpe_ratio": float(
                    (returns - risk_free).mean() / volatility * np.sqrt(12.0)
                ),
                "compound_annual_growth": float(
                    np.prod(1.0 + returns) ** (12.0 / len(returns)) - 1.0
                ),
                "maximum_drawdown": float(np.min(wealth / peak - 1.0)),
                "mean_monthly_target_turnover": float(target_turn),
                "mean_monthly_rebalance_turnover": float(rebalance_turn),
            })
        path_values[design] = metrics
        level = {}
        for metric in (
            "annualized_total_return", "annualized_total_volatility", "sharpe_ratio",
            "compound_annual_growth", "maximum_drawdown",
            "mean_monthly_target_turnover", "mean_monthly_rebalance_turnover",
        ):
            values = np.asarray([row[metric] for row in metrics])
            level[metric] = {
                "mean": float(values.mean()),
                "assignment_95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
            }
        level["n_fixed_assignments"] = len(metrics)
        levels[design] = level

    net_return_sensitivity = {}
    for cost_bps in (10, 25, 50):
        label = str(cost_bps)
        net_return_sensitivity[label] = {}
        for design in OPERATING_PORTFOLIOS:
            values = np.asarray([
                row["annualized_total_return"]
                - 12.0 * row["mean_monthly_rebalance_turnover"] * cost_bps / 10_000.0
                for row in path_values[design]
            ])
            net_return_sensitivity[label][design] = {
                "mean": float(values.mean()),
                "assignment_95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
            }
    break_even = {}
    for design in ("mixed", "coherent"):
        gross_shortfall = (
            levels["shared"]["annualized_total_return"]["mean"]
            - levels[design]["annualized_total_return"]["mean"]
        )
        annual_turnover_saving = 12.0 * (
            levels["shared"]["mean_monthly_rebalance_turnover"]["mean"]
            - levels[design]["mean_monthly_rebalance_turnover"]["mean"]
        )
        break_even[design] = float(gross_shortfall / annual_turnover_saving * 10_000.0)
    return {
        "team_size": TEAM_SIZE,
        "n_dates": len(packet_rows),
        "evaluation_windows": windows,
        "entry_rule": "first available close strictly after the signal date",
        "exit_rule": "last available close on or before the next calendar month-end",
        "assignment_rule": "one model-to-information-seat assignment is fixed across all dates",
        "active_book_centering": "1/14 within-universe weight; geometric normalization only",
        "comparison_control": "shared access over identical holding windows",
        "equal_weight_reference": "constant 1/14 target weights over the same holding windows",
        "turnover_definition": "half the absolute trade from return-drifted end weights to next-month target weights",
        "levels": levels,
        "allocation_decomposition": _allocation_decomposition(seat_returns),
        "net_return_sensitivity": net_return_sensitivity,
        "break_even_one_way_cost_bps_vs_shared": break_even,
    }


def _collection_diagnostics(rows: list[dict]) -> dict:
    retries = [row for row in rows if int(row.get("format_retry_count", 0)) > 0]
    return {
        "rows": len(rows),
        "response_ids_unique": len({row["response_id"] for row in rows}),
        "response_id_coverage": sum(bool(row.get("response_id")) for row in rows) / len(rows),
        "response_provider_coverage": sum(bool(row.get("response_provider")) for row in rows) / len(rows),
        "system_fingerprint_coverage": sum(bool(row.get("system_fingerprint")) for row in rows) / len(rows),
        "seed_supported_rows": sum(bool(row.get("seed_supported")) for row in rows),
        "retried_rows": len(retries),
        "retry_rate": len(retries) / len(rows),
        "reported_cost": float(sum(row.get("usage", {}).get("cost", 0.0) or 0.0 for row in rows)),
        "terminal_failures": 0,
    }


def _flat_rows(date_rows: list[dict]) -> list[dict]:
    output = []
    for row in date_rows:
        common = {key: row[key] for key in (
            "sample", "snapshot_id", "date", "return_regime", "volatility_regime", "team_size"
        )}
        for design in DESIGNS:
            output.append({**common, "design": design, **row[design]})
    return output


def main() -> None:
    packet_rows, rows, vendors = _inputs()
    date_rows = _date_rows(packet_rows, rows, vendors)
    result = {
        "schema_version": "information-capacity-audited-long-only-results-v1",
        "integrity_correction_id": INTEGRITY_CORRECTION_ID,
        "design_status": "post-audit long-only mandate-transfer rerun; pooled 34-date panel",
        "portfolio": "fully invested long-only",
        "analysis_representation": "within-universe centered active books; shared-access operating control",
        "team_size": TEAM_SIZE,
        "n_dates": len(date_rows),
        "plan_path": str(PLAN_PATH.relative_to(Path(DATA).parent)),
        "collection_diagnostics": _collection_diagnostics(rows),
        "per_date": date_rows,
        **_summaries(date_rows),
        "operating_performance": _operating_performance(packet_rows, rows, vendors),
    }
    _atomic_json(RESULT_PATH, result)
    pd.DataFrame(_flat_rows(date_rows)).to_csv(SUMMARY_PATH, index=False)
    print("Audit-corrected K=7 long-only result")
    for design in DESIGNS:
        level = result["levels"][design]
        print(
            f"  {design}: N_eff={level['N_eff']['mean']:.3f}, "
            f"active overlap={level['same_side_overlap']['mean']:.3f}, "
            f"holdings overlap={level['holdings_overlap']['mean']:.3f}, "
            f"Q={level['convex_order_concentration']['mean']:.3f}"
        )
    for comparison in ("mixed_minus_shared", "coherent_minus_shared", "coherent_minus_mixed"):
        values = result["paired_comparisons"][comparison]
        print(
            f"  {comparison}: dN={values['N_eff']['mean']:+.3f} "
            f"CI={values['N_eff']['date_bootstrap_95']}; "
            f"dO={values['same_side_overlap']['mean']:+.3f}"
        )
    performance = result["operating_performance"]["levels"]
    for design in DESIGNS:
        print(
            f"  {design} operating: return="
            f"{performance[design]['annualized_total_return']['mean']:.3%}, "
            f"vol={performance[design]['annualized_total_volatility']['mean']:.3%}, "
            f"Sharpe={performance[design]['sharpe_ratio']['mean']:.3f}, "
            f"turnover={performance[design]['mean_monthly_rebalance_turnover']['mean']:.3%}"
        )
    reference = performance["equal_weight"]
    print(
        "  equal_weight operating: return="
        f"{reference['annualized_total_return']['mean']:.3%}, "
        f"vol={reference['annualized_total_volatility']['mean']:.3%}, "
        f"Sharpe={reference['sharpe_ratio']['mean']:.3f}, "
        f"turnover={reference['mean_monthly_rebalance_turnover']['mean']:.3%}"
    )
    allocation = result["operating_performance"]["allocation_decomposition"]
    for design in ("mixed", "coherent"):
        comparison = allocation["comparisons"][f"{design}_minus_shared"]
        print(
            f"  {design} allocation: variance reduction="
            f"{comparison['team_variance_reduction_fraction']:.1%}; "
            f"covariance share={comparison['covariance_share_of_variance_reduction']:.1%}"
        )
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
