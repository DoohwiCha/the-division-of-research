"""Analyze the mandate-matched long-only Information-Capacity Curve.

The geometry follows the signed-book primary experiment exactly: every model
subset and packet-to-model permutation is averaged within a date, and the date
is the inference unit.  Long-only holdings are converted to benchmark-relative
active weights before independence and overlap are measured.  Raw holdings are
retained for descriptive return, tracking-error, active-share, and turnover
calculations.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA, RESULTS
from cohort_crossvendor_hardening import matched_null
from cohort_information_capacity_analysis import (
    BOOTSTRAP_DRAWS,
    METRICS as CAPACITY_METRICS,
    _metrics as capacity_metrics,
)
from cohort_information_capacity_experiment import ARCHITECTURES, TEAM_SIZES
from cohort_information_capacity_longonly import CACHE_PATH, PLAN_PATH
from cohort_information_diversity import PILOT_MODELS, _atomic_json, _sign_flip_p
from cohort_information_diversity_longonly import (
    CACHE_PATH as SHARED_LONGONLY_CACHE_PATH,
    _strict_complete,
)
from cohort_information_diversity_random_placebo import load_packets


RESULT_PATH = Path(RESULTS) / "information_capacity_longonly_results.json"
SUMMARY_PATH = Path(RESULTS) / "information_capacity_longonly_summary.csv"
RETURN_PRICE_PATH = Path(DATA) / "jfs_global_prices_united_states.csv"

OPERATING_METRICS = (
    "team_active_share",
    "team_max_weight",
    "active_return",
    "total_return",
)
METRICS = CAPACITY_METRICS + OPERATING_METRICS
PERFORMANCE_METRICS = (
    "annualized_active_return",
    "annualized_tracking_error",
    "information_ratio",
    "annualized_total_return",
    "annualized_total_volatility",
    "sharpe_ratio",
)
TRANSACTION_COST_SCENARIOS_BPS = (10, 25, 50)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _job_key(row: dict) -> tuple:
    return row["snapshot_id"], row["condition"], row["vendor"]


def _portfolio(row: dict, symbols: list[str]) -> np.ndarray:
    raw = {item["symbol"]: float(item["weight"]) for item in row["weights"]}
    vector = np.asarray([raw.get(symbol, np.nan) for symbol in symbols], dtype=float)
    if not np.isfinite(vector).all() or (vector < -1e-12).any():
        raise ValueError(f"invalid long-only weights in {_job_key(row)}")
    total = float(vector.sum())
    if total <= 0:
        raise ValueError(f"non-positive long-only gross in {_job_key(row)}")
    vector = np.maximum(vector, 0.0) / total
    if not np.isclose(vector.sum(), 1.0, atol=1e-10):
        raise AssertionError("long-only normalization failed")
    return vector


def _active_book(portfolio: np.ndarray) -> np.ndarray:
    raw = portfolio - 1.0 / len(portfolio)
    gross = float(np.abs(raw).sum())
    if gross <= 1e-12:
        raise ValueError("equal-weight response has no active book")
    return raw / gross


def _configuration_metrics(
    rows: list[dict],
    symbols: list[str],
    full_consensus: np.ndarray,
    forward: np.ndarray,
    adv: np.ndarray,
    null_mean: float,
) -> dict:
    holdings = np.column_stack([_portfolio(row, symbols) for row in rows])
    active = np.column_stack([_active_book(holdings[:, index]) for index in range(holdings.shape[1])])
    result = capacity_metrics(active, full_consensus, forward, adv, null_mean, rows)
    team = holdings.mean(axis=1)
    benchmark = np.full(len(symbols), 1.0 / len(symbols))
    team_active = team - benchmark
    result.update({
        "team_active_share": float(0.5 * np.abs(team_active).sum()),
        "team_max_weight": float(team.max()),
        "active_return": float(team_active.dot(forward)),
        "total_return": float(team.dot(forward)),
    })
    return result


def _mean_metrics(rows: list[dict]) -> dict:
    return {metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}


def _selected(rows: list[dict], sample: str) -> list[dict]:
    return rows if sample == "all" else [row for row in rows if row.get("sample") == sample]


def _load_inputs(sample: str) -> tuple[list[dict], list[dict], list[dict], dict]:
    packets = _selected(load_packets(), sample)
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"{CACHE_PATH} is missing")
    partitioned = _selected(_load(CACHE_PATH), sample)
    incomplete = [row for row in partitioned if not _strict_complete(row)]
    if incomplete:
        raise ValueError(f"long-only capacity cache has {len(incomplete)} incomplete rows")
    shared = _selected([
        row for row in _load(SHARED_LONGONLY_CACHE_PATH)
        if row.get("condition") == "rich" and int(row.get("replicate", 0)) == 0
    ], sample)
    incomplete_shared = [row for row in shared if not _strict_complete(row)]
    if incomplete_shared:
        raise ValueError(f"shared long-only cache has {len(incomplete_shared)} incomplete rows")
    plan = _load(PLAN_PATH)
    expected_dates = len(packets)
    expected_partitioned = expected_dates * 378
    expected_shared = expected_dates * len(PILOT_MODELS)
    if len(partitioned) != expected_partitioned:
        raise ValueError(
            f"sample={sample}: expected {expected_partitioned} partitioned rows, "
            f"found {len(partitioned)}"
        )
    if len(shared) != expected_shared:
        raise ValueError(
            f"sample={sample}: expected {expected_shared} shared rows, found {len(shared)}"
        )
    return packets, partitioned, shared, plan


def _date_rows(sample: str) -> tuple[list[dict], dict]:
    packet_rows, partitioned, shared, plan = _load_inputs(sample)
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    lookup = {(row["date"], row["vendor"], row["condition"]): row for row in partitioned}
    shared_lookup = {(row["date"], row["vendor"]): row for row in shared}
    null_means = {
        k: matched_null(k, [f"A{index}" for index in range(14)], B=10_000, seed=2026071700 + k)["mean"]
        for k in TEAM_SIZES
    }
    output = []
    for item in packet_rows:
        snapshot = item["snapshot"]
        date = snapshot["date"]
        symbols = snapshot["symbols"]
        forward = np.asarray([snapshot["forward_return_21d"][symbol] for symbol in symbols])
        adv = np.asarray([snapshot["features"]["market"][symbol]["adv_bn"] for symbol in symbols])
        shared_rows = {vendor: shared_lookup[date, vendor] for vendor in vendors}
        full_consensus = np.mean(
            [_active_book(_portfolio(shared_rows[vendor], symbols)) for vendor in vendors],
            axis=0,
        )
        for k in TEAM_SIZES:
            shared_configurations = [
                _configuration_metrics(
                    [shared_rows[vendor] for vendor in combination],
                    symbols, full_consensus, forward, adv, null_means[k],
                )
                for combination in itertools.combinations(vendors, k)
            ]
            row = {
                "sample": item["sample"],
                "snapshot_id": snapshot["snapshot_id"],
                "date": date,
                "return_regime": snapshot["return_regime"],
                "volatility_regime": snapshot["volatility_regime"],
                "team_size": k,
                "shared": _mean_metrics(shared_configurations),
                "shared_configurations": len(shared_configurations),
            }
            for architecture in ARCHITECTURES:
                configurations = []
                for combination in itertools.combinations(vendors, k):
                    for assignment in itertools.permutations(range(1, k + 1)):
                        rows = [
                            lookup[
                                date,
                                vendor,
                                f"capacity_longonly_{architecture}_k{k}_p{packet}",
                            ]
                            for vendor, packet in zip(combination, assignment)
                        ]
                        configurations.append(
                            _configuration_metrics(
                                rows, symbols, full_consensus, forward, adv, null_means[k]
                            )
                        )
                row[architecture] = _mean_metrics(configurations)
                row[f"{architecture}_configurations"] = len(configurations)
            output.append(row)
            print(f"  analyzed long-only {date} K={k}", flush=True)
    return output, plan


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


def _capacity_summaries(date_rows: list[dict]) -> dict:
    levels = {}
    comparisons = {}
    seed = 2026072100
    for k in TEAM_SIZES:
        rows = [row for row in date_rows if row["team_size"] == k]
        levels[str(k)] = {}
        for design in ("shared", "mixed", "coherent"):
            levels[str(k)][design] = {}
            for metric in METRICS:
                levels[str(k)][design][metric] = _bootstrap(
                    np.asarray([row[design][metric] for row in rows]), seed
                )
                seed += 1
        comparisons[str(k)] = {}
        for treatment, control in (
            ("mixed", "shared"),
            ("coherent", "mixed"),
            ("coherent", "shared"),
        ):
            label = f"{treatment}_minus_{control}"
            comparisons[str(k)][label] = {}
            for metric in METRICS:
                comparisons[str(k)][label][metric] = _paired(
                    rows, treatment, control, metric, seed
                )
                seed += 1
    return {"levels": levels, "paired_comparisons": comparisons}


def _performance_metrics(active: np.ndarray, total: np.ndarray, risk_free: np.ndarray) -> dict:
    active_sd = float(active.std(ddof=1))
    total_sd = float(total.std(ddof=1))
    wealth = np.cumprod(1.0 + total)
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    return {
        "annualized_active_return": float(active.mean() * 12.0),
        "annualized_tracking_error": float(active_sd * np.sqrt(12.0)),
        "information_ratio": float(active.mean() / active_sd * np.sqrt(12.0)),
        "annualized_total_return": float(total.mean() * 12.0),
        "annualized_total_volatility": float(total_sd * np.sqrt(12.0)),
        "sharpe_ratio": float((total - risk_free).mean() / total_sd * np.sqrt(12.0)),
        "compound_annual_growth": float(np.prod(1.0 + total) ** (12.0 / len(total)) - 1.0),
        "maximum_drawdown": float(np.min(wealth / peak - 1.0)),
        "active_return_hit_rate": float(np.mean(active > 0.0)),
    }


def _month_end_forward_returns(
    snapshots: list[dict], symbols: list[str]
) -> tuple[np.ndarray, list[dict[str, str]]]:
    prices = pd.read_csv(RETURN_PRICE_PATH, index_col=0, parse_dates=True).sort_index()
    values = []
    windows = []
    for snapshot in snapshots:
        decision_date = pd.Timestamp(snapshot["date"])
        start_candidates = prices.index[prices.index > decision_date]
        target_month_end = decision_date + pd.offsets.MonthEnd(1)
        end_candidates = prices.index[prices.index <= target_month_end]
        if len(start_candidates) == 0 or len(end_candidates) == 0:
            raise ValueError(f"missing month-end return observations at {snapshot['date']}")
        start = start_candidates[0]
        end = end_candidates[-1]
        if end <= start:
            raise ValueError(f"non-positive post-signal holding window at {snapshot['date']}")
        values.append(
            (prices.loc[end, symbols] / prices.loc[start, symbols] - 1.0).to_numpy(float)
        )
        windows.append({
            "decision_date": snapshot["date"],
            "start_session": start.date().isoformat(),
            "end_session": end.date().isoformat(),
        })
    ordered = sorted(windows, key=lambda row: row["decision_date"])
    for left, right in zip(ordered, ordered[1:]):
        if left["end_session"] > right["start_session"]:
            raise AssertionError(f"overlapping month-end windows: {left} and {right}")
    return np.asarray(values), windows


def _bootstrap_performance(
    active: np.ndarray,
    total: np.ndarray,
    risk_free: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    active_draws = active[indices]
    total_draws = total[indices]
    risk_free_draws = risk_free[indices]
    active_sd = active_draws.std(axis=1, ddof=1)
    total_sd = total_draws.std(axis=1, ddof=1)
    return {
        "annualized_active_return": active_draws.mean(axis=1) * 12.0,
        "annualized_tracking_error": active_sd * np.sqrt(12.0),
        "information_ratio": np.divide(
            active_draws.mean(axis=1) * np.sqrt(12.0),
            active_sd,
            out=np.full_like(active_sd, np.nan),
            where=active_sd > 0.0,
        ),
        "annualized_total_return": total_draws.mean(axis=1) * 12.0,
        "annualized_total_volatility": total_sd * np.sqrt(12.0),
        "sharpe_ratio": np.divide(
            (total_draws - risk_free_draws).mean(axis=1) * np.sqrt(12.0),
            total_sd,
            out=np.full_like(total_sd, np.nan),
            where=total_sd > 0.0,
        ),
    }


def _performance_summary(date_rows: list[dict], packet_rows: list[dict]) -> dict:
    snapshots = {item["snapshot"]["date"]: item["snapshot"] for item in packet_rows}
    output = {}
    seed = 2026072600
    for k in TEAM_SIZES:
        rows = sorted(
            (row for row in date_rows if row["team_size"] == k),
            key=lambda row: row["date"],
        )
        risk_free = np.asarray([
            snapshots[row["date"]]["features"]["macro"]["three_month_yield"] / 1200.0
            for row in rows
        ])
        series = {
            design: {
                "active": np.asarray([row[design]["active_return"] for row in rows]),
                "total": np.asarray([row[design]["total_return"] for row in rows]),
            }
            for design in ("shared", "mixed", "coherent")
        }
        levels = {
            design: _performance_metrics(values["active"], values["total"], risk_free)
            for design, values in series.items()
        }
        comparisons = {}
        rng = np.random.default_rng(seed + k)
        indices = rng.integers(0, len(rows), size=(BOOTSTRAP_DRAWS, len(rows)))
        bootstrap = {
            design: _bootstrap_performance(
                values["active"], values["total"], risk_free, indices
            )
            for design, values in series.items()
        }
        for treatment, control in (
            ("mixed", "shared"),
            ("coherent", "mixed"),
            ("coherent", "shared"),
        ):
            label = f"{treatment}_minus_{control}"
            comparisons[label] = {}
            point_t = levels[treatment]
            point_c = levels[control]
            for metric in PERFORMANCE_METRICS:
                delta = bootstrap[treatment][metric] - bootstrap[control][metric]
                comparisons[label][metric] = {
                    "mean_difference": float(point_t[metric] - point_c[metric]),
                    "date_bootstrap_95": [
                        float(value) for value in np.quantile(delta, [0.025, 0.975])
                    ],
                }
        output[str(k)] = {
            "frequency": f"{len(rows)} post-signal monthly forward-return windows",
            "annualization": "arithmetic monthly mean and standard deviation annualized by 12 and sqrt(12)",
            "risk_free_rate": "point-in-time 3-month Treasury yield divided by 12",
            "levels": levels,
            "comparisons": comparisons,
        }
    return output


def _fixed_assignment_performance(sample: str) -> dict:
    """Evaluate implementable K=7 paths from the next session to month-end.

    The date-level capacity estimator averages packet-to-model permutations.
    That averaging is harmless for mean return, which is linear, but it would
    understate path-dependent quantities such as volatility and drawdown.  For
    operating outcomes, each of the 7! packet assignments is therefore held
    fixed across all dates before path statistics are computed.
    """
    packet_rows, partitioned, shared, _plan = _load_inputs(sample)
    snapshots = sorted((item["snapshot"] for item in packet_rows), key=lambda row: row["date"])
    dates = [row["date"] for row in snapshots]
    symbols = snapshots[0]["symbols"]
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    forward, evaluation_windows = _month_end_forward_returns(snapshots, symbols)
    benchmark = forward.mean(axis=1)
    risk_free = np.asarray([
        snapshot["features"]["macro"]["three_month_yield"] / 1200.0
        for snapshot in snapshots
    ])
    shared_lookup = {(row["date"], row["vendor"]): row for row in shared}
    split_lookup = {
        (row["date"], row["vendor"], row["architecture"], int(row["packet_index"])): row
        for row in partitioned
        if int(row["team_size"]) == 7
    }

    targets: dict[str, np.ndarray] = {
        "shared": np.asarray([[
            np.mean(
                [_portfolio(shared_lookup[date, vendor], symbols) for vendor in vendors],
                axis=0,
            )
            for date in dates
        ]])
    }
    assignments = list(itertools.permutations(range(1, 8)))
    for architecture in ARCHITECTURES:
        paths = []
        for assignment in assignments:
            paths.append([
                np.mean([
                    _portfolio(
                        split_lookup[date, vendor, architecture, packet_index], symbols
                    )
                    for vendor, packet_index in zip(vendors, assignment)
                ], axis=0)
                for date in dates
            ])
        targets[architecture] = np.asarray(paths)

    levels = {}
    for design, paths in targets.items():
        total_returns = np.einsum("cta,ta->ct", paths, forward)
        active_returns = total_returns - benchmark[None, :]
        path_metrics = [
            _performance_metrics(active, total, risk_free)
            for active, total in zip(active_returns, total_returns)
        ]
        levels[design] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in path_metrics])),
                "assignment_95": [
                    float(value) for value in np.quantile(
                        [row[metric] for row in path_metrics], [0.025, 0.975]
                    )
                ],
            }
            for metric in (*PERFORMANCE_METRICS, "compound_annual_growth", "maximum_drawdown", "active_return_hit_rate")
        }
        levels[design]["n_fixed_assignments"] = len(path_metrics)

    return {
        "team_size": 7,
        "n_dates": len(dates),
        "frequency": f"{len(dates)} next-session-to-next-month-end forward-return windows",
        "entry_rule": "first available close strictly after the signal date",
        "exit_rule": "last available close on or before the next calendar month-end",
        "evaluation_windows": evaluation_windows,
        "assignment_rule": "one model-to-packet assignment is held fixed across all dates",
        "annualization": "arithmetic window mean and standard deviation annualized by 12 and sqrt(12)",
        "returns_are_gross_of_trading_costs": True,
        "levels": levels,
    }


def _turnover_summary(sample: str) -> dict:
    packet_rows, partitioned, shared, _plan = _load_inputs(sample)
    snapshots = sorted((item["snapshot"] for item in packet_rows), key=lambda row: row["date"])
    dates = [row["date"] for row in snapshots]
    symbols = snapshots[0]["symbols"]
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    vendor_index = {vendor: index for index, vendor in enumerate(vendors)}
    shared_lookup = {(row["date"], row["vendor"]): row for row in shared}
    split_lookup = {(row["date"], row["vendor"], row["condition"]): row for row in partitioned}
    output = {}
    for k in TEAM_SIZES:
        combinations = list(itertools.combinations(vendors, k))
        shared_indices = np.asarray([
            [vendor_index[vendor] for vendor in combination] for combination in combinations
        ])
        shared_by_date = {}
        for date in dates:
            matrix = np.asarray([_portfolio(shared_lookup[date, vendor], symbols) for vendor in vendors])
            shared_by_date[date] = matrix[shared_indices].mean(axis=1)
        design_transitions = {"shared": []}
        for left, right in zip(dates[:-1], dates[1:]):
            values = 0.5 * np.abs(shared_by_date[right] - shared_by_date[left]).sum(axis=1)
            design_transitions["shared"].append(float(values.mean()))

        for architecture in ARCHITECTURES:
            vendor_rows = []
            packet_rows_index = []
            for combination in combinations:
                for assignment in itertools.permutations(range(1, k + 1)):
                    vendor_rows.append([vendor_index[vendor] for vendor in combination])
                    packet_rows_index.append([packet - 1 for packet in assignment])
            vendor_rows_array = np.asarray(vendor_rows)
            packet_rows_array = np.asarray(packet_rows_index)
            portfolios = {}
            for date in dates:
                cube = np.empty((len(vendors), k, len(symbols)), dtype=float)
                for vendor_position, vendor in enumerate(vendors):
                    for packet in range(1, k + 1):
                        condition = f"capacity_longonly_{architecture}_k{k}_p{packet}"
                        cube[vendor_position, packet - 1] = _portfolio(
                            split_lookup[date, vendor, condition], symbols
                        )
                selected = cube[vendor_rows_array, packet_rows_array]
                portfolios[date] = selected.mean(axis=1)
            design_transitions[architecture] = []
            for left, right in zip(dates[:-1], dates[1:]):
                values = 0.5 * np.abs(portfolios[right] - portfolios[left]).sum(axis=1)
                design_transitions[architecture].append(float(values.mean()))

        levels = {
            design: {
                "mean_monthly_turnover": float(np.mean(values)),
                "median_monthly_turnover": float(np.median(values)),
            }
            for design, values in design_transitions.items()
        }
        comparisons = {}
        for treatment, control in (
            ("mixed", "shared"),
            ("coherent", "mixed"),
            ("coherent", "shared"),
        ):
            delta = np.asarray(design_transitions[treatment]) - np.asarray(design_transitions[control])
            comparisons[f"{treatment}_minus_{control}"] = {
                "mean_difference": float(delta.mean()),
                "transitions_negative": int(np.sum(delta < 0)),
                "transitions_positive": int(np.sum(delta > 0)),
                "n_transitions": len(delta),
            }
        output[str(k)] = {
            "definition": "one-way turnover: half the absolute change in fully invested team-portfolio weights",
            "levels": levels,
            "comparisons": comparisons,
        }
    return output


def _transaction_cost_sensitivity(
    fixed_assignment_performance: dict,
    turnover: dict,
) -> dict:
    """Translate one-way turnover into steady-state annualized net returns."""
    performance_levels = fixed_assignment_performance["levels"]
    turnover_levels = turnover["7"]["levels"]
    designs = ("shared", "mixed", "coherent")
    levels = {}
    for design in designs:
        gross_return = float(
            performance_levels[design]["annualized_total_return"]["mean"]
        )
        annualized_turnover = float(
            12.0 * turnover_levels[design]["mean_monthly_turnover"]
        )
        levels[design] = {
            "annualized_gross_return": gross_return,
            "annualized_one_way_turnover": annualized_turnover,
            "annualized_net_return": {
                str(cost_bps): float(
                    gross_return - annualized_turnover * cost_bps / 10_000.0
                )
                for cost_bps in TRANSACTION_COST_SCENARIOS_BPS
            },
        }

    shared = levels["shared"]
    break_even = {}
    for design in ("mixed", "coherent"):
        treatment = levels[design]
        turnover_saving = (
            shared["annualized_one_way_turnover"]
            - treatment["annualized_one_way_turnover"]
        )
        gross_shortfall = (
            shared["annualized_gross_return"]
            - treatment["annualized_gross_return"]
        )
        if turnover_saving <= 0.0:
            raise AssertionError(f"{design} does not save turnover versus shared access")
        break_even[design] = float(gross_shortfall / turnover_saving * 10_000.0)

    return {
        "definition": (
            "steady-state annualized net return equals annualized gross return minus "
            "12 times mean monthly one-way turnover times cost per dollar traded"
        ),
        "cost_unit": "basis points per dollar of one-way turnover",
        "initial_formation_trade_included": False,
        "scenarios_bps": list(TRANSACTION_COST_SCENARIOS_BPS),
        "levels": levels,
        "break_even_one_way_cost_bps_vs_shared": break_even,
    }


def _flat_rows(date_rows: list[dict]) -> list[dict]:
    output = []
    for row in date_rows:
        common = {key: row[key] for key in (
            "sample", "snapshot_id", "date", "return_regime", "volatility_regime", "team_size"
        )}
        for design in ("shared", "mixed", "coherent"):
            output.append({**common, "design": design, **row[design]})
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=("pilot", "held-out", "all"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    date_rows, plan = _date_rows(args.sample)
    packet_rows = _selected(load_packets(), args.sample)
    summaries = _capacity_summaries(date_rows)
    performance = _performance_summary(date_rows, packet_rows)
    fixed_assignment_performance = _fixed_assignment_performance(args.sample)
    turnover = _turnover_summary(args.sample)
    transaction_cost_sensitivity = _transaction_cost_sensitivity(
        fixed_assignment_performance,
        turnover,
    )
    cache_rows = _selected(_load(CACHE_PATH), args.sample)
    result = {
        "schema_version": "information-capacity-long-only-results-v1",
        "sample": args.sample,
        "design_status": "frozen before long-only model collection",
        "source_signed_assignment_sha256": plan["source_signed_plan"]["assignment_sha256"],
        "date_is_inference_unit": True,
        "team_information_union_fixed": True,
        "portfolio": "fully invested long-only; geometry uses equal-weight benchmark-relative active weights",
        "configuration_averaging": (
            "all model subsets and all packet-to-model permutations are averaged within date"
        ),
        "n_dates": len({row["date"] for row in date_rows}),
        "collection": {
            "n_partitioned": len(cache_rows),
            "n_parse_ok": sum(_strict_complete(row) for row in cache_rows),
            "reported_cost": float(sum(row.get("usage", {}).get("cost", 0.0) or 0.0 for row in cache_rows)),
        },
        "per_date": date_rows,
        **summaries,
        "operating_performance": performance,
        "fixed_assignment_operating_performance": fixed_assignment_performance,
        "turnover": turnover,
        "transaction_cost_sensitivity": transaction_cost_sensitivity,
    }
    if args.sample == "all":
        output_path = RESULT_PATH
        summary_path = SUMMARY_PATH
    else:
        suffix = args.sample.replace("-", "_")
        output_path = RESULT_PATH.with_name(f"information_capacity_longonly_results_{suffix}.json")
        summary_path = SUMMARY_PATH.with_name(f"information_capacity_longonly_summary_{suffix}.csv")
    _atomic_json(output_path, result)
    pd.DataFrame(_flat_rows(date_rows)).to_csv(summary_path, index=False)
    print("Long-only Information-Capacity Curve")
    for k in TEAM_SIZES:
        row = summaries["levels"][str(k)]
        print(
            f"  K={k}: shared={row['shared']['N_eff']['mean']:.3f}, "
            f"mixed={row['mixed']['N_eff']['mean']:.3f}, "
            f"coherent={row['coherent']['N_eff']['mean']:.3f}"
        )
    endpoint = fixed_assignment_performance["levels"]
    print(
        "  K=7 tracking error: "
        f"shared={endpoint['shared']['annualized_tracking_error']['mean']:.3%}, "
        f"mixed={endpoint['mixed']['annualized_tracking_error']['mean']:.3%}, "
        f"coherent={endpoint['coherent']['annualized_tracking_error']['mean']:.3%}"
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
