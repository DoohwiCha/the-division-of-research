"""Collect and analyze the fixed-calendar U.S.-equity monthly panel.

The design uses every available month-end label from July 2023 through May
2026.  Each panel has the same 14-stock universe, model roster, personas,
prompt, and temperature.  Market states are descriptive: return regimes are
terciles of the broad-universe trailing three-month return, and high
volatility is the top tercile of trailing one-month realized volatility.

Live collection (resumable):
    OPENROUTER_API_KEY=... PYTHONPATH=src python3 src/cohort_equity_monthly.py

Offline analysis:
    PYTHONPATH=src python3 src/cohort_equity_monthly.py --analyze-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA, RESULTS
from cohort_crossvendor import OPENROUTER_MODELS, PERSONAS
from cohort_analyze import n_eff
from cohort_crossvendor_hardening import books, matched_null, same_side_overlap
from cohort_equity_locked import (
    DEFAULT_MAX_WORKERS,
    REPEATED_TEMPERATURE,
    _atomic_json,
    _load_records,
    _panel_metrics,
    _sha256,
    _summarize_rows,
    collect,
)
from cohort_equity_multiregime import EQUITY_PROMPT_SPEC_VERSION, build_equity_prompt


SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    "WMT", "PG", "JPM", "BAC", "XOM", "JNJ", "UNH",
]
SNAPSHOT_PATH = Path(DATA) / "equity_monthly_snapshots.json"
CACHE_PATH = Path(DATA) / "cohort_agents_equity_monthly.json"
RESULT_PATH = Path(RESULTS) / "cohort_equity_monthly.json"
PRICE_PATH = Path(DATA) / "jfs_global_prices_united_states.csv"
GLOBAL_SNAPSHOT_PATH = Path(DATA) / "jfs_global_equity_snapshots.json"
EXPERIMENT = "monthly_panel"


def _feature_snapshot(prices: pd.DataFrame, date: str) -> tuple[str, str]:
    history = prices.loc[prices.index <= pd.Timestamp(date), SYMBOLS].ffill().dropna()
    if len(history) < 64:
        raise ValueError(f"fewer than 64 complete observations through {date}")
    history = history.iloc[-64:]
    ret_7d = (history.iloc[-1] / history.iloc[-6] - 1.0) * 100.0
    ret_30d = (history.iloc[-1] / history.iloc[-22] - 1.0) * 100.0
    ret_90d = (history.iloc[-1] / history.iloc[-64] - 1.0) * 100.0
    vol_30d = np.log(history / history.shift(1)).iloc[-21:].std() * math.sqrt(252.0) * 100.0
    drawdown = (history.iloc[-1] / history.iloc[-63:].max() - 1.0) * 100.0
    header = (
        f"{'ticker':<7}{'ret_7d':>8}{'ret_30d':>9}{'ret_90d':>9}"
        f"{'vol_30d_ann':>13}{'drawdown_from_90d_high':>24}"
    )
    rows = [header]
    for symbol in SYMBOLS:
        rows.append(
            f"{symbol:<7}{ret_7d[symbol]:>8.1f}{ret_30d[symbol]:>9.1f}"
            f"{ret_90d[symbol]:>9.1f}{vol_30d[symbol]:>13.1f}{drawdown[symbol]:>24.1f}"
        )
    return "\n".join(rows), str(history.index[-1].date())


def _market_state(prices: pd.DataFrame, date: str) -> tuple[float, float]:
    broad_returns = prices.pct_change(fill_method=None).mean(axis=1, skipna=True)
    returns = broad_returns.loc[broad_returns.index <= pd.Timestamp(date)].dropna()
    if len(returns) < 63:
        raise ValueError(f"fewer than 63 broad-market returns through {date}")
    trailing = returns.iloc[-63:]
    market_return_3m = float(np.prod(1.0 + trailing.to_numpy(dtype=float)) - 1.0)
    market_vol_1m = float(trailing.iloc[-21:].std() * math.sqrt(252.0))
    return market_return_3m, market_vol_1m


def build_snapshots() -> list[dict]:
    prices = pd.read_csv(PRICE_PATH, index_col=0, parse_dates=True).sort_index()
    missing = sorted(set(SYMBOLS) - set(prices.columns))
    if missing:
        raise ValueError(f"monthly equity universe missing from price data: {missing}")
    calendar = json.loads(GLOBAL_SNAPSHOT_PATH.read_text(encoding="utf-8"))["dates"]
    rows = []
    for date in calendar:
        table, observed_through = _feature_snapshot(prices, date)
        market_return_3m, market_vol_1m = _market_state(prices, date)
        rows.append({
            "date": date,
            "observed_through": observed_through,
            "syms": SYMBOLS,
            "snapshot": table,
            "market_return_3m": market_return_3m,
            "market_vol_1m": market_vol_1m,
        })
    returns = np.asarray([row["market_return_3m"] for row in rows], dtype=float)
    volatility = np.asarray([row["market_vol_1m"] for row in rows], dtype=float)
    return_q1, return_q2 = np.quantile(returns, [1.0 / 3.0, 2.0 / 3.0])
    volatility_q2 = float(np.quantile(volatility, 2.0 / 3.0))
    for row in rows:
        value = row["market_return_3m"]
        if value <= return_q1:
            row["return_regime"] = "weak"
        elif value <= return_q2:
            row["return_regime"] = "middle"
        else:
            row["return_regime"] = "strong"
        row["volatility_regime"] = "high" if row["market_vol_1m"] >= volatility_q2 else "normal"
    payload = {
        "schema_version": "equity-monthly-snapshots-v1",
        "source_prices": str(PRICE_PATH.relative_to(Path(DATA).parent)),
        "date_rule": "all stored month-end labels from 2023-07-31 through 2026-05-31",
        "regime_rule": {
            "return": "terciles of broad 50-stock equal-weight trailing 3-month return",
            "return_q1": float(return_q1),
            "return_q2": float(return_q2),
            "volatility": "high is the top tercile of broad 50-stock trailing 1-month annualized volatility",
            "volatility_q2": volatility_q2,
        },
        "n_snapshots": len(rows),
        "symbols": SYMBOLS,
        "snapshots": rows,
    }
    _atomic_json(SNAPSHOT_PATH, payload)
    return rows


def load_snapshots() -> list[dict]:
    rows = build_snapshots()
    if len(rows) != 35 or len({row["date"] for row in rows}) != 35:
        raise ValueError("expected 35 distinct monthly U.S.-equity snapshots")
    if any(pd.Timestamp(row["observed_through"]) > pd.Timestamp(row["date"]) for row in rows):
        raise ValueError("monthly snapshot contains look-ahead data")
    return rows


def make_jobs(snapshots: list[dict]) -> list[dict]:
    jobs = []
    for snapshot in snapshots:
        for vendor, model in OPENROUTER_MODELS:
            for persona in PERSONAS:
                prompt = build_equity_prompt(persona, snapshot["syms"], snapshot["snapshot"])
                jobs.append({
                    "experiment": EXPERIMENT,
                    "date": snapshot["date"],
                    "regime": snapshot["return_regime"],
                    "replicate": 0,
                    "temperature": REPEATED_TEMPERATURE,
                    "vendor": vendor,
                    "requested_model": model,
                    "persona": persona,
                    "syms": snapshot["syms"],
                    "table": snapshot["snapshot"],
                    "prompt_spec_version": EQUITY_PROMPT_SPEC_VERSION,
                    "prompt_sha256": _sha256(prompt),
                    "snapshot_sha256": _sha256(snapshot["snapshot"]),
                })
    return jobs


def _association(rows: list[dict], x_key: str, y_key: str) -> dict:
    complete = [row for row in rows if row.get("complete")]
    if len(complete) < 2:
        return {"n": int(len(complete)), "pearson": None, "spearman": None}
    x = pd.Series([row[x_key] for row in complete], dtype=float)
    y = pd.Series([row[y_key] for row in complete], dtype=float)
    return {
        "n": int(len(complete)),
        "pearson": float(x.corr(y, method="pearson")),
        "spearman": float(x.corr(y, method="spearman")),
    }


def _family_average_metrics(agents: list[dict], syms: list[str]) -> dict:
    clean = [row for row in agents if row.get("parse_ok") and row.get("weights")]
    W = books(clean, syms)
    family_books = {}
    for vendor, _ in OPENROUTER_MODELS:
        columns = [f"{persona}:{vendor}" for persona in PERSONAS if f"{persona}:{vendor}" in W.columns]
        if len(columns) != len(PERSONAS):
            continue
        vector = W[columns].mean(axis=1)
        vector = vector - vector.mean()
        gross = float(vector.abs().sum())
        if gross > 1e-12:
            family_books[vendor] = vector / gross
    F = pd.DataFrame(family_books, index=syms)
    if F.shape[1] < 4:
        return {"M": int(F.shape[1]), "complete": False}
    C = np.corrcoef(F.to_numpy(dtype=float).T)
    iu = np.triu_indices(F.shape[1], 1)
    null = matched_null(F.shape[1], syms)
    value = float(n_eff(C))
    return {
        "M": int(F.shape[1]),
        "complete": F.shape[1] == len(OPENROUTER_MODELS),
        "N_eff": value,
        "mean_abs_corr": float(np.abs(C[iu]).mean()),
        "same_side_overlap": float(same_side_overlap(F)),
        "null": null,
        "below_null_p2_5": bool(value < null["p2_5"]),
    }


def _group_contrast(
    rows: list[dict],
    group_key: str,
    focal: str,
    comparison: set[str],
    metric: str,
    seed: int,
    draws: int = 20_000,
) -> dict:
    focal_values = np.asarray(
        [row[metric] for row in rows if row.get("complete") and row[group_key] == focal], dtype=float
    )
    comparison_values = np.asarray(
        [row[metric] for row in rows if row.get("complete") and row[group_key] in comparison], dtype=float
    )
    observed = float(focal_values.mean() - comparison_values.mean())
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=float)
    pooled = np.concatenate([focal_values, comparison_values])
    permutations = np.empty(draws, dtype=float)
    n_focal = len(focal_values)
    for index in range(draws):
        bootstrap[index] = float(
            rng.choice(focal_values, n_focal, replace=True).mean()
            - rng.choice(comparison_values, len(comparison_values), replace=True).mean()
        )
        shuffled = rng.permutation(pooled)
        permutations[index] = float(shuffled[:n_focal].mean() - shuffled[n_focal:].mean())
    return {
        "contrast": f"{focal} minus {'+'.join(sorted(comparison))}",
        "metric": metric,
        "n_focal": int(n_focal),
        "n_comparison": int(len(comparison_values)),
        "difference": observed,
        "bootstrap_95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "permutation_draws": draws,
        "permutation_p_two_sided": float(
            (np.sum(np.abs(permutations) >= abs(observed)) + 1) / (draws + 1)
        ),
    }


def analyze(snapshots: list[dict]) -> dict:
    records = list(_load_records(CACHE_PATH).values())
    null_cache: dict[int, dict] = {}
    panels = []
    family_average_panels = []
    for snapshot in snapshots:
        agents = [row for row in records if row["date"] == snapshot["date"]]
        metrics = _panel_metrics(agents, snapshot["syms"], null_cache)
        panels.append({
            "date": snapshot["date"],
            "observed_through": snapshot["observed_through"],
            "return_regime": snapshot["return_regime"],
            "volatility_regime": snapshot["volatility_regime"],
            "market_return_3m": snapshot["market_return_3m"],
            "market_vol_1m": snapshot["market_vol_1m"],
            **metrics,
        })
        family_average_panels.append({
            "date": snapshot["date"],
            "return_regime": snapshot["return_regime"],
            **_family_average_metrics(agents, snapshot["syms"]),
        })
    return_groups = {
        regime: _summarize_rows([row for row in panels if row["return_regime"] == regime])
        for regime in ("weak", "middle", "strong")
    }
    volatility_groups = {
        regime: _summarize_rows([row for row in panels if row["volatility_regime"] == regime])
        for regime in ("normal", "high")
    }
    usage_rows = [row.get("usage", {}) for row in records if row.get("parse_ok")]
    result = {
        "schema_version": "cohort-equity-monthly-v1",
        "prompt_spec_version": EQUITY_PROMPT_SPEC_VERSION,
        "snapshot_sha256": hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest(),
        "design": {
            "n_dates": len(snapshots),
            "dates": [row["date"] for row in snapshots],
            "temperature": REPEATED_TEMPERATURE,
            "draws_per_date": 1,
            "vendors": [vendor for vendor, _ in OPENROUTER_MODELS],
            "models": [model for _, model in OPENROUTER_MODELS],
            "personas": list(PERSONAS),
            "assets_per_panel": len(SYMBOLS),
            "fixed_universe": SYMBOLS,
            "date_rule": "all available month-end labels; no outcome-based date selection",
        },
        "collection": {
            "n_records": len(records),
            "n_parse_ok": sum(bool(row.get("parse_ok")) for row in records),
            "n_parse_failures": sum(not row.get("parse_ok") for row in records),
            "prompt_tokens": int(sum(row.get("prompt_tokens", 0) for row in usage_rows)),
            "completion_tokens": int(sum(row.get("completion_tokens", 0) for row in usage_rows)),
        },
        "panels": panels,
        "overall": _summarize_rows(panels),
        "family_average_panels": family_average_panels,
        "family_average_overall": _summarize_rows(family_average_panels),
        "by_return_regime": return_groups,
        "by_volatility_regime": volatility_groups,
        "associations": {
            "N_eff_vs_market_return_3m": _association(panels, "market_return_3m", "N_eff"),
            "N_eff_vs_market_vol_1m": _association(panels, "market_vol_1m", "N_eff"),
            "overlap_vs_market_return_3m": _association(panels, "market_return_3m", "same_side_overlap"),
            "overlap_vs_market_vol_1m": _association(panels, "market_vol_1m", "same_side_overlap"),
        },
        "regime_contrasts": {
            "weak_vs_other_N_eff": _group_contrast(
                panels, "return_regime", "weak", {"middle", "strong"}, "N_eff", 2026071401
            ),
            "weak_vs_other_overlap": _group_contrast(
                panels, "return_regime", "weak", {"middle", "strong"}, "same_side_overlap", 2026071402
            ),
            "high_vs_normal_N_eff": _group_contrast(
                panels, "volatility_regime", "high", {"normal"}, "N_eff", 2026071403
            ),
            "high_vs_normal_overlap": _group_contrast(
                panels, "volatility_regime", "high", {"normal"}, "same_side_overlap", 2026071404
            ),
        },
    }
    _atomic_json(RESULT_PATH, result)
    return result


def print_result(result: dict) -> None:
    overall = result["overall"]
    print("\nFixed-calendar monthly U.S.-equity audit")
    print(
        f"  complete panels: {overall['n_complete_panels']}/{overall['n_panels']}; "
        f"below matched-null p2.5: {overall['n_below_null_p2_5']}/{overall['n_complete_panels']}"
    )
    if "N_eff" in overall:
        print(
            f"  N_eff={overall['N_eff']['mean']:.3f} +/- {overall['N_eff']['sd']:.3f}; "
            f"O={overall['same_side_overlap']['mean']:.3f} +/- {overall['same_side_overlap']['sd']:.3f}"
        )
    for regime, row in result["by_return_regime"].items():
        if "N_eff" in row:
            print(
                f"  return={regime:<6}: n={row['n_complete_panels']:2d}; "
                f"N_eff={row['N_eff']['mean']:.3f}; O={row['same_side_overlap']['mean']:.3f}"
            )
    for regime, row in result["by_volatility_regime"].items():
        if "N_eff" in row:
            print(
                f"  volatility={regime:<6}: n={row['n_complete_panels']:2d}; "
                f"N_eff={row['N_eff']['mean']:.3f}; O={row['same_side_overlap']['mean']:.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-new-calls", type=int)
    args = parser.parse_args()

    snapshots = load_snapshots()
    jobs = make_jobs(snapshots)
    if args.dry_run:
        print(f"monthly snapshots: {len(snapshots)}")
        print(f"monthly calls: {len(jobs)}")
        return
    if args.build_only:
        print(f"wrote {SNAPSHOT_PATH} with {len(snapshots)} snapshots")
        return
    if not args.analyze_only and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required for collection")
    if not args.analyze_only:
        collect(jobs, CACHE_PATH, args.max_workers, args.max_new_calls)
    result = analyze(snapshots)
    print_result(result)


if __name__ == "__main__":
    main()
