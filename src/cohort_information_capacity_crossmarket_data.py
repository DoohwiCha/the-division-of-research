"""Build harmonized point-in-time packets for cross-market replication.

The U.S. information-capacity experiment uses a rich research union that
includes SEC filings.  Europe and crypto do not share that disclosure system.
This module therefore builds a second, harmonized 32-field union from only
price, volume, and risk histories available in both markets.  The information
architecture, not a market-specific field, is the treatment.

Run:
    PYTHONPATH=src python3 src/cohort_information_capacity_crossmarket_data.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA, RESULTS
from cohort_information_diversity_data import (
    RAW_CLOSE_PATH as US_RAW_CLOSE_PATH,
    _write_csv,
    _yahoo_equity_history,
)


SCHEMA_VERSION = "information-capacity-crossmarket-packets-v2-pit-corrected"
CRYPTO_VOLUME_PATH = Path(DATA) / "jfs_crypto_perp_volume.csv"
EUROPE_RAW_CLOSE_PATH = Path(DATA) / "information_capacity_europe_split_adjusted_close.csv"
EUROPE_PRICE_BASIS_PATH = Path(DATA) / "information_capacity_europe_price_basis.json"
PRICE_BASIS_CACHE = Path(
    os.environ.get(
        "INFORMATION_CAPACITY_PRICE_BASIS_CACHE",
        "/tmp/information-access-price-basis",
    )
)

# Fixed before cross-market LLM collection.  The dates span the common period
# with at least 252 prior observations and one full forward month.
REPLICATION_DATES = (
    "2024-09-30",
    "2024-10-31",
    "2024-12-31",
    "2025-02-28",
    "2025-04-30",
    "2025-06-30",
    "2025-08-31",
    "2025-10-31",
    "2025-12-31",
    "2026-01-31",
    "2026-02-28",
    "2026-04-30",
)

EUROPE_SYMBOLS = (
    "ASML.AS", "SAP.DE", "MC.PA", "TTE.PA", "SIE.DE", "AIR.PA", "SAN.MC",
    "SU.PA", "OR.PA", "ALV.DE", "DTE.DE", "ENEL.MI", "BNP.PA", "BAS.DE",
)
US_SYMBOLS = (
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA",
    "WMT", "PG", "JPM", "BAC", "XOM", "JNJ", "UNH",
)
CRYPTO_SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT",
    "ADAUSDT", "LINKUSDT", "AVAXUSDT", "TRXUSDT", "LTCUSDT", "BCHUSDT",
)

# Every field is asset-specific.  This prevents a coherent packet from being
# unable to rank assets merely because it contains market-wide scalar values.
FIELD_SPECS = (
    ("trend", "ret_5", "ret_5%", "percent"),
    ("trend", "ret_21", "ret_21%", "percent"),
    ("trend", "ret_63", "ret_63%", "percent"),
    ("trend", "ret_126", "ret_126%", "percent"),
    ("trend", "ma_gap_21_63", "MA21-63%", "percent"),
    ("trend", "distance_high_63", "from_high63%", "percent"),
    ("reversal", "ret_1", "ret_1%", "percent"),
    ("reversal", "ret_5_minus_21", "ret5-ret21", "percent"),
    ("reversal", "rsi_14", "RSI14", "number"),
    ("reversal", "up_share_21", "up_days21%", "percent"),
    ("volatility", "vol_21", "vol21%", "percent"),
    ("volatility", "vol_63", "vol63%", "percent"),
    ("volatility", "downside_vol_63", "downvol63%", "percent"),
    ("volatility", "vol_ratio_21_126", "vol21/126", "number"),
    ("volatility", "skew_63", "skew63", "number"),
    ("tail", "drawdown_63", "dd63%", "percent"),
    ("tail", "drawdown_252", "dd252%", "percent"),
    ("tail", "worst_day_63", "worst63%", "percent"),
    ("tail", "expected_shortfall_63", "ES63%", "percent"),
    ("systematic", "beta_63", "beta63", "number"),
    ("systematic", "beta_252", "beta252", "number"),
    ("systematic", "corr_63", "corr63", "number"),
    ("systematic", "corr_252", "corr252", "number"),
    ("systematic", "idio_vol_63", "idio63%", "percent"),
    ("liquidity", "volume_ratio_21_126", "volm21/126", "number"),
    ("liquidity", "volume_cv_63", "volmCV63", "number"),
    ("liquidity", "dollar_volume_ratio_21_126", "dvol21/126", "number"),
    ("liquidity", "amihud_ratio_21_126", "illiq21/126", "number"),
    ("liquidity", "zero_return_share_63", "zero_ret63%", "percent"),
    ("relative", "momentum_rank", "mom_rank", "rank"),
    ("relative", "low_vol_rank", "lowvol_rank", "rank"),
    ("relative", "liquidity_rank", "liq_rank", "rank"),
)
FIELD_IDS = tuple(f"{group}.{field}" for group, field, _, _ in FIELD_SPECS)

MARKETS = {
    "us": {
        "label": "U.S. large-cap equities",
        "asset_class": "equity",
        "symbols": US_SYMBOLS,
        "prices": Path(DATA) / "jfs_global_prices_united_states.csv",
        "volumes": Path(DATA) / "jfs_global_volumes_united_states.csv",
        "annualization": 252,
    },
    "europe": {
        "label": "European large-cap equities",
        "asset_class": "equity",
        "symbols": EUROPE_SYMBOLS,
        "prices": Path(DATA) / "jfs_global_prices_europe.csv",
        "volumes": Path(DATA) / "jfs_global_volumes_europe.csv",
        "annualization": 252,
    },
    "crypto": {
        "label": "crypto perpetuals",
        "asset_class": "crypto",
        "symbols": CRYPTO_SYMBOLS,
        "prices": Path(DATA) / "jfs_crypto_perp_prices.csv",
        "volumes": CRYPTO_VOLUME_PATH,
        "annualization": 365,
    },
}


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wide_csv(path: Path, symbols: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame = frame.loc[:, list(symbols)].apply(pd.to_numeric, errors="coerce")
    return frame.sort_index()


def _fetch_crypto_volume() -> None:
    if CRYPTO_VOLUME_PATH.exists():
        return
    columns = {}
    for symbol in CRYPTO_SYMBOLS:
        encoded = urllib.parse.quote(symbol)
        url = (
            "https://fapi.binance.com/fapi/v1/klines?"
            f"symbol={encoded}&interval=1d&startTime=1704067200000&limit=1500"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "InformationAccessStudy/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            rows = json.loads(response.read())
        columns[symbol] = pd.Series(
            {pd.Timestamp(int(row[0]), unit="ms").normalize(): float(row[5]) for row in rows}
        )
    frame = pd.DataFrame(columns).sort_index()
    frame.index.name = "date"
    CRYPTO_VOLUME_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CRYPTO_VOLUME_PATH)


def _asof_index(index: pd.DatetimeIndex, date: str) -> pd.Timestamp:
    eligible = index[index <= pd.Timestamp(date)]
    if not len(eligible):
        raise ValueError(f"no observation on or before {date}")
    return eligible[-1]


def _ret(values: pd.Series, sessions: int) -> float:
    return float(values.iloc[-1] / values.iloc[-1 - sessions] - 1.0)


def _annualized_vol(values: pd.Series, annualization: int) -> float:
    return float(values.std(ddof=1) * math.sqrt(annualization))


def _drawdown(prices: pd.Series) -> float:
    return float((prices / prices.cummax() - 1.0).min())


def _rsi(returns: pd.Series, sessions: int = 14) -> float:
    values = returns.iloc[-sessions:]
    gains = values.clip(lower=0).mean()
    losses = -values.clip(upper=0).mean()
    if losses <= 1e-12:
        return 100.0
    return float(100.0 - 100.0 / (1.0 + gains / losses))


def _beta_corr(asset: pd.Series, market: pd.Series) -> tuple[float, float, float]:
    paired = pd.concat([asset, market], axis=1).dropna()
    x = paired.iloc[:, 0].to_numpy(dtype=float)
    y = paired.iloc[:, 1].to_numpy(dtype=float)
    variance = float(np.var(y, ddof=1))
    beta = 0.0 if variance <= 1e-15 else float(np.cov(x, y, ddof=1)[0, 1] / variance)
    corr = 0.0 if np.std(x) <= 1e-12 or np.std(y) <= 1e-12 else float(np.corrcoef(x, y)[0, 1])
    residual = x - beta * y
    return beta, corr, float(np.std(residual, ddof=1))


def _base_features(
    prices: pd.DataFrame,
    raw_closes: pd.DataFrame,
    volumes: pd.DataFrame,
    asof: pd.Timestamp,
    symbols: tuple[str, ...],
    annualization: int,
) -> dict[str, dict[str, float]]:
    p = prices.loc[:asof, list(symbols)].ffill()
    raw = raw_closes.loc[:asof, list(symbols)].ffill()
    v = volumes.loc[:asof, list(symbols)].fillna(0.0)
    returns = p.pct_change(fill_method=None)
    market = returns.mean(axis=1)
    output: dict[str, dict[str, float]] = {}
    for symbol in symbols:
        ps = p[symbol].dropna()
        rs = returns[symbol].dropna()
        vs = v[symbol].reindex(ps.index).fillna(0.0)
        if len(ps) < 254 or len(rs) < 252:
            raise ValueError(f"{symbol} has insufficient history at {asof.date()}")
        r63, r252 = rs.iloc[-63:], rs.iloc[-252:]
        market63 = market.reindex(r63.index).dropna()
        market252 = market.reindex(r252.index).dropna()
        b63, c63, idio63 = _beta_corr(r63, market63)
        b252, c252, _ = _beta_corr(r252, market252)
        raw_symbol = raw[symbol].reindex(ps.index).ffill()
        dvol = raw_symbol * vs
        absret = rs.abs().replace(0.0, np.nan)
        illiq = absret / dvol.reindex(rs.index).replace(0.0, np.nan)
        es_count = max(1, int(math.ceil(0.05 * len(r63))))
        feature = {
            "trend.ret_5": _ret(ps, 5),
            "trend.ret_21": _ret(ps, 21),
            "trend.ret_63": _ret(ps, 63),
            "trend.ret_126": _ret(ps, 126),
            "trend.ma_gap_21_63": float(ps.iloc[-21:].mean() / ps.iloc[-63:].mean() - 1.0),
            "trend.distance_high_63": float(ps.iloc[-1] / ps.iloc[-63:].max() - 1.0),
            "reversal.ret_1": _ret(ps, 1),
            "reversal.ret_5_minus_21": _ret(ps, 5) - _ret(ps, 21),
            "reversal.rsi_14": _rsi(rs),
            "reversal.up_share_21": float((rs.iloc[-21:] > 0).mean()),
            "volatility.vol_21": _annualized_vol(rs.iloc[-21:], annualization),
            "volatility.vol_63": _annualized_vol(r63, annualization),
            "volatility.downside_vol_63": _annualized_vol(r63.clip(upper=0), annualization),
            "volatility.vol_ratio_21_126": float(rs.iloc[-21:].std(ddof=1) / max(rs.iloc[-126:].std(ddof=1), 1e-12)),
            "volatility.skew_63": float(r63.skew()),
            "tail.drawdown_63": _drawdown(ps.iloc[-63:]),
            "tail.drawdown_252": _drawdown(ps.iloc[-252:]),
            "tail.worst_day_63": float(r63.min()),
            "tail.expected_shortfall_63": float(np.sort(r63.to_numpy())[:es_count].mean()),
            "systematic.beta_63": b63,
            "systematic.beta_252": b252,
            "systematic.corr_63": c63,
            "systematic.corr_252": c252,
            "systematic.idio_vol_63": float(idio63 * math.sqrt(annualization)),
            "liquidity.volume_ratio_21_126": float(vs.iloc[-21:].mean() / max(vs.iloc[-126:].mean(), 1e-12)),
            "liquidity.volume_cv_63": float(vs.iloc[-63:].std(ddof=1) / max(vs.iloc[-63:].mean(), 1e-12)),
            "liquidity.dollar_volume_ratio_21_126": float(dvol.iloc[-21:].mean() / max(dvol.iloc[-126:].mean(), 1e-12)),
            "liquidity.amihud_ratio_21_126": float(illiq.iloc[-21:].mean() / max(illiq.iloc[-126:].mean(), 1e-30)),
            "liquidity.zero_return_share_63": float((r63.abs() < 1e-12).mean()),
        }
        output[symbol] = feature
    values = pd.DataFrame(output).T
    output_rank = {
        "relative.momentum_rank": values["trend.ret_63"].rank(pct=True),
        "relative.low_vol_rank": (-values["volatility.vol_63"]).rank(pct=True),
        "relative.liquidity_rank": values["liquidity.dollar_volume_ratio_21_126"].rank(pct=True),
    }
    for field_id, ranks in output_rank.items():
        for symbol in symbols:
            output[symbol][field_id] = float(ranks[symbol])
    return output


def _format(value: float, kind: str) -> str:
    if kind == "percent":
        return f"{100.0 * value:.2f}"
    if kind == "rank":
        return f"{100.0 * value:.1f}"
    return f"{value:.3f}"


def _full_table(features: dict[str, dict[str, float]], symbols: tuple[str, ...]) -> str:
    labels = [label for _, _, label, _ in FIELD_SPECS]
    lines = ["ticker | " + " | ".join(labels)]
    for symbol in symbols:
        values = []
        for group, field, _, kind in FIELD_SPECS:
            values.append(_format(features[symbol][f"{group}.{field}"], kind))
        lines.append(symbol + " | " + " | ".join(values))
    return "\n".join(lines)


def _raw_close_basis(
    market: str,
    prices: pd.DataFrame,
    symbols: tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    if market == "crypto":
        config_path = MARKETS[market]["prices"]
        return prices, {
            "basis": "native perpetual close",
            "sha256": _sha256(config_path),
            "file": config_path.name,
        }
    if market == "us":
        frame = _wide_csv(US_RAW_CLOSE_PATH, symbols).reindex(prices.index).ffill()
        return frame, {
            "basis": "Yahoo dividend-unadjusted, split-adjusted quote close",
            "file": US_RAW_CLOSE_PATH.name,
            "sha256": _sha256(US_RAW_CLOSE_PATH),
        }

    closes = {}
    sources = {}
    for symbol in symbols:
        close, _splits, source = _yahoo_equity_history(symbol, PRICE_BASIS_CACHE)
        closes[symbol] = close
        sources[symbol] = source
    frame = pd.DataFrame(closes).sort_index().reindex(prices.index).ffill()
    if frame[list(symbols)].isna().any().any():
        raise ValueError("European raw-close basis contains missing observations")
    _write_csv(EUROPE_RAW_CLOSE_PATH, frame)
    metadata = {
        "schema_version": "information-capacity-europe-price-basis-v1",
        "basis": "Yahoo dividend-unadjusted, split-adjusted quote close",
        "sources": sources,
        "sha256": _sha256(EUROPE_RAW_CLOSE_PATH),
    }
    _atomic_json(EUROPE_PRICE_BASIS_PATH, metadata)
    return frame, {
        "basis": metadata["basis"],
        "file": EUROPE_RAW_CLOSE_PATH.name,
        "sha256": metadata["sha256"],
        "source_manifest": EUROPE_PRICE_BASIS_PATH.name,
    }


def _build_market(market: str, config: dict) -> dict:
    if market == "crypto":
        _fetch_crypto_volume()
    symbols = tuple(config["symbols"])
    prices = _wide_csv(config["prices"], symbols)
    volumes = _wide_csv(config["volumes"], symbols)
    common_index = prices.index.intersection(volumes.index)
    prices, volumes = prices.loc[common_index], volumes.loc[common_index]
    raw_closes, raw_close_source = _raw_close_basis(market, prices, symbols)
    raw_closes = raw_closes.reindex(common_index).ffill()
    prices_asof = prices.ffill()
    snapshots = []
    for index, date in enumerate(REPLICATION_DATES, start=1):
        asof = _asof_index(common_index, date)
        features = _base_features(
            prices, raw_closes, volumes, asof, symbols, int(config["annualization"])
        )
        trailing_prices = raw_closes.loc[:asof, list(symbols)].iloc[-21:]
        trailing_volumes = volumes.loc[:asof, list(symbols)].fillna(0.0).iloc[-21:]
        raw_depth = (trailing_prices * trailing_volumes).mean(axis=0)
        median_depth = max(float(raw_depth.median()), 1e-12)
        liquidity_depth = {
            symbol: float(raw_depth[symbol] / median_depth) for symbol in symbols
        }
        future_index = prices.index[prices.index > asof][:21]
        if len(future_index) < 21:
            raise ValueError(f"{market} lacks 21 forward observations after {date}")
        forward = {
            symbol: float(
                prices_asof.loc[future_index[-1], symbol]
                / prices_asof.loc[asof, symbol]
                - 1.0
            )
            for symbol in symbols
        }
        market_returns = prices.loc[:asof, list(symbols)].pct_change(fill_method=None).mean(axis=1)
        ret63 = float((1.0 + market_returns.iloc[-63:]).prod() - 1.0)
        vol21 = float(market_returns.iloc[-21:].std(ddof=1) * math.sqrt(config["annualization"]))
        snapshots.append({
            "snapshot_id": f"{market[:2].upper()}{index:02d}",
            "market": market,
            "market_label": config["label"],
            "asset_class": config["asset_class"],
            "date": date,
            "observed_through": asof.date().isoformat(),
            "symbols": list(symbols),
            "common_core": "ticker\n" + "\n".join(symbols),
            "features": features,
            "liquidity_depth": liquidity_depth,
            "full_table": _full_table(features, symbols),
            "forward_return_21d": forward,
            "market_return_63": ret63,
            "market_volatility_21": vol21,
        })
    returns = np.asarray([row["market_return_63"] for row in snapshots])
    vols = np.asarray([row["market_volatility_21"] for row in snapshots])
    q1, q2 = np.quantile(returns, [1 / 3, 2 / 3])
    vmid = float(np.median(vols))
    for row in snapshots:
        value = row["market_return_63"]
        row["return_regime"] = "weak" if value <= q1 else "strong" if value >= q2 else "middle"
        row["volatility_regime"] = "high" if row["market_volatility_21"] >= vmid else "normal"
    return {
        "schema_version": SCHEMA_VERSION,
        "design_frozen_before_llm_collection": True,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "market": market,
        "market_label": config["label"],
        "design": {
            "dates": list(REPLICATION_DATES),
            "date_rule": "fixed common calendar month-ends before LLM collection",
            "symbols": list(symbols),
            "horizon": "one month; 21 observations for realized-return diagnostics",
            "n_fields": len(FIELD_SPECS),
            "field_specs": [
                {"id": f"{group}.{field}", "label": label, "kind": kind}
                for group, field, label, kind in FIELD_SPECS
            ],
            "field_union_is_identical_across_replication_markets": True,
        },
        "sources": {
            "return_prices": {
                "file": config["prices"].name,
                "sha256": _sha256(config["prices"]),
                "purpose": "returns and risk statistics only",
            },
            "liquidity_closes": raw_close_source,
            "volumes": {"file": config["volumes"].name, "sha256": _sha256(config["volumes"])},
        },
        "snapshots": snapshots,
    }


def _validate(packet: dict) -> dict:
    issues = []
    if len(packet["snapshots"]) != len(REPLICATION_DATES):
        issues.append("wrong snapshot count")
    for snapshot in packet["snapshots"]:
        if snapshot["observed_through"] > snapshot["date"]:
            issues.append(f"future observation at {snapshot['date']}")
        if set(snapshot["forward_return_21d"]) != set(snapshot["symbols"]):
            issues.append(f"incomplete forward return at {snapshot['date']}")
        for symbol in snapshot["symbols"]:
            values = snapshot["features"].get(symbol, {})
            if set(values) != set(FIELD_IDS):
                issues.append(f"field mismatch {snapshot['date']} {symbol}")
            if any(not math.isfinite(float(value)) for value in values.values()):
                issues.append(f"nonfinite field {snapshot['date']} {symbol}")
    return {
        "schema_version": "information-capacity-crossmarket-data-validation-v1",
        "market": packet["market"],
        "n_snapshots": len(packet["snapshots"]),
        "n_assets": len(packet["snapshots"][0]["symbols"]),
        "n_fields": len(FIELD_IDS),
        "issues": issues,
        "passed": not issues,
    }


def main() -> None:
    for market, config in MARKETS.items():
        packet = _build_market(market, config)
        validation = _validate(packet)
        if not validation["passed"]:
            raise ValueError(f"{market} packet validation failed: {validation['issues'][:5]}")
        output = Path(DATA) / f"information_capacity_{market}_packets.json"
        validation_path = Path(RESULTS) / f"information_capacity_{market}_data_validation.json"
        _atomic_json(output, packet)
        _atomic_json(validation_path, validation)
        print(
            f"{market}: {validation['n_snapshots']} dates, {validation['n_assets']} assets, "
            f"{validation['n_fields']} fields -> {output}"
        )


if __name__ == "__main__":
    main()
