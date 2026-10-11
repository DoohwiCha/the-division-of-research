"""Build point-in-time information packets for the information-diversity pilot.

The pilot holds the U.S. equity universe, investment objective, horizon, model
roster, and output constraints fixed.  It changes only which information
channels an analyst may use.  Public price/volume data already frozen in this
repository are combined with point-in-time SEC facts and filing metadata and
public market-context series.

Run:
    PYTHONPATH=src python3 src/cohort_information_diversity_data.py

The derived packet is intentionally compact and machine-readable.  Raw SEC
responses are cached outside the repository by default because company-facts
files are large and can be downloaded again from the recorded URLs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _paths import DATA, RESULTS
from cohort_equity_monthly import SYMBOLS


SCHEMA_VERSION = "information-diversity-packets-v2-pit-corrected"
SEC_HISTORY_SCHEMA_VERSION = "information-diversity-packets-v3-sec-history-corrected"
OUTPUT_PATH = Path(DATA) / "information_diversity_packets.json"
VALIDATION_PATH = Path(RESULTS) / "information_diversity_data_validation.json"
CONFIRMATORY_OUTPUT_PATH = Path(DATA) / "information_diversity_confirmatory_packets.json"
CONFIRMATORY_VALIDATION_PATH = Path(RESULTS) / "information_diversity_confirmatory_data_validation.json"
PRICE_PATH = Path(DATA) / "jfs_global_prices_united_states.csv"
VOLUME_PATH = Path(DATA) / "jfs_global_volumes_united_states.csv"
RAW_CLOSE_PATH = Path(DATA) / "information_diversity_us_split_adjusted_close.csv"
SPLIT_PATH = Path(DATA) / "information_diversity_us_splits.json"
PRICE_BASIS_PATH = Path(DATA) / "information_diversity_us_price_basis.json"
MONTHLY_PATH = Path(DATA) / "equity_monthly_snapshots.json"
EQUITY_HISTORY_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
EQUITY_HISTORY_END = datetime(2026, 6, 19, tzinfo=timezone.utc)
CORPORATE_ACTION_AVAILABLE_THROUGH = datetime(2026, 7, 20, tzinfo=timezone.utc)
CORPORATE_ACTION_REQUEST_END_EXCLUSIVE = datetime(2026, 7, 21, tzinfo=timezone.utc)

# Chosen before looking at any information-diversity output: four dates from
# each return tercile and six dates from each volatility state.
PILOT_DATES = (
    "2023-08-31",  # strong / high
    "2023-09-30",  # weak / normal
    "2023-10-31",  # weak / high
    "2023-11-30",  # middle / normal
    "2024-02-29",  # strong / normal
    "2024-04-30",  # middle / high
    "2024-08-31",  # strong / high
    "2024-11-30",  # middle / high
    "2025-04-30",  # weak / high
    "2025-07-31",  # strong / normal
    "2025-11-30",  # middle / normal
    "2026-01-31",  # weak / normal
)

COMPANIES = {
    "AAPL": ("Apple", "Information Technology"),
    "MSFT": ("Microsoft", "Information Technology"),
    "NVDA": ("NVIDIA", "Information Technology"),
    "GOOGL": ("Alphabet", "Communication Services"),
    "META": ("Meta Platforms", "Communication Services"),
    "AMZN": ("Amazon", "Consumer Discretionary"),
    "TSLA": ("Tesla", "Consumer Discretionary"),
    "WMT": ("Walmart", "Consumer Staples"),
    "PG": ("Procter & Gamble", "Consumer Staples"),
    "JPM": ("JPMorgan Chase", "Financials"),
    "BAC": ("Bank of America", "Financials"),
    "XOM": ("Exxon Mobil", "Energy"),
    "JNJ": ("Johnson & Johnson", "Health Care"),
    "UNH": ("UnitedHealth Group", "Health Care"),
}

# Exxon Mobil's historical filings use the long-standing issuer CIK.  The SEC
# ticker map currently points XOM at a 2026 successor shell, which has no
# filings for the pilot dates.
CIK_OVERRIDES = {"XOM": 34088}

MACRO_TICKERS = {
    "ten_year_yield": "^TNX",
    "three_month_yield": "^IRX",
    "vix": "^VIX",
    "equity_market": "SPY",
    "high_yield_bonds": "HYG",
}

EVENT_LABELS = {
    "1.01": "material agreement",
    "1.02": "agreement termination",
    "2.01": "acquisition or disposal",
    "2.02": "earnings release",
    "2.05": "restructuring costs",
    "2.06": "material impairment",
    "5.02": "director or officer change",
    "7.01": "regulation FD disclosure",
    "8.01": "other material event",
}

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "Information-access academic research contact research@example.com"
)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _download(url: str, cache_path: Path, user_agent: str, attempts: int = 5) -> bytes:
    if cache_path.exists():
        return cache_path.read_bytes()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            accept = "text/csv" if cache_path.suffix == ".csv" else "application/json"
            request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": accept})
            with urllib.request.urlopen(request, timeout=120) as response:
                value = response.read()
            cache_path.write_bytes(value)
            time.sleep(0.12)
            return value
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def _json_download(url: str, cache_path: Path, user_agent: str) -> tuple[dict, str]:
    raw = _download(url, cache_path, user_agent)
    return json.loads(raw), _sha256_bytes(raw)


def _yahoo_series(ticker: str, cache_dir: Path) -> tuple[pd.Series, dict]:
    period1 = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp())
    period2 = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp())
    encoded = urllib.parse.quote(ticker, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?"
        f"period1={period1}&period2={period2}&interval=1d&events=history"
    )
    safe_name = ticker.replace("^", "index_")
    raw = _download(url, cache_dir / f"yahoo_{safe_name}.json", "Mozilla/5.0")
    payload = json.loads(raw)["chart"]["result"][0]
    timestamps = payload["timestamp"]
    closes = payload["indicators"]["quote"][0]["close"]
    values = {
        pd.Timestamp(value, unit="s").normalize(): float(close)
        for value, close in zip(timestamps, closes)
        if close is not None and math.isfinite(float(close))
    }
    return pd.Series(values).sort_index(), {"url": url, "sha256": _sha256_bytes(raw)}


def _yahoo_equity_history(
    ticker: str, cache_dir: Path
) -> tuple[pd.Series, list[dict[str, Any]], dict[str, str]]:
    period1 = int(EQUITY_HISTORY_START.timestamp())
    period2 = int(CORPORATE_ACTION_REQUEST_END_EXCLUSIVE.timestamp())
    encoded = urllib.parse.quote(ticker, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?"
        f"period1={period1}&period2={period2}&interval=1d&events=div%2Csplits"
    )
    raw = _download(url, cache_dir / f"yahoo_equity_{ticker}.json", "Mozilla/5.0")
    payload = json.loads(raw)["chart"]["result"][0]
    timestamps = payload["timestamp"]
    closes = payload["indicators"]["quote"][0]["close"]
    values = {
        pd.Timestamp(value, unit="s").normalize(): float(close)
        for value, close in zip(timestamps, closes)
        if close is not None and math.isfinite(float(close))
    }
    splits = []
    for event in (payload.get("events", {}).get("splits", {}) or {}).values():
        numerator = float(event.get("numerator", 0.0))
        denominator = float(event.get("denominator", 0.0))
        factor = numerator / denominator if denominator > 0 else math.nan
        if not math.isfinite(factor) or factor <= 0:
            ratio = str(event.get("splitRatio", ""))
            left, separator, right = ratio.partition(":")
            factor = float(left) / float(right) if separator and float(right) > 0 else math.nan
        if math.isfinite(factor) and factor > 0:
            splits.append({
                "date": pd.Timestamp(int(event["date"]), unit="s").normalize().date().isoformat(),
                "factor": float(factor),
                "split_ratio": str(event.get("splitRatio", f"{factor}:1")),
            })
    return (
        pd.Series(values).sort_index(),
        sorted(splits, key=lambda row: row["date"]),
        {"url": url, "sha256": _sha256_bytes(raw)},
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, float_format="%.12g", date_format="%Y-%m-%d")
    os.replace(temp, path)


def _build_equity_price_basis(
    cache_dir: Path, target_index: pd.DatetimeIndex
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]], dict[str, Any]]:
    closes = {}
    splits = {}
    sources = {}
    for symbol in SYMBOLS:
        close, symbol_splits, source = _yahoo_equity_history(symbol, cache_dir)
        closes[symbol] = close
        splits[symbol] = symbol_splits
        sources[symbol] = source
    frame = pd.DataFrame(closes).sort_index()
    frame = frame.loc[frame.index < pd.Timestamp(EQUITY_HISTORY_END.replace(tzinfo=None))]
    frame = frame.reindex(target_index).ffill()
    missing = {
        symbol: int(frame[symbol].isna().sum())
        for symbol in SYMBOLS
        if frame[symbol].isna().any()
    }
    if missing:
        raise ValueError(f"raw close basis has missing observations: {missing}")
    _write_csv(RAW_CLOSE_PATH, frame)
    _atomic_json(SPLIT_PATH, {
        "schema_version": "information-diversity-split-events-v1",
        "basis_cutoff": CORPORATE_ACTION_AVAILABLE_THROUGH.date().isoformat(),
        "splits": splits,
    })
    metadata = {
        "schema_version": "information-diversity-price-basis-v1",
        "return_price_basis": (
            "repository adjusted close; used only for returns, volatility, drawdown, and beta"
        ),
        "valuation_price_basis": (
            "Yahoo quote close; dividend-unadjusted and split-adjusted to the recorded corporate-action cutoff"
        ),
        "share_basis": (
            "as-filed SEC shares multiplied by splits after the share measurement date"
        ),
        "corporate_action_cutoff": CORPORATE_ACTION_AVAILABLE_THROUGH.date().isoformat(),
        "sources": sources,
        "raw_close_sha256": _sha256_bytes(RAW_CLOSE_PATH.read_bytes()),
        "split_sha256": _sha256_bytes(SPLIT_PATH.read_bytes()),
    }
    _atomic_json(PRICE_BASIS_PATH, metadata)
    return frame, splits, metadata


def _asof(series: pd.Series, date: str) -> float | None:
    values = series.loc[series.index <= pd.Timestamp(date)].dropna()
    return float(values.iloc[-1]) if len(values) else None


def _trailing_return(series: pd.Series, date: str, sessions: int = 63) -> float | None:
    values = series.loc[series.index <= pd.Timestamp(date)].dropna()
    if len(values) <= sessions:
        return None
    return float(values.iloc[-1] / values.iloc[-1 - sessions] - 1.0)


def _fact_rows(companyfacts: dict, namespace: str, tags: tuple[str, ...], unit: str) -> list[dict]:
    facts = companyfacts.get("facts", {}).get(namespace, {})
    rows = []
    for priority, tag in enumerate(tags):
        units = facts.get(tag, {}).get("units", {})
        if unit in units:
            rows.extend(dict(row, _tag=tag, _tag_priority=priority) for row in units[unit])
    return rows


def _eligible_fact_rows(rows: list[dict], snapshot_date: str) -> list[dict]:
    cutoff = pd.Timestamp(snapshot_date)
    out = []
    for row in rows:
        if row.get("form") not in {"10-Q", "10-K", "10-Q/A", "10-K/A"}:
            continue
        if not row.get("filed") or not row.get("end"):
            continue
        if pd.Timestamp(row["filed"]) > cutoff or pd.Timestamp(row["end"]) > cutoff:
            continue
        try:
            value = float(row["val"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            out.append(row)
    return out


def _latest_instant(companyfacts: dict, tags: tuple[str, ...], unit: str, snapshot_date: str) -> dict | None:
    namespaces = ("us-gaap", "dei") if unit == "shares" else ("us-gaap",)
    candidates = []
    for namespace in namespaces:
        candidates.extend(
            dict(row, _namespace=namespace)
            for row in _fact_rows(companyfacts, namespace, tags, unit)
        )
    rows = _eligible_fact_rows(candidates, snapshot_date)
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            row["filed"],
            row["end"],
            -int(row.get("_tag_priority", 0)),
            row.get("accn", ""),
        ),
    )


def _annual_series(
    companyfacts: dict, tags: tuple[str, ...], snapshot_date: str, unit: str = "USD"
) -> list[dict]:
    cutoff = pd.Timestamp(snapshot_date)
    rows = []
    for row in _eligible_fact_rows(_fact_rows(companyfacts, "us-gaap", tags, unit), snapshot_date):
        if row.get("form") not in {"10-K", "10-K/A"} or not row.get("start"):
            continue
        duration = (pd.Timestamp(row["end"]) - pd.Timestamp(row["start"])).days
        if 300 <= duration <= 390 and pd.Timestamp(row["filed"]) <= cutoff:
            rows.append(row)
    by_end: dict[str, dict] = {}
    for row in rows:
        end = row["end"]
        candidate = (row["filed"], -int(row.get("_tag_priority", 0)), row.get("accn", ""))
        incumbent = by_end.get(end)
        incumbent_key = (
            incumbent["filed"], -int(incumbent.get("_tag_priority", 0)), incumbent.get("accn", "")
        ) if incumbent else None
        if incumbent is None or candidate > incumbent_key:
            by_end[end] = row
    return sorted(by_end.values(), key=lambda row: row["end"])


def _value(row: dict | None) -> float | None:
    return float(row["val"]) if row else None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= 1e-12:
        return None
    value = numerator / denominator
    return float(value) if math.isfinite(value) else None


def _split_factor_after(split_events: list[dict[str, Any]], measurement_date: str) -> float:
    return float(np.prod([
        float(event["factor"])
        for event in split_events
        if pd.Timestamp(event["date"]) > pd.Timestamp(measurement_date)
    ]))


def _fundamentals(
    companyfacts: dict,
    snapshot_date: str,
    price: float,
    split_events: list[dict[str, Any]],
) -> dict[str, Any]:
    assets_row = _latest_instant(companyfacts, ("Assets",), "USD", snapshot_date)
    equity_row = _latest_instant(
        companyfacts,
        ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        "USD",
        snapshot_date,
    )
    cash_row = _latest_instant(
        companyfacts,
        ("CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
        "USD",
        snapshot_date,
    )
    debt_row = _latest_instant(
        companyfacts,
        ("LongTermDebtAndFinanceLeaseObligations", "LongTermDebt", "LongTermDebtNoncurrent"),
        "USD",
        snapshot_date,
    )
    shares_row = _latest_instant(
        companyfacts,
        ("CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"),
        "shares",
        snapshot_date,
    )
    diluted_shares = _annual_series(
        companyfacts,
        ("WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"),
        snapshot_date,
        unit="shares",
    )
    revenues = _annual_series(
        companyfacts,
        ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"),
        snapshot_date,
    )
    incomes = _annual_series(companyfacts, ("NetIncomeLoss", "ProfitLoss"), snapshot_date)
    revenue = _value(revenues[-1]) if revenues else None
    prior_revenue = _value(revenues[-2]) if len(revenues) >= 2 else None
    net_income = _value(incomes[-1]) if incomes else None
    assets = _value(assets_row)
    equity = _value(equity_row)
    cash = _value(cash_row)
    debt = _value(debt_row)
    selected_shares_row = shares_row or (diluted_shares[-1] if diluted_shares else None)
    shares_as_filed = _value(selected_shares_row)
    split_factor = (
        _split_factor_after(split_events, selected_shares_row["end"])
        if selected_shares_row is not None
        else None
    )
    shares = (
        shares_as_filed * split_factor
        if shares_as_filed is not None and split_factor is not None
        else None
    )
    market_cap = price * shares if shares is not None else None
    pe = _safe_ratio(market_cap, net_income) if net_income is not None and net_income > 0 else None
    return {
        "pe": pe,
        "price_to_sales": _safe_ratio(market_cap, revenue),
        "price_to_book": _safe_ratio(market_cap, equity),
        "roa": _safe_ratio(net_income, assets),
        "debt_to_assets": _safe_ratio(debt, assets),
        "cash_to_assets": _safe_ratio(cash, assets),
        "revenue_growth": (
            _safe_ratio(revenue - prior_revenue, abs(prior_revenue))
            if revenue is not None and prior_revenue is not None
            else None
        ),
        "accounting_period_end": revenues[-1]["end"] if revenues else None,
        "accounting_filed": revenues[-1]["filed"] if revenues else None,
        "shares_as_filed": shares_as_filed,
        "shares_split_adjustment": split_factor,
        "shares_on_price_basis": shares,
        "shares_period_end": selected_shares_row.get("end") if selected_shares_row else None,
        "shares_filed": selected_shares_row.get("filed") if selected_shares_row else None,
        "shares_tag": selected_shares_row.get("_tag") if selected_shares_row else None,
        "shares_namespace": (
            selected_shares_row.get("_namespace", "us-gaap")
            if selected_shares_row
            else None
        ),
    }


def _filing_table_rows(table: dict) -> list[dict]:
    """Convert an SEC column-oriented filing table to row records."""
    if not table:
        return []
    keys = [key for key, values in table.items() if isinstance(values, list)]
    if not keys:
        return []
    lengths = {len(table[key]) for key in keys}
    if len(lengths) != 1:
        raise ValueError(f"SEC filing columns have inconsistent lengths: {sorted(lengths)}")
    return [
        {key: table[key][index] for key in keys}
        for index in range(next(iter(lengths)))
    ]


def _submission_rows(submissions: dict) -> list[dict]:
    """Return the complete SEC filing history, including paginated archives."""
    rows = _filing_table_rows(submissions.get("filings", {}).get("recent", {}))
    for archived in submissions.get("_historical_filing_tables", []):
        rows.extend(_filing_table_rows(archived))

    # SEC archive files can overlap the rolling recent table around page boundaries.
    # Accession number is stable; the fallback protects older records that omit it.
    deduplicated: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row.get("accessionNumber"),
            row.get("filingDate"),
            row.get("form"),
            row.get("primaryDocument"),
        )
        deduplicated[key] = row
    return sorted(
        deduplicated.values(),
        key=lambda row: (str(row.get("filingDate", "")), str(row.get("accessionNumber", ""))),
    )


def _load_submission_history(
    submissions: dict,
    cik: str,
    cache_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Download every SEC submissions archive named by the issuer index."""
    tables = []
    sources = []
    for descriptor in submissions.get("filings", {}).get("files", []) or []:
        name = Path(str(descriptor.get("name", ""))).name
        if not name or name != descriptor.get("name") or not name.endswith(".json"):
            raise ValueError(f"unsafe SEC submissions archive name for CIK{cik}: {descriptor!r}")
        url = f"https://data.sec.gov/submissions/{name}"
        payload, payload_hash = _json_download(
            url,
            cache_dir / name,
            SEC_USER_AGENT,
        )
        tables.append(payload)
        sources.append({
            "name": name,
            "url": url,
            "sha256": payload_hash,
            "filing_count": int(descriptor.get("filingCount", 0) or 0),
            "filing_from": descriptor.get("filingFrom"),
            "filing_to": descriptor.get("filingTo"),
        })
    return tables, sources


def _events(submissions: dict, snapshot_date: str) -> dict[str, Any]:
    cutoff = pd.Timestamp(snapshot_date)
    rows = [
        row for row in _submission_rows(submissions)
        if row.get("filingDate") and pd.Timestamp(row["filingDate"]) <= cutoff
    ]
    filings_8k = [row for row in rows if row.get("form") in {"8-K", "8-K/A"}]

    def within(row: dict, days: int) -> bool:
        delta = (cutoff - pd.Timestamp(row["filingDate"])).days
        return 0 <= delta <= days

    recent_90 = [row for row in filings_8k if within(row, 90)]
    recent_30 = [row for row in filings_8k if within(row, 30)]
    item_codes = []
    for row in recent_90:
        item_codes.extend(code.strip() for code in str(row.get("items", "")).split(",") if code.strip())
    latest_event = max(recent_90, key=lambda row: row["filingDate"]) if recent_90 else None
    periodic = [row for row in rows if row.get("form") in {"10-Q", "10-K"}]
    latest_periodic = max(periodic, key=lambda row: row["filingDate"]) if periodic else None
    latest_labels = []
    if latest_event:
        for code in str(latest_event.get("items", "")).split(","):
            code = code.strip()
            if code:
                latest_labels.append(EVENT_LABELS.get(code, f"item {code}"))
    return {
        "eight_k_30d": len(recent_30),
        "eight_k_90d": len(recent_90),
        "earnings_release_90d": sum(code == "2.02" for code in item_codes),
        "material_event_90d": sum(code in {"1.01", "1.02", "2.01", "2.05", "2.06", "5.02", "8.01"} for code in item_codes),
        "days_since_latest_8k": (
            (cutoff - pd.Timestamp(latest_event["filingDate"])).days if latest_event else None
        ),
        "latest_8k_items": "; ".join(latest_labels[:3]) if latest_labels else "none in prior 90d",
        "days_since_periodic_filing": (
            (cutoff - pd.Timestamp(latest_periodic["filingDate"])).days if latest_periodic else None
        ),
        "latest_periodic_form": latest_periodic.get("form") if latest_periodic else None,
    }


def _market_features(
    prices: pd.DataFrame,
    raw_closes: pd.DataFrame,
    volumes: pd.DataFrame,
    snapshot_date: str,
) -> dict[str, dict]:
    history = prices.loc[prices.index <= pd.Timestamp(snapshot_date), SYMBOLS].ffill().dropna()
    volume = volumes.reindex(history.index).reindex(columns=SYMBOLS).fillna(0.0)
    raw_history = raw_closes.reindex(history.index).reindex(columns=SYMBOLS).ffill()
    if raw_history.isna().any().any():
        raise ValueError(f"missing raw close through {snapshot_date}")
    if len(history) < 253:
        raise ValueError(f"insufficient price history through {snapshot_date}")
    ret_7d = history.iloc[-1] / history.iloc[-6] - 1.0
    ret_30d = history.iloc[-1] / history.iloc[-22] - 1.0
    ret_90d = history.iloc[-1] / history.iloc[-64] - 1.0
    log_returns = np.log(history / history.shift(1))
    vol_30d = log_returns.iloc[-21:].std() * math.sqrt(252.0)
    drawdown_90d = history.iloc[-1] / history.iloc[-63:].max() - 1.0
    adv = (raw_history.iloc[-21:] * volume.iloc[-21:]).mean() / 1e9
    return {
        symbol: {
            "price": float(raw_history.iloc[-1][symbol]),
            "ret_7d": float(ret_7d[symbol]),
            "ret_30d": float(ret_30d[symbol]),
            "ret_90d": float(ret_90d[symbol]),
            "vol_30d": float(vol_30d[symbol]),
            "drawdown_90d": float(drawdown_90d[symbol]),
            "adv_bn": float(adv[symbol]),
        }
        for symbol in SYMBOLS
    }


def _risk_features(prices: pd.DataFrame, snapshot_date: str) -> dict[str, dict]:
    history = prices.loc[prices.index <= pd.Timestamp(snapshot_date)].ffill().iloc[-253:]
    returns = history.pct_change(fill_method=None).dropna()
    market = returns.mean(axis=1, skipna=True)
    market_variance = float(market.var())
    downside = market < 0
    out: dict[str, dict] = {}
    for symbol in SYMBOLS:
        stock = returns[symbol].reindex(market.index).dropna()
        aligned_market = market.reindex(stock.index)
        beta = float(stock.cov(aligned_market) / market_variance) if market_variance > 1e-12 else 0.0
        down_market = aligned_market[downside.reindex(aligned_market.index).fillna(False)]
        down_stock = stock.reindex(down_market.index)
        down_variance = float(down_market.var())
        downside_beta = float(down_stock.cov(down_market) / down_variance) if down_variance > 1e-12 else 0.0
        residual = stock - beta * aligned_market
        running_max = history[symbol].cummax()
        max_drawdown = float((history[symbol] / running_max - 1.0).min())
        out[symbol] = {
            "market_beta": beta,
            "downside_beta": downside_beta,
            "market_corr": float(stock.corr(aligned_market)),
            "idio_vol": float(residual.std() * math.sqrt(252.0)),
            "max_drawdown_1y": max_drawdown,
        }
    return out


def _forward_returns(prices: pd.DataFrame, snapshot_date: str, horizon: int = 21) -> dict[str, float]:
    before = prices.loc[prices.index <= pd.Timestamp(snapshot_date), SYMBOLS].ffill()
    after = prices.loc[prices.index > pd.Timestamp(snapshot_date), SYMBOLS].ffill()
    if before.empty or len(after) < horizon:
        raise ValueError(f"fewer than {horizon} forward sessions after {snapshot_date}")
    value = after.iloc[horizon - 1] / before.iloc[-1] - 1.0
    return {symbol: float(value[symbol]) for symbol in SYMBOLS}


def _fmt(value: float | None, scale: float = 1.0, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value) * scale:.{digits}f}"


def _core_table() -> str:
    rows = [f"{'ticker':<7}{'company':<22}{'sector':<26}"]
    for symbol in SYMBOLS:
        company, sector = COMPANIES[symbol]
        rows.append(f"{symbol:<7}{company:<22}{sector:<26}")
    return "\n".join(rows)


def _simple_table(market: dict[str, dict]) -> str:
    rows = [f"{'ticker':<7}{'ret_7d%':>10}{'ret_30d%':>11}{'ret_90d%':>11}{'vol_30d%':>12}{'dd_90d%':>11}"]
    for symbol in SYMBOLS:
        row = market[symbol]
        rows.append(
            f"{symbol:<7}{_fmt(row['ret_7d'],100):>10}{_fmt(row['ret_30d'],100):>11}"
            f"{_fmt(row['ret_90d'],100):>11}{_fmt(row['vol_30d'],100):>12}{_fmt(row['drawdown_90d'],100):>11}"
        )
    return "\n".join(rows)


def _market_table(market: dict[str, dict]) -> str:
    rows = [f"{'ticker':<7}{'ret_7d%':>10}{'ret_30d%':>11}{'ret_90d%':>11}{'vol_30d%':>12}{'dd_90d%':>11}{'ADV_$bn':>10}"]
    for symbol in SYMBOLS:
        row = market[symbol]
        rows.append(
            f"{symbol:<7}{_fmt(row['ret_7d'],100):>10}{_fmt(row['ret_30d'],100):>11}"
            f"{_fmt(row['ret_90d'],100):>11}{_fmt(row['vol_30d'],100):>12}"
            f"{_fmt(row['drawdown_90d'],100):>11}{_fmt(row['adv_bn']):>10}"
        )
    return "\n".join(rows)


def _fundamental_table(rows_by_symbol: dict[str, dict]) -> str:
    rows = [
        f"{'ticker':<7}{'P/E':>9}{'P/S':>9}{'P/B':>9}{'ROA%':>9}{'debt/A%':>10}{'cash/A%':>10}{'rev_g%':>9}{'age_d':>8}"
    ]
    for symbol in SYMBOLS:
        row = rows_by_symbol[symbol]
        age = None
        if row.get("accounting_filed"):
            age = (pd.Timestamp(row["snapshot_date"]) - pd.Timestamp(row["accounting_filed"])).days
        rows.append(
            f"{symbol:<7}{_fmt(row['pe']):>9}{_fmt(row['price_to_sales']):>9}{_fmt(row['price_to_book']):>9}"
            f"{_fmt(row['roa'],100):>9}{_fmt(row['debt_to_assets'],100):>10}{_fmt(row['cash_to_assets'],100):>10}"
            f"{_fmt(row['revenue_growth'],100):>9}{('NA' if age is None else str(age)):>8}"
        )
    return "\n".join(rows)


def _event_table(rows_by_symbol: dict[str, dict]) -> str:
    rows = [
        f"{'ticker':<7}{'8K_30d':>8}{'8K_90d':>8}{'earn_90d':>10}{'material':>10}{'last_8K_d':>11}{'periodic_d':>12}  latest event"
    ]
    for symbol in SYMBOLS:
        row = rows_by_symbol[symbol]
        rows.append(
            f"{symbol:<7}{row['eight_k_30d']:>8}{row['eight_k_90d']:>8}{row['earnings_release_90d']:>10}"
            f"{row['material_event_90d']:>10}{str(row['days_since_latest_8k'] or 'NA'):>11}"
            f"{str(row['days_since_periodic_filing'] or 'NA'):>12}  {row['latest_8k_items']}"
        )
    return "\n".join(rows)


def _risk_table(rows_by_symbol: dict[str, dict], macro: dict[str, float | None]) -> str:
    curve = None
    if macro.get("ten_year_yield") is not None and macro.get("three_month_yield") is not None:
        curve = macro["ten_year_yield"] - macro["three_month_yield"]
    macro_line = (
        "Macro state: "
        f"10y={_fmt(macro.get('ten_year_yield'))}%, 3m={_fmt(macro.get('three_month_yield'))}%, "
        f"10y-3m={_fmt(curve)}pp, VIX={_fmt(macro.get('vix'))}, "
        f"SPY_3m={_fmt(macro.get('equity_market_return_3m'),100)}%, "
        f"HYG_3m={_fmt(macro.get('high_yield_return_3m'),100)}%."
    )
    rows = [macro_line, f"{'ticker':<7}{'beta':>9}{'down_beta':>11}{'mkt_corr':>11}{'idio_vol%':>12}{'max_dd%':>10}"]
    for symbol in SYMBOLS:
        row = rows_by_symbol[symbol]
        rows.append(
            f"{symbol:<7}{_fmt(row['market_beta']):>9}{_fmt(row['downside_beta']):>11}"
            f"{_fmt(row['market_corr']):>11}{_fmt(row['idio_vol'],100):>12}{_fmt(row['max_drawdown_1y'],100):>10}"
        )
    return "\n".join(rows)


def build(
    cache_dir: Path,
    selected_dates: tuple[str, ...] = PILOT_DATES,
    output_path: Path = OUTPUT_PATH,
    validation_path: Path = VALIDATION_PATH,
    snapshot_prefix: str = "S",
    date_selection_note: str = "four dates per return tercile and six dates per volatility state; selected without information-diversity outcomes",
    schema_version: str = SCHEMA_VERSION,
) -> dict:
    prices = pd.read_csv(PRICE_PATH, index_col=0, parse_dates=True).sort_index()
    volumes = pd.read_csv(VOLUME_PATH, index_col=0, parse_dates=True).sort_index()
    raw_closes, split_events, price_basis = _build_equity_price_basis(cache_dir, prices.index)
    monthly = json.loads(MONTHLY_PATH.read_text(encoding="utf-8"))["snapshots"]
    monthly_by_date = {row["date"]: row for row in monthly}
    ticker_url = "https://www.sec.gov/files/company_tickers.json"
    ticker_map, ticker_hash = _json_download(ticker_url, cache_dir / "company_tickers.json", SEC_USER_AGENT)
    cik_by_ticker = {
        str(row["ticker"]).upper(): int(row["cik_str"])
        for row in ticker_map.values()
    }
    cik_by_ticker.update(CIK_OVERRIDES)
    missing_cik = sorted(set(SYMBOLS) - set(cik_by_ticker))
    if missing_cik:
        raise ValueError(f"missing SEC CIK values: {missing_cik}")

    sec_sources: dict[str, dict] = {}
    companyfacts: dict[str, dict] = {}
    submissions: dict[str, dict] = {}
    for symbol in SYMBOLS:
        cik = f"{cik_by_ticker[symbol]:010d}"
        facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        companyfacts[symbol], facts_hash = _json_download(
            facts_url, cache_dir / f"CIK{cik}_companyfacts.json", SEC_USER_AGENT
        )
        submissions[symbol], submissions_hash = _json_download(
            submissions_url, cache_dir / f"CIK{cik}_submissions.json", SEC_USER_AGENT
        )
        history_tables, history_sources = _load_submission_history(
            submissions[symbol], cik, cache_dir
        )
        submissions[symbol]["_historical_filing_tables"] = history_tables
        sec_sources[symbol] = {
            "cik": cik,
            "companyfacts_url": facts_url,
            "companyfacts_sha256": facts_hash,
            "submissions_url": submissions_url,
            "submissions_sha256": submissions_hash,
            "submission_history_files": history_sources,
            "complete_submission_rows": len(_submission_rows(submissions[symbol])),
        }

    macro_series: dict[str, pd.Series] = {}
    macro_sources: dict[str, dict] = {}
    for label, ticker in MACRO_TICKERS.items():
        macro_series[label], macro_sources[label] = _yahoo_series(ticker, cache_dir)

    core = _core_table()
    snapshots = []
    validation_rows = []
    for index, snapshot_date in enumerate(selected_dates, start=1):
        monthly_row = monthly_by_date[snapshot_date]
        market = _market_features(prices, raw_closes, volumes, snapshot_date)
        risk = _risk_features(prices, snapshot_date)
        macro = {
            "ten_year_yield": _asof(macro_series["ten_year_yield"], snapshot_date),
            "three_month_yield": _asof(macro_series["three_month_yield"], snapshot_date),
            "vix": _asof(macro_series["vix"], snapshot_date),
            "equity_market_return_3m": _trailing_return(macro_series["equity_market"], snapshot_date),
            "high_yield_return_3m": _trailing_return(macro_series["high_yield_bonds"], snapshot_date),
        }
        fundamentals: dict[str, dict] = {}
        events: dict[str, dict] = {}
        for symbol in SYMBOLS:
            row = _fundamentals(
                companyfacts[symbol],
                snapshot_date,
                market[symbol]["price"],
                split_events[symbol],
            )
            row["snapshot_date"] = snapshot_date
            fundamentals[symbol] = row
            events[symbol] = _events(submissions[symbol], snapshot_date)
        tables = {
            "simple": _simple_table(market),
            "market": _market_table(market),
            "fundamentals": _fundamental_table(fundamentals),
            "events": _event_table(events),
            "risk": _risk_table(risk, macro),
        }
        rich = "\n\n".join(
            [
                "MARKET BEHAVIOR AND LIQUIDITY\n" + tables["market"],
                "POINT-IN-TIME FUNDAMENTALS AND VALUATION\n" + tables["fundamentals"],
                "RECENT SEC FILING EVENTS\n" + tables["events"],
                "RISK AND MACRO CONTEXT\n" + tables["risk"],
            ]
        )
        fwd = _forward_returns(prices, snapshot_date)
        missing_fundamental = {
            key: sum(fundamentals[symbol].get(key) is None for symbol in SYMBOLS)
            for key in ("pe", "price_to_sales", "price_to_book", "roa", "debt_to_assets", "revenue_growth")
        }
        validation_rows.append({
            "date": snapshot_date,
            "return_regime": monthly_row["return_regime"],
            "volatility_regime": monthly_row["volatility_regime"],
            "missing_fundamental_values": missing_fundamental,
            "symbols_without_recent_8k": sum(events[symbol]["eight_k_90d"] == 0 for symbol in SYMBOLS),
            "forward_return_count": len(fwd),
            "future_share_filings": sum(
                bool(fundamentals[symbol].get("shares_filed"))
                and pd.Timestamp(fundamentals[symbol]["shares_filed"]) > pd.Timestamp(snapshot_date)
                for symbol in SYMBOLS
            ),
            "stale_share_filings_over_400d": sum(
                bool(fundamentals[symbol].get("shares_filed"))
                and (
                    pd.Timestamp(snapshot_date)
                    - pd.Timestamp(fundamentals[symbol]["shares_filed"])
                ).days > 400
                for symbol in SYMBOLS
            ),
        })
        snapshots.append({
            "snapshot_id": f"{snapshot_prefix}{index:02d}",
            "date": snapshot_date,
            "return_regime": monthly_row["return_regime"],
            "volatility_regime": monthly_row["volatility_regime"],
            "symbols": list(SYMBOLS),
            "common_core": core,
            "tables": {**tables, "rich": rich},
            "features": {
                "market": market,
                "fundamentals": fundamentals,
                "events": events,
                "risk": risk,
                "macro": macro,
            },
            "forward_return_21d": fwd,
        })

    payload = {
        "schema_version": schema_version,
        "design_frozen_before_llm_collection": True,
        "design": {
            "dates": list(selected_dates),
            "date_selection": date_selection_note,
            "symbols": list(SYMBOLS),
            "horizon": "one month; 21-session forward return used only for ex-post quality checks",
            "channels": ["market", "fundamentals", "events", "risk"],
            "simple_condition": "returns, volatility, and drawdown only",
            "rich_condition": "union of all four source packets",
            "specialized_condition": "one source packet per analyst; common objective and constraints unchanged",
        },
        "sources": {
            "return_prices": {
                "path": str(PRICE_PATH.relative_to(Path(DATA).parent)),
                "sha256": _sha256_bytes(PRICE_PATH.read_bytes()),
                "purpose": "returns and risk statistics only",
            },
            "valuation_closes": {
                "path": str(RAW_CLOSE_PATH.relative_to(Path(DATA).parent)),
                "sha256": _sha256_bytes(RAW_CLOSE_PATH.read_bytes()),
                "purpose": "valuation and average daily dollar volume",
            },
            "corporate_actions": {
                "path": str(SPLIT_PATH.relative_to(Path(DATA).parent)),
                "sha256": _sha256_bytes(SPLIT_PATH.read_bytes()),
            },
            "price_basis": price_basis,
            "volumes": {"path": str(VOLUME_PATH.relative_to(Path(DATA).parent)), "sha256": _sha256_bytes(VOLUME_PATH.read_bytes())},
            "sec_ticker_map": {"url": ticker_url, "sha256": ticker_hash},
            "sec": sec_sources,
            "macro_market_series": macro_sources,
        },
        "snapshots": snapshots,
    }
    validation = {
        "schema_version": "information-diversity-data-validation-v1",
        "packet_sha256": None,
        "n_dates": len(snapshots),
        "return_regime_counts": pd.Series([row["return_regime"] for row in snapshots]).value_counts().to_dict(),
        "volatility_regime_counts": pd.Series([row["volatility_regime"] for row in snapshots]).value_counts().to_dict(),
        "rows": validation_rows,
    }
    _atomic_json(output_path, payload)
    validation["packet_sha256"] = _sha256_bytes(output_path.read_bytes())
    _atomic_json(validation_path, validation)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("INFORMATION_DIVERSITY_SOURCE_CACHE", "/tmp/information-access-sources")),
    )
    parser.add_argument("--confirmatory", action="store_true")
    args = parser.parse_args()
    if args.confirmatory:
        monthly = json.loads(MONTHLY_PATH.read_text(encoding="utf-8"))["snapshots"]
        selected_dates = tuple(
            row["date"] for row in monthly
            if row["date"] not in PILOT_DATES and row["date"] <= "2026-04-30"
        )
        if len(selected_dates) != 22:
            raise ValueError(f"expected 22 confirmatory dates, got {len(selected_dates)}")
        output_path = CONFIRMATORY_OUTPUT_PATH
        validation_path = CONFIRMATORY_VALIDATION_PATH
        payload = build(
            args.cache_dir,
            selected_dates=selected_dates,
            output_path=output_path,
            validation_path=validation_path,
            snapshot_prefix="H",
            date_selection_note=(
                "all pre-existing U.S. month-ends not used in the 12-date pilot, excluding 2026-05-31 "
                "because 21 forward sessions were unavailable; frozen before confirmatory LLM collection"
            ),
        )
    else:
        output_path = OUTPUT_PATH
        payload = build(args.cache_dir)
    print(f"wrote {output_path.relative_to(Path(DATA).parent)}")
    print(f"dates={len(payload['snapshots'])}; symbols={len(SYMBOLS)}; channels=4")


if __name__ == "__main__":
    main()
