"""Measure how information specialization scales with analyst-team size.

This analysis uses only the frozen pilot and held-out books.  For matched team
sizes k=2,3,4 it compares (i) distinct model families that all receive the
same rich packet with (ii) one model receiving k distinct research channels.
It also reports a convex co-order concentration index, with and without
average-dollar-volume scaling.  No new LLM calls are made.

Run:
    PYTHONPATH=src python3 src/cohort_information_diversity_scaling.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import DATA, RESULTS
from cohort_information_diversity import (
    BOOTSTRAP_DRAWS,
    CHANNELS,
    PILOT_MODELS,
    _atomic_json,
    _matrix,
    _roster_metrics,
    _sign_flip_p,
    _strict_complete,
)


PILOT_PACKET = Path(DATA) / "information_diversity_packets.json"
HELDOUT_PACKET = Path(DATA) / "information_diversity_confirmatory_packets.json"
PILOT_CACHE = Path(DATA) / "cohort_agents_information_diversity.json"
HELDOUT_CACHE = Path(DATA) / "cohort_agents_information_diversity_confirmatory.json"
OUTPUT_PATH = Path(RESULTS) / "information_diversity_scaling.json"

TEAM_SIZES = (2, 3, 4)
IMPACT_DELTA = 0.5
EXTRA_METRICS = (
    "convex_coorder_unweighted",
    "convex_coorder_adv_weighted",
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _strict_rows(path: Path) -> list[dict]:
    rows = _load_json(path)
    bad = [row for row in rows if not _strict_complete(row)]
    if bad:
        raise ValueError(f"{path} contains {len(bad)} incomplete rows")
    return rows


def _coorder_index(frame: pd.DataFrame, adv: pd.Series | None, delta: float) -> float:
    """Convex cost of the simultaneous aggregate relative to separate books.

    Every input book is dollar neutral and unit gross.  A value above one
    means that aggregation reinforces positions enough to raise the convex
    order-size term; a lower value means that more positions offset.
    """
    values = frame.to_numpy(dtype=float)
    if adv is None:
        depth = np.ones(frame.shape[0], dtype=float)
    else:
        depth = adv.reindex(frame.index).to_numpy(dtype=float)
        if not np.all(np.isfinite(depth)) or np.any(depth <= 0):
            raise ValueError("ADV must be finite and positive")
        depth = depth / np.median(depth)
    scale = np.power(depth, delta)
    aggregate = np.sum(values, axis=1)
    numerator = np.sum(np.power(np.abs(aggregate), 1.0 + delta) / scale)
    denominator = np.sum(np.power(np.abs(values), 1.0 + delta) / scale[:, None])
    return float(numerator / denominator)


def _metrics(
    frame: pd.DataFrame,
    forward: dict[str, float],
    adv: pd.Series,
    delta: float,
    nulls: dict[int, dict],
) -> dict:
    values = _roster_metrics(frame, forward, nulls)
    values["convex_coorder_unweighted"] = _coorder_index(frame, None, delta)
    values["convex_coorder_adv_weighted"] = _coorder_index(frame, adv, delta)
    return values


def _mean(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _paired(rows: list[dict], treatment: str, control: str, metric: str, seed: int) -> dict:
    t = np.asarray([row[treatment][metric] for row in rows], dtype=float)
    c = np.asarray([row[control][metric] for row in rows], dtype=float)
    difference = t - c
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        rng.choice(difference, size=len(difference), replace=True).mean()
        for _ in range(BOOTSTRAP_DRAWS)
    ])
    return {
        "metric": metric,
        "n_dates": len(rows),
        "treatment_mean": float(t.mean()),
        "control_mean": float(c.mean()),
        "mean_difference": float(difference.mean()),
        "bootstrap_95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "dates_treatment_higher": int(np.sum(difference > 0)),
        "dates_treatment_lower": int(np.sum(difference < 0)),
        "dates_equal": int(np.sum(difference == 0)),
        "sign_flip_p_two_sided": _sign_flip_p(difference),
    }


def _date_rows(packet_path: Path, cache_path: Path, sample: str, delta: float) -> list[dict]:
    packet = _load_json(packet_path)
    records = _strict_rows(cache_path)
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    nulls: dict[int, dict] = {}
    output = []
    for snapshot in packet["snapshots"]:
        sid = snapshot["snapshot_id"]
        current = [row for row in records if row["snapshot_id"] == sid]
        symbols = snapshot["symbols"]
        forward = snapshot["forward_return_21d"]
        adv = pd.Series({
            symbol: snapshot["features"]["market"][symbol]["adv_bn"]
            for symbol in symbols
        })
        row = {
            "sample": sample,
            "snapshot_id": sid,
            "date": snapshot["date"],
            "return_regime": snapshot["return_regime"],
            "volatility_regime": snapshot["volatility_regime"],
        }
        for k in TEAM_SIZES:
            model_metrics = []
            for combination in itertools.combinations(vendors, k):
                selected = [
                    item for item in current
                    if item["condition"] == "rich"
                    and int(item["replicate"]) == 0
                    and item["vendor"] in combination
                ]
                frame = _matrix(selected, symbols, ("vendor",))
                model_metrics.append(_metrics(frame, forward, adv, delta, nulls))

            information_metrics = []
            for vendor in vendors:
                for channels in itertools.combinations(CHANNELS, k):
                    selected = [
                        item for item in current
                        if item["vendor"] == vendor and item["condition"] in channels
                    ]
                    frame = _matrix(selected, symbols, ("condition",))
                    information_metrics.append(_metrics(frame, forward, adv, delta, nulls))

            row[f"models_shared_rich_m{k}"] = _mean(model_metrics)
            row[f"information_same_model_m{k}"] = _mean(information_metrics)
        output.append(row)
    return output


def _provider_pilot_check(delta: float) -> list[dict]:
    packet = _load_json(PILOT_PACKET)
    records = _strict_rows(PILOT_CACHE)
    nulls: dict[int, dict] = {}
    rows = []
    for snapshot in packet["snapshots"]:
        sid = snapshot["snapshot_id"]
        current = [row for row in records if row["snapshot_id"] == sid]
        symbols = snapshot["symbols"]
        forward = snapshot["forward_return_21d"]
        adv = pd.Series({
            symbol: snapshot["features"]["market"][symbol]["adv_bn"]
            for symbol in symbols
        })
        for vendor, _ in PILOT_MODELS:
            information = [
                row for row in current
                if row["vendor"] == vendor and row["condition"] in CHANNELS
            ]
            sampling = [
                row for row in current
                if row["vendor"] == vendor and row["condition"] == "rich"
            ]
            rows.append({
                "date": snapshot["date"],
                "vendor": vendor,
                "information": _metrics(
                    _matrix(information, symbols, ("condition",)), forward, adv, delta, nulls
                ),
                "repeated_rich": _metrics(
                    _matrix(sampling, symbols, ("replicate",)), forward, adv, delta, nulls
                ),
            })
    output = []
    for vendor, _ in PILOT_MODELS:
        current = [row for row in rows if row["vendor"] == vendor]
        metrics = {}
        for metric in ("N_eff", "same_side_overlap", "surviving_gross", *EXTRA_METRICS):
            metrics[metric] = _paired(current, "information", "repeated_rich", metric, 2026071640)
        output.append({"vendor": vendor, "n_dates": len(current), "comparisons": metrics})
    return output


def _channel_pair_rows(packet: dict, records: list[dict], heldout: list[dict], delta: float) -> list[dict]:
    """Evaluate every two-channel specialist team against two shared-packet models."""
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    nulls: dict[int, dict] = {}
    controls = {row["snapshot_id"]: row["models_shared_rich_m2"] for row in heldout}
    output = []
    for snapshot in packet["snapshots"]:
        sid = snapshot["snapshot_id"]
        current = [row for row in records if row["snapshot_id"] == sid]
        symbols = snapshot["symbols"]
        forward = snapshot["forward_return_21d"]
        adv = pd.Series({
            symbol: snapshot["features"]["market"][symbol]["adv_bn"]
            for symbol in symbols
        })
        row = {
            "snapshot_id": sid,
            "date": snapshot["date"],
            "models_shared_rich_m2": controls[sid],
        }
        for channels in itertools.combinations(CHANNELS, 2):
            key = "__".join(channels)
            pair_metrics = []
            for vendor in vendors:
                selected = [
                    item for item in current
                    if item["vendor"] == vendor and item["condition"] in channels
                ]
                frame = _matrix(selected, symbols, ("condition",))
                pair_metrics.append(_metrics(frame, forward, adv, delta, nulls))
            row[key] = _mean(pair_metrics)
        output.append(row)
    return output


def _channel_pair_effects(rows: list[dict]) -> list[dict]:
    output = []
    seed = 2026071690
    for channels in itertools.combinations(CHANNELS, 2):
        key = "__".join(channels)
        effects = {}
        for metric in ("N_eff", "same_side_overlap", "surviving_gross"):
            effects[metric] = _paired(rows, key, "models_shared_rich_m2", metric, seed)
            seed += 1
        output.append({"channels": list(channels), "key": key, "effects": effects})
    return output


def _representative_case(packet: dict, records: list[dict], heldout: list[dict], delta: float) -> dict:
    """Select a non-cherry-picked date/provider and retain its four normalized books."""
    deltas = np.asarray([
        row["information_same_model_m4"]["N_eff"] - row["models_shared_rich_m4"]["N_eff"]
        for row in heldout
    ])
    target = float(np.median(deltas))
    case_row = min(
        heldout,
        key=lambda row: (
            abs(
                row["information_same_model_m4"]["N_eff"]
                - row["models_shared_rich_m4"]["N_eff"]
                - target
            ),
            row["date"],
        ),
    )
    snapshot = next(row for row in packet["snapshots"] if row["snapshot_id"] == case_row["snapshot_id"])
    current = [row for row in records if row["snapshot_id"] == case_row["snapshot_id"]]
    symbols = snapshot["symbols"]
    forward = snapshot["forward_return_21d"]
    adv = pd.Series({
        symbol: snapshot["features"]["market"][symbol]["adv_bn"]
        for symbol in symbols
    })
    nulls: dict[int, dict] = {}
    provider_frames = {}
    provider_metrics = {}
    for vendor, _ in PILOT_MODELS:
        selected = sorted(
            [row for row in current if row["vendor"] == vendor and row["condition"] in CHANNELS],
            key=lambda row: CHANNELS.index(row["condition"]),
        )
        frame = _matrix(selected, symbols, ("condition",)).reindex(columns=CHANNELS)
        provider_frames[vendor] = frame
        provider_metrics[vendor] = _metrics(frame, forward, adv, delta, nulls)
    provider_mean = float(np.mean([row["N_eff"] for row in provider_metrics.values()]))
    provider = min(
        provider_metrics,
        key=lambda name: (abs(provider_metrics[name]["N_eff"] - provider_mean), name),
    )
    frame = provider_frames[provider]
    books = []
    for channel in CHANNELS:
        values = frame[channel].sort_values()
        books.append({
            "channel": channel,
            "weights": {symbol: float(frame.loc[symbol, channel]) for symbol in symbols},
            "top_longs": [
                {"symbol": symbol, "weight": float(frame.loc[symbol, channel])}
                for symbol in values.index[-2:][::-1]
            ],
            "top_shorts": [
                {"symbol": symbol, "weight": float(frame.loc[symbol, channel])}
                for symbol in values.index[:2]
            ],
        })
    return {
        "selection_rule": (
            "held-out date nearest the median date-level N_eff gain; provider nearest that date's "
            "seven-provider mean specialist N_eff"
        ),
        "snapshot_id": case_row["snapshot_id"],
        "date": case_row["date"],
        "return_regime": case_row["return_regime"],
        "volatility_regime": case_row["volatility_regime"],
        "target_median_delta_N_eff": target,
        "date_delta_N_eff": float(
            case_row["information_same_model_m4"]["N_eff"]
            - case_row["models_shared_rich_m4"]["N_eff"]
        ),
        "provider": provider,
        "provider_specialist_metrics": provider_metrics[provider],
        "symbols": symbols,
        "books": books,
    }


def _regime_effects(packet: dict, heldout: list[dict]) -> dict:
    """Describe held-out treatment effects across return, volatility, and filing states."""
    snapshots = {row["snapshot_id"]: row for row in packet["snapshots"]}
    enriched = []
    for row in heldout:
        snapshot = snapshots[row["snapshot_id"]]
        recent_filings = sum(
            event.get("days_since_periodic_filing") is not None
            and float(event["days_since_periodic_filing"]) <= 30
            for event in snapshot["features"]["events"].values()
        )
        enriched.append({
            "snapshot_id": row["snapshot_id"],
            "date": row["date"],
            "return_regime": row["return_regime"],
            "volatility_regime": row["volatility_regime"],
            "recent_periodic_filings_30d": int(recent_filings),
            "delta_N_eff": float(
                row["information_same_model_m4"]["N_eff"]
                - row["models_shared_rich_m4"]["N_eff"]
            ),
            "delta_overlap": float(
                row["information_same_model_m4"]["same_side_overlap"]
                - row["models_shared_rich_m4"]["same_side_overlap"]
            ),
            "delta_adv_concentration": float(
                row["information_same_model_m4"]["convex_coorder_adv_weighted"]
                - row["models_shared_rich_m4"]["convex_coorder_adv_weighted"]
            ),
        })
    filing_median = float(np.median([row["recent_periodic_filings_30d"] for row in enriched]))
    for row in enriched:
        row["filing_activity"] = (
            "high" if row["recent_periodic_filings_30d"] > filing_median else "low"
        )

    specs = (
        ("weak_return", lambda row: row["return_regime"] == "weak"),
        ("other_return", lambda row: row["return_regime"] != "weak"),
        ("high_volatility", lambda row: row["volatility_regime"] == "high"),
        ("normal_volatility", lambda row: row["volatility_regime"] != "high"),
        ("filing_heavy", lambda row: row["filing_activity"] == "high"),
        ("filing_quiet", lambda row: row["filing_activity"] == "low"),
    )
    groups = {}
    for name, predicate in specs:
        selected = [row for row in enriched if predicate(row)]
        groups[name] = {
            "n_dates": len(selected),
            "mean_delta_N_eff": float(np.mean([row["delta_N_eff"] for row in selected])),
            "mean_delta_overlap": float(np.mean([row["delta_overlap"] for row in selected])),
            "mean_delta_adv_concentration": float(
                np.mean([row["delta_adv_concentration"] for row in selected])
            ),
            "dates_N_eff_higher": sum(row["delta_N_eff"] > 0 for row in selected),
            "dates_overlap_lower": sum(row["delta_overlap"] < 0 for row in selected),
        }
    return {
        "analysis_status": "post-hoc descriptive heterogeneity on the held-out dates",
        "filing_activity_definition": (
            "high when the number of universe firms with a 10-Q or 10-K in the prior 30 days "
            "exceeds the held-out date median"
        ),
        "filing_count_median": filing_median,
        "groups": groups,
        "per_date": enriched,
    }


def main() -> None:
    # The square-root impact convention implies total execution cost
    # proportional to |q|^(1 + 0.5).  This is a relative concentration index,
    # not a calibrated estimate of realized trading cost.
    delta = IMPACT_DELTA
    pilot = _date_rows(PILOT_PACKET, PILOT_CACHE, "pilot", delta)
    heldout = _date_rows(HELDOUT_PACKET, HELDOUT_CACHE, "held_out", delta)
    heldout_packet = _load_json(HELDOUT_PACKET)
    heldout_records = _strict_rows(HELDOUT_CACHE)
    channel_pair_rows = _channel_pair_rows(heldout_packet, heldout_records, heldout, delta)
    samples = {"pilot": pilot, "held_out": heldout, "pooled": pilot + heldout}
    comparisons = {}
    seed = 2026071660
    for sample, rows in samples.items():
        comparisons[sample] = {}
        for k in TEAM_SIZES:
            name = f"information_vs_models_m{k}"
            comparisons[sample][name] = {}
            for metric in (
                "N_eff", "N_eff_over_null", "same_side_overlap", "surviving_gross",
                "forward_return_mean", *EXTRA_METRICS,
            ):
                comparisons[sample][name][metric] = _paired(
                    rows,
                    f"information_same_model_m{k}",
                    f"models_shared_rich_m{k}",
                    metric,
                    seed,
                )
                seed += 1

    result = {
        "schema_version": "information-diversity-scaling-v2",
        "analysis_status": "post-hoc analysis of frozen books; no new LLM calls",
        "design": {
            "team_sizes": list(TEAM_SIZES),
            "pilot_dates": len(pilot),
            "held_out_dates": len(heldout),
            "matched_comparison": (
                "k distinct models sharing the rich packet versus one model receiving k distinct channels"
            ),
            "convex_exponent": delta,
            "convex_cost_power": 1.0 + delta,
            "interpretation": (
                "relative co-order concentration under the square-root impact convention; "
                "not a realized-cost estimate"
            ),
            "adv_unit": "billions of U.S. dollars; rescaled by the cross-sectional median within date",
            "inference_unit": "date; model combinations, channel subsets, and providers averaged within date",
        },
        "per_date": pilot + heldout,
        "paired_date_comparisons": comparisons,
        "pilot_provider_checks": _provider_pilot_check(delta),
        "heldout_channel_pair_effects": _channel_pair_effects(channel_pair_rows),
        "representative_heldout_case": _representative_case(
            heldout_packet, heldout_records, heldout, delta
        ),
        "heldout_regime_effects": _regime_effects(heldout_packet, heldout),
    }
    _atomic_json(OUTPUT_PATH, result)

    print("Information-specialization scaling")
    print(f"  convex exponent={delta:.3f}; dates={len(pilot) + len(heldout)}")
    for sample in ("held_out", "pooled"):
        print(f"  {sample}")
        for k in TEAM_SIZES:
            block = comparisons[sample][f"information_vs_models_m{k}"]
            nrow = block["N_eff"]
            arow = block["convex_coorder_adv_weighted"]
            print(
                f"    k={k}: dN={nrow['mean_difference']:+.3f} "
                f"[{nrow['bootstrap_95'][0]:+.3f},{nrow['bootstrap_95'][1]:+.3f}], "
                f"dADV-stress={arow['mean_difference']:+.3f} "
                f"[{arow['bootstrap_95'][0]:+.3f},{arow['bootstrap_95'][1]:+.3f}]"
            )
    print(f"  output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
