"""Analyze harmonized Europe/crypto information-capacity experiments."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import RESULTS
from cohort_crossvendor_hardening import matched_null
from cohort_information_capacity_analysis import METRICS, _mean_metrics, _metrics
from cohort_information_capacity_crossmarket_experiment import (
    ARCHITECTURES,
    PILOT_MODELS,
    TEAM_SIZES,
    paths,
    usable_response,
)
from cohort_information_diversity import _atomic_json, _sign_flip_p, _strict_complete


BOOTSTRAP_DRAWS = 20_000
FORMAT_STABLE_RETRY_THRESHOLD = 0.01


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _book(row: dict, symbols: list[str]) -> np.ndarray:
    raw = {item["symbol"]: float(item["weight"]) for item in row["weights"]}
    vector = np.asarray([raw.get(symbol, 0.0) for symbol in symbols], dtype=float)
    vector -= vector.mean()
    gross = float(np.abs(vector).sum())
    if gross <= 1e-12:
        raise ValueError(
            f"zero-gross book: {row.get('market')} {row.get('date')} "
            f"{row.get('vendor')} {row.get('condition')}"
        )
    return vector / gross


def _select(rows: list[dict], sample: str) -> list[dict]:
    return rows if sample == "all" else [row for row in rows if row.get("sample") == sample]


def _load_inputs(market: str, sample: str, allow_partial: bool):
    packet_path, plan_path, cache_path = paths(market)
    packet = _load(packet_path)
    plan = _load(plan_path)
    snapshots = packet["snapshots"]
    if sample != "all":
        pilot_dates = {row["date"] for row in snapshots[:4]}
        snapshots = [
            row for row in snapshots
            if (row["date"] in pilot_dates) == (sample == "pilot")
        ]
    cache = _select(_load(cache_path), sample)
    bad = [row for row in cache if not usable_response(row)]
    if bad and not allow_partial:
        raise ValueError(f"{market} {sample}: {len(bad)} incomplete rows")
    cache = [row for row in cache if usable_response(row)]
    expected_dates = len(snapshots)
    expected = expected_dates * 385
    if len(cache) != expected and not allow_partial:
        raise ValueError(f"{market} {sample}: expected {expected}, found {len(cache)}")
    if allow_partial:
        complete_dates = {
            date for date in {row["date"] for row in cache}
            if sum(row["date"] == date for row in cache) == 385
        }
        snapshots = [row for row in snapshots if row["date"] in complete_dates]
        cache = [row for row in cache if row["date"] in complete_dates]
        if not snapshots:
            raise ValueError(f"{market}: no complete 385-call date for partial analysis")
    return snapshots, cache, plan


def _date_rows(market: str, sample: str, allow_partial: bool) -> list[dict]:
    snapshots, cache, _ = _load_inputs(market, sample, allow_partial)
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    lookup = {(row["date"], row["vendor"], row["condition"]): row for row in cache}
    n_assets = len(snapshots[0]["symbols"])
    null_means = {
        k: matched_null(
            k,
            [f"A{index}" for index in range(n_assets)],
            B=10_000,
            seed=2026071800 + 10 * n_assets + k,
        )["mean"]
        for k in TEAM_SIZES
    }
    output = []
    for snapshot in snapshots:
        date = snapshot["date"]
        symbols = snapshot["symbols"]
        forward = np.asarray([snapshot["forward_return_21d"][symbol] for symbol in symbols])
        depth = np.asarray([snapshot["liquidity_depth"][symbol] for symbol in symbols])
        shared_rows = {vendor: lookup[date, vendor, "shared_full"] for vendor in vendors}
        shared_books = {vendor: _book(row, symbols) for vendor, row in shared_rows.items()}
        full_consensus = np.mean(list(shared_books.values()), axis=0)
        for k in TEAM_SIZES:
            shared_configs = []
            for combination in itertools.combinations(vendors, k):
                rows = [shared_rows[vendor] for vendor in combination]
                matrix = np.column_stack([shared_books[vendor] for vendor in combination])
                shared_configs.append(
                    _metrics(matrix, full_consensus, forward, depth, null_means[k], rows)
                )
            date_row = {
                "market": market,
                "sample": "pilot" if snapshot in snapshots[:4] and sample != "held-out" else "held-out",
                "snapshot_id": snapshot["snapshot_id"],
                "date": date,
                "return_regime": snapshot["return_regime"],
                "volatility_regime": snapshot["volatility_regime"],
                "team_size": k,
                "shared": _mean_metrics(shared_configs),
                "shared_configurations": len(shared_configs),
            }
            for architecture in ARCHITECTURES:
                books = {
                    (vendor, packet): _book(
                        lookup[date, vendor, f"{architecture}_k{k}_p{packet}"], symbols
                    )
                    for vendor in vendors for packet in range(1, k + 1)
                }
                configs = []
                for combination in itertools.combinations(vendors, k):
                    for assignment in itertools.permutations(range(1, k + 1)):
                        rows = [
                            lookup[date, vendor, f"{architecture}_k{k}_p{packet}"]
                            for vendor, packet in zip(combination, assignment)
                        ]
                        matrix = np.column_stack([
                            books[vendor, packet]
                            for vendor, packet in zip(combination, assignment)
                        ])
                        configs.append(
                            _metrics(matrix, full_consensus, forward, depth, null_means[k], rows)
                        )
                date_row[architecture] = _mean_metrics(configs)
                date_row[f"{architecture}_configurations"] = len(configs)
            output.append(date_row)
        print(f"{market}: analyzed {date}")
    return output


def _bootstrap(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def _collection_diagnostics(cache: list[dict]) -> dict:
    vendors = sorted({row["vendor"] for row in cache})
    by_vendor = {}
    for vendor in vendors:
        rows = [row for row in cache if row["vendor"] == vendor]
        retried = sum(int(row.get("format_retry_count", 0)) > 0 for row in rows)
        by_vendor[vendor] = {
            "rows": len(rows),
            "retried_rows": retried,
            "retry_rate": retried / len(rows),
        }
    stable = [
        vendor for vendor in vendors
        if by_vendor[vendor]["retry_rate"] <= FORMAT_STABLE_RETRY_THRESHOLD
    ]
    retried = sum(int(row.get("format_retry_count", 0)) > 0 for row in cache)
    return {
        "collection_wave": "earlier harmonized transfer wave",
        "rows": len(cache),
        "format_retry_rows": retried,
        "format_retry_rate": retried / len(cache),
        "generation_id_coverage": sum(bool(row.get("response_id")) for row in cache) / len(cache),
        "request_payload_hash_coverage": sum(
            bool(row.get("request_payload_sha256")) for row in cache
        ) / len(cache),
        "retry_semantics": (
            "format-only follow-up appended after an invalid structured response; market data, "
            "mandate, model route, and assigned fields were unchanged"
        ),
        "by_vendor": by_vendor,
        "format_stable_vendor_rule": f"vendor retry rate <= {FORMAT_STABLE_RETRY_THRESHOLD:.0%}",
        "format_stable_vendors": stable,
    }


def _format_stable_sensitivity(
    snapshots: list[dict],
    cache: list[dict],
    diagnostics: dict,
) -> dict:
    vendors = diagnostics["format_stable_vendors"]
    if len(vendors) < 3:
        raise ValueError("format-stable sensitivity requires at least three routes")
    vendors = vendors[:3]
    k = len(vendors)
    lookup = {(row["date"], row["vendor"], row["condition"]): row for row in cache}
    n_assets = len(snapshots[0]["symbols"])
    null_mean = matched_null(
        k,
        [f"A{index}" for index in range(n_assets)],
        B=10_000,
        seed=2026071880 + 10 * n_assets + k,
    )["mean"]
    date_rows = []
    used_rows = {}
    for snapshot in snapshots:
        date = snapshot["date"]
        symbols = snapshot["symbols"]
        forward = np.asarray([snapshot["forward_return_21d"][symbol] for symbol in symbols])
        depth = np.asarray([snapshot["liquidity_depth"][symbol] for symbol in symbols])
        shared_rows = [lookup[date, vendor, "shared_full"] for vendor in vendors]
        shared_matrix = np.column_stack([_book(row, symbols) for row in shared_rows])
        full_consensus = shared_matrix.mean(axis=1)
        output = {
            "date": date,
            "shared": _metrics(
                shared_matrix, full_consensus, forward, depth, null_mean, shared_rows
            ),
        }
        used_rows.update({id(row): row for row in shared_rows})
        for architecture in ARCHITECTURES:
            configurations = []
            for assignment in itertools.permutations(range(1, k + 1)):
                rows = [
                    lookup[date, vendor, f"{architecture}_k{k}_p{packet}"]
                    for vendor, packet in zip(vendors, assignment)
                ]
                used_rows.update({id(row): row for row in rows})
                matrix = np.column_stack([_book(row, symbols) for row in rows])
                configurations.append(
                    _metrics(matrix, full_consensus, forward, depth, null_mean, rows)
                )
            output[architecture] = _mean_metrics(configurations)
        date_rows.append(output)
    levels = {
        design: {
            metric: float(np.mean([row[design][metric] for row in date_rows]))
            for metric in METRICS
        }
        for design in ("shared", *ARCHITECTURES)
    }
    contrasts = {}
    seed = 2026071960
    for treatment in ARCHITECTURES:
        contrasts[f"{treatment}_minus_shared"] = {}
        for metric in ("N_eff", "same_side_overlap", "convex_order_concentration"):
            values = np.asarray([
                row[treatment][metric] - row["shared"][metric] for row in date_rows
            ])
            contrasts[f"{treatment}_minus_shared"][metric] = {
                "mean": float(values.mean()),
                "ci95": _bootstrap(values, seed),
                "n_dates": len(values),
            }
            seed += 1
    used = list(used_rows.values())
    return {
        "status": "post-collection format-stability sensitivity",
        "selection_rule": diagnostics["format_stable_vendor_rule"],
        "vendors": vendors,
        "team_size": k,
        "n_dates": len(date_rows),
        "used_rows": len(used),
        "used_rows_retried": sum(
            int(row.get("format_retry_count", 0)) > 0 for row in used
        ),
        "levels": levels,
        "contrasts": contrasts,
    }


def _summarize(
    rows: list[dict],
    market: str,
    sample: str,
    plan: dict,
    snapshots: list[dict],
    cache: list[dict],
) -> dict:
    curve = []
    for k in TEAM_SIZES:
        selected = [row for row in rows if row["team_size"] == k]
        for architecture in ("shared", *ARCHITECTURES):
            curve.append({
                "team_size": k,
                "architecture": architecture,
                **{
                    metric: float(np.mean([row[architecture][metric] for row in selected]))
                    for metric in METRICS
                },
            })
    k7 = [row for row in rows if row["team_size"] == 7]
    contrasts = {}
    index = 0
    for left, right in (
        ("mixed", "shared"),
        ("coherent", "shared"),
        ("coherent", "mixed"),
    ):
        contrasts[f"{left}_minus_{right}"] = {}
        for metric in ("N_eff", "same_side_overlap", "convex_order_concentration", "consensus_forward_return"):
            values = np.asarray([row[left][metric] - row[right][metric] for row in k7])
            contrasts[f"{left}_minus_{right}"][metric] = {
                "mean": float(values.mean()),
                "ci95": _bootstrap(values, 2026071900 + index),
                "sign_flip_p": _sign_flip_p(values),
                "n_dates": len(values),
            }
            index += 1
    diagnostics = _collection_diagnostics(cache)
    return {
        "schema_version": "information-capacity-crossmarket-results-v1",
        "market": market,
        "sample": sample,
        "n_dates": len({row["date"] for row in rows}),
        "assignment_sha256": plan["assignment_sha256"],
        "job_manifest_sha256": plan["job_manifest_sha256"],
        "curve": curve,
        "k7_contrasts": contrasts,
        "collection_diagnostics": diagnostics,
        "format_stable_route_sensitivity": _format_stable_sensitivity(
            snapshots, cache, diagnostics
        ),
        "date_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=("us", "europe", "crypto"), required=True)
    parser.add_argument("--sample", choices=("pilot", "held-out", "all"), default="all")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    rows = _date_rows(args.market, args.sample, args.allow_partial)
    snapshots, cache, plan = _load_inputs(args.market, args.sample, args.allow_partial)
    result = _summarize(rows, args.market, args.sample, plan, snapshots, cache)
    suffix = "" if args.sample == "all" else f"_{args.sample.replace('-', '_')}"
    if args.allow_partial:
        suffix += "_partial"
    output = Path(RESULTS) / f"information_capacity_{args.market}_results{suffix}.json"
    summary = Path(RESULTS) / f"information_capacity_{args.market}_summary{suffix}.csv"
    _atomic_json(output, result)
    pd.DataFrame(result["curve"]).to_csv(summary, index=False)
    print(f"wrote {output}")
    for row in result["curve"]:
        if row["team_size"] == 7:
            print(
                row["architecture"],
                f"N_eff={row['N_eff']:.3f}",
                f"O={row['same_side_overlap']:.3f}",
                f"Q={row['convex_order_concentration']:.3f}",
            )


if __name__ == "__main__":
    main()
