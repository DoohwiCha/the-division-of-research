"""Describe all K=7 route-to-packet assignments without recollection or inference.

Run: python3 src/cohort_information_capacity_assignment_dispersion.py
Writes only the assignment-dispersion JSON and the v7 review-package tables.
The 34 dates remain the analysis units; assignments are exhaustive sensitivity
configurations, not independent observations. Table quantiles are arithmetic
means of the 34 within-date quantiles, NOT quantiles of a pooled sample or CIs.
"""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np

from _paths import RESULTS, ROOT
from cohort_crossvendor_hardening import matched_null
from cohort_information_capacity_analysis import METRICS, _book, _metrics
from cohort_information_capacity_audited_analysis import (
    AUDITED_CACHE_PATH,
    INTEGRITY_CACHE_PATH,
    PLAN_PATH,
    RESULT_PATH as PUBLISHED_ARCHITECTURE_PATH,
    _current,
    _load,
)
from cohort_information_capacity_audited_curve_analysis import (
    RESULT_PATH as PUBLISHED_CURVE_PATH,
)
from cohort_information_capacity_integrity import (
    CONFIRMATORY_OUTPUT,
    INTEGRITY_CORRECTION_ID,
    PILOT_OUTPUT,
    TEAM_SIZE,
    _load_packets,
)
from cohort_information_diversity import PILOT_MODELS, _atomic_json


RESULT_PATH = Path(RESULTS) / "information_capacity_assignment_dispersion_results.json"
TABLE_DIR = Path(ROOT) / "submissions/icaif-2026-v7-camera-ready-review/tables"
N_DATES = 34
N_ASSIGNMENTS = 5040
DESIGNS = ("mixed", "coherent")
QUANTILES = ("q05", "q50", "q95")
ATOL = 1e-10
RTOL = 1e-12


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _distribution(values: np.ndarray) -> dict:
    _require(bool(np.isfinite(values).all()), "non-finite assignment metric")
    quantiles = np.quantile(values, [0.05, 0.50, 0.95], method="linear")
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        **{key: float(value) for key, value in zip(QUANTILES, quantiles)},
        "max": float(values.max()),
    }


def _published_rows(path: Path) -> dict:
    result = _load(path)
    _require(result["integrity_correction_id"] == INTEGRITY_CORRECTION_ID,
             f"{path.name}: incorrect integrity protocol")
    rows = [row for row in result["per_date"] if row["team_size"] == TEAM_SIZE]
    indexed = {row["date"]: row for row in rows}
    _require(len(rows) == len(indexed) == N_DATES,
             f"{path.name}: expected exactly 34 unique K=7 dates")
    for row in rows:
        for design in DESIGNS:
            _require(row["configuration_counts"][design] == N_ASSIGNMENTS,
                     f"{path.name}: incorrect published assignment count")
    return indexed


def _check_means(actual: dict, expected: dict, context: str) -> dict:
    errors = {}
    for metric in METRICS:
        value, reference = actual[metric], expected[metric]
        _require(bool(np.isfinite(reference)) and bool(np.isclose(
            value, reference, atol=ATOL, rtol=RTOL)),
            f"{context} {metric}: reconstructed {value!r}, published {reference!r}")
        errors[metric] = abs(value - reference)
    return errors


def _date_quantile_summary(rows: list[dict], design: str, metric: str) -> dict:
    return {
        key: {
            "mean_across_dates": float(values.mean()),
            "min_across_dates": float(values.min()),
            "median_across_dates": float(np.median(values)),
            "max_across_dates": float(values.max()),
        }
        for key in QUANTILES
        for values in [np.asarray([row[design][metric][key] for row in rows])]
    }


def _summarize(rows: list[dict], design: str) -> dict:
    worst = sorted(rows, key=lambda row: (row[design]["N_eff_minus_shared"]["min"], row["date"]))
    fractions = np.asarray([row[design]["fraction_above_shared"] for row in rows])
    return {
        "mean_N_eff_across_dates": float(np.mean([row[design]["metrics"]["N_eff"]["mean"] for row in rows])),
        "mean_gap_across_dates": float(np.mean([row[design]["N_eff_minus_shared"]["mean"] for row in rows])),
        "N_eff_date_quantiles": _date_quantile_summary(
            [{**row, design: row[design]["metrics"]} for row in rows], design, "N_eff"),
        "gap_date_quantiles": _date_quantile_summary(rows, design, "N_eff_minus_shared"),
        "fraction_above_shared": {
            "mean_across_dates": float(fractions.mean()),
            "min_across_dates": float(fractions.min()),
            "max_across_dates": float(fractions.max()),
        },
        "assignments_above_shared": sum(row[design]["counts_vs_shared"]["above"] for row in rows),
        "assignments_equal_shared": sum(row[design]["counts_vs_shared"]["equal"] for row in rows),
        "assignments_below_shared": sum(row[design]["counts_vs_shared"]["below"] for row in rows),
        "dates_with_negative_single_assignment_gap": [
            row["date"] for row in rows if row[design]["counts_vs_shared"]["below"] > 0
        ],
        "worst_dates_by_minimum_gap": [
            {
                "date": row["date"],
                "min_gap": row[design]["N_eff_minus_shared"]["min"],
                "q05_gap": row[design]["N_eff_minus_shared"]["q05"],
                "fraction_above_shared": row[design]["fraction_above_shared"],
                "counts_vs_shared": row[design]["counts_vs_shared"],
                "minimum_gap_assignment": row[design]["minimum_gap_assignment"],
            }
            for row in worst[:5]
        ],
    }


def _write_tables(result: dict) -> None:
    summaries = result["summary"]
    lines = [
        "% Generated by src/cohort_information_capacity_assignment_dispersion.py.",
        r"\footnotesize",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcc}",
        r"\toprule",
        r"Assignment sensitivity & Mixed & Coherent \\",
        r"\midrule",
    ]
    for label, field, signed in (
        (r"$N_{\mathrm{eff}}$: $q_{.05}/q_{.50}/q_{.95}$", "N_eff_date_quantiles", False),
        (r"$\Delta N_{\mathrm{eff}}$: $q_{.05}/q_{.50}/q_{.95}$", "gap_date_quantiles", True),
    ):
        cells = [" / ".join(format(summaries[design][field][q]["mean_across_dates"],
                                   "+.2f" if signed else ".2f") for q in QUANTILES)
                 for design in DESIGNS]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    for label, cells in (
        ("Above shared: mean date fraction", [f"{100 * summaries[d]['fraction_above_shared']['mean_across_dates']:.3f}\\%" for d in DESIGNS]),
        ("Above shared: lowest date fraction", [f"{100 * summaries[d]['fraction_above_shared']['min_across_dates']:.2f}\\%" for d in DESIGNS]),
        (r"Smallest single-assignment $\Delta N_{\mathrm{eff}}$", [f"{summaries[d]['worst_dates_by_minimum_gap'][0]['min_gap']:+.3f}" for d in DESIGNS]),
        ("Date of smallest gap", [summaries[d]["worst_dates_by_minimum_gap"][0]["date"] for d in DESIGNS]),
        ("Dates with any negative gap", [f"{len(summaries[d]['dates_with_negative_single_assignment_gap'])}/34" for d in DESIGNS]),
        ("Assignments below shared", [f"{summaries[d]['assignments_below_shared']:,}" for d in DESIGNS]),
    ):
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\par\vspace{2pt}\begingroup\scriptsize",
        r"All $7!=5{,}040$ one-to-one route-to-packet assignments per design and date; 34 dates. "
        r"The first two rows are arithmetic means across dates of within-date quantiles, not pooled quantiles or confidence intervals. "
        r"$\Delta$ subtracts that date's shared $N_{\mathrm{eff}}$. Fractions use strict $>$; "
        r"counts below shared are out of $171{,}360$ configurations per design, not independent observations. "
        r"No new inference is performed.\par\endgroup",
    ])
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "assignment_dispersion.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    macros = ["% Generated assignment sensitivity numbers; quantiles are means of date-level quantiles."]
    for design in DESIGNS:
        prefix = "AD" + design.title()
        summary = summaries[design]
        numbers = {
            "MeanNeff": f"{summary['mean_N_eff_across_dates']:.2f}",
            "MeanGap": f"{summary['mean_gap_across_dates']:+.2f}",
            "AbovePercent": f"{100 * summary['fraction_above_shared']['mean_across_dates']:.1f}",
            "MinAbovePercent": f"{100 * summary['fraction_above_shared']['min_across_dates']:.2f}",
            "MinGap": f"{summary['worst_dates_by_minimum_gap'][0]['min_gap']:+.3f}",
            "WorstDate": summary["worst_dates_by_minimum_gap"][0]["date"],
            "NegativeDates": str(len(summary["dates_with_negative_single_assignment_gap"])),
            "BelowCount": str(summary["assignments_below_shared"]),
        }
        for name, q in zip(("Low", "Median", "High"), QUANTILES):
            numbers["Neff" + name] = f"{summary['N_eff_date_quantiles'][q]['mean_across_dates']:.2f}"
            numbers["Gap" + name] = f"{summary['gap_date_quantiles'][q]['mean_across_dates']:+.2f}"
        macros.extend("\\newcommand{\\" + prefix + name + "}{" + value + "}" for name, value in numbers.items())
    (TABLE_DIR / "assignment_dispersion_numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")


def main() -> None:
    _require(TEAM_SIZE == 7 and len(PILOT_MODELS) == 7, "analysis requires seven routes")
    vendors = [vendor for vendor, _ in PILOT_MODELS]
    _require(len(set(vendors)) == 7, "route labels must be unique")
    published = {path.name: _published_rows(path) for path in (
        PUBLISHED_ARCHITECTURE_PATH, PUBLISHED_CURVE_PATH)}
    packets = _load_packets()
    dates = {item["snapshot"]["date"] for item in packets}
    _require(len(packets) == len(dates) == N_DATES, "expected 34 unique dates")
    for name, reference in published.items():
        _require(set(reference) == dates, f"{name}: date inclusion differs from audited packets")
    shared_rows = [row for row in _load(INTEGRITY_CACHE_PATH)
                   if row.get("architecture") == "shared" and _current(row)]
    audited_rows = [row for row in _load(AUDITED_CACHE_PATH) if _current(row)]
    shared_lookup = {(row["date"], row["vendor"]): row for row in shared_rows}
    audited_lookup = {(row["date"], row["vendor"], row["architecture"], int(row["packet_index"])): row
                      for row in audited_rows}
    expected_shared = {(date, vendor) for date in dates for vendor in vendors}
    expected_audited = {(date, vendor, design + "_audited", packet)
                        for date in dates for vendor in vendors for design in DESIGNS
                        for packet in range(1, TEAM_SIZE + 1)}
    _require(len(shared_rows) == 238 and set(shared_lookup) == expected_shared,
             "shared cache must contain each of the 238 required keys exactly once")
    _require(len(audited_rows) == 3332 and set(audited_lookup) == expected_audited,
             "audited cache must contain each of the 3332 required keys exactly once")
    assignments = np.asarray(list(itertools.permutations(range(TEAM_SIZE))), dtype=int)
    _require(assignments.shape == (N_ASSIGNMENTS, TEAM_SIZE), "expected 5040 complete assignments")
    null_mean = matched_null(TEAM_SIZE, [f"A{index}" for index in range(14)],
                             B=10_000, seed=2026071700 + TEAM_SIZE)["mean"]
    errors = {name: {metric: 0.0 for metric in METRICS} for name in published}
    output = []
    route_indices = np.arange(TEAM_SIZE)
    for item in packets:
        snapshot = item["snapshot"]
        date, symbols = snapshot["date"], snapshot["symbols"]
        _require(len(symbols) == len(set(symbols)) == 14, f"{date}: expected 14 assets")
        date_shared = [shared_lookup[date, vendor] for vendor in vendors]
        shared_matrix = np.column_stack([_book(row, symbols) for row in date_shared])
        full_consensus = shared_matrix.mean(axis=1)
        forward = np.asarray([snapshot["forward_return_21d"][symbol] for symbol in symbols])
        adv = np.asarray([snapshot["features"]["market"][symbol]["adv_bn"] for symbol in symbols])
        shared = _metrics(shared_matrix, full_consensus, forward, adv, null_mean, date_shared)
        row = {"date": date, "sample": item["sample"], "snapshot_id": snapshot["snapshot_id"],
               "team_size": TEAM_SIZE, "shared": shared}
        reconstructed = {"shared": shared}
        for design in DESIGNS:
            cached = [[audited_lookup[date, vendor, design + "_audited", packet]
                       for packet in range(1, TEAM_SIZE + 1)] for vendor in vendors]
            books = np.asarray([[_book(book, symbols) for book in route] for route in cached])
            values = np.empty((N_ASSIGNMENTS, len(METRICS)))
            for index, assignment in enumerate(assignments):
                matrix = books[route_indices, assignment].T
                usage = [cached[route][packet] for route, packet in enumerate(assignment)]
                metrics = _metrics(matrix, full_consensus, forward, adv, null_mean, usage)
                values[index] = [metrics[metric] for metric in METRICS]
            _require(bool(np.isfinite(values).all()), f"{date} {design}: non-finite metric")
            distributions = {metric: _distribution(values[:, index])
                             for index, metric in enumerate(METRICS)}
            reconstructed[design] = {metric: distributions[metric]["mean"] for metric in METRICS}
            gap = values[:, METRICS.index("N_eff")] - shared["N_eff"]
            counts = {"above": int(np.count_nonzero(gap > 0)),
                      "equal": int(np.count_nonzero(gap == 0)),
                      "below": int(np.count_nonzero(gap < 0))}
            _require(sum(counts.values()) == N_ASSIGNMENTS, f"{date} {design}: incomplete assignments")
            row[design] = {
                "n_assignments": len(values), "metrics": distributions,
                "N_eff_minus_shared": _distribution(gap), "counts_vs_shared": counts,
                "fraction_above_shared": counts["above"] / N_ASSIGNMENTS,
                "minimum_gap_assignment": {
                    vendor: int(packet) + 1
                    for vendor, packet in zip(vendors, assignments[int(np.argmin(gap))])
                },
            }
        for name, reference in published.items():
            for key in ("sample", "snapshot_id", "team_size"):
                _require(row[key] == reference[date][key], f"{date}: published {key} mismatch")
            for design, means in reconstructed.items():
                difference = _check_means(means, reference[date][design], f"{name} {date} {design}")
                for metric, value in difference.items():
                    errors[name][metric] = max(errors[name][metric], value)
        output.append(row)
        print(f"  generated assignment dispersion {date}: 5040 per design", flush=True)
    source_paths = [INTEGRITY_CACHE_PATH, AUDITED_CACHE_PATH, PILOT_OUTPUT, CONFIRMATORY_OUTPUT,
                    PLAN_PATH, PUBLISHED_ARCHITECTURE_PATH, PUBLISHED_CURVE_PATH,
                    Path(__file__), Path(__file__).with_name("cohort_information_capacity_analysis.py"),
                    Path(__file__).with_name("cohort_information_capacity_audited_analysis.py")]
    result = {
        "schema_version": "information-capacity-assignment-dispersion-v1",
        "integrity_correction_id": INTEGRITY_CORRECTION_ID,
        "team_size": TEAM_SIZE, "n_dates": len(output),
        "n_assignments_per_date_per_design": N_ASSIGNMENTS,
        "date_is_inference_unit": True, "new_model_calls": 0,
        "interpretation": "Exhaustive within-date assignment sensitivity; no new inferential tests or intervals.",
        "quantile_method": "numpy.quantile(method='linear'); q05, q50, q95 within each date and design",
        "summary_quantiles": "Arithmetic mean, minimum, median and maximum across the 34 date-level quantiles; never pooled quantiles.",
        "pairing": "Each assignment minus the same date's single seven-route shared N_eff.",
        "comparison_rule": "Strict >, ==, < zero gap; no tolerance used to classify gaps.",
        "assignment_order": "Lexicographic permutations of packet indices 1..7 in route_order.",
        "route_order": vendors, "random_book_null_mean": float(null_mean),
        "source_sha256": {str(path.relative_to(Path(ROOT))): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_paths},
        "invariants": {"published_mean_atol": ATOL, "published_mean_rtol": RTOL,
                       "all_metrics_all_dates_all_designs_match_published": True,
                       "max_absolute_mean_error": errors},
        "per_date": output,
        "summary": {design: _summarize(output, design) for design in DESIGNS},
    }
    _atomic_json(RESULT_PATH, result)
    _write_tables(result)
    for design, summary in result["summary"].items():
        worst = summary["worst_dates_by_minimum_gap"][0]
        print(f"{design}: mean N_eff={summary['mean_N_eff_across_dates']:.12f}; "
              f"below shared={summary['assignments_below_shared']}; "
              f"minimum gap={worst['min_gap']:.12f} ({worst['date']})")
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
