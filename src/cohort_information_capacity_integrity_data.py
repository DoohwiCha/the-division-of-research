"""Build SEC-history-corrected packets without replacing the published cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _paths import DATA, RESULTS
from cohort_information_diversity_data import (
    CONFIRMATORY_OUTPUT_PATH,
    MONTHLY_PATH,
    OUTPUT_PATH,
    PILOT_DATES,
    SEC_HISTORY_SCHEMA_VERSION,
    build,
)


PILOT_OUTPUT = Path(DATA) / "information_diversity_integrity_packets.json"
CONFIRMATORY_OUTPUT = Path(DATA) / "information_diversity_integrity_confirmatory_packets.json"
PILOT_VALIDATION = Path(RESULTS) / "information_diversity_integrity_data_validation.json"
CONFIRMATORY_VALIDATION = (
    Path(RESULTS) / "information_diversity_integrity_confirmatory_data_validation.json"
)
DIFF_OUTPUT = Path(RESULTS) / "information_diversity_integrity_event_diff.json"


def _event_rows(packet: dict) -> dict[tuple[str, str], dict]:
    return {
        (snapshot["date"], symbol): values
        for snapshot in packet["snapshots"]
        for symbol, values in snapshot["features"]["events"].items()
    }


def _diff(old_paths: tuple[Path, Path], new_paths: tuple[Path, Path]) -> dict:
    old = {}
    new = {}
    for path in old_paths:
        old.update(_event_rows(json.loads(path.read_text(encoding="utf-8"))))
    for path in new_paths:
        new.update(_event_rows(json.loads(path.read_text(encoding="utf-8"))))
    fields = sorted(next(iter(new.values())))
    changed_rows = []
    field_counts = {field: 0 for field in fields}
    for key in sorted(new):
        differences = {
            field: {"old": old[key].get(field), "new": new[key].get(field)}
            for field in fields
            if old[key].get(field) != new[key].get(field)
        }
        if differences:
            changed_rows.append({"date": key[0], "symbol": key[1], "differences": differences})
            for field in differences:
                field_counts[field] += 1
    return {
        "schema_version": "information-capacity-integrity-event-diff-v1",
        "old_packet_paths": [str(path.relative_to(Path(DATA).parent)) for path in old_paths],
        "new_packet_paths": [str(path.relative_to(Path(DATA).parent)) for path in new_paths],
        "n_symbol_dates": len(new),
        "n_changed_symbol_dates": len(changed_rows),
        "n_changed_field_values": sum(field_counts.values()),
        "changed_field_counts": field_counts,
        "affected_dates": sorted({row["date"] for row in changed_rows}),
        "affected_symbols": sorted({row["symbol"] for row in changed_rows}),
        "rows": changed_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/information-access-sources"),
    )
    args = parser.parse_args()

    monthly = json.loads(MONTHLY_PATH.read_text(encoding="utf-8"))["snapshots"]
    confirmation_dates = tuple(
        row["date"]
        for row in monthly
        if row["date"] not in PILOT_DATES and row["date"] <= "2026-04-30"
    )
    if len(confirmation_dates) != 22:
        raise ValueError(f"expected 22 complement dates, found {len(confirmation_dates)}")

    build(
        args.cache_dir,
        selected_dates=PILOT_DATES,
        output_path=PILOT_OUTPUT,
        validation_path=PILOT_VALIDATION,
        snapshot_prefix="S",
        date_selection_note="12 previously designated development dates",
        schema_version=SEC_HISTORY_SCHEMA_VERSION,
    )
    build(
        args.cache_dir,
        selected_dates=confirmation_dates,
        output_path=CONFIRMATORY_OUTPUT,
        validation_path=CONFIRMATORY_VALIDATION,
        snapshot_prefix="H",
        date_selection_note="22 complementary month-ends; not described as untouched held-out data",
        schema_version=SEC_HISTORY_SCHEMA_VERSION,
    )

    diff = _diff(
        (OUTPUT_PATH, CONFIRMATORY_OUTPUT_PATH),
        (PILOT_OUTPUT, CONFIRMATORY_OUTPUT),
    )
    DIFF_OUTPUT.write_text(json.dumps(diff, indent=2, allow_nan=False), encoding="utf-8")
    print(
        f"SEC correction changed {diff['n_changed_field_values']} values in "
        f"{diff['n_changed_symbol_dates']} symbol-date rows"
    )
    print(f"wrote {PILOT_OUTPUT} and {CONFIRMATORY_OUTPUT}")


if __name__ == "__main__":
    main()
