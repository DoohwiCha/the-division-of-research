"""Explicit authorization and provenance for the 2026-07-20 PIT correction."""
from __future__ import annotations

import os


PIT_CORRECTION_ID = "2026-07-20-price-share-basis-v2"
PIT_CORRECTION_ENV = "INFORMATION_ACCESS_PIT_CORRECTION"


def enabled() -> bool:
    return os.environ.get(PIT_CORRECTION_ENV) == PIT_CORRECTION_ID


def require_enabled(context: str) -> None:
    if not enabled():
        raise ValueError(
            f"{context}; set {PIT_CORRECTION_ENV}={PIT_CORRECTION_ID} "
            "to authorize the recorded PIT correction"
        )


def cache_record_is_current(row: dict) -> bool:
    """During the correction run, reject records from an earlier collection wave."""
    return not enabled() or row.get("pit_correction_id") == PIT_CORRECTION_ID


def response_marker() -> dict[str, str]:
    """Attach a stable collection-wave marker to every corrected API response."""
    return {"pit_correction_id": PIT_CORRECTION_ID}


def record() -> dict[str, str]:
    return {
        "id": PIT_CORRECTION_ID,
        "reason": (
            "align dividend-unadjusted split-adjusted closes with split-adjusted "
            "as-filed SEC shares and select the newest eligible US-GAAP/DEI share fact"
        ),
        "scope": "input correction only; dates, fields, models, prompts, and assignments unchanged",
    }
