"""Export exact rendered prompt examples from the frozen primary design."""
from __future__ import annotations

import json
from pathlib import Path

from _paths import DATA, RESULTS, ROOT
from cohort_information_capacity_experiment import FIELD_BY_ID, PILOT_MODELS
from cohort_information_capacity_integrity import (
    _atomic_json,
    _base_job,
    _load_packets,
    _model_catalog,
    _sha256_file,
)
from cohort_information_capacity_audited_architecture import (
    PLAN_PATH as AUDITED_PLAN_PATH,
    _frozen_assignment,
)


OUTPUT_PATH = Path(RESULTS) / "information_capacity_prompt_examples.json"
PACKET_PATHS = (
    Path(DATA) / "information_diversity_integrity_packets.json",
    Path(DATA) / "information_diversity_integrity_confirmatory_packets.json",
)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(Path(ROOT).resolve()))


def _example(job: dict) -> dict:
    return {
        "date": job["date"],
        "sample": job["sample"],
        "architecture": job["architecture"],
        "condition": job["condition"],
        "packet_index": job["packet_index"],
        "field_ids": job["field_ids"],
        "requested_model": job["requested_model"],
        "temperature": job["temperature"],
        "prompt_spec_version": job["prompt_spec_version"],
        "prompt_sha256": job["prompt_sha256"],
        "packet_sha256": job["packet_sha256"],
        "request_payload_sha256": job["request_payload_sha256"],
        "rendered_prompt": job["prompt"],
    }


def main() -> None:
    packets = _load_packets()
    item = packets[0]
    vendor, model = PILOT_MODELS[0]
    catalog = _model_catalog()
    catalog_by_model = {row["requested_model"]: row for row in catalog["models"]}
    assignment = _frozen_assignment()

    jobs = {
        "shared_full": _base_job(
            item=item,
            architecture="shared",
            packet_index=0,
            field_ids=list(FIELD_BY_ID),
            vendor=vendor,
            model=model,
            catalog_by_model=catalog_by_model,
        ),
        "mixed_seat_1": _base_job(
            item=item,
            architecture="mixed_audited",
            packet_index=1,
            field_ids=list(assignment["mixed"][0]["field_ids"]),
            vendor=vendor,
            model=model,
            catalog_by_model=catalog_by_model,
        ),
        "coherent_seat_1": _base_job(
            item=item,
            architecture="coherent_audited",
            packet_index=1,
            field_ids=list(assignment["coherent"][0]["field_ids"]),
            vendor=vendor,
            model=model,
            catalog_by_model=catalog_by_model,
        ),
    }
    output = {
        "schema_version": "information-capacity-prompt-examples-v1",
        "purpose": (
            "Exact prompt renderings for reproducibility; no model call is made. The calendar date "
            "is hidden in every rendered prompt, while ticker identities remain visible."
        ),
        "example_selection": {
            "rule": "earliest primary packet and first frozen model route",
            "date": item["snapshot"]["date"],
            "requested_model": model,
        },
        "sources": {
            "packet_files": [
                {"path": _relative(path), "sha256": _sha256_file(path)}
                for path in PACKET_PATHS
            ],
            "assignment_plan": {
                "path": _relative(AUDITED_PLAN_PATH),
                "sha256": _sha256_file(AUDITED_PLAN_PATH),
                "assignment_sha256": json.loads(
                    AUDITED_PLAN_PATH.read_text(encoding="utf-8")
                )["assignment_sha256"],
            },
        },
        "examples": {label: _example(job) for label, job in jobs.items()},
    }
    _atomic_json(OUTPUT_PATH, output)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
