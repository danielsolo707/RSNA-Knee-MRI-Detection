"""Strict validator for RSNA Knee Abnormality Detection submissions."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ID = "StudyInstanceUID"
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
EXPECTED = [ID, *TARGETS]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        return reader.fieldnames, list(reader)


def validate(submission: Path, test: Path) -> None:
    columns, rows = read_csv(submission)
    test_columns, test_rows = read_csv(test)

    if columns != EXPECTED:
        raise ValueError(f"columns/order mismatch: expected {EXPECTED}, got {columns}")
    if ID not in test_columns:
        raise ValueError(f"test file is missing {ID}")

    expected_ids = [row[ID] for row in test_rows]
    actual_ids = [row[ID] for row in rows]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("submission contains duplicate StudyInstanceUID values")
    if set(actual_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(actual_ids))[:5]
        extra = sorted(set(actual_ids) - set(expected_ids))[:5]
        raise ValueError(f"UID mismatch; missing={missing}, extra={extra}")

    for row_number, row in enumerate(rows, start=2):
        for target in TARGETS:
            try:
                value = float(row[target])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"row {row_number}, {target}: not numeric") from exc
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"row {row_number}, {target}: expected finite [0, 1], got {value}"
                )

    if len(rows) != len(test_rows):
        raise ValueError(f"row count mismatch: {len(rows)} vs {len(test_rows)}")
    print(f"OK: {submission} has {len(rows)} studies and 12 valid targets")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("test", type=Path)
    args = parser.parse_args()
    validate(args.submission, args.test)


if __name__ == "__main__":
    main()
