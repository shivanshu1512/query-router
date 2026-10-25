"""Validates data/queries.jsonl before anything else reads it.

Run this after every editing session. A broken line or a duplicate query
is far cheaper to fix now than after 200 model calls have been made
against the file.

    uv run python scripts/validate_dataset.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "queries.jsonl"

REQUIRED_FIELDS = {"id", "category", "query"}
EXPECTED_PER_CATEGORY = 40


def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation.

    Used only for duplicate detection, so that 'What is SQL?' and
    'what is sql' are recognised as the same question.
    """
    return " ".join(text.lower().split()).rstrip("?.!")


def main() -> int:
    if not DATA.exists():
        print(f"FAIL: {DATA} does not exist")
        return 1

    errors: list[str] = []
    warnings: list[str] = []
    records: list[dict] = []

    # --- Parse, reporting the line number of anything broken ---
    for lineno, raw in enumerate(DATA.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue  # blank lines are harmless
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as e:
            errors.append(f"line {lineno}: not valid JSON — {e.msg}")
            continue

        missing = REQUIRED_FIELDS - rec.keys()
        if missing:
            errors.append(f"line {lineno}: missing field(s) {sorted(missing)}")
            continue

        extra = rec.keys() - REQUIRED_FIELDS
        if extra:
            # Not fatal, but worth flagging — a stray 'difficulty' field here
            # would mean someone started hand-labelling, which we do not do.
            warnings.append(f"line {lineno}: unexpected field(s) {sorted(extra)}")

        if not str(rec["query"]).strip():
            errors.append(f"line {lineno}: empty query")

        records.append(rec)

    # --- Unique ids ---
    for rid, count in Counter(r["id"] for r in records).items():
        if count > 1:
            errors.append(f"duplicate id {rid!r} appears {count} times")

    # --- Duplicate / near-duplicate queries ---
    seen: dict[str, str] = {}
    for r in records:
        key = normalise(r["query"])
        if key in seen:
            errors.append(f"{r['id']} duplicates {seen[key]}: {r['query']!r}")
        else:
            seen[key] = r["id"]

    # --- Category counts ---
    by_cat = Counter(r["category"] for r in records)

    # --- Report ---
    print(f"Read {len(records)} records from {DATA.name}\n")

    print("Per category:")
    for cat, n in sorted(by_cat.items()):
        flag = "" if n == EXPECTED_PER_CATEGORY else f"  (expected {EXPECTED_PER_CATEGORY})"
        print(f"  {cat:12s} {n:3d}{flag}")

    lengths = [len(r["query"].split()) for r in records]
    if lengths:
        print(
            f"\nQuery length (words): min {min(lengths)}, "
            f"max {max(lengths)}, mean {sum(lengths) / len(lengths):.1f}"
        )

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  ! {w}")

    if errors:
        print(f"\n{len(errors)} ERROR(S):")
        for e in errors:
            print(f"  x {e}")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())