"""Teacher-grade manifests from large-scale human graded judgments.

ESCI's query-product judgments (Exact/Substitute/Complement/Irrelevant) teach
ordinal decision structure, the same 0-3 shape as the quality score. This is
deliberately NOT the reviewed manifest: there are no screenshots, no
independent evaluator receipts, and no pixel binding here. It teaches grading
structure, never visual facts. Visual claims still require reviewed records.
"""

import json
import random
from pathlib import Path

SCHEMA = "mesen.teacher-grades.v1"

ESCI_TO_GRADE = {"E": 3, "S": 2, "C": 1, "I": 0}


def build_teacher_manifest(
    rows,
    seed,
    per_class_cap,
    source="esci",
    source_sha256="",
    image_urls=None,
):
    """Sample graded rows into a manifest.

    Stratified by label with a per-class cap so the 65% Exact majority cannot
    drown the 3% Complement minority. Deterministic in `seed`. Unknown labels
    raise instead of guessing. `image_urls` optionally maps product_id to an
    image URL (e.g. SQID); products without one simply carry none.
    """
    image_urls = image_urls or {}
    by_label = {}
    skipped = 0
    for row in rows:
        label = row.get("esci_label")
        if label is None:
            skipped += 1
            continue
        if label not in ESCI_TO_GRADE:
            raise ValueError(f"Unknown esci_label: {label!r}")
        by_label.setdefault(label, []).append(row)

    rng = random.Random(seed)
    records = []
    for label in sorted(by_label):
        pool = list(by_label[label])
        rng.shuffle(pool)
        for row in pool[:per_class_cap]:
            records.append(
                {
                    "id": str(row["example_id"]),
                    "grade": ESCI_TO_GRADE[label],
                    "source_label": label,
                    "query": row.get("query", ""),
                    "product_id": str(row.get("product_id", "")),
                    "image_url": image_urls.get(str(row.get("product_id", ""))),
                    "locale": row.get("product_locale", ""),
                    "split": row.get("split", ""),
                }
            )

    return {
        "schema": SCHEMA,
        "source": source,
        "source_sha256": source_sha256,
        "sampling": {
            "seed": seed,
            "per_class_cap": per_class_cap,
            "input_rows": len(rows),
            "skipped_null_label": skipped,
        },
        "records": records,
    }


def load_teacher_manifest(manifest_path):
    """Read back a manifest, rejecting anything that is not this schema."""
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("Unsupported teacher manifest schema")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Teacher manifest needs a records list")
    return records, {
        "source": payload.get("source", ""),
        "source_sha256": payload.get("source_sha256", ""),
        "sampling": payload.get("sampling", {}),
    }
