"""Sample ESCI teacher grades without materializing 2.6M dicts.

Vectorized stratified sample in pandas, then build manifest records only
for the sampled rows.
"""

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from mesen.data.teacher_grades import build_teacher_manifest

PARQUET = "/Users/gloomcheng/Workspace/data/esci/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
PER_CLASS_CAP = 5000
SEED = 7

t0 = time.perf_counter()
d = pd.read_parquet(
    PARQUET,
    columns=[
        "example_id",
        "query",
        "product_id",
        "product_locale",
        "esci_label",
        "split",
    ],
)
t1 = time.perf_counter()
parts = []
for label, group in d.groupby("esci_label"):
    parts.append(group.sample(n=min(len(group), PER_CLASS_CAP), random_state=SEED))
sampled = (
    pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
)
t2 = time.perf_counter()
sha = hashlib.sha256()
with open(PARQUET, "rb") as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
        sha.update(chunk)
m = build_teacher_manifest(
    sampled.to_dict("records"),
    seed=SEED,
    per_class_cap=PER_CLASS_CAP,
    source="esci",
    source_sha256=sha.hexdigest(),
)
t3 = time.perf_counter()
print(
    f"rows={len(d)} read={t1 - t0:.1f}s sample={t2 - t1:.1f}s manifest={t3 - t2:.1f}s", flush=True
)
print("grades:", sorted((r["grade"], r["id"]) for r in m["records"])[:2], "...", flush=True)
print("grade counts:", dict(sorted(Counter(r["grade"] for r in m["records"]).items())), flush=True)
print("locales:", dict(sorted(Counter(r["locale"] for r in m["records"]).items())), flush=True)
out = Path("/tmp/teacher-manifest.json")
out.write_text(
    json.dumps({**m, "records": m["records"][:5], "record_count": len(m["records"])}),
    encoding="utf-8",
)
print("wrote", out, flush=True)
