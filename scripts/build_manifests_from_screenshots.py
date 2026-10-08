"""Build data/synthetic/train.json and val.json from rendered screenshots on disk.
Runs in milliseconds without invoking headless browsers.
"""

import glob
import json
import os
import random
from collections import Counter

from mesen.data.mutator import MutationType, mutate_html
from mesen.data.templates import TEMPLATES


def build_manifests(
    images_dir: str = "data/synthetic/screenshots",
    output_dir: str = "data/synthetic",
    val_ratio: float = 0.2,
    seed: int = 42,
):
    random.seed(seed)
    files = glob.glob(f"{images_dir}/*.png")

    sample_files = {}
    for f in files:
        base = os.path.basename(f)
        if "real_clinical" in base or "mirror_" in base:
            continue
        parts = base.split("_")
        vp_idx = -1
        for i, p in enumerate(parts):
            if p in ("mobile", "tablet", "laptop", "desktop"):
                vp_idx = i
                break
        if vp_idx > 0:
            prefix = "_".join(parts[:vp_idx])
            sample_files.setdefault(prefix, []).append(f)

    vp_order = ["mobile", "tablet", "laptop", "desktop"]

    def sort_vp(p):
        for i, v in enumerate(vp_order):
            if f"_{v}_" in p:
                return i
        return 99

    mut_types = sorted(list(MutationType), key=lambda x: len(x.value), reverse=True)
    tpl_keys = sorted(list(TEMPLATES.keys()), key=lambda x: len(x), reverse=True)

    samples = []
    for prefix, paths in sample_files.items():
        if len(paths) < 4:
            continue
        paths.sort(key=sort_vp)
        tpl_key = next((k for k in tpl_keys if prefix.startswith(k)), None)
        if not tpl_key:
            continue
        mut_val = next(
            (m for m in mut_types if f"_{m.value}_" in prefix or prefix.endswith(f"_{m.value}")),
            None,
        )
        if not mut_val:
            continue

        tpl_info = TEMPLATES[tpl_key]
        category = tpl_info["category"]
        title = tpl_info["title"]
        mutated_html, labels, score = mutate_html(tpl_info["html"], mut_val)

        witness_state = {
            "imageOrder": [
                "mobile_375x812",
                "tablet_768x1024",
                "laptop_1024x768",
                "desktop_1440x900",
            ],
            "product": category,
            "route": f"/{category}/{tpl_key}",
            "contract": {
                "title": title,
                "mutation": mut_val.value,
                "is_clean": mut_val in (MutationType.CLEAN, MutationType.CLEAN_EXCELLENT),
            },
            "accessibilityViolations": (
                [{"rule": "color-contrast", "severity": "serious"}]
                if mut_val == MutationType.LOW_CONTRAST
                else []
            ),
            "geometryAnomalies": (
                [{"type": "horizontal-overflow", "viewport": "mobile"}]
                if mut_val == MutationType.RESPONSIVE_BREAK
                else []
            ),
            "consoleErrors": (
                ["Fatal error rendering view"] if mut_val == MutationType.EMPTY_STATE else []
            ),
        }
        samples.append(
            {
                "id": prefix,
                "state": witness_state,
                "image_paths": paths,
                "labels": {**labels, "overall_quality": score},
                "mutation_type": mut_val.value,
            }
        )

    random.shuffle(samples)
    val_count = int(len(samples) * val_ratio)
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(samples)} synthetic UI/UX samples across {len(tpl_keys)} templates.")
    print(f"  - Train: {len(train_samples)} samples -> {train_path}")
    print(f"  - Val:   {len(val_samples)} samples -> {val_path}")
    scores = [s["labels"]["overall_quality"] for s in samples]
    print("Score distribution:", Counter(scores))


if __name__ == "__main__":
    build_manifests()
