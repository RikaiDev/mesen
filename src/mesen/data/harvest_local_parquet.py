"""
Fast Local Parquet Harvester for Real Mobile UI Dataset.
Extracts real mobile screenshots directly from local Parquet files (dev & train shards of RICO),
generates multi-orientation samples (native portrait + 21:9 landscape viewport variations),
and evaluates physical grounding metrics using the Evidence Engine.
"""

import argparse
import io
import json
import os
import time

import pyarrow.parquet as pq
from PIL import Image

from mesen.engine.evidence import EvidenceEngine
from mesen.rules.registry import RULE_ID_LIST, RULE_TO_INDEX


def extract_real_mobile_dataset(
    parquet_path: str,
    output_dir: str = "data/real/mobile_rico",
    max_samples: int = 1000,
    prefix: str = "rico",
):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print(f"Reading Parquet: {parquet_path}...")
    table = pq.read_table(parquet_path)
    images_col = table.column("image")
    text_col = table.column("text")
    total_in_table = table.num_rows
    print(f"Total available rows in table: {total_in_table}")

    evidence_engine = EvidenceEngine(default_dpi=440)
    manifest: list[dict] = []
    num_to_process = min(max_samples, total_in_table)

    print(f"Processing {num_to_process} real screenshots...")
    start_time = time.time()

    for i in range(num_to_process):
        try:
            img_struct = images_col[i].as_py()
            raw_bytes = img_struct["bytes"]
            desc = text_col[i].as_py()

            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            w, h = pil_img.size

            # Save real native portrait image
            img_filename = f"{prefix}_{i:05d}.png"
            img_path = os.path.join(images_dir, img_filename)
            pil_img.save(img_path)

            aspect_ratio = round(w / float(h), 2)
            elements = evidence_engine.extract_and_measure_elements(img_path, dpi=440)

            rule_vec = [0.0] * len(RULE_ID_LIST)
            active_rules: list[str] = []
            target_bbox = [0.0, 0.0, 1.0, 1.0]

            has_contrast_fail = False
            for el in elements:
                if el.contrast_ratio < 3.0 and el.text:
                    has_contrast_fail = True
                    active_rules.append("accessibility/contrast-ratio-insufficient")
                    idx = RULE_TO_INDEX.get("accessibility/contrast-ratio-insufficient")
                    if idx is not None:
                        rule_vec[idx] = 1.0
                    target_bbox = el.text_bbox
                    break

            has_font_fail = False
            for el in elements:
                if el.estimated_sp < 10.0 and el.text:
                    has_font_fail = True
                    active_rules.append("accessibility/font-size-insufficient")
                    idx = RULE_TO_INDEX.get("accessibility/font-size-insufficient")
                    if idx is not None:
                        rule_vec[idx] = 1.0
                    if not has_contrast_fail:
                        target_bbox = el.text_bbox
                    break

            visual_integrity = "no" if has_contrast_fail else "yes"
            operator_clarity = "no" if (has_contrast_fail or has_font_fail) else "yes"
            responsive_consistency = "yes"
            primary_action_reachable = "yes"
            evidence_consistency = "yes"
            overall_quality = 1 if (has_contrast_fail or has_font_fail) else 3

            manifest.append(
                {
                    "image_path": img_path,
                    "width": w,
                    "height": h,
                    "aspect_ratio": aspect_ratio,
                    "description": desc,
                    "labels": {
                        "primary_action_reachable": primary_action_reachable,
                        "visual_integrity": visual_integrity,
                        "responsive_consistency": responsive_consistency,
                        "evidence_consistency": evidence_consistency,
                        "operator_clarity": operator_clarity,
                        "overall_quality": overall_quality,
                    },
                    "rule_targets": rule_vec,
                    "active_rules": active_rules,
                    "bbox_targets": target_bbox,
                    "num_detected_elements": len(elements),
                }
            )

            # Create unadapted 21:9 landscape variant every 3 samples
            if i % 3 == 0:
                lw, lh = 2424, 1080
                landscape_img = Image.new("RGB", (lw, lh), color=(245, 245, 245))
                scale = lh / float(h)
                scaled_w = int(w * scale)
                resized_p = pil_img.resize((scaled_w, lh), Image.Resampling.LANCZOS)
                offset_x = (lw - scaled_w) // 2
                landscape_img.paste(resized_p, (offset_x, 0))

                l_filename = f"{prefix}_landscape_{i:05d}.png"
                l_path = os.path.join(images_dir, l_filename)
                landscape_img.save(l_path)

                # Responsive layout desert
                l_rule_vec = [0.0] * len(RULE_ID_LIST)
                l_active_rules = ["layout/horizontal-space-desert", "layout/aspect-ratio-mismatch"]
                for r_id in l_active_rules:
                    idx = RULE_TO_INDEX.get(r_id)
                    if idx is not None:
                        l_rule_vec[idx] = 1.0

                min_x = round(offset_x / float(lw), 3)
                max_x = round((offset_x + scaled_w) / float(lw), 3)

                manifest.append(
                    {
                        "image_path": l_path,
                        "width": lw,
                        "height": lh,
                        "aspect_ratio": 2.24,
                        "description": f"{desc} (unadapted on 21:9 landscape)",
                        "labels": {
                            "primary_action_reachable": "yes",
                            "visual_integrity": "no",
                            "responsive_consistency": "no",
                            "evidence_consistency": "yes",
                            "operator_clarity": "yes",
                            "overall_quality": 1,
                        },
                        "rule_targets": l_rule_vec,
                        "active_rules": l_active_rules,
                        "bbox_targets": [0.0, min_x, 1.0, max_x],
                        "num_detected_elements": len(elements),
                    }
                )

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[{i + 1}/{num_to_process}] processed ({elapsed:.1f}s, {elapsed / (i + 1):.2f}s/sample)"
                )

        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    manifest_file = os.path.join(output_dir, f"{prefix}_manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(manifest)} processed samples to {manifest_file}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/real/mobile_rico")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--prefix", type=str, default="rico")
    args = parser.parse_args()

    extract_real_mobile_dataset(
        parquet_path=args.parquet,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        prefix=args.prefix,
    )
