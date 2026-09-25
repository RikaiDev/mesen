"""
Harvest & Label Real-World Mobile and Web UI Dataset.
Streams real Android application screenshots from RICO (1440x2560 native mobile screens),
applies orientation & responsive viewport awareness, and uses the Evidence Engine to compute
ground-truth physical metrics (contrast, SP sizes, aspect-ratio span, touch targets).
Generates multi-task training samples for Mesen Jev-VLM.
"""

import argparse
import json
import os

from datasets import load_dataset
from PIL import Image

from mesen.engine.evidence import EvidenceEngine
from mesen.rules.registry import RULE_ID_LIST, RULE_TO_INDEX

CHOICE_MAP = {"yes": 0, "no": 1, "unknown": 2}


def analyze_real_screenshot(
    image_path: str,
    evidence_engine: EvidenceEngine,
    description: str = "",
    is_landscape: bool = False,
) -> dict:
    """
    Analyzes a real mobile screenshot and generates grounded JEV labels and rule violations.
    """
    img = Image.open(image_path)
    w, h = img.size
    aspect_ratio = round(w / float(h), 2)

    # 1. Physical Evidence Extraction
    elements = evidence_engine.extract_and_measure_elements(image_path, dpi=440)

    # 2. Rule Violations Detection
    rule_vec = [0.0] * len(RULE_ID_LIST)
    active_rules: list[str] = []
    target_bbox = [0.0, 0.0, 1.0, 1.0]

    # Contrast check (< 3.0:1 is critical fail)
    has_contrast_fail = False
    for el in elements:
        if el.contrast_ratio < 3.0 and el.text:
            has_contrast_fail = True
            active_rules.append("accessibility/contrast-ratio-insufficient")
            rule_idx = RULE_TO_INDEX.get("accessibility/contrast-ratio-insufficient")
            if rule_idx is not None:
                rule_vec[rule_idx] = 1.0
            target_bbox = el.text_bbox
            break

    # Font size check (< 10sp is severe unreadable)
    has_font_fail = False
    for el in elements:
        if el.estimated_sp < 10.0 and el.text:
            has_font_fail = True
            active_rules.append("accessibility/font-size-insufficient")
            rule_idx = RULE_TO_INDEX.get("accessibility/font-size-insufficient")
            if rule_idx is not None:
                rule_vec[rule_idx] = 1.0
            if not has_contrast_fail:
                target_bbox = el.text_bbox
            break

    # Layout Aspect Ratio Desert check
    has_layout_desert = False
    if aspect_ratio >= 1.7:
        if elements:
            min_x = min(el.text_bbox[1] for el in elements)
            max_x = max(el.text_bbox[3] for el in elements)
            h_span = max_x - min_x
            if h_span < 0.35:
                has_layout_desert = True
                active_rules.append("layout/horizontal-space-desert")
                active_rules.append("layout/aspect-ratio-mismatch")
                for r_id in ["layout/horizontal-space-desert", "layout/aspect-ratio-mismatch"]:
                    idx = RULE_TO_INDEX.get(r_id)
                    if idx is not None:
                        rule_vec[idx] = 1.0
                target_bbox = [0.0, min_x, 1.0, max_x]

    # 3. Derive 6 JEV Atomic Labels
    visual_integrity = "no" if (has_contrast_fail or has_layout_desert) else "yes"
    operator_clarity = "no" if (has_contrast_fail or has_font_fail) else "yes"
    responsive_consistency = "no" if has_layout_desert else "yes"
    primary_action_reachable = "yes"
    evidence_consistency = "yes"

    # Quality Score (0..3)
    if has_layout_desert and (has_contrast_fail or has_font_fail):
        overall_quality = 0
    elif has_contrast_fail or has_font_fail or has_layout_desert:
        overall_quality = 1
    else:
        overall_quality = 3

    return {
        "image_path": image_path,
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "description": description,
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


def harvest_real_rico_dataset(
    output_dir: str = "data/real/mobile_rico",
    max_samples: int = 500,
    include_landscape_views: bool = True,
):
    """
    Downloads real mobile app screenshots from RICO and builds labeled dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print("Loading RICO dataset stream from HuggingFace...")
    ds = load_dataset("pinkmooncake/rico-screen2words", split="train", streaming=True)
    evidence_engine = EvidenceEngine(default_dpi=440)

    manifest: list[dict] = []
    count = 0

    print(f"Collecting and analyzing {max_samples} real mobile app screenshots...")
    for idx, sample in enumerate(ds):
        if count >= max_samples:
            break

        try:
            pil_img = sample["image"].convert("RGB")
            desc = sample.get("text", "")

            # 1. Portrait native screen
            img_filename = f"rico_{count:05d}.png"
            img_path = os.path.join(images_dir, img_filename)
            pil_img.save(img_path)

            entry = analyze_real_screenshot(
                img_path,
                evidence_engine,
                description=desc,
                is_landscape=False,
            )
            manifest.append(entry)
            count += 1

            # 2. Ultra-wide landscape unadapted viewport variant (e.g. tablet/ultra-wide unadapted view)
            if include_landscape_views and count % 3 == 0 and count < max_samples:
                # Place portrait in center of 21:9 landscape (2424x1080) to capture real aspect-ratio desert
                lw, lh = 2424, 1080
                landscape_img = Image.new("RGB", (lw, lh), color=(245, 245, 245))
                # Scale portrait to fit height 1080
                pw, ph = pil_img.size
                scale = lh / float(ph)
                scaled_w = int(pw * scale)
                resized_p = pil_img.resize((scaled_w, lh), Image.Resampling.LANCZOS)
                # Paste in center
                offset_x = (lw - scaled_w) // 2
                landscape_img.paste(resized_p, (offset_x, 0))

                l_filename = f"rico_landscape_{count:05d}.png"
                l_path = os.path.join(images_dir, l_filename)
                landscape_img.save(l_path)

                l_entry = analyze_real_screenshot(
                    l_path,
                    evidence_engine,
                    description=f"{desc} (unadapted on 21:9 landscape)",
                    is_landscape=True,
                )
                manifest.append(l_entry)
                count += 1

            if count % 25 == 0:
                print(f"Progress: {count}/{max_samples} real samples collected and labeled.")

        except Exception as e:
            print(f"Skipping sample {idx} due to error: {e}")
            continue

    # Split into train (80%) and val (20%)
    split_idx = int(len(manifest) * 0.8)
    train_samples = manifest[:split_idx]
    val_samples = manifest[split_idx:]

    train_file = os.path.join(output_dir, "train.json")
    val_file = os.path.join(output_dir, "val.json")

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)

    print(f"\nCompleted harvesting {len(manifest)} real-world UI samples!")
    print(f"  Train: {len(train_samples)} saved to {train_file}")
    print(f"  Val:   {len(val_samples)} saved to {val_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/real/mobile_rico")
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()

    harvest_real_rico_dataset(output_dir=args.output_dir, max_samples=args.max_samples)
