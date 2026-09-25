"""
Data Synthesizer & Collector.
Combines templates, adversarial layout mutators, and headless browser rendering
to produce balanced UI/UX dataset packets containing:
1. Multi-viewport screenshots (mobile, tablet, laptop, desktop)
2. Witness state (DOM contract, product, route, accessibility facts)
3. Ground-truth 6-head labels with exact defect classifications
"""

import argparse
import json
import os
import random

from mesen.data.mutator import MutationType, mutate_html
from mesen.data.renderer import render_html_viewports
from mesen.data.templates import TEMPLATES


def generate_synthetic_dataset(
    output_dir: str,
    num_variations: int = 20,
    render_screenshots: bool = True,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, str]:
    """
    Generates a full synthetic dataset with clean and mutated UI samples.
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "screenshots")
    os.makedirs(images_dir, exist_ok=True)

    samples: list[dict] = []
    sample_idx = 0

    mutation_types = list(MutationType)

    for template_key, tpl_info in TEMPLATES.items():
        base_html = tpl_info["html"]
        category = tpl_info["category"]
        title = tpl_info["title"]

        for var_idx in range(num_variations):
            for mut_type in mutation_types:
                sample_id = f"{template_key}_var{var_idx}_{mut_type.value}_{sample_idx:04d}"
                mutated_html, labels, score = mutate_html(base_html, mut_type)

                # Render multi-viewport screenshots if requested
                screenshot_paths = []
                if render_screenshots:
                    try:
                        screenshot_paths = render_html_viewports(
                            mutated_html,
                            output_dir=images_dir,
                            prefix=sample_id,
                        )
                    except Exception:
                        # Fallback if rendering tool fails in headless environment
                        screenshot_paths = []

                # Synthesize realistic witness state
                witness_state = {
                    "imageOrder": [
                        "mobile_375x812",
                        "tablet_768x1024",
                        "laptop_1024x768",
                        "desktop_1440x900",
                    ],
                    "product": category,
                    "route": f"/{category}/{template_key}",
                    "contract": {
                        "title": title,
                        "mutation": mut_type.value,
                        "is_clean": mut_type == MutationType.CLEAN,
                    },
                    "accessibilityViolations": (
                        [{"rule": "color-contrast", "severity": "serious"}]
                        if mut_type == MutationType.LOW_CONTRAST
                        else []
                    ),
                    "geometryAnomalies": (
                        [{"type": "horizontal-overflow", "viewport": "mobile"}]
                        if mut_type == MutationType.RESPONSIVE_BREAK
                        else []
                    ),
                    "consoleErrors": (
                        ["Fatal error rendering view"]
                        if mut_type == MutationType.EMPTY_STATE
                        else []
                    ),
                }

                all_labels = {
                    **labels,
                    "overall_quality": score,
                }

                samples.append(
                    {
                        "id": sample_id,
                        "state": witness_state,
                        "image_paths": screenshot_paths,
                        "labels": all_labels,
                        "mutation_type": mut_type.value,
                    }
                )
                sample_idx += 1

    print(f"Generated {len(samples)} synthetic UI/UX samples across {len(TEMPLATES)} templates.")
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

    print("Dataset saved:")
    print(f"  - Train: {len(train_samples)} samples -> {train_path}")
    print(f"  - Val:   {len(val_samples)} samples -> {val_path}")

    # Calculate summary defect ratio
    defects_count = sum(1 for s in samples if s["mutation_type"] != "clean")
    print(
        f"Balance: {defects_count} defects ({defects_count / len(samples) * 100:.1f}%), {len(samples) - defects_count} clean samples."
    )

    return {"train_path": train_path, "val_path": val_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic UI/UX training dataset")
    parser.add_argument("--output-dir", default="data/synthetic", help="Output directory")
    parser.add_argument("--num-variations", type=int, default=5, help="Variations per template")
    parser.add_argument("--no-render", action="store_true", help="Skip screenshot rendering")
    args = parser.parse_args()

    generate_synthetic_dataset(
        output_dir=args.output_dir,
        num_variations=args.num_variations,
        render_screenshots=not args.no_render,
    )
