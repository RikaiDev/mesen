"""
WebSight Ingestion & Adversarial Mutation Pipeline.
Fetches real HTML/CSS websites from HuggingFaceM4/WebSight streaming API,
applies systematic UI/UX defect mutations, and produces massive paired training data.
"""

import argparse
import json
import os
import urllib.request

from mesen.data.mutator import MutationType, mutate_html

WEBSIGHT_API = "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceM4%2FWebSight&config=v0.2&split=train"


def fetch_websight_batch(offset: int = 0, limit: int = 100) -> list[dict]:
    """Fetches a batch of HTML/CSS web designs from HuggingFace WebSight."""
    url = f"{WEBSIGHT_API}&offset={offset}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mesen-UI-DataCollector/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return [row["row"] for row in data.get("rows", [])]


def harvest_websight_mutations(
    output_file: str,
    target_count: int = 100,
    offset_start: int = 0,
) -> int:
    """
    Downloads clean websites from WebSight, applies clean + 5 defect mutations,
    and saves to output JSONL.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    total_written = 0
    current_offset = offset_start
    batch_size = 50

    mutation_types = list(MutationType)

    with open(output_file, "a", encoding="utf-8") as out_f:
        while total_written < target_count:
            print(f"Fetching WebSight batch from offset {current_offset}...")
            try:
                rows = fetch_websight_batch(offset=current_offset, limit=batch_size)
            except Exception as e:
                print(f"Error fetching batch at offset {current_offset}: {e}")
                break

            if not rows:
                break

            for i, row in enumerate(rows):
                raw_html = row.get("text", "")
                idea = row.get("llm_generated_idea", "Generic Web Layout")
                img_info = row.get("image", {})
                orig_img_url = img_info.get("src") if isinstance(img_info, dict) else None

                if not raw_html or len(raw_html) < 50:
                    continue

                site_id = f"websight_off{current_offset}_idx{i}"

                # Generate clean and mutated variants
                for mut in mutation_types:
                    mutated_html, labels, score = mutate_html(raw_html, mut)

                    witness_state = {
                        "imageOrder": ["desktop_preview"],
                        "product": "open_web",
                        "route": f"/{site_id}",
                        "contract": {
                            "idea": idea[:120],
                            "mutation": mut.value,
                            "is_clean": mut == MutationType.CLEAN,
                        },
                        "accessibilityViolations": (
                            [{"rule": "color-contrast", "severity": "serious"}]
                            if mut == MutationType.LOW_CONTRAST
                            else []
                        ),
                        "geometryAnomalies": (
                            [{"type": "horizontal-overflow"}]
                            if mut == MutationType.RESPONSIVE_BREAK
                            else []
                        ),
                        "consoleErrors": (
                            ["Failed to load DOM subtree"]
                            if mut == MutationType.EMPTY_STATE
                            else []
                        ),
                    }

                    record = {
                        "id": f"{site_id}_{mut.value}",
                        "state": witness_state,
                        "original_image_url": orig_img_url if mut == MutationType.CLEAN else None,
                        "labels": {
                            **labels,
                            "overall_quality": score,
                        },
                        "mutation_type": mut.value,
                    }

                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_written += 1

                    if total_written >= target_count:
                        break

                if total_written >= target_count:
                    break

            current_offset += len(rows)

    print(f"Successfully harvested {total_written} balanced WebSight samples into {output_file}.")
    return total_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvest and mutate WebSight layouts")
    parser.add_argument("--output", default="data/websight_mutations.jsonl")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    harvest_websight_mutations(args.output, target_count=args.count, offset_start=args.offset)
