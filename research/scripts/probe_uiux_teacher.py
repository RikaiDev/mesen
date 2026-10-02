"""Run a pinned specialist UX teacher on one public screenshot defect question."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

REVISION = "846401e482f7b6fb8d61ab9fb7bc30d4e90e1c4b"
QUESTION = (
    "Core Task: Determine whether a floating overlay blocks a clickable primary action "
    "in this screenshot. Options: A. No floating overlay is present. "
    "B. A floating overlay blocks the primary action. "
    "C. A floating overlay is present but does not block the primary action. "
    "Output Format: $\\boxed{X}$ where X is one of A-C."
)


def sha256(path: Path) -> str:
    """Return the digest of a regular local input file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Load only local pinned weights and save the raw and parsed one-case result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    image = args.image.resolve()
    if REVISION[:12] not in model_dir.name or not image.is_file():
        raise ValueError("Pinned model directory or screenshot is missing")

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True,
    )
    model.eval()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": QUESTION},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    response = processor.decode(generated[0][len(inputs.input_ids[0]) :], skip_special_tokens=True)
    choices = re.findall(r"\\boxed\{([ABC])\}", response)
    result = {
        "model_revision": REVISION,
        "image": str(image),
        "image_sha256": sha256(image),
        "question": QUESTION,
        "response": response,
        "choice": choices[-1] if len(choices) == 1 else None,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"choice": result["choice"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
