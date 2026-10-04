"""Ask the UI-UX teacher the four rubric axes, verbatim, and hold it to the rubric.

The previous attempt asked it one global 0-3 question and graded that. That was
the wrong shape: mesen was designed as a four-axis judgment model
(`schema.DesignCriterion`, `research/contracts/four_axis_rubric.json`) and a
single ordinal collapses four independent judgements into one number.

This asks each axis its own rubric question, unmodified, and treats the rubric's
`unknown` option as a real answer. That matters: the rubric says originality
cannot be established from a single screenshot and functionality cannot be
established without interaction evidence. A teacher that answers those with
yes/no is not rubric-compliant, and its labels on those axes are not
distillable, however good it looks elsewhere.

  label     run every axis, recording the raw answer
  validate  abstention compliance, and craft against measured consistency
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, "src")

REVISION = "846401e482f7b6fb8d61ab9fb7bc30d4e90e1c4b"
RUBRIC_PATH = "research/contracts/four_axis_rubric.json"
ANSWER_RE = re.compile(r"\\boxed\{(yes|no|unknown|YES|NO|UNKNOWN)\}")

PREAMBLE = (
    "You are reviewing a web page interface as a senior UI/UX designer. "
    "Answer with one word: yes, no, or unknown. Use unknown when the supplied "
    "evidence cannot establish the answer. Do not explain. "
    "Reply in the form $\\boxed{yes}$, $\\boxed{no}$ or $\\boxed{unknown}$."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    label = sub.add_parser("label")
    label.add_argument("--model-dir", required=True)
    label.add_argument("--pages", required=True, help="JSON list of {images:[...]} page groups")
    label.add_argument("--out", required=True)
    label.add_argument("--max-new-tokens", type=int, default=8)

    validate = sub.add_parser("validate")
    validate.add_argument("--labels", required=True)
    return parser.parse_args()


def load_axes():
    with open(RUBRIC_PATH, encoding="utf-8") as handle:
        rubric = json.load(handle)
    return {name: spec["question"] for name, spec in rubric["criteria"].items()}


def ask(model, processor, image_paths, prompt, args_max_new_tokens=8):
    """One question against every supplied state of a page."""
    import torch

    content = [{"type": "image", "image": path} for path in image_paths]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    prefix = "$\\boxed{"
    prefix_ids = torch.tensor(
        [processor.tokenizer.encode(prefix, add_special_tokens=False)],
        device=inputs["input_ids"].device,
    )
    inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix_ids], dim=1)
    prompt_len = inputs["input_ids"].shape[1] - prefix_ids.shape[1]
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(prefix_ids)], dim=1
        )
    for key in [k for k in inputs if k.endswith("_ids")]:
        if inputs[key].shape[-1] == prompt_len:
            inputs[key] = torch.cat([inputs[key], torch.zeros_like(prefix_ids)], dim=1)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args_max_new_tokens, do_sample=False)
    completion = processor.decode(
        generated[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )
    return prefix + completion


def stage_label(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    axes = load_axes()
    print(f"axes: {list(axes)}", flush=True)
    with open(args.pages, encoding="utf-8") as handle:
        groups = json.load(handle)

    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    )
    model.eval()

    records = []
    for index, group in enumerate(groups):
        images = [p for p in group["images"] if os.path.exists(p)]
        if not images:
            continue
        axes_answers = {}
        for axis, question in axes.items():
            prompt = f"{PREAMBLE}\n\nAxis: {axis}\nQuestion: {question}"
            raw = ask(model, processor, images, prompt, args.max_new_tokens)
            found = ANSWER_RE.findall(raw)
            axes_answers[axis] = {
                "answer": found[-1].lower() if len(found) == 1 else None,
                "raw": raw.strip(),
            }
        records.append(
            {
                "page": group.get("page", images[0]),
                "n_states": len(images),
                "images": images,
                "axes": axes_answers,
            }
        )
        if (index + 1) % 10 == 0:
            print(f"  {index + 1}/{len(groups)} page groups", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {
        "teacher_model_id": "afx-team/UI-UX",
        "teacher_revision": REVISION,
        "rubric": RUBRIC_PATH,
        "axes": list(axes),
        "preamble": PREAMBLE,
        "labelled_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    summary = {}
    for axis in axes:
        answers = [r["axes"][axis]["answer"] for r in records]
        parsed = [a for a in answers if a]
        summary[axis] = {
            "parsed": len(parsed),
            "parse_rate": round(len(parsed) / max(len(records), 1), 4),
            "distribution": {v: parsed.count(v) for v in ("yes", "no", "unknown")},
        }
    receipt = {
        "labels": args.out,
        "page_groups": len(records),
        "per_axis": summary,
    }
    with open(os.path.splitext(args.out)[0] + ".receipt.json", "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def stage_validate(args):
    with open(args.labels, encoding="utf-8") as handle:
        payload = json.load(handle)
    axes = payload["axes"]
    records = payload["records"]

    single = [r for r in records if r["n_states"] == 1]
    multi = [r for r in records if r["n_states"] > 1]

    def distribution(rows, axis):
        answers = [r["axes"][axis]["answer"] for r in rows if r["axes"][axis]["answer"]]
        return {v: answers.count(v) for v in ("yes", "no", "unknown")}

    report = {
        "teacher_revision": REVISION,
        "page_groups": len(records),
        "single_state_groups": len(single),
        "multi_state_groups": len(multi),
    }

    # The rubric is explicit that a single screenshot cannot establish
    # originality, and that functionality needs interaction evidence. A teacher
    # answering yes/no there is not following the rubric, and its labels on those
    # axes are proposals only.
    rubric_requires_unknown_on_single_state = ("originality", "functionality")
    compliance = {}
    for axis in rubric_requires_unknown_on_single_state:
        dist = distribution(single, axis) if single else {}
        decisive = dist.get("yes", 0) + dist.get("no", 0)
        compliance[axis] = {
            "single_state_distribution": dist,
            "decisive_answers": decisive,
            "abstention_rate": round(dist.get("unknown", 0) / max(len(single), 1), 4),
            "rubric_compliant": decisive == 0,
        }
    report["abstention_compliance"] = compliance

    # Craft is the one axis a single screenshot can answer. Check whether the
    # teacher's craft verdict tracks measured consistency at all, using the
    # strongest measurable proxy available: contrast spread across text
    # elements, which is high when spacing and colour relationships are sloppy.
    craft_rows = []
    for record in records:
        answer = record["axes"].get("craft", {}).get("answer")
        if answer not in ("yes", "no"):
            continue
        craft_rows.append((answer, record["images"][0]))

    if craft_rows:
        from mesen.engine.evidence import EvidenceEngine

        engine = EvidenceEngine()
        contrasts = []
        for answer, image in craft_rows:
            elements = engine.extract_and_measure_elements(image)
            if not elements:
                continue
            ratios = [el.contrast_ratio for el in elements]
            contrasts.append((answer, float(np.std(ratios)), float(np.mean(ratios))))
        if len({c[0] for c in contrasts}) > 1:
            labels = np.asarray([1 if c[0] == "yes" else 0 for c in contrasts])
            spread = np.asarray([c[1] for c in contrasts])
            mean_ratio = np.asarray([c[2] for c in contrasts])
            report["craft_vs_contrast_spread_r"] = round(
                float(np.corrcoef(labels, spread)[0, 1]), 4
            )
            report["craft_vs_mean_contrast_r"] = round(
                float(np.corrcoef(labels, mean_ratio)[0, 1]), 4
            )
            report["craft_pages_scored"] = len(contrasts)

    for axis in axes:
        report.setdefault("per_axis_distribution_all_states", {})[axis] = distribution(
            records, axis
        )

    report["verdict"] = (
        "every axis the rubric says needs more evidence returned unknown, so the "
        "teacher follows the rubric and only craft is distillable"
        if all(v["rubric_compliant"] for v in compliance.values())
        else "teacher gave decisive answers on axes the rubric says a screenshot "
        "cannot establish; those axes are proposals, not labels"
    )

    out = os.path.splitext(args.labels)[0] + ".validation.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.stage == "label":
        stage_label(parsed_args)
    else:
        stage_validate(parsed_args)
