"""Label UI screenshots with the pinned UI-UX teacher, then hold it to account.

A model-authored label is only worth training on if something independent can
check it. Human agreement is not available here and not required: WCAG
measurements on real pixels are facts, so the teacher can be graded against
them. That is what `validate_against_pixels` does, and its verdict gates whether
the ordinal grades may be distilled at all.

Two questions, deliberately kept separable:

  grade   0-3 ordinal quality, the System 1 score target
  defect  yes/no on one measurable proposition, used only for validation

If the teacher's defect answers track measured pixels better than chance, its
grade carries signal. If they do not, the grade is decoration.

  label    run the teacher over pages, write raw responses plus a receipt
  validate score the defect answers against EvidenceEngine measurements
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

GRADE_PROMPT = (
    "You are a senior UI/UX designer reviewing a screenshot of a web page. "
    "Judge the overall interface quality.\n"
    "0 = broken or unusable: unreadable text, overlapping or clipped controls, "
    "no usable hierarchy\n"
    "1 = confusing: the page works but hierarchy, spacing or legibility make it "
    "hard to use\n"
    "2 = usable and clear: ordinary, competent commercial UI\n"
    "3 = excellent: deliberate hierarchy, consistent spacing, effortless to scan\n"
    "Reply with the answer only, no analysis, in the form $\\boxed{N}$ where N "
    "is 0, 1, 2 or 3."
)

# Kept narrow on purpose: a yes/no proposition System 2 can measure exactly, so
# the teacher can be graded rather than trusted.
DEFECT_PROMPT = (
    "Core Task: determine whether any text in this screenshot has a contrast "
    "ratio below 4.5:1 against its background, which fails the WCAG 2.1 AA "
    "requirement for body text.\n"
    "Options:\n"
    "A. All text meets 4.5:1.\n"
    "B. Some text is below 4.5:1.\n"
    "Reply with the answer only, no analysis, in the form $\\boxed{X}$ where X "
    "is A or B."
)

GRADE_RE = re.compile(r"\\boxed\{([0-3])\}")
DEFECT_RE = re.compile(r"\\boxed\{([AB])\}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    label = sub.add_parser("label")
    label.add_argument("--model-dir", required=True)
    label.add_argument("--images", required=True)
    label.add_argument("--out", required=True)
    label.add_argument("--limit", type=int, default=200)
    label.add_argument("--max-new-tokens", type=int, default=8)
    label.add_argument("--seed", type=int, default=42)

    validate = sub.add_parser("validate")
    validate.add_argument("--labels", required=True)
    validate.add_argument("--images", required=True)
    return parser.parse_args()


def find_images(root: str) -> list[str]:
    found = []
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d != ".cache"]
        for name in sorted(names):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                found.append(os.path.join(base, name))
    found.sort()
    return found


def ask(model, processor, image_path: str, prompt: str, max_new_tokens: int) -> str:
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # The teacher is a Qwen3.5 hybrid-reasoning model. Left in thinking mode it
    # writes a long structured analysis and the boxed answer falls past any
    # usable token budget: 200 pages came back with zero parseable answers at 24
    # tokens and again at 192. `enable_thinking=False` makes it answer directly,
    # which is the only thing a labeler needs.
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    # Prefilling the answer format removes the last source of unparseable output.
    # Even in non-thinking mode the teacher opens with "Looking at this
    # screenshot, I need to..." and the boxed answer lands past a usable token
    # budget. Continuing from `$\boxed{` leaves only the character to predict.
    prefix = "$\\boxed{"
    prefix_ids = torch.tensor(
        [processor.tokenizer.encode(prefix, add_special_tokens=False)],
        device=inputs["input_ids"].device,
    )
    inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix_ids], dim=1)
    prompt_length = inputs["input_ids"].shape[1]
    if "attention_mask" in inputs:
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(prefix_ids)], dim=1
        )
    # Qwen3.5 derives rope positions from token_type_ids to tell vision tokens
    # from text tokens, so any id tensor aligned with the prompt has to grow by
    # the same amount. Prompt text is type 0.
    for key in [k for k in inputs if k.endswith("_ids")]:
        if inputs[key].shape[-1] == prompt_length - prefix_ids.shape[1]:
            inputs[key] = torch.cat([inputs[key], torch.zeros_like(prefix_ids)], dim=1)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    completion = processor.decode(
        generated[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )
    return prefix + completion


def stage_label(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    files = find_images(args.images)
    rng = np.random.default_rng(args.seed)
    if len(files) > args.limit:
        files = [files[i] for i in rng.choice(len(files), args.limit, replace=False)]
    files.sort()
    print(f"pages to label: {len(files)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True
    )
    model.eval()

    records = []
    for index, path in enumerate(files):
        grade_text = ask(model, processor, path, GRADE_PROMPT, args.max_new_tokens)
        defect_text = ask(model, processor, path, DEFECT_PROMPT, args.max_new_tokens)
        grades = GRADE_RE.findall(grade_text)
        defects = DEFECT_RE.findall(defect_text)
        records.append(
            {
                "image": path,
                "grade": int(grades[-1]) if len(grades) == 1 else None,
                "grade_raw": grade_text.strip(),
                "has_low_contrast_text": (defects[-1] == "B") if len(defects) == 1 else None,
                "defect_raw": defect_text.strip(),
            }
        )
        if (index + 1) % 20 == 0:
            parsed = sum(1 for r in records if r["grade"] is not None)
            print(f"  {index + 1}/{len(files)} labelled, {parsed} parsed", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {
        "teacher_model_id": "afx-team/UI-UX",
        "teacher_revision": REVISION,
        "grade_prompt": GRADE_PROMPT,
        "defect_prompt": DEFECT_PROMPT,
        "images_root": args.images,
        "labelled_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    grades = [r["grade"] for r in records if r["grade"] is not None]
    counts = np.bincount(grades, minlength=4).tolist() if grades else []
    receipt = {
        "labels": args.out,
        "pages": len(records),
        "grade_parsed": len(grades),
        "grade_parse_rate": round(len(grades) / max(len(records), 1), 4),
        "grade_distribution": counts,
        "defect_parsed": sum(1 for r in records if r["has_low_contrast_text"] is not None),
        "grade_prompt_sha_note": "verbatim in the labels file",
    }
    with open(os.path.splitext(args.out)[0] + ".receipt.json", "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def stage_validate(args):
    from mesen.engine.evidence import EvidenceEngine

    with open(args.labels, encoding="utf-8") as handle:
        payload = json.load(handle)
    engine = EvidenceEngine()

    rows = []
    for record in payload["records"]:
        if record["has_low_contrast_text"] is None:
            continue
        if not os.path.exists(record["image"]):
            continue
        elements = engine.extract_and_measure_elements(record["image"])
        if not elements:
            continue
        ratios = [el.contrast_ratio for el in elements]
        rows.append(
            {
                "teacher_says_failing": record["has_low_contrast_text"],
                "measured_has_failing": any(r < 4.5 for r in ratios),
                "measured_min_contrast": round(float(min(ratios)), 2),
                "n_elements": len(elements),
                "grade": record["grade"],
            }
        )

    if not rows:
        raise SystemExit("no comparable rows: the teacher's answers never parsed")

    teacher = np.asarray([r["teacher_says_failing"] for r in rows], dtype=int)
    truth = np.asarray([r["measured_has_failing"] for r in rows], dtype=int)
    severity = np.asarray([r["measured_min_contrast"] for r in rows], dtype=float)
    pos, neg = int(truth.sum()), int(len(truth) - truth.sum())
    if pos == 0 or neg == 0:
        raise SystemExit("degenerate ground truth: every page measured the same way")

    agreement = float((teacher == truth).mean())
    tp = int(((teacher == 1) & (truth == 1)).sum())
    fp = int(((teacher == 1) & (truth == 0)).sum())
    fn = int(((teacher == 0) & (truth == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    base_rate = pos / len(truth)
    majority_agreement = max(base_rate, 1 - base_rate)

    # Agreement is the wrong yardstick when one class dominates: 84% of these
    # pages measured failing, so "always yes" already scores 0.84 agreement. The
    # question is whether the teacher beats that trivial predictor, tested by
    # shuffling its answers to get the chance distribution of its own precision.
    rng = np.random.default_rng(0)
    trials, at_least = 2000, 0
    for _ in range(trials):
        shuffled = rng.permutation(teacher)
        flagged = int((shuffled == 1).sum())
        if not flagged:
            continue
        if float(((shuffled == 1) & (truth == 1)).sum()) / flagged >= precision:
            at_least += 1
    p_value = (at_least + 1) / (trials + 1)

    # Whether the answer tracks measured severity, not just the label.
    severity_corr = float(np.corrcoef(teacher, severity)[0, 1]) if severity.std() > 0 else 0.0

    beats_trivial = bool(precision > base_rate and agreement >= majority_agreement - 0.02)
    verdict = {
        "pages_compared": len(rows),
        "measured_failing": pos,
        "measured_failing_base_rate": round(base_rate, 4),
        "teacher_says_failing": int(teacher.sum()),
        "agreement": round(agreement, 4),
        "majority_class_agreement": round(majority_agreement, 4),
        "random_guess_agreement": round((pos * pos + neg * neg) / (len(rows) ** 2), 4),
        "precision": round(precision, 4),
        "precision_lift_over_base_rate": round(precision - base_rate, 4),
        "recall": round(recall, 4),
        "precision_permutation_p": round(p_value, 5),
        "teacher_vs_measured_min_contrast_r": round(severity_corr, 4),
        "beats_trivial_predictor": beats_trivial,
    }

    teacher = np.asarray([r["teacher_says_failing"] for r in rows], dtype=int)
    truth = np.asarray([r["measured_has_failing"] for r in rows], dtype=int)
    order = np.argsort([r["measured_min_contrast"] for r in rows])
    ranks = np.empty(len(rows), float)
    ranks[order] = np.arange(1, len(rows) + 1)
    pos, neg = truth.sum(), len(truth) - truth.sum()
    if pos == 0 or neg == 0:
        raise SystemExit("degenerate ground truth: every page measured the same way")

    agreement = float((teacher == truth).mean())
    tp = int(((teacher == 1) & (truth == 1)).sum())
    # The defect prompt was validated above. The ordinal grade is what would
    # actually be distilled, so it gets its own test: do higher grades go with
    # better measured pixels, or is the grade a constant with noise on top?
    graded = [r for r in rows if r["grade"] is not None]
    grade_values = [r["grade"] for r in graded]
    verdict["grade_distribution"] = (
        np.bincount(grade_values, minlength=4).tolist() if grade_values else []
    )
    if len(set(grade_values)) > 1 and len({r["measured_min_contrast"] for r in graded}) > 1:

        def _rank(values):
            order = np.argsort(values, kind="mergesort")
            out = np.empty(len(values), float)
            i = 0
            while i < len(values):
                j = i
                while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
                    j += 1
                out[order[i : j + 1]] = (i + j) / 2.0 + 1.0
                i = j + 1
            return out

        g = _rank(np.asarray(grade_values, dtype=float))
        c = _rank(np.asarray([r["measured_min_contrast"] for r in graded], dtype=float))
        g = g - g.mean()
        c = c - c.mean()
        denom = np.sqrt((g * g).sum() * (c * c).sum())
        rho = float((g * c).sum() / denom) if denom else 0.0
        verdict["grade_vs_measured_contrast_spearman"] = round(rho, 4)
        by_grade = {}
        for row in graded:
            by_grade.setdefault(row["grade"], []).append(row["measured_min_contrast"])
        verdict["mean_measured_contrast_by_grade"] = {
            str(k): round(float(np.mean(v)), 2) for k, v in sorted(by_grade.items())
        }
        verdict["grade_carries_signal"] = bool(abs(rho) > 0.15)
    verdict["teacher_revision"] = REVISION
    verdict["verdict"] = (
        "teacher carries signal beyond the base rate; its grades may be distilled"
        if beats_trivial
        else "teacher does not beat a trivial predictor on measured pixels; "
        "its grades may not be distilled"
    )

    out = os.path.splitext(args.labels)[0] + ".validation.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=2, sort_keys=True)
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.stage == "label":
        stage_label(parsed_args)
    else:
        stage_validate(parsed_args)
