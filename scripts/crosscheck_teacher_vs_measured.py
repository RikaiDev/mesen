"""Cross-check the UI-UX teacher against System 2 on real pages.

The point of a two-system judge is that the two views disagree sometimes. If
System 1 measured the same thing as System 2, the cross-check would confirm a
number with itself: the previous triage head reached Spearman -0.024 against
System 2's measured violation count on yana, saturated at ~1.0 on seven of eight
pages, which made `evidence_consistency` vacuous.

This puts the teacher's four-axis reading next to the pixel measurements and
reports where they agree, where only one of them sees a problem, and where the
teacher contradicts a measurement. Those three cases are different findings and
only the last two are informative.

  --teacher  four-axis labels from label_four_axes.py
  --witness  directory of witness states, for System 2's measurements
"""

import argparse
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--witness", required=True)
    parser.add_argument("--images", required=True, help="root holding the screenshots")
    return parser.parse_args()


def main():
    args = parse_args()
    from mesen.engine.jev_evaluator import JevEvaluator
    from mesen.schema import ContextSpec, WitnessState

    with open(args.teacher, encoding="utf-8") as handle:
        teacher_payload = json.load(handle)
    evaluator = JevEvaluator()

    rows = []
    for record in teacher_payload["records"]:
        page = record["page"]
        route = page.split("@")[0]
        viewport = page.split("@")[1] if "@" in page else ""
        state_path = os.path.join(args.witness, route, f"state-{viewport}.json")
        if not os.path.exists(state_path):
            continue
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
        image = os.path.join(args.images, route, f"{viewport}.png")
        if not os.path.exists(image):
            continue

        report = evaluator.evaluate_screenshot(
            image, ContextSpec(**state["context"]), witness=WitnessState(**state)
        )
        measured = defaultdict(int)
        for violation in report.violations:
            measured[violation.rule_id.split("/")[-1]] += 1
        rows.append(
            {
                "page": page,
                "route": route,
                "viewport": viewport,
                "axes": {
                    axis: record["axes"].get(axis, {}).get("answer")
                    for axis in teacher_payload["axes"]
                },
                "measured": dict(measured),
                "n_violations": len(report.violations),
                "system2_verdict": report.verdict,
                "system2_score": report.summary_score,
            }
        )

    if not rows:
        raise SystemExit("no comparable pages: teacher labels and witness states do not line up")

    by_route = defaultdict(list)
    for row in rows:
        by_route[row["route"]].append(row)

    print(f"pages cross-checked: {len(rows)} across {len(by_route)} routes\n")
    header = f"{'route':28s} {'vp':>5s}  " + "  ".join(
        f"{a[:9]:>9s}" for a in teacher_payload["axes"]
    )
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: (r["route"], r["viewport"])):
        axes = "  ".join(f"{(row['axes'][a] or '?'):>9s}" for a in teacher_payload["axes"])
        print(f"{row['route'][:27]:28s} {row['viewport']:>5s}  {axes}   s2={row['n_violations']}")

    summary = {"pages": len(rows), "routes": len(by_route), "axes": {}}
    for axis in teacher_payload["axes"]:
        answers = [r["axes"][axis] for r in rows if r["axes"][axis]]
        summary["axes"][axis] = {
            "parsed": len(answers),
            "distribution": {v: answers.count(v) for v in ("yes", "no", "unknown")},
        }

    # The informative cases. A teacher "no" on a page System 2 measured as
    # clean is System 1 seeing something pixels cannot; a teacher "yes" on a
    # page with measured defects is the two views agreeing, which is worth
    # recording but is not new information.
    both_see = []
    only_teacher = []
    only_pixels = []
    for row in rows:
        teacher_says_problem = any(row["axes"].get(a) == "no" for a in teacher_payload["axes"])
        pixels_say_problem = row["n_violations"] > 0
        if teacher_says_problem and pixels_say_problem:
            both_see.append(row["page"])
        elif teacher_says_problem:
            only_teacher.append(row["page"])
        elif pixels_say_problem:
            only_pixels.append(row["page"])
    summary["agreement"] = {
        "both_systems_see_a_problem": len(both_see),
        "only_teacher_sees_a_problem": len(only_teacher),
        "only_measured_defects": len(only_pixels),
        "neither": len(rows) - len(both_see) - len(only_teacher) - len(only_pixels),
        "only_teacher_pages": only_teacher,
        "note": (
            "only_teacher_sees_a_problem is the load-bearing bucket: a page the "
            "expert model rejects while pixel measurement finds nothing is a "
            "finding System 2 structurally cannot produce."
        ),
    }

    out = os.path.splitext(args.teacher)[0] + ".crosscheck.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print("\n" + json.dumps(summary["agreement"], indent=2, sort_keys=True))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
