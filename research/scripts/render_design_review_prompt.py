"""Render the fixed four-axis design rubric for an independent VLM reviewer."""

import argparse
import json
from pathlib import Path

AXES = ("design_quality", "originality", "craft", "functionality")
RUBRIC_PATH = Path(__file__).resolve().parents[1] / "contracts" / "four_axis_rubric.json"


def render_system_prompt(rubric_path: Path = RUBRIC_PATH) -> str:
    """Return the versioned rubric without inventing unsupported positive labels."""
    rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    criteria = rubric.get("criteria", {})
    if rubric.get("schema") != "mesen.design-rubric.v1" or set(criteria) != set(AXES):
        raise ValueError("Incomplete four-axis rubric")
    lines = [
        "You are an independent UI/UX reviewer. Inspect the supplied images, brief, "
        "browser witness, and comparison context. Return JSON only.",
        *rubric["common_rules"],
    ]
    for axis in AXES:
        criterion = criteria[axis]
        lines.extend(
            [
                f"{axis}: {criterion['question']}",
                f"yes: {criterion['yes']}",
                f"no: {criterion['no']}",
                f"unknown: {criterion['unknown']}",
            ]
        )
    lines.append(
        "Return exactly these keys: "
        + ", ".join(AXES)
        + '. Each value is {"choice":"yes|no|unknown","evidence_ref":"..."}. '
        "The evidence_ref must identify a supplied source or the specific missing source."
    )
    return "\n".join(lines)


def main() -> None:
    """Print the current canonical reviewer system prompt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rubric", type=Path, default=RUBRIC_PATH)
    args = parser.parse_args()
    print(render_system_prompt(args.rubric))


if __name__ == "__main__":
    main()
