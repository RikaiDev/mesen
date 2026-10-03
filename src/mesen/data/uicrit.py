"""
UICrit human ratings: real labeled UI screenshots.

The System 1 heads were fit to `data/synthetic`, 420 rendered template
screens whose labels are the generator's own mutation type. This module
loads the opposite: 1000 real app screens, each rated by human critics
(Google Research UICrit). One screen is the unit of supervision, because
all task rows of a screen share one set of ratings.
"""

import ast
import csv
import re
from dataclasses import dataclass

# Ratings a human critic gave the screen, averaged over that screen's tasks.
RATING_COLUMNS = ("aesthetics_rating", "usability_rating", "design_quality_rating")

# design_quality_rating runs 1..9. mesen's overall_quality runs 0..3
# (Blocked / Confusing / Usable / Excellent). The bins below are a mapping
# decision, not a fact: they are recorded here so every metric that depends
# on them names the boundary it used. Rank metrics (Spearman) do not use them.
QUALITY_UPPER_BOUNDS = ((4, 0), (5, 1), (6, 2), (99, 3))


@dataclass(frozen=True)
class ScreenRating:
    """One screen's human ratings. `ratings` maps a column name to its mean."""

    rico_id: str
    ratings: dict[str, float]
    n_tasks: int

    @property
    def design_quality(self) -> float:
        return self.ratings["design_quality_rating"]

    @property
    def overall_quality_class(self) -> int:
        """mesen overall_quality (0..3) from the human design_quality_rating."""
        return ordinal_quality_class(self.design_quality)


def ordinal_quality_class(design_quality: float) -> int:
    """Map a 1..9 human design rating onto mesen's 0..3 ordinal scale."""
    for upper, label in QUALITY_UPPER_BOUNDS:
        if design_quality <= upper:
            return label
    raise AssertionError("QUALITY_UPPER_BOUNDS must cover the rating range")


def load_screen_ratings(csv_path: str) -> dict[str, ScreenRating]:
    """Map rico_id to its averaged human ratings.

    Rows whose rating cell is not a number are skipped rather than coerced:
    a silently imputed rating would become a training label nobody wrote.
    """
    per_screen: dict[str, list[dict[str, float]]] = {}
    with open(csv_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                values = {col: float(row[col]) for col in RATING_COLUMNS}
            except (KeyError, TypeError, ValueError):
                continue
            per_screen.setdefault(row["rico_id"], []).append(values)

    screens: dict[str, ScreenRating] = {}
    for rico_id, rows in per_screen.items():
        screens[rico_id] = ScreenRating(
            rico_id=rico_id,
            ratings={col: sum(row[col] for row in rows) / len(rows) for col in RATING_COLUMNS},
            n_tasks=len(rows),
        )
    return screens


BOX_PATTERN = re.compile(r"Bounding Box:\s*\[([^\]]+)\]")


def human_defect_box_count(csv_path: str, sources=("human", "both")) -> dict[str, int]:
    """Count human-annotated defect boxes per screen.

    A screen with at least one box is a screen a human critic marked as
    having a specific visual defect; screens without boxes are unlabeled,
    not negative. Used only as a weak signal, never as a yes/no label.
    """
    counts: dict[str, int] = {}
    with open(csv_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                comment_sources = ast.literal_eval(row["comments_source"])
                comments = ast.literal_eval(row["comments"])
            except (KeyError, ValueError, SyntaxError):
                continue
            if not isinstance(comments, list):
                comments = [comments]
            for source, comment in zip(comment_sources, comments):
                if source not in sources or not isinstance(comment, str):
                    continue
                counts[row["rico_id"]] = counts.get(row["rico_id"], 0) + len(
                    BOX_PATTERN.findall(comment)
                )
    return counts
