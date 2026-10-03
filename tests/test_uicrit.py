"""UICrit rating parsing: per-screen aggregation and ordinal mapping."""

import csv

from mesen.data.uicrit import (
    ScreenRating,
    human_defect_box_count,
    load_screen_ratings,
    ordinal_quality_class,
)

HEADER = "rico_id,task,aesthetics_rating,usability_rating,design_quality_rating,comments_source,comments\n"


def write_csv(path, lines):
    path.write_text(HEADER + "".join(lines), encoding="utf-8")
    return str(path)


def test_ratings_average_over_the_screens_tasks(tmp_path):
    csv_path = write_csv(
        tmp_path / "u.csv",
        [
            '1001,t,5,5,5,"[\'human\']","[]"\n',
            '1001,t2,7,7,7,"[\'human\']","[]"\n',
            '1002,t,4,4,6,"[\'human\']","[]"\n',
        ],
    )
    screens = load_screen_ratings(csv_path)
    assert screens["1001"].design_quality == 6.0
    assert screens["1001"].n_tasks == 2
    assert screens["1002"].n_tasks == 1


def test_non_numeric_rating_is_skipped_not_imputed(tmp_path):
    csv_path = write_csv(
        tmp_path / "u.csv",
        ['1001,t,5,5,,"[\'human\']","[]"\n', '1002,t,4,4,4,"[\'human\']","[]"\n'],
    )
    screens = load_screen_ratings(csv_path)
    assert "1001" not in screens
    assert screens["1002"].design_quality == 4.0


def test_quality_scale_maps_the_documented_bounds():
    assert ordinal_quality_class(3) == 0
    assert ordinal_quality_class(4) == 0
    assert ordinal_quality_class(4.5) == 1
    assert ordinal_quality_class(5) == 1
    assert ordinal_quality_class(6) == 2
    assert ordinal_quality_class(6.5) == 3
    assert ordinal_quality_class(9) == 3


def test_screen_rating_class_uses_its_own_design_quality():
    screen = ScreenRating(rico_id="1", ratings={"design_quality_rating": 7.0}, n_tasks=1)
    assert screen.overall_quality_class == 3


def test_defect_boxes_count_only_human_sources(tmp_path):
    human_comment = (
        "Comment 1\nA. Bounding Box: [0.1, 0.2, 0.3, 0.4]\nB. Bounding Box: [0.5, 0.6, 0.7, 0.8]\n"
    )
    # Real UICrit cells are Python-list literals written by csv.writer, so build
    # them with repr() instead of hand-escaping quotes and newlines.
    csv_path = tmp_path / "u.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rico_id",
                "task",
                "aesthetics_rating",
                "usability_rating",
                "design_quality_rating",
                "comments_source",
                "comments",
            ]
        )
        writer.writerow(["1001", "t", 5, 5, 5, "['human']", [human_comment]])
        writer.writerow(["1002", "t", 5, 5, 5, "['llm']", [human_comment]])
    assert human_defect_box_count(str(csv_path)) == {"1001": 2}


def test_missing_image_column_yields_no_ratings(tmp_path):
    csv_path = tmp_path / "u.csv"
    csv_path.write_text("rico_id\n1001\n", encoding="utf-8")
    assert load_screen_ratings(str(csv_path)) == {}
