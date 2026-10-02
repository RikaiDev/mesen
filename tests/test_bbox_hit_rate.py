"""BBox hit-rate math: identical, contained, disjoint, degenerate."""

import pytest

from mesen.evaluation.bbox_hit_rate import hit_rate, iou, parse_uicrit_boxes


def test_identical_boxes_score_one():
    box = [0.1, 0.2, 0.3, 0.5]
    assert iou(box, box) == pytest.approx(1.0)


def test_disjoint_boxes_score_zero():
    assert iou([0.0, 0.0, 0.1, 0.1], [0.9, 0.9, 1.0, 1.0]) == 0.0


def test_half_overlap_scores_one_third():
    assert iou([0.0, 0.0, 0.5, 0.5], [0.25, 0.0, 0.75, 0.5]) == pytest.approx(1 / 3)


def test_degenerate_box_never_hits():
    assert iou([0.2, 0.2, 0.2, 0.5], [0.0, 0.0, 1.0, 1.0]) == 0.0


def test_hit_rate_counts_best_box_only(tmp_path):
    predictions = [
        ([0.0, 0.0, 0.5, 0.5], [[0.0, 0.0, 0.5, 0.5], [0.9, 0.9, 1.0, 1.0]]),
        ([0.0, 0.0, 0.1, 0.1], [[0.9, 0.9, 1.0, 1.0]]),
        ([0.0, 0.0, 0.1, 0.1], []),
    ]
    assert hit_rate(predictions) == pytest.approx(0.5)


def test_parse_skips_llm_only_and_malformed(tmp_path):
    csv_path = tmp_path / "mini.csv"
    csv_path.write_text(
        "rico_id,comments_source,comments\n"
        '1001,"[' + "'human'" + ']","Comment 1\nToo small. Bounding Box: [0.1, 0.2, 0.3, 0.4]\n"\n'
        '1002,"[' + "'llm'" + ']","Comment 1\nToo small. Bounding Box: [0.1, 0.2, 0.3, 0.4]\n"\n'
        '1003,"[' + "'human'" + ']","Comment 1\nNo box here.\n"\n',
        encoding="utf-8",
    )
    boxes = parse_uicrit_boxes(str(csv_path))
    assert boxes == {"1001": [[0.1, 0.2, 0.3, 0.4]]}
