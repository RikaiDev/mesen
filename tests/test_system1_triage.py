"""System 1 triage: nine tiles, a trained head, and an honest silence.

The failure modes this guards against are the ones that shipped: one squashed
224px frame, and a head answering without a receipt. Both are checked here
against a stub session so the geometry and the abstention rules are pinned
independently of any model file.
"""

import json

import numpy as np
import pytest

from mesen.engine.system1_triage import (
    GRID,
    TILE,
    TriageHead,
    normalized_box,
    page_regions,
    tile_batch,
    triage_page,
)


@pytest.fixture
def page(tmp_path):
    from PIL import Image

    path = tmp_path / "page.png"
    # Non-square and not tile-sized, so any hardcoded 224 assumption shows up.
    Image.new("RGB", (900, 1800), (250, 245, 235)).save(path)
    return str(path)


class StubSession:
    """Returns one distinct feature vector per tile, in tile order."""

    def __init__(self, feature_dim=8, exposes_hidden=True):
        self.feature_dim = feature_dim
        self.exposes_hidden = exposes_hidden
        self.seen_batch = None

    def run(self, _outputs, feed):
        self.seen_batch = feed["screenshot"]
        batch = feed["screenshot"].shape[0]
        base = np.arange(batch * self.feature_dim, dtype=np.float32).reshape(
            batch, 1, self.feature_dim
        )
        return [base] if self.exposes_hidden else []

    def get_outputs(self):
        names = ["hidden_states"] if self.exposes_hidden else ["logits_overall_quality"]
        return [type("O", (), {"name": n})() for n in names]


def test_page_is_cut_into_nine_equal_regions():
    regions = page_regions(900, 1800)
    assert len(regions) == GRID * GRID == 9
    assert regions[0].box == (0, 0, 300, 600)
    assert regions[8].box == (600, 1200, 900, 1800)
    # Every pixel is covered exactly once.
    covered = sum((b[2] - b[0]) * (b[3] - b[1]) for b in (r.box for r in regions))
    assert covered == 900 * 1800


def test_tiles_are_cropped_before_resized(page):
    batch, (width, height) = tile_batch(page)
    assert batch.shape == (9, 3, TILE, TILE)
    assert (width, height) == (900, 1800)


def test_normalized_box_is_page_relative_and_yxxy():
    assert normalized_box((0, 0, 450, 900), 900, 1800) == [0.0, 0.0, 0.5, 0.5]
    assert normalized_box((300, 600, 600, 1200), 900, 1800) == [
        0.3333,
        0.3333,
        0.6667,
        0.6667,
    ]


def test_missing_head_checkpoint_yields_no_signal_not_a_clean_page(page, tmp_path):
    head = TriageHead(str(tmp_path / "nope.pt"))
    result = triage_page(page, StubSession(), head)
    assert result["signal"] is False
    assert result["tiles"] == []
    assert "checkpoint" in result["reason"]


def test_graph_without_hidden_states_yields_no_signal(page, tmp_path):
    head = TriageHead(str(tmp_path / "nope.pt"))
    session = StubSession(exposes_hidden=False)
    result = triage_page(page, session, head)
    assert result["signal"] is False
    assert "hidden_states" in result["reason"]


def test_triage_scores_every_tile_and_locates_flagged_ones(page, tmp_path, monkeypatch):
    head = TriageHead(str(tmp_path / "heads.pt"))
    probabilities = np.linspace(0.1, 0.9, 9, dtype=np.float32)

    monkeypatch.setattr(head, "load", lambda: object())
    monkeypatch.setattr(
        head,
        "score",
        lambda features: (probabilities, np.full(9, 5.0, dtype=np.float32)),
    )
    session = StubSession()
    session.run(None, {"screenshot": np.zeros((9, 3, TILE, TILE), dtype=np.float32)})
    result = triage_page(page, session, head)

    assert result["signal"] is True
    assert len(result["tiles"]) == 9
    assert result["flagged_tiles"] == [5, 6, 7, 8]
    worst = result["tiles"][-1]
    assert worst["normalized_box"] == [0.6667, 0.6667, 1.0, 1.0]
    assert worst["worth_measuring"] is True
    assert "System 2" in result["role"], "triage must not present itself as a verdict"


def test_batch_is_actually_sent_to_the_session(page, tmp_path, monkeypatch):
    head = TriageHead(str(tmp_path / "heads.pt"))
    monkeypatch.setattr(head, "load", lambda: object())
    monkeypatch.setattr(head, "score", lambda f: (np.zeros(9), np.zeros(9)))
    session = StubSession()
    triage_page(page, session, head)
    assert session.seen_batch.shape == (9, 3, TILE, TILE)


def test_triage_head_reports_its_own_provenance(tmp_path):
    checkpoint = tmp_path / "heads.pt"
    torch = pytest.importorskip("torch")
    from torch import nn

    torch.save(
        {
            "trunk": nn.Sequential(nn.Linear(8, 512), nn.LayerNorm(512), nn.GELU()).state_dict(),
            "fail_head": nn.Linear(512, 2).state_dict(),
            "contrast_head": nn.Linear(512, 1).state_dict(),
            "feature_mean": torch.zeros(1, 8),
            "feature_std": torch.ones(1, 8),
            "input_dim": 8,
        },
        checkpoint,
    )
    (tmp_path / "triage_receipt.json").write_text(
        json.dumps({"val_auc_has_failing_text": 0.81}), encoding="utf-8"
    )
    head = TriageHead(str(checkpoint))
    assert head.available
    head.load()
    assert head.receipt["val_auc_has_failing_text"] == 0.81


def test_threshold_is_above_chance_not_above_nothing(page, tmp_path, monkeypatch):
    from mesen.engine.system1_triage import TRIAGE_REPORT_MIN

    assert TRIAGE_REPORT_MIN >= 0.55, "below this a random head would flag tiles"
    head = TriageHead(str(tmp_path / "heads.pt"))
    monkeypatch.setattr(head, "load", lambda: object())
    # A head that is barely better than chance must not send tiles to System 2.
    probs = np.array([0.3, 0.5, 0.55] + [0.1] * 6, dtype=np.float32)
    monkeypatch.setattr(head, "score", lambda f: (probs, np.ones(9, dtype=np.float32)))
    result = triage_page(page, StubSession(), head)
    assert result["flagged_tiles"] == []


def test_wiring_reports_no_signal_when_nothing_is_installed(tmp_path):
    from mesen.engine.triage_wiring import triage_for_image

    result = triage_for_image("/nonexistent.png", str(tmp_path))
    assert result["signal"] is False
    assert "hidden_states" in result["reason"]


def test_wiring_tolerates_a_missing_models_dir():
    from mesen.engine.triage_wiring import triage_for_image

    result = triage_for_image("/nonexistent.png", None)
    assert result["signal"] is False


def test_triage_is_the_only_registered_calibrated_head():
    from mesen.engine.head_calibration import CALIBRATION, is_calibrated
    from mesen.engine.triage_wiring import is_triage_calibrated

    assert is_calibrated("triage")
    assert is_triage_calibrated()
    calibrated = {h for h, c in CALIBRATION.items() if c.calibrated}
    # evidence_consistency is derived, not neural; triage is the only fitted head.
    assert calibrated == {"evidence_consistency", "triage"}


def test_payload_is_json_serialisable(tmp_path, monkeypatch):
    from mesen.engine.triage_wiring import describe_triage, triage_to_payload

    payload = triage_to_payload(
        {
        "signal": True,
        "tiles": [],
        "flagged_tiles": [],
        "receipt": {"val_auc_has_failing_text": 0.7454},
    }
    )
    assert json.loads(json.dumps(payload))["signal"] is True
    assert "held-out AUC 0.7454" in describe_triage(payload)


def test_describe_states_absence_as_absence_of_opinion():
    from mesen.engine.triage_wiring import describe_triage

    line = describe_triage({"signal": False, "reason": "no graph"})
    assert "no signal" in line
    assert "clean" not in line
