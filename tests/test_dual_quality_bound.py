"""Dual-mode quality is bounded above by measured evidence.

System 1 reads a 224px impression; System 2 measures real pixels. When they
disagree on quality, the measurement wins: a page cannot be better than its
verified defects allow, whatever the impression.
"""

from mesen.schema import ContextSpec


def _require_artifact():
    import os

    artifact = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "onnx", "mesen_jev_vlm.onnx"
    )
    if not os.path.exists(artifact):
        raise AssertionError(
            f"model artifact not staged: {artifact} (run: bash scripts/fetch_models.sh)"
        )


def _white_png(path, w=800, h=600):
    import cv2
    import numpy as np

    cv2.imwrite(path, np.full((h, w, 3), 255, dtype=np.uint8))


def test_dual_quality_never_exceeds_measured_evidence(tmp_path):
    _require_artifact()
    from mesen.engine.jev_vlm_engine import JevVlmEngine

    img = str(tmp_path / "blank.png")
    _white_png(img)
    engine = JevVlmEngine()
    context = ContextSpec(
        cohort="general_public",
        modality="desktop_web",
        interaction_mode="pointer",
        locale="zh-TW",
    )
    report, answers = engine.evaluate(img, context=context)
    assert answers.overall_quality.score <= report.summary_score


def test_fast_mode_keeps_raw_neural_score(tmp_path):
    # fast mode is documented raw System 1 with no evidence pass; the bound
    # applies to dual mode only.
    _require_artifact()
    from mesen.engine.jev_vlm_engine import JevVlmEngine

    img = str(tmp_path / "blank.png")
    _white_png(img)
    engine = JevVlmEngine()
    report, answers = engine.evaluate(img, mode="fast")
    assert 0 <= answers.overall_quality.score <= 3
    assert report.summary_score == answers.overall_quality.score
