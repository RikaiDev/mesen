"""Tests for FastAPI serving daemon (src/mesen/serve/server.py)."""

import base64
import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import mesen.serve.server as server_module
from mesen.engine.jev_vlm_engine import JevVlmEngine
from mesen.schema import (
    ChoiceAnswer,
    ConsultantReport,
    ContextSpec,
    JudgeAnswers,
    ScoreAnswer,
)


@pytest.fixture(autouse=True)
def reset_server_engine():
    """Ensure clean engine state between tests."""
    original = server_module._engine
    yield
    server_module._engine = original


def _mock_answers() -> JudgeAnswers:
    return JudgeAnswers(
        primary_action_reachable=ChoiceAnswer(choice="yes", confidence=0.99),
        visual_integrity=ChoiceAnswer(choice="yes", confidence=0.99),
        responsive_consistency=ChoiceAnswer(choice="yes", confidence=0.99),
        evidence_consistency=ChoiceAnswer(choice="yes", confidence=0.99),
        operator_clarity=ChoiceAnswer(choice="yes", confidence=0.99),
        overall_quality=ScoreAnswer(score=3, confidence=0.99),
    )


def _mock_engine():
    engine = JevVlmEngine.__new__(JevVlmEngine)
    report = ConsultantReport(
        context=ContextSpec(),
        violations=[],
        verdict="pass",
        summary_score=3,
    )
    engine.evaluate = lambda image_path, context=None, dpr=None, mode="dual", witness=None: (
        report,
        _mock_answers(),
    )
    return engine


def test_health_check_unloaded():
    server_module._engine = None
    client = TestClient(server_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["backend"] == "unloaded"
    assert data["model_loaded"] is False


def test_health_check_loaded():
    server_module._engine = _mock_engine()
    client = TestClient(server_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["backend"] == "jev_vlm_engine"
    assert data["model_loaded"] is True


def test_judge_unloaded_raises_503():
    server_module._engine = None
    client = TestClient(server_module.app)
    req = {
        "state": {
            "product": "test",
            "context": {"modality": "desktop_web"},
        },
        "image_paths": ["any_image.png"],
    }
    resp = client.post("/v1/judge", json=req)
    assert resp.status_code == 503
    assert "not loaded" in resp.json()["detail"]


def test_judge_no_image_raises_400():
    server_module._engine = _mock_engine()
    client = TestClient(server_module.app)
    req = {
        "state": {
            "product": "test",
            "context": {"modality": "desktop_web"},
        },
        "image_paths": ["non_existent_file.png"],
    }
    resp = client.post("/v1/judge", json=req)
    assert resp.status_code == 400
    assert "At least one valid image" in resp.json()["detail"]


def test_judge_with_base64_image_cleans_tempfile():
    server_module._engine = _mock_engine()
    client = TestClient(server_module.app)

    # Generate small 16x16 PNG in memory
    buf = io.BytesIO()
    img = Image.new("RGB", (16, 16), color="red")
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    captured_temp_path = None
    original_evaluate = server_module._engine.evaluate

    def spy_evaluate(image_path, **kwargs):
        nonlocal captured_temp_path
        captured_temp_path = image_path
        assert os.path.exists(image_path)
        return original_evaluate(image_path, **kwargs)

    server_module._engine.evaluate = spy_evaluate

    req = {
        "state": {
            "product": "portal",
            "context": {"modality": "desktop_web"},
        },
        "images_base64": [f"data:image/png;base64,{b64_str}"],
    }
    resp = client.post("/v1/judge", json=req)
    assert resp.status_code == 200
    assert captured_temp_path is not None
    # Verify temporary file was deleted after request completion
    assert not os.path.exists(captured_temp_path)


def test_load_model_wiring():
    from mesen.hub import get_default_models_dir

    onnx_path = os.path.join(get_default_models_dir(), "mesen_jev_vlm.onnx")
    if not os.path.exists(onnx_path):
        pytest.skip("mesen_jev_vlm.onnx not staged")
    server_module.load_model(onnx_path)
    assert server_module._engine is not None
    assert isinstance(server_module._engine, JevVlmEngine)
