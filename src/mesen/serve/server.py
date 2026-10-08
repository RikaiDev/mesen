"""
FastAPI On-Prem Serving Daemon for Mesen UX Consultant & Decision Engine.
Equipped with:
1. Ultra-fast Sub-millisecond ONNX Runtime Inference
2. Zero-Fluff Standardized Rule Registry Grounding (WCAG / ISO / Ambient Mirror)
3. Prescriptive UX Consultation Reports with Bounding Box Coordinates
"""

import base64
import os
import tempfile

import uvicorn
from fastapi import FastAPI, HTTPException

from mesen.engine.dual_judge import compute_agreement, derive_system_two
from mesen.engine.jev_vlm_engine import JevVlmEngine
from mesen.rules.registry import RULE_ID_LIST
from mesen.schema import (
    ContextSpec,
    JudgeRequest,
    JudgeResponse,
)

app = FastAPI(title="Mesen UX Consultant & Decision Daemon", version="0.3.0")

_engine: JevVlmEngine | None = None


def load_model(checkpoint_path: str | None = None):
    global _engine
    default_path = "models/onnx/mesen_jev_vlm.onnx"
    target_path = checkpoint_path or default_path
    if os.path.exists(target_path):
        print(f"Loading Mesen JevVlmEngine from {target_path}...")
        _engine = JevVlmEngine(onnx_model_path=target_path)
        print("Mesen JevVlmEngine ready!")
    else:
        print(f"Warning: Model checkpoint not found at {target_path}. Running unloaded.")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "mesen-ux-consultant",
        "backend": "jev_vlm_engine" if _engine is not None else "unloaded",
        "model_loaded": _engine is not None,
        "rules_registered": len(RULE_ID_LIST),
    }


def _resolve_image_path(request: JudgeRequest) -> tuple[str, bool]:
    """Resolves an image from request (paths, base64, or viewport facts).
    Returns (image_path, is_temp). If is_temp is True, caller should delete image_path when done.
    """
    # 1. Direct file paths
    if request.image_paths:
        for p in request.image_paths:
            if p and os.path.isfile(p):
                return p, False

    # 2. Base64 images
    if request.images_base64:
        for b64_str in request.images_base64:
            if not b64_str:
                continue
            try:
                # Strip data URI header if present
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                data = base64.b64decode(b64_str)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    f.write(data)
                    return f.name, True
            except Exception:
                continue

    # 3. Viewport facts screenshot references
    if request.state and request.state.viewport_facts:
        for fact in request.state.viewport_facts:
            if not fact.screenshot:
                continue
            if os.path.isfile(fact.screenshot):
                return fact.screenshot, False
            # Check if screenshot is base64
            if fact.screenshot.startswith("data:image/") or len(fact.screenshot) > 256:
                try:
                    s = fact.screenshot
                    if "," in s:
                        s = s.split(",", 1)[1]
                    data = base64.b64decode(s)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        f.write(data)
                        return f.name, True
                except Exception:
                    continue

    raise HTTPException(
        status_code=400,
        detail="At least one valid image (image_paths, images_base64, or viewport screenshot) is required for UI judgment.",
    )


@app.post("/v1/judge", response_model=JudgeResponse)
def judge_ui(request: JudgeRequest):
    if _engine is None:
        raise HTTPException(status_code=503, detail="ONNX model not loaded")

    image_path, is_temp = _resolve_image_path(request)
    try:
        context = request.state.context or ContextSpec()
        result = _engine.evaluate(
            image_path=image_path,
            context=context,
            witness=request.state,
        )
        report = getattr(result, "report", result[0])
        answers = getattr(result, "answers", result[1])
        system_two = getattr(result, "system_two", None)
        if system_two is None:
            system_two = derive_system_two(report, witness=request.state)
        agreement = getattr(result, "agreement", None)
        if agreement is None:
            agreement = compute_agreement(answers, system_two)

        return JudgeResponse(
            answers=answers,
            consultation=report,
            system_two=system_two,
            agreement=agreement,
        )
    finally:
        if is_temp and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass


def run_server(
    host: str = "0.0.0.0",
    port: int = 8089,
    checkpoint: str | None = None,
):
    load_model(checkpoint)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
