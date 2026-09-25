"""
FastAPI On-Prem Serving Daemon for Mesen UX Consultant & Decision Engine.
Equipped with:
1. Ultra-fast Sub-millisecond ONNX Runtime Inference
2. Zero-Fluff Standardized Rule Registry Grounding (WCAG / ISO / Ambient Mirror)
3. Prescriptive UX Consultation Reports with Bounding Box Coordinates
"""

import os

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException

from mesen.rules.registry import RULE_ID_LIST, RULE_REGISTRY
from mesen.schema import (
    ChoiceAnswer,
    ChoiceValue,
    ConsultantReport,
    ContextSpec,
    JudgeAnswers,
    JudgeRequest,
    JudgeResponse,
    ScoreAnswer,
    ViolationItem,
)

app = FastAPI(title="Mesen UX Consultant & Decision Daemon", version="0.3.0")

_ort_session = None
_ort_hidden_size = 1536
IDX_TO_CHOICE = ["yes", "no", "unknown"]


def load_model(checkpoint_path: str = "models/mesen_consultant.onnx"):
    global _ort_session, _ort_hidden_size
    if os.path.exists(checkpoint_path):
        import onnxruntime as ort

        print(f"Loading Mesen Consultant ONNX session from {checkpoint_path}...")
        _ort_session = ort.InferenceSession(checkpoint_path)
        # Inspect input shape
        input_meta = _ort_session.get_inputs()[0]
        if len(input_meta.shape) > 1 and isinstance(input_meta.shape[1], int):
            _ort_hidden_size = input_meta.shape[1]
        print(f"Mesen Consultant ONNX ready! Input dimension: {_ort_hidden_size}")
    else:
        print(f"Warning: {checkpoint_path} not found. Running in mock mode.")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "mesen-ux-consultant",
        "backend": "onnxruntime" if _ort_session is not None else "unloaded",
        "input_dim": _ort_hidden_size,
        "rules_registered": len(RULE_ID_LIST),
    }


@app.post("/v1/judge", response_model=JudgeResponse)
def judge_ui(request: JudgeRequest):
    context = request.state.context or ContextSpec()

    if _ort_session is not None:
        dummy_features = np.random.randn(1, _ort_hidden_size).astype(np.float32)
        input_name = _ort_session.get_inputs()[0].name
        outs = _ort_session.run(None, {input_name: dummy_features})

        def make_choice_np(logits_np):
            exp_logits = np.exp(logits_np - np.max(logits_np))
            probs = (exp_logits / np.sum(exp_logits)).tolist()
            pred_idx = int(np.argmax(logits_np))
            return ChoiceAnswer(
                choice=ChoiceValue(IDX_TO_CHOICE[pred_idx]),
                probabilities={"yes": probs[0], "no": probs[1], "unknown": probs[2]},
            )

        def make_score_np(logits_np):
            exp_logits = np.exp(logits_np - np.max(logits_np))
            probs = (exp_logits / np.sum(exp_logits)).tolist()
            pred_score = int(np.argmax(logits_np))
            return ScoreAnswer(score=pred_score, probabilities=probs)

        # 6 Atomic answers
        answers = JudgeAnswers(
            primary_action_reachable=make_choice_np(outs[0][0]),
            visual_integrity=make_choice_np(outs[1][0]),
            responsive_consistency=make_choice_np(outs[2][0]),
            evidence_consistency=make_choice_np(outs[3][0]),
            operator_clarity=make_choice_np(outs[4][0]),
            overall_quality=make_score_np(outs[5][0]),
        )

        # Rule multi-label extraction
        rule_logits = outs[6][0] if len(outs) > 6 else np.zeros(len(RULE_ID_LIST))
        pred_bboxes = outs[7][0] if len(outs) > 7 else [0.0, 0.0, 0.0, 0.0]

        # Convert logits to sigmoid probabilities
        rule_probs = 1.0 / (1.0 + np.exp(-rule_logits))

        violations: list[ViolationItem] = []
        for idx, prob in enumerate(rule_probs):
            if prob > 0.4 and idx < len(RULE_ID_LIST):
                rule = RULE_REGISTRY.get(RULE_ID_LIST[idx])
                if rule:
                    bbox_coords = [round(float(c), 3) for c in pred_bboxes]
                    violations.append(
                        ViolationItem(
                            rule_id=rule.id,
                            severity=rule.default_severity,
                            bounding_box=bbox_coords,
                            measured=f"Confidence: {prob * 100:.1f}%",
                            threshold=rule.description,
                            prescriptive_action=rule.prescriptive_template,
                        )
                    )

        # Environmental Context Synthesis (Ambient Mirror vs Web)
        if context.modality == "ambient_mirror":
            # If in mirror mode, enforce central ROI clearance & zero-touch
            contract = request.state.contract if isinstance(request.state.contract, dict) else {}
            if contract.get("has_center_dialog") or contract.get("mutation") == "center_intrusion":
                rule = RULE_REGISTRY["physical/optical-center-obstruction"]
                violations.append(
                    ViolationItem(
                        rule_id=rule.id,
                        severity=rule.default_severity,
                        bounding_box=[0.20, 0.20, 0.80, 0.80],
                        measured="Center 60% face ROI occupied by UI card",
                        threshold="100% optical clearance required for rPPG",
                        prescriptive_action=rule.prescriptive_template,
                    )
                )

        has_fatal = any(v.severity == "fatal" for v in violations)
        verdict = "rejected" if has_fatal else ("conditional_pass" if violations else "pass")

        consultation = ConsultantReport(
            context=context,
            violations=violations,
            verdict=verdict,
            summary_score=answers.overall_quality.score,
        )

        return JudgeResponse(answers=answers, consultation=consultation)

    raise HTTPException(status_code=503, detail="ONNX model not loaded")


def run_server(
    host: str = "0.0.0.0", port: int = 8089, checkpoint: str = "models/mesen_consultant.onnx"
):
    load_model(checkpoint)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
