"""
FastAPI On-Prem Serving Daemon for vlm-jev UI decision engine.
"""

import os
from typing import Optional
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
import uvicorn

from mesen.schema import (
    ChoiceAnswer,
    ChoiceValue,
    JudgeAnswers,
    JudgeRequest,
    JudgeResponse,
    ScoreAnswer,
)
from mesen.model.vlm_jev import VlmJevModel
from mesen.pipeline.dataset import IDX_TO_CHOICE

app = FastAPI(title="mesen (vlm-jev) UI Decision Daemon", version="0.2.0")

_model: Optional[VlmJevModel] = None
_device: Optional[torch.device] = None


def load_model(checkpoint_path: Optional[str] = None, base_model: str = "Qwen/Qwen3.5-2B-Base"):
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading vlm-jev model onto {_device}...")
    _model = VlmJevModel(base_model_name_or_path=base_model)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from {checkpoint_path}...")
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
        _model.load_state_dict(state_dict, strict=False)
    _model.to(_device)
    _model.eval()
    print("vlm-jev model ready for inference!")


@app.get("/health")
def health_check():
    return {"status": "ok", "device": str(_device), "model_ready": _model is not None}


@app.post("/v1/judge", response_model=JudgeResponse)
def judge_ui(request: JudgeRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized")

    # Construct input tokens from witness state
    state = request.state
    # In real pipeline, images are processed and embedded via vision encoder
    # Here we mock the latent feature pass or perform single forward pass
    with torch.no_grad():
        dummy_features = torch.randn(1, _model.hidden_dim, device=_device)
        logits_dict = _model(latent_features=dummy_features)

        def make_choice(key: str) -> ChoiceAnswer:
            logits = logits_dict[key].squeeze(0)
            probs = F.softmax(logits, dim=-1).cpu().tolist()
            pred_idx = int(torch.argmax(logits).item())
            choice_str = IDX_TO_CHOICE[pred_idx]
            return ChoiceAnswer(
                choice=ChoiceValue(choice_str),
                probabilities={"yes": probs[0], "no": probs[1], "unknown": probs[2]},
            )

        def make_score(key: str) -> ScoreAnswer:
            logits = logits_dict[key].squeeze(0)
            probs = F.softmax(logits, dim=-1).cpu().tolist()
            pred_score = int(torch.argmax(logits).item())
            return ScoreAnswer(score=pred_score, probabilities=probs)

        answers = JudgeAnswers(
            primary_action_reachable=make_choice("primary_action_reachable"),
            visual_integrity=make_choice("visual_integrity"),
            responsive_consistency=make_choice("responsive_consistency"),
            evidence_consistency=make_choice("evidence_consistency"),
            operator_clarity=make_choice("operator_clarity"),
            overall_quality=make_score("overall_quality"),
        )

    return JudgeResponse(answers=answers)


def run_server(host: str = "0.0.0.0", port: int = 8088, checkpoint: Optional[str] = None):
    load_model(checkpoint)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
