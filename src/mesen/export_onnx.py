"""
ONNX Exporter for Mesen UX Consultant Model.
Exports the trained multi-task consultant model to ONNX for 0.2ms edge inference.
"""

import os

import torch

from mesen.model.consultant_model import MesenConsultantModel
from mesen.rules.registry import RULE_DEFINITIONS


def export_consultant_onnx(
    checkpoint_path: str = "checkpoints/best_consultant_model.pt",
    onnx_path: str = "models/mesen_consultant.onnx",
    hidden_size: int = 1536,
):
    print(f"Loading checkpoint from {checkpoint_path}...")
    model = MesenConsultantModel(hidden_size=hidden_size, num_rules=len(RULE_DEFINITIONS))

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print("Model weights successfully loaded.")
    else:
        print(
            f"Warning: Checkpoint {checkpoint_path} not found. Exporting randomly initialized model."
        )

    model.eval()

    dummy_input = torch.randn(1, hidden_size, dtype=torch.float32)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    input_names = ["hidden_states"]
    output_names = [
        "primary_action_reachable",
        "visual_integrity",
        "responsive_consistency",
        "evidence_consistency",
        "operator_clarity",
        "overall_quality",
        "rule_logits",
        "pred_bboxes",
    ]

    class ExportWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            out = self.m(x)
            return (
                out["atomic_logits"]["primary_action_reachable"],
                out["atomic_logits"]["visual_integrity"],
                out["atomic_logits"]["responsive_consistency"],
                out["atomic_logits"]["evidence_consistency"],
                out["atomic_logits"]["operator_clarity"],
                out["atomic_logits"]["overall_quality"],
                out["rule_logits"],
                out["pred_bboxes"],
            )

    wrapped_model = ExportWrapper(model)

    print(f"Exporting to ONNX at {onnx_path}...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"hidden_states": {0: "batch_size"}},
    )
    print(f"Successfully exported Mesen Consultant ONNX model to {onnx_path}!")


if __name__ == "__main__":
    export_consultant_onnx()
