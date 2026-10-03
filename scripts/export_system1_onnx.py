"""Export the ViT+heads graph with `hidden_states` exposed.

The shipped graph starts at `screenshot` and ends at the eight head outputs,
so the 1536-d fused representation the heads consume is invisible from outside.
Every head experiment therefore had to re-export the whole 375MB model just to
read it. Adding `hidden_states` as an extra output makes the representation
readable without changing what the existing outputs mean, so head retraining
becomes a small artifact instead of a model re-export.

The head weights come from a checkpoint when one is given. Without one they are
left at initialization and every head abstains, which is the honest state and
the reason `mesen/engine/head_calibration.py` exists.
"""

import argparse
import os

import torch

from mesen.model.vit_consultant import MesenViTConsultantModel


def build_model(head_checkpoint: str | None, pretrained: bool = True):
    model = MesenViTConsultantModel(pretrained=pretrained)
    receipt = {"head_checkpoint": head_checkpoint or None, "heads_loaded": False}
    if head_checkpoint:
        if not os.path.exists(head_checkpoint):
            raise FileNotFoundError(f"head checkpoint not found: {head_checkpoint}")
        blob = torch.load(head_checkpoint, map_location="cpu", weights_only=False)
        state = blob.get("model_state_dict") or blob
        missing, unexpected = model.consultant_heads.load_state_dict(state, strict=False)
        receipt.update(
            heads_loaded=True,
            missing_keys=list(missing),
            unexpected_keys=list(unexpected),
        )
    return model, receipt


class ExportWrapper(torch.nn.Module):
    """Head outputs plus the fused representation they were computed from."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, screenshot):
        hidden = self.model.extract_patch_features(screenshot)
        out = self.model.consultant_heads(hidden_states=hidden)
        logits = out["atomic_logits"]
        return (
            hidden,
            logits["primary_action_reachable"],
            logits["visual_integrity"],
            logits["responsive_consistency"],
            logits["evidence_consistency"],
            logits["operator_clarity"],
            logits["overall_quality"],
            out["rule_logits"],
            out["pred_bboxes"],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="destination .onnx")
    parser.add_argument("--head-checkpoint", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    model, receipt = build_model(args.head_checkpoint, pretrained=not args.no_pretrained)
    model = model.float().cpu().eval()

    names = [
        "hidden_states",
        "logits_primary_action",
        "logits_visual_integrity",
        "logits_responsive_consistency",
        "logits_evidence_consistency",
        "logits_operator_clarity",
        "logits_overall_quality",
        "rule_logits",
        "pred_bboxes",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        ExportWrapper(model),
        torch.randn(1, 3, 224, 224),
        args.out,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["screenshot"],
        output_names=names,
        dynamic_axes={"screenshot": {0: "batch"}, "hidden_states": {0: "batch"}},
    )

    import json

    import onnx
    import onnxruntime as ort

    graph = onnx.load(args.out).graph
    receipt.update(
        onnx_path=args.out,
        outputs=[o.name for o in graph.output],
        bytes=os.path.getsize(args.out),
        verified=bool(ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"wrote {args.out} ({receipt['bytes'] / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
