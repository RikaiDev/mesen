"""
Export VlmJevModel to ONNX graph and perform INT8 quantization.
"""

import argparse
import os

import torch

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
except ImportError:
    onnx = None
    ort = None


def export_vlm_jev_to_onnx(
    model: torch.nn.Module,
    output_onnx_path: str,
    hidden_dim: int = 2048,
    quantize: bool = True,
    quantized_path: str | None = None,
):
    """
    Exports VlmJevModel (or its decision head wrapper) to ONNX format.
    """
    model = model.float().cpu()
    model.eval()
    dummy_input = torch.randn(1, hidden_dim, dtype=torch.float32)

    class ExportWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, features):
            res = self.m(latent_features=features)
            return (
                res["primary_action_reachable"],
                res["visual_integrity"],
                res["responsive_consistency"],
                res["evidence_consistency"],
                res["operator_clarity"],
                res["overall_quality"],
            )

    wrapper = ExportWrapper(model)

    output_names = [
        "logits_primary_action",
        "logits_visual_integrity",
        "logits_responsive_consistency",
        "logits_evidence_consistency",
        "logits_operator_clarity",
        "logits_overall_quality",
    ]

    print(f"Exporting ONNX model to {output_onnx_path}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["latent_features"],
        output_names=output_names,
        dynamic_axes={
            "latent_features": {0: "batch_size"},
            "logits_primary_action": {0: "batch_size"},
            "logits_visual_integrity": {0: "batch_size"},
            "logits_responsive_consistency": {0: "batch_size"},
            "logits_evidence_consistency": {0: "batch_size"},
            "logits_operator_clarity": {0: "batch_size"},
            "logits_overall_quality": {0: "batch_size"},
        },
    )
    print("ONNX export succeeded!")

    if quantize and ort is not None:
        if quantized_path is None:
            base, ext = os.path.splitext(output_onnx_path)
            quantized_path = f"{base}.int8{ext}"
        print(f"Quantizing ONNX model to INT8: {quantized_path}...")
        quantize_dynamic(
            model_input=output_onnx_path,
            model_output=quantized_path,
            weight_type=QuantType.QInt8,
        )
        print("INT8 Dynamic Quantization complete!")
        return quantized_path

    return output_onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        type=str,
        default="/mnt/model-cache/vlm-jev/Qwen3.5-2B-Base",
        help="Path to base model",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint safetensors"
    )
    parser.add_argument("--output", type=str, default="vlm_jev.onnx", help="Output ONNX filename")
    parser.add_argument("--quantize", action="store_true", help="Perform dynamic INT8 quantization")
    args = parser.parse_args()

    from safetensors.torch import load_file

    from mesen.model.vlm_jev import VlmJevModel

    print(f"Loading model checkpoint from {args.checkpoint}...")
    model = VlmJevModel(base_model_name_or_path=args.base_model)
    state_dict = load_file(args.checkpoint)
    model.load_state_dict(state_dict)

    export_vlm_jev_to_onnx(
        model=model,
        output_onnx_path=args.output,
        hidden_dim=2048,
        quantize=args.quantize,
    )
