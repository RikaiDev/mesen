"""System 1's triage rides alongside the measured verdict, never over it.

Wiring rule: triage adds a field to the answer payload and changes no measured
value. System 2 measured 184 real findings on the audited pages; a triage
probability must not add to, subtract from, or reorder those. Anything else
would let a fast approximate head acquire authority it was denied by
`head_calibration`.
"""

import json

from mesen.engine.head_calibration import is_calibrated
from mesen.engine.system1_triage import TriageHead, triage_page

MODEL_WITH_HIDDEN_STATES = "mesen_jev_vlm_v3.onnx"
TRIAGE_CHECKPOINT = "triage_v1/triage_heads.pt"


def build_triage(models_dir: str | None) -> tuple[TriageHead, str | None]:
    """Resolve the triage head and, if present, the graph that exposes features.

    Returns the head plus the ONNX path to use for it. A missing head is not an
    error: the caller gets signal=False and says so.
    """
    if not models_dir:
        return TriageHead(None), None
    checkpoint = f"{models_dir}/{TRIAGE_CHECKPOINT}"
    head = TriageHead(checkpoint if _exists(checkpoint) else None)
    graph = f"{models_dir}/{MODEL_WITH_HIDDEN_STATES}"
    return head, (graph if _exists(graph) else None)


def _exists(path: str) -> bool:
    import os

    return os.path.exists(path)


def triage_for_image(image_path: str, models_dir: str | None) -> dict:
    """Run System 1 triage on one screenshot.

    Falls back to the default session when the feature-exposing graph is
    absent, in which case triage reports no signal rather than guessing.
    """
    head, graph = build_triage(models_dir)
    if graph is None:
        return {"signal": False, "reason": "no graph exposing hidden_states is installed"}

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(graph, options, providers=["CPUExecutionProvider"])
    result = triage_page(image_path, session, head)
    if result.get("signal"):
        result["receipt"] = head.receipt
    return result


def is_triage_calibrated() -> bool:
    """Triage speaks only when a fitted head with a receipt is installed."""
    return is_calibrated("triage")


def describe_triage(triage: dict) -> str:
    """One line for a report. Absence of signal reads as absence of opinion."""
    if not triage.get("signal"):
        return f"system1_triage: no signal ({triage.get('reason', 'unknown')})"
    receipt = triage.get("receipt", {})
    flagged = triage.get("flagged_tiles", [])
    auc = receipt.get("val_auc_has_failing_text", "?")
    return (
        f"system1_triage: {len(flagged)}/{len(triage.get('tiles', []))} tiles worth measuring "
        f"(held-out AUC {auc}, label-free target)"
    )


def triage_to_payload(triage: dict) -> dict:
    """Serializable form for the verdict file, with any numpy types flattened."""
    return json.loads(json.dumps(triage, default=float))
