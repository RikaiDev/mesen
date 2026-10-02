"""Score a sealed UI/UX holdout without tuning model or prompt parameters."""

import json
import math
from collections import Counter
from pathlib import Path

from mesen.data.reviewed_manifest import file_sha256, load_reviewed_manifest
from mesen.schema import DesignCriterion

AXES = tuple(item.value for item in DesignCriterion)
CHOICES = ("yes", "no", "unknown")


def _verified_ref(base: Path, ref: dict) -> Path:
    if not isinstance(ref, dict) or not ref.get("path") or not ref.get("sha256"):
        raise ValueError("Model or reviewer reference needs path and sha256")
    path = (base / ref["path"]).resolve()
    if file_sha256(path) != ref["sha256"]:
        raise ValueError(f"Reference hash mismatch: {path}")
    return path


def _validated_prediction(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("choice") not in CHOICES:
        raise ValueError("Prediction needs yes/no/unknown choice")
    probabilities = value.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(CHOICES):
        raise ValueError("Prediction needs all three class probabilities")
    if (
        any(
            type(probabilities[choice]) not in (float, int)
            or not math.isfinite(probabilities[choice])
            or not 0 <= probabilities[choice] <= 1
            for choice in CHOICES
        )
        or abs(sum(probabilities.values()) - 1) > 0.001
    ):
        raise ValueError("Invalid prediction probabilities")
    if value["choice"] != max(CHOICES, key=lambda choice: probabilities[choice]):
        raise ValueError("Prediction choice contradicts probabilities")
    return value


def _macro_f1(truths: list[str], predictions: list[str]) -> float:
    scores = []
    for choice in ("yes", "no"):
        tp = sum(t == choice and p == choice for t, p in zip(truths, predictions))
        fp = sum(t != choice and p == choice for t, p in zip(truths, predictions))
        fn = sum(t == choice and p != choice for t, p in zip(truths, predictions))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(scores) / 2


def _ece(truths: list[str], predictions: list[dict]) -> float:
    if not truths:
        return 1.0
    bins = [[] for _ in range(10)]
    for truth, prediction in zip(truths, predictions):
        confidence = max(prediction["probabilities"].values())
        bin_index = min(9, int(confidence * 10))
        bins[bin_index].append((confidence, int(prediction["choice"] == truth)))
    return sum(
        len(entries)
        / len(truths)
        * abs(
            sum(correct for _, correct in entries) / len(entries)
            - sum(confidence for confidence, _ in entries) / len(entries)
        )
        for entries in bins
        if entries
    )


def _axis_metrics(records: list[dict], predictions: dict[str, dict], axis: str) -> dict:
    known = [
        record
        for record in records
        if record["label"].criteria[DesignCriterion(axis)].choice.value != "unknown"
    ]
    truths = [record["label"].criteria[DesignCriterion(axis)].choice.value for record in known]
    teacher = [predictions[record["id"]]["teacher"][axis] for record in known]
    student = [predictions[record["id"]]["onnx"][axis] for record in known]
    confusion = Counter(
        f"{truth}->{prediction['choice']}" for truth, prediction in zip(truths, student)
    )
    return {
        "known_cases": len(known),
        "truth_counts": dict(Counter(truths)),
        "student_macro_f1": _macro_f1(truths, [item["choice"] for item in student]),
        "student_ece": _ece(truths, student),
        "teacher_accuracy": sum(item["choice"] == truth for item, truth in zip(teacher, truths))
        / len(truths)
        if truths
        else 0.0,
        "student_accuracy": sum(item["choice"] == truth for item, truth in zip(student, truths))
        / len(truths)
        if truths
        else 0.0,
        "confusion": dict(sorted(confusion.items())),
        "abstentions": sum(item["choice"] == "unknown" for item in student),
    }


def evaluate_blind_holdout(
    manifest_path: str | Path, predictions_path: str | Path, contract_path: str | Path
) -> dict:
    """Check every predeclared acceptance metric against verified blind labels."""
    records, holdout_receipt = load_reviewed_manifest(manifest_path)
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    if contract.get("schema") != "mesen.uiux-vlm.acceptance.v1":
        raise ValueError("Unsupported acceptance contract")
    prediction_file = Path(predictions_path).resolve()
    payload = json.loads(prediction_file.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != "mesen.blind-predictions.v1"
        or payload.get("holdout_manifest_sha256") != holdout_receipt["manifest_sha256"]
    ):
        raise ValueError("Predictions are not bound to this sealed holdout")
    checkpoint = _verified_ref(prediction_file.parent, payload.get("student_checkpoint"))
    onnx_model = _verified_ref(prediction_file.parent, payload.get("onnx_model"))
    qualification_refs = payload.get("teacher_qualifications")
    if not isinstance(qualification_refs, dict) or set(qualification_refs) != set(AXES):
        raise ValueError("Each design axis needs a teacher qualification receipt")
    for axis, ref in qualification_refs.items():
        qualification = json.loads(
            _verified_ref(prediction_file.parent, ref).read_text(encoding="utf-8")
        )
        checkpoint_hash = qualification.get("checkpoint_sha256")
        qualification_manifest_hash = qualification.get("evaluation_manifest_sha256")
        reviewed_cases = qualification.get("human_reviewed_cases")
        reviewed_accuracy = qualification.get("human_accuracy")
        if (
            qualification.get("schema") != "mesen.teacher-qualification.v1"
            or qualification.get("axis") != axis
            or qualification.get("verdict") != "pass"
            or not isinstance(checkpoint_hash, str)
            or len(checkpoint_hash) != 64
            or any(digit not in "0123456789abcdef" for digit in checkpoint_hash)
            or not qualification.get("source_model_id")
            or not isinstance(qualification_manifest_hash, str)
            or len(qualification_manifest_hash) != 64
            or any(digit not in "0123456789abcdef" for digit in qualification_manifest_hash)
            or qualification_manifest_hash == holdout_receipt["manifest_sha256"]
            or type(reviewed_cases) is not int
            or reviewed_cases < contract["split_policy"]["minimum_evidence_for_each_axis"]
            or type(reviewed_accuracy) not in (float, int)
            or not math.isfinite(reviewed_accuracy)
            or not 0 <= reviewed_accuracy <= 1
            or not qualification.get("reviewer_id")
            or qualification.get("reviewer_id") == qualification.get("teacher_maker_id")
        ):
            raise ValueError(f"Unqualified teacher for {axis}")

    raw_predictions = payload.get("records")
    if not isinstance(raw_predictions, list) or len(raw_predictions) != len(records):
        raise ValueError("Every holdout case needs exactly one prediction")
    predictions = {}
    latencies = []
    for item in raw_predictions:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or item["id"] in predictions
        ):
            raise ValueError("Missing or duplicate prediction id")
        for model_name in ("teacher", "torch", "onnx"):
            if not isinstance(item.get(model_name), dict) or set(item[model_name]) != set(AXES):
                raise ValueError(f"Incomplete {model_name} design-axis output")
            for axis in AXES:
                _validated_prediction(item[model_name][axis])
        latency = item.get("onnx_cpu_latency_ms")
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError("Missing valid ONNX CPU latency")
        latencies.append(latency)
        predictions[item["id"]] = item
    if set(predictions) != {record["id"] for record in records}:
        raise ValueError("Prediction IDs differ from reviewed holdout IDs")

    split = contract["split_policy"]
    thresholds = contract["acceptance"]
    failures = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    critical = [record for record in records if record["critical_axes"]]
    controls = [record for record in records if record["clean_control"]]
    require(len(records) >= split["minimum_cases"], "holdout case count")
    require(
        len({record["group"][0] for record in records}) >= split["minimum_site_families"],
        "site family count",
    )
    require(len(critical) >= split["minimum_positive_critical_cases"], "critical case count")
    require(len(controls) >= split["minimum_clean_controls"], "clean control count")
    per_axis = {axis: _axis_metrics(records, predictions, axis) for axis in AXES}
    for axis, metrics in per_axis.items():
        require(
            metrics["known_cases"] >= split["minimum_evidence_for_each_axis"],
            f"{axis} labeled count",
        )
        require(
            all(metrics["truth_counts"].get(choice, 0) > 0 for choice in ("yes", "no")),
            f"{axis} class coverage",
        )
        require(
            metrics["student_macro_f1"] >= thresholds["four_axis_macro_f1_minimum_each"],
            f"{axis} macro F1",
        )
        require(metrics["student_ece"] <= thresholds["calibration_ece_maximum_each"], f"{axis} ECE")
        require(
            metrics["student_accuracy"] - metrics["teacher_accuracy"]
            >= thresholds["student_accuracy_minus_teacher_minimum_each"] - 1e-12,
            f"{axis} teacher parity",
        )

    critical_false_negatives = sum(
        predictions[record["id"]]["onnx"][axis]["choice"] != "no"
        for record in critical
        for axis in record["critical_axes"]
    )
    unsupported_accepts = sum(
        predictions[record["id"]]["onnx"][axis]["choice"] == "yes"
        and predictions[record["id"]]["onnx"][axis]["probabilities"]["yes"] >= 0.9
        for record in records
        for axis in AXES
        if record["label"].criteria[DesignCriterion(axis)].choice.value == "unknown"
    )
    require(
        critical_false_negatives <= thresholds["critical_false_negative_count"],
        "critical false negatives",
    )
    require(
        unsupported_accepts <= thresholds["confident_unsupported_accept_count"],
        "unsupported confident accepts",
    )

    total_axes = len(records) * len(AXES)
    parity = (
        sum(
            predictions[record["id"]]["torch"][axis]["choice"]
            == predictions[record["id"]]["onnx"][axis]["choice"]
            for record in records
            for axis in AXES
        )
        / total_axes
    )
    probability_error = sum(
        abs(
            predictions[record["id"]]["torch"][axis]["probabilities"][choice]
            - predictions[record["id"]]["onnx"][axis]["probabilities"][choice]
        )
        for record in records
        for axis in AXES
        for choice in CHOICES
    ) / (total_axes * len(CHOICES))
    p95_latency = sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1]
    model_size_mb = onnx_model.stat().st_size / 1_000_000
    require(parity >= thresholds["onnx_argmax_parity_minimum"], "ONNX argmax parity")
    require(
        probability_error <= thresholds["onnx_probability_mean_absolute_error_maximum"],
        "ONNX probability parity",
    )
    require(model_size_mb <= thresholds["onnx_size_megabytes_maximum"], "ONNX model size")
    require(
        p95_latency <= thresholds["onnx_cpu_p95_latency_milliseconds_maximum"],
        "ONNX CPU p95 latency",
    )
    return {
        "accepted": not failures,
        "failures": failures,
        "holdout_receipt": holdout_receipt,
        "predictions_sha256": file_sha256(prediction_file),
        "contract_sha256": file_sha256(Path(contract_path)),
        "student_checkpoint_sha256": file_sha256(checkpoint),
        "onnx_sha256": file_sha256(onnx_model),
        "per_axis": per_axis,
        "critical_false_negatives": critical_false_negatives,
        "confident_unsupported_accepts": unsupported_accepts,
        "onnx_argmax_parity": parity,
        "onnx_probability_mean_absolute_error": probability_error,
        "onnx_size_megabytes": model_size_mb,
        "onnx_cpu_p95_latency_milliseconds": p95_latency,
    }
