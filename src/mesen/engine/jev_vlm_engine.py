"""
Unified Jev-VLM Engine for Mesen.
Executes the true multimodal ONNX Jev-VLM model (ViT Backbone + 6 JEV Heads + Rule Classifier + BBox Regressor),
fused with deterministic evidence grounding.
Runs 100% offline, single-machine ONNX CPU inference. Zero external server / GPU dependency.
"""

import os

import numpy as np
import onnxruntime as ort
from PIL import Image

from mesen.engine.dual_judge import (
    adjudicate_evidence_consistency,
    adjudicate_violation_verdict,
    compute_agreement,
    derive_system_two,
    document_overflow_measurements,
)
from mesen.engine.evidence import DEFAULT_DEVICE_PIXEL_RATIO, EvidenceEngine
from mesen.engine.head_calibration import abstention_reason, is_calibrated
from mesen.engine.jev_evaluator import JevEvaluator
from mesen.schema import (
    ChoiceAnswer,
    ConsultantReport,
    ContextSpec,
    JudgeAnswers,
    ScoreAnswer,
    ViolationItem,
    WitnessState,
)

CHOICE_LABELS = ["yes", "no", "unknown"]


class EvaluationResult(tuple):
    """Backwards-compatible (report, answers) 2-tuple preserving pre-adjudication diagnostic state."""

    report: ConsultantReport
    answers: JudgeAnswers
    agreement: dict[str, bool] | None
    system_two: JudgeAnswers | None

    def __new__(
        cls,
        report: ConsultantReport,
        answers: JudgeAnswers,
        agreement: dict[str, bool] | None = None,
        system_two: JudgeAnswers | None = None,
    ):
        inst = super().__new__(cls, (report, answers))
        inst.report = report
        inst.answers = answers
        inst.agreement = agreement
        inst.system_two = system_two
        return inst


class JevVlmEngine:
    def __init__(
        self,
        models_dir: str | None = None,
        default_dpr: float = DEFAULT_DEVICE_PIXEL_RATIO,
        onnx_model_path: str | None = None,
    ):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            models_dir = os.path.join(base_dir, "models", "onnx")

        # An explicit model file always wins; otherwise resolve the bundled default.
        # The artifact is gitignored; scripts/fetch_models.sh stages it and owns its hash.
        self.vlm_model_path = (
            onnx_model_path if onnx_model_path else os.path.join(models_dir, "mesen_jev_vlm.onnx")
        )
        self.models_dir = models_dir

        # Initialize ONNX inference session
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_mem_pattern = False
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.vlm_model_path, opts, providers=["CPUExecutionProvider"]
        )
        self.evidence_engine = EvidenceEngine(default_dpr=default_dpr, models_dir=models_dir)
        self.system_two = JevEvaluator(
            default_dpr=default_dpr,
            models_dir=models_dir,
            evidence_engine=self.evidence_engine,
        )

    def _preprocess_screenshot(self, image_path: str) -> np.ndarray:
        """Preprocesses screenshot for the Vision Transformer backbone."""
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0
        # ImageNet normalization
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        blob = arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        return blob

    def judge_system1(
        self,
        image_path: str,
        context: ContextSpec | None = None,
    ) -> tuple[JudgeAnswers, list[float], list[float], int]:
        """System 1: single neural forward pass only. No OCR, no rules."""
        if context is None:
            context = ContextSpec(
                cohort="general_mobile", modality="mobile_app", interaction_mode="touch"
            )

        # -------------------------------------------------------------
        # Track 1: True Jev-VLM Multimodal Neural Network Forward Pass
        # -------------------------------------------------------------
        blob = self._preprocess_screenshot(image_path)
        raw_outputs = self.session.run(None, {"screenshot": blob})

        out_map = {o.name: raw_outputs[idx] for idx, o in enumerate(self.session.get_outputs())}

        # Parse 6 Atomic JEV Decision Heads
        def parse_choice(logits: np.ndarray) -> tuple[str, float, dict[str, float]]:
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            pred_idx = int(np.argmax(logits))
            prob_dict = {
                CHOICE_LABELS[i]: round(float(probs[0, i]), 4) for i in range(len(CHOICE_LABELS))
            }
            return CHOICE_LABELS[pred_idx], float(probs[0, pred_idx]), prob_dict

        pa_choice, pa_conf, pa_probs = parse_choice(out_map["logits_primary_action"])
        vi_choice, vi_conf, vi_probs = parse_choice(out_map["logits_visual_integrity"])
        rc_choice, rc_conf, rc_probs = parse_choice(out_map["logits_responsive_consistency"])
        ec_choice, ec_conf, ec_probs = parse_choice(out_map["logits_evidence_consistency"])
        oc_choice, oc_conf, oc_probs = parse_choice(out_map["logits_operator_clarity"])

        # Score Head (0..3)
        score_logits = out_map["logits_overall_quality"]
        score_probs = np.exp(score_logits) / np.sum(np.exp(score_logits), axis=-1, keepdims=True)
        pred_score = int(np.argmax(score_logits))
        score_conf = float(score_probs[0, pred_score])
        score_prob_list = [round(float(p), 4) for p in score_probs[0]]

        # Rule classifier logits
        rule_logits = out_map["rule_logits"][0]
        rule_probs = 1.0 / (1.0 + np.exp(-rule_logits))  # Sigmoid

        # Predicted regression BBox
        pred_bbox = [round(float(c), 4) for c in out_map["pred_bboxes"][0]]

        # A softmax over untrained weights is confident by construction, so a
        # head without a calibration receipt abstains rather than publishing a
        # number indistinguishable from a real judgment. See
        # mesen/engine/head_calibration.py.
        def gated_choice(head: str, choice: str, confidence: float, probs) -> ChoiceAnswer:
            if is_calibrated(head):
                return ChoiceAnswer(
                    choice=choice,
                    confidence=round(confidence, 3),
                    probabilities=probs,
                    reasoning=f"Calibrated JEV head predicted {choice} at {confidence:.3f}.",
                )
            return ChoiceAnswer(
                choice="unknown",
                confidence=0.0,
                reasoning=abstention_reason(head),
            )

        judge_answers = JudgeAnswers(
            primary_action_reachable=gated_choice(
                "primary_action_reachable", pa_choice, pa_conf, pa_probs
            ),
            visual_integrity=gated_choice("visual_integrity", vi_choice, vi_conf, vi_probs),
            responsive_consistency=gated_choice(
                "responsive_consistency", rc_choice, rc_conf, rc_probs
            ),
            evidence_consistency=gated_choice("evidence_consistency", ec_choice, ec_conf, ec_probs),
            operator_clarity=gated_choice("operator_clarity", oc_choice, oc_conf, oc_probs),
            overall_quality=(
                ScoreAnswer(
                    score=pred_score,
                    confidence=round(score_conf, 3),
                    probabilities=score_prob_list,
                    reasoning=(
                        f"Calibrated JEV score head outputted {pred_score}/3 at {score_conf:.3f}."
                    ),
                )
                if is_calibrated("overall_quality")
                else ScoreAnswer(
                    score=pred_score,
                    confidence=0.0,
                    reasoning=(
                        abstention_reason("overall_quality")
                        + " System 2's measured verdict is the quality signal."
                    ),
                )
            ),
        )
        return judge_answers, list(rule_probs), pred_bbox, pred_score

    def evaluate(
        self,
        image_path: str,
        context: ContextSpec | None = None,
        dpr: float | None = None,
        mode: str = "dual",
        witness: WitnessState | None = None,
    ) -> tuple[ConsultantReport, JudgeAnswers]:
        """Judge in fast (System 1), deep (System 2), or dual mode (default)."""
        if context is None:
            context = ContextSpec(
                cohort="general_mobile", modality="mobile_app", interaction_mode="touch"
            )
        if mode not in ("fast", "deep", "dual"):
            raise ValueError(f"Unknown judge mode: {mode}")

        if mode == "deep":
            report = self.system_two.evaluate_screenshot(image_path, context, dpr, witness=witness)
            s2 = derive_system_two(report, witness=witness)
            return EvaluationResult(report, s2, None, s2)

        answers, rule_probs, pred_bbox, pred_score = self.judge_system1(image_path, context)
        violations = self._neural_rule_violations(rule_probs, pred_bbox)

        if mode == "fast":
            report = ConsultantReport(
                context=context,
                violations=violations,
                verdict=self._score_verdict(pred_score),
                summary_score=pred_score,
            )
            return EvaluationResult(report, answers, None, None)

        deep_report = self.system_two.evaluate_screenshot(image_path, context, dpr, witness=witness)
        all_violations = deep_report.violations + violations
        verdict, score = adjudicate_violation_verdict(all_violations)
        report = ConsultantReport(
            context=context,
            violations=all_violations,
            verdict=verdict,
            summary_score=score,
        )
        system_two_answers = derive_system_two(report, witness=witness)
        agreement = compute_agreement(answers, system_two_answers)
        answers.evidence_consistency = adjudicate_evidence_consistency(
            answers, system_two_answers, agreement
        )
        # A claimed quality score is bounded above by measured evidence.
        # System 1 reads a 224px impression; System 2 measured real pixels.
        # When they disagree on quality, the measurement wins: a page cannot
        # be better than its verified defects allow, whatever the impression.
        #
        # That bound only means something if System 1 scored the page. An
        # uncalibrated head carries no evidence, so it must not cap a
        # measurement either: min(0, 3) would let a random head veto a page
        # that System 2 found clean. When System 1 abstains, the measured
        # verdict simply is the quality answer.
        if not is_calibrated("overall_quality"):
            answers.overall_quality = ScoreAnswer(
                score=report.summary_score,
                confidence=1.0,
                reasoning=(
                    f"System 1 has no calibration receipt, so this is System 2's "
                    f"measured verdict: {report.summary_score} "
                    f"({report.verdict}, {len(all_violations)} violation(s))."
                ),
            )
        elif report.summary_score < answers.overall_quality.score:
            answers.overall_quality = ScoreAnswer(
                score=report.summary_score,
                confidence=1.0,
                reasoning=(
                    f"System 1 scored {pred_score}; measured evidence supports "
                    f"at most {report.summary_score} "
                    f"({report.verdict}, {len(all_violations)} violation(s))."
                ),
            )
        if system_two_answers.responsive_consistency.choice == "no":
            measured = "; ".join(document_overflow_measurements(witness))
            answers.responsive_consistency = ChoiceAnswer(
                choice="no", confidence=1.0, reasoning=measured
            )
            answers.visual_integrity = ChoiceAnswer(choice="no", confidence=1.0, reasoning=measured)
            answers.overall_quality = ScoreAnswer(
                score=min(answers.overall_quality.score, report.summary_score),
                confidence=1.0,
                reasoning=f"Measured responsive defect: {measured}",
            )
        elif witness is not None:
            answers.responsive_consistency = ChoiceAnswer(
                choice="unknown",
                confidence=1.0,
                reasoning="A single screenshot cannot establish semantic consistency across viewports.",
            )
        return EvaluationResult(
            report=report,
            answers=answers,
            agreement=agreement,
            system_two=system_two_answers,
        )

    @staticmethod
    def _score_verdict(pred_score: int) -> str:
        if pred_score == 0:
            return "rejected"
        if pred_score in (1, 2):
            return "conditional_pass"
        return "pass"

    @staticmethod
    def _neural_rule_violations(
        rule_probs: list[float], pred_bbox: list[float]
    ) -> list[ViolationItem]:
        """Rule findings from the neural head, only once that head has a receipt."""
        # A rule head at initialization still emits a probability, and a
        # threshold on that probability is not evidence. Rule findings may only
        # come from a head with a receipt.
        if not is_calibrated("rule_classifier"):
            return []

        rule_9_prob = rule_probs[9] if len(rule_probs) > 9 else 0.0
        if rule_9_prob <= 0.4:
            return []
        return [
            ViolationItem(
                rule_id="cognitive/interaction-affordance-deficit",
                severity="critical",
                target_selector="center_interactive_subject",
                bounding_box=pred_bbox,
                measured=f"Affordance Deficit (Model Prob: {rule_9_prob:.2f})",
                threshold="明確可見之互動施力點",
                prescriptive_action=(
                    "神經網路判定畫面中央核心目標缺乏可感知的操作暗示（可信度 "
                    f"{rule_9_prob:.2f}）。請加上持續可見的按壓目標：明確邊界＋文字或通用圖示，"
                    "尺寸不小於 24x24 CSS px（WCAG 2.5.8），並提供 reduced-motion 安全版本（WCAG 2.3.3）。"
                ),
            )
        ]
