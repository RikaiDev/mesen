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
)
from mesen.engine.evidence import EvidenceEngine
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


class JevVlmEngine:
    def __init__(
        self,
        models_dir: str | None = None,
        default_dpi: int = 440,
        onnx_model_path: str | None = None,
    ):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            models_dir = os.path.join(base_dir, "models", "onnx")

        # An explicit model file always wins; otherwise resolve the bundled default.
        self.vlm_model_path = (
            onnx_model_path if onnx_model_path else os.path.join(models_dir, "mesen_jev_vlm.onnx")
        )
        if not os.path.exists(self.vlm_model_path):
            alt_path = "/home/gloomcheng/Workspace/RikaiDev/mesen/models/onnx/mesen_jev_vlm.onnx"
            if os.path.exists(alt_path):
                self.vlm_model_path = alt_path

        # Initialize ONNX inference session
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            self.vlm_model_path, opts, providers=["CPUExecutionProvider"]
        )
        self.evidence_engine = EvidenceEngine(default_dpi=default_dpi, models_dir=models_dir)
        self.system_two = JevEvaluator(
            default_dpi=default_dpi,
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

        # Construct structured JudgeAnswers directly from model logits
        judge_answers = JudgeAnswers(
            primary_action_reachable=ChoiceAnswer(
                choice=pa_choice,
                confidence=round(pa_conf, 3),
                probabilities=pa_probs,
                reasoning=f"Neural JEV Head predicted {pa_choice} with confidence {pa_conf:.3f}.",
            ),
            visual_integrity=ChoiceAnswer(
                choice=vi_choice,
                confidence=round(vi_conf, 3),
                probabilities=vi_probs,
                reasoning=f"Neural JEV Head predicted {vi_choice} with confidence {vi_conf:.3f}.",
            ),
            responsive_consistency=ChoiceAnswer(
                choice=rc_choice,
                confidence=round(rc_conf, 3),
                probabilities=rc_probs,
                reasoning=f"Neural JEV Head predicted {rc_choice} with confidence {rc_conf:.3f}.",
            ),
            evidence_consistency=ChoiceAnswer(
                choice=ec_choice,
                confidence=round(ec_conf, 3),
                probabilities=ec_probs,
                reasoning=f"Neural JEV Head predicted {ec_choice} with confidence {ec_conf:.3f}.",
            ),
            operator_clarity=ChoiceAnswer(
                choice=oc_choice,
                confidence=round(oc_conf, 3),
                probabilities=oc_probs,
                reasoning=f"Neural JEV Head predicted {oc_choice} with confidence {oc_conf:.3f}.",
            ),
            overall_quality=ScoreAnswer(
                score=pred_score,
                confidence=round(score_conf, 3),
                probabilities=score_prob_list,
                reasoning=f"Neural JEV Score Head outputted quality score {pred_score}/3 (confidence {score_conf:.3f}).",
            ),
        )
        return judge_answers, list(rule_probs), pred_bbox, pred_score

    def evaluate(
        self,
        image_path: str,
        context: ContextSpec | None = None,
        dpi: int | None = None,
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
            report = self.system_two.evaluate_screenshot(image_path, context, dpi, witness=witness)
            return report, derive_system_two(report)

        answers, rule_probs, pred_bbox, pred_score = self.judge_system1(image_path, context)
        violations = self._neural_rule_violations(rule_probs, pred_bbox)

        if mode == "fast":
            report = ConsultantReport(
                context=context,
                violations=violations,
                verdict=self._score_verdict(pred_score),
                summary_score=pred_score,
            )
            return report, answers

        deep_report = self.system_two.evaluate_screenshot(image_path, context, dpi, witness=witness)
        all_violations = deep_report.violations + violations
        verdict, score = adjudicate_violation_verdict(all_violations)
        report = ConsultantReport(
            context=context,
            violations=all_violations,
            verdict=verdict,
            summary_score=score,
        )
        system_two_answers = derive_system_two(report)
        agreement = compute_agreement(answers, system_two_answers)
        answers.evidence_consistency = adjudicate_evidence_consistency(
            answers, system_two_answers, agreement
        )
        return report, answers

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
        # Affordance violation from neural head (rule index 9)
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
