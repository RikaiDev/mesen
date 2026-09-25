"""
Unit & Integration Verification Suite for Mesen Jev-VLM.
Tests pure ONNX graph integrity, latency, multimodal inference, and evidence grounding.
Run with: python3 -m unittest tests/test_jev_vlm.py
"""

import os
import time
import unittest

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from mesen.engine.jev_vlm_engine import JevVlmEngine
from mesen.schema import ConsultantReport, ContextSpec, JudgeAnswers


class TestJevVlmEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        cls.models_dir = os.path.join(base_dir, "models", "onnx")
        cls.vlm_model_path = os.path.join(cls.models_dir, "mesen_jev_vlm.onnx")

        # Fallback path if running in remote dev environment
        if not os.path.exists(cls.vlm_model_path):
            cls.vlm_model_path = (
                "/home/gloomcheng/Workspace/RikaiDev/mesen/models/onnx/mesen_jev_vlm.onnx"
            )
            cls.models_dir = "/home/gloomcheng/Workspace/RikaiDev/mesen/models/onnx"

        cls.engine = JevVlmEngine(models_dir=cls.models_dir, default_dpi=440)
        cls.real_pixel_image = "/tmp/fansee-pixel10-ui-review.png"

        # Create a synthetic test image for positive baseline
        cls.clean_test_image = "/tmp/test_clean_ui.png"
        img = Image.new("RGB", (1920, 1080), color=(250, 250, 250))
        draw = ImageDraw.Draw(img)
        # Clean high-contrast title and button
        draw.rectangle([700, 400, 1220, 520], fill=(20, 20, 20))
        draw.text((750, 440), "Confirm Action", fill=(255, 255, 255))
        img.save(cls.clean_test_image)

    def test_01_onnx_graph_contract(self):
        """Verify the ONNX model inputs, outputs, and tensor shapes."""
        self.assertTrue(
            os.path.exists(self.vlm_model_path), f"Missing ONNX model: {self.vlm_model_path}"
        )
        sess = ort.InferenceSession(self.vlm_model_path, providers=["CPUExecutionProvider"])

        # Inputs
        inputs = sess.get_inputs()
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].name, "screenshot")
        self.assertEqual(inputs[0].shape, ["batch", 3, 224, 224])

        # Outputs
        expected_outputs = {
            "logits_primary_action": [1, 3],
            "logits_visual_integrity": [1, 3],
            "logits_responsive_consistency": [1, 3],
            "logits_evidence_consistency": [1, 3],
            "logits_operator_clarity": [1, 3],
            "logits_overall_quality": [1, 4],
            "rule_logits": [1, 17],
            "pred_bboxes": [1, 4],
        }

        actual_outputs = {o.name: o.shape for o in sess.get_outputs()}
        for name, exp_shape in expected_outputs.items():
            self.assertIn(name, actual_outputs, f"Output {name} not found in ONNX graph")
            self.assertEqual(actual_outputs[name], exp_shape, f"Shape mismatch for output {name}")

    def test_02_pure_onnx_inference_stability(self):
        """Verify numerical stability (no NaNs, bounded coordinates, valid logits)."""
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        sess = ort.InferenceSession(self.vlm_model_path, providers=["CPUExecutionProvider"])

        start = time.perf_counter()
        outputs = sess.run(None, {"screenshot": dummy_input})
        latency = (time.perf_counter() - start) * 1000

        self.assertLess(latency, 1500, f"Inference took too long: {latency:.2f}ms")

        # Verify no NaN or Inf
        for val in outputs:
            self.assertFalse(np.isnan(val).any(), "ONNX output contains NaN")
            self.assertFalse(np.isinf(val).any(), "ONNX output contains Inf")

        # Verify pred_bboxes are in [0, 1]
        bboxes = outputs[7]
        self.assertTrue(
            (bboxes >= 0.0).all() and (bboxes <= 1.0).all(), f"BBox out of [0, 1]: {bboxes}"
        )

    def test_03_real_pixel10_inspection(self):
        """Verify real Pixel 10 mobile screenshot evaluation."""
        if not os.path.exists(self.real_pixel_image):
            self.skipTest(f"Real screenshot {self.real_pixel_image} not present")

        context = ContextSpec(
            cohort="general_mobile", modality="mobile_app", interaction_mode="touch"
        )

        start = time.perf_counter()
        report, judge = self.engine.evaluate(self.real_pixel_image, context=context, dpi=440)
        total_time = (time.perf_counter() - start) * 1000

        # Latency budget: single forward pass must stay interactive on CPU.
        self.assertLess(total_time, 10000, f"evaluate took {total_time:.0f}ms")

        # Assert structured types
        self.assertIsInstance(report, ConsultantReport)
        self.assertIsInstance(judge, JudgeAnswers)

        # Assert Neural Model Decisions
        # Model detected flaws in visual integrity and operator clarity
        self.assertEqual(judge.visual_integrity.choice, "no")
        self.assertEqual(judge.operator_clarity.choice, "no")
        self.assertEqual(judge.overall_quality.score, 0)
        self.assertEqual(report.verdict, "rejected")

        # Assert Detected Violations
        rule_ids = [v.rule_id for v in report.violations]
        self.assertIn("accessibility/contrast-ratio-insufficient", rule_ids)
        self.assertIn("accessibility/font-size-insufficient", rule_ids)
        self.assertIn("layout/horizontal-space-desert", rule_ids)
        self.assertIn("cognitive/interaction-affordance-deficit", rule_ids)

        # Assert Bounding Box Geometry Validity
        for v in report.violations:
            ymin, xmin, ymax, xmax = v.bounding_box
            self.assertTrue(0.0 <= ymin <= ymax <= 1.0, f"Invalid bbox Y: {v.bounding_box}")
            self.assertTrue(0.0 <= xmin <= xmax <= 1.0, f"Invalid bbox X: {v.bounding_box}")

    def test_04_synthetic_clean_ui_no_false_contrast_alarms(self):
        """Verify that on clean high-contrast text, contrast violation is not triggered."""
        context = ContextSpec(
            cohort="general_mobile", modality="mobile_app", interaction_mode="touch"
        )
        report, judge = self.engine.evaluate(self.clean_test_image, context=context, dpi=440)

        # High contrast white on black should not have contrast violation
        contrast_violations = [
            v for v in report.violations if v.rule_id == "accessibility/contrast-ratio-insufficient"
        ]
        self.assertEqual(
            len(contrast_violations),
            0,
            "Clean high contrast text incorrectly triggered contrast violation",
        )


if __name__ == "__main__":
    unittest.main()
