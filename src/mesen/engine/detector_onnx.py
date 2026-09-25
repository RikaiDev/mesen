"""
High-Resolution Text & UI Element Detection + Recognition via Standalone Local ONNX.
Zero external server / GPU requirements. Runs completely offline in ~30ms on CPU.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class DetectedTextItem:
    text: str
    confidence: float
    bbox: List[float]  # [ymin, xmin, ymax, xmax] normalized 0..1
    pixel_bbox: Tuple[int, int, int, int]  # (x, y, w, h)


class OnnxTextDetector:
    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            # Default to repo models/onnx directory
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            models_dir = os.path.join(base_dir, "models", "onnx")

        self.det_model_path = os.path.join(models_dir, "ch_PP-OCRv4_det.onnx")
        self.rec_model_path = os.path.join(models_dir, "ch_PP-OCRv4_rec.onnx")
        self.keys_path = os.path.join(models_dir, "ppocr_keys_v1.txt")

        # Fallback to internal path if not present
        if not os.path.exists(self.det_model_path):
            alt_path = "/home/gloomcheng/Workspace/RikaiDev/mesen/models/onnx"
            if os.path.exists(alt_path):
                self.det_model_path = os.path.join(alt_path, "ch_PP-OCRv4_det.onnx")
                self.rec_model_path = os.path.join(alt_path, "ch_PP-OCRv4_rec.onnx")
                self.keys_path = os.path.join(alt_path, "ppocr_keys_v1.txt")

        # Initialize ONNX runtime sessions
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.det_session = ort.InferenceSession(self.det_model_path, opts, providers=["CPUExecutionProvider"])
        self.rec_session = ort.InferenceSession(self.rec_model_path, opts, providers=["CPUExecutionProvider"])

        # Load dictionary
        with open(self.keys_path, "r", encoding="utf-8") as f:
            keys = [line.strip("\r\n") for line in f.readlines()]
        self.char_list = ["blank"] + keys + [" "]

    def detect_and_recognize(self, img_bgr: np.ndarray) -> List[DetectedTextItem]:
        h, w, _ = img_bgr.shape

        # 1. Preprocess for DBNet Det
        # Resize maintaining aspect ratio so max dim is ~960 and divisible by 32
        scale = 960.0 / max(h, w)
        target_w = int(round(w * scale / 32) * 32)
        target_h = int(round(h * scale / 32) * 32)
        resized = cv2.resize(img_bgr, (target_w, target_h))

        blob = resized.astype(np.float32) / 255.0
        blob = (blob - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        # Run Detection
        pred_map = self.det_session.run(None, {"x": blob})[0][0, 0]

        # Binarize and extract contours
        mask = (pred_map > 0.3).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale_x = w / float(target_w)
        scale_y = h / float(target_h)

        boxes: List[Tuple[int, int, int, int]] = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)
            orig_w = int(bw * scale_x)
            orig_h = int(bh * scale_y)
            if orig_w > 40 and orig_h >= 10:
                boxes.append((orig_x, orig_y, orig_w, orig_h))

        # Sort top-to-bottom
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

        items: List[DetectedTextItem] = []
        for bx, by, bw, bh in boxes:
            pad = 2
            crop = img_bgr[max(0, by - pad) : min(h, by + bh + pad), max(0, bx - pad) : min(w, bx + bw + pad)]
            ch, cw, _ = crop.shape
            if ch < 4 or cw < 4:
                continue

            target_rec_h = 48
            target_rec_w = max(16, int(round(cw * (target_rec_h / float(ch)))))
            resized_crop = cv2.resize(crop, (target_rec_w, target_rec_h))

            rec_blob = resized_crop.astype(np.float32) / 255.0
            rec_blob = (rec_blob - 0.5) / 0.5
            rec_blob = rec_blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

            preds = self.rec_session.run(None, {"x": rec_blob})[0]
            pred_indices = np.argmax(preds[0], axis=-1)
            confs = np.max(preds[0], axis=-1)

            decoded_chars = []
            char_confs = []
            last_idx = 0
            for idx, c_idx in enumerate(pred_indices):
                if c_idx != 0 and c_idx != last_idx:
                    if c_idx < len(self.char_list):
                        decoded_chars.append(self.char_list[c_idx])
                        char_confs.append(confs[idx])
                last_idx = c_idx

            text = "".join(decoded_chars).strip()
            conf = float(np.mean(char_confs)) if char_confs else 0.0

            norm_box = [
                round(by / float(h), 4),
                round(bx / float(w), 4),
                round((by + bh) / float(h), 4),
                round((bx + bw) / float(w), 4),
            ]

            items.append(
                DetectedTextItem(
                    text=text,
                    confidence=round(conf, 2),
                    bbox=norm_box,
                    pixel_bbox=(bx, by, bw, bh),
                )
            )

        return items
