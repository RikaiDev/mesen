"""
Dataset loading and preprocessing for multi-viewport screenshots and witness states.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset
from PIL import Image

CHOICE_TO_IDX = {"yes": 0, "no": 1, "unknown": 2}
IDX_TO_CHOICE = {0: "yes", 1: "no", 2: "unknown"}


class UiEvidenceDataset(Dataset):
    """
    Dataset storing UI capture packets containing:
    1. Multi-viewport screenshots (e.g. mobile 375px, tablet 768px, laptop 1024px, desktop 1440px)
    2. Witness state JSON (DOM contract, accessibility, geometry facts)
    3. Ground truth or teacher pseudo-labels
    """

    def __init__(
        self,
        samples: List[Dict],
        tokenizer=None,
        processor=None,
        max_seq_length: int = 1024,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # 1. Format text prompt / witness state representation
        state = sample.get("state", {})
        state_str = json.dumps(state, ensure_ascii=False)
        text_prompt = (
            f"Judge UI Evidence for route {state.get('route', 'unknown')}.\n"
            f"Contract: {json.dumps(state.get('contract', {}))}\n"
            f"Accessibility: {json.dumps(state.get('accessibilityViolations', []))}\n"
            f"Geometry: {json.dumps(state.get('geometryAnomalies', []))}\n"
        )

        tokens = None
        if self.tokenizer is not None:
            enc = self.tokenizer(
                text_prompt,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
        else:
            input_ids = torch.zeros(self.max_seq_length, dtype=torch.long)
            attention_mask = torch.ones(self.max_seq_length, dtype=torch.long)

        # 2. Extract targets if present
        labels = sample.get("labels", {})
        targets = {}
        for key in [
            "primary_action_reachable",
            "visual_integrity",
            "responsive_consistency",
            "evidence_consistency",
            "operator_clarity",
        ]:
            val = labels.get(key, "unknown")
            targets[key] = torch.tensor(CHOICE_TO_IDX.get(val, 2), dtype=torch.long)

        score_val = labels.get("overall_quality", 2)
        targets["overall_quality"] = torch.tensor(int(score_val), dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets,
            "sample_id": sample.get("id", str(idx)),
        }
