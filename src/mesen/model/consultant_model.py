"""
Mesen Consultant Multi-Task PyTorch Model Architecture.
Equipped with:
1. Atomic Heuristic Heads (5 choices + 1 score) for CI/CD gates
2. Rule Violation Multi-Label Head (grounded in standardized Rule Registry)
3. Bounding Box Spatial Grounding Head (locating violation coordinates)
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from mesen.rules.registry import RULE_DEFINITIONS, RULE_ID_LIST


class MesenConsultantModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1536,
        num_rules: int = len(RULE_DEFINITIONS),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_rules = num_rules

        # Context & Multimodal projection layer
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 1. Atomic CI/CD Gate Heads (3-way: yes=0, no=1, unknown=2)
        self.atomic_heads = nn.ModuleDict({
            "primary_action_reachable": nn.Linear(hidden_size, 3),
            "visual_integrity": nn.Linear(hidden_size, 3),
            "responsive_consistency": nn.Linear(hidden_size, 3),
            "evidence_consistency": nn.Linear(hidden_size, 3),
            "operator_clarity": nn.Linear(hidden_size, 3),
            "overall_quality": nn.Linear(hidden_size, 4),  # 4 scores: 0, 1, 2, 3
        })

        # 2. Rule Violation Multi-label Head (13 standard rules)
        self.rule_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_rules),
        )

        # 3. Spatial Grounding Head: Bounding Box Regressor [ymin, xmin, ymax, xmax] in [0, 1]
        self.bbox_regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Sigmoid(),  # Bound coordinates in [0, 1]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        atomic_labels: Optional[Dict[str, torch.Tensor]] = None,
        rule_targets: Optional[torch.Tensor] = None,
        bbox_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        hidden_states: (batch_size, hidden_size) pooled embedding of multimodal state
        """
        feats = self.feature_proj(hidden_states)

        # Forward atomic heads
        atomic_logits = {
            name: head(feats) for name, head in self.atomic_heads.items()
        }

        # Forward rule multi-label head
        rule_logits = self.rule_classifier(feats)

        # Forward bounding box regressor
        pred_bboxes = self.bbox_regressor(feats)

        output = {
            "atomic_logits": atomic_logits,
            "rule_logits": rule_logits,
            "pred_bboxes": pred_bboxes,
        }

        # Compute multi-task losses if targets provided
        if atomic_labels is not None and rule_targets is not None:
            total_loss = torch.tensor(0.0, device=feats.device)

            # Atomic cross-entropy losses
            for head_name, targets in atomic_labels.items():
                loss = F.cross_entropy(atomic_logits[head_name], targets)
                total_loss = total_loss + loss

            # Multi-label BCE loss with positive weighting
            pos_weight = torch.tensor([3.0] * self.num_rules, device=feats.device)
            rule_loss = F.binary_cross_entropy_with_logits(
                rule_logits, rule_targets.float(), pos_weight=pos_weight
            )
            total_loss = total_loss + 2.0 * rule_loss

            # Bounding box regression loss (only for samples with violations)
            if bbox_targets is not None:
                has_violation = (rule_targets.sum(dim=-1) > 0).unsqueeze(-1)
                if has_violation.any():
                    bbox_loss = F.smooth_l1_loss(
                        pred_bboxes * has_violation,
                        bbox_targets * has_violation,
                        reduction="sum"
                    ) / (has_violation.sum() * 4 + 1e-6)
                    total_loss = total_loss + 1.5 * bbox_loss

            output["loss"] = total_loss

        return output
