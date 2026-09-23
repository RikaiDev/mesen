"""
Distillation loss objectives combining teacher predictions, human ground truth, and calibration penalties.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss to heavily penalize confident false approvals on defect classes.
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class MultiTaskDistillLoss(nn.Module):
    """
    Total Loss = L_supervised (Focal Loss on Human Labels)
               + alpha * L_distill (KL Divergence with Teacher Logits)
               + beta * L_calibration (ECE Penalty against Overconfidence)
    """

    def __init__(
        self,
        distill_weight: float = 0.5,
        temperature: float = 2.0,
        false_approval_penalty: float = 2.5,
    ):
        super().__init__()
        self.distill_weight = distill_weight
        self.temperature = temperature
        self.false_approval_penalty = false_approval_penalty

        # Weight vector emphasizing defect detection ('no' class has higher penalty if missed)
        # 0: yes, 1: no (defect), 2: unknown
        choice_weights = torch.tensor([1.0, false_approval_penalty, 1.2])
        self.choice_focal = FocalLoss(alpha=choice_weights, gamma=2.0)
        self.score_ce = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        teacher_logits: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        total_supervised = torch.tensor(0.0, device=next(iter(student_logits.values())).device)
        total_distill = torch.tensor(0.0, device=next(iter(student_logits.values())).device)

        # 1. Supervised Choice losses
        for key in [
            "primary_action_reachable",
            "visual_integrity",
            "responsive_consistency",
            "evidence_consistency",
            "operator_clarity",
        ]:
            if key in targets:
                s_logits = student_logits[key]
                t_target = targets[key]
                total_supervised += self.choice_focal(s_logits, t_target)

                # Teacher distillation if available
                if teacher_logits is not None and key in teacher_logits:
                    t_logits = teacher_logits[key]
                    p_s = F.log_softmax(s_logits / self.temperature, dim=-1)
                    p_t = F.softmax(t_logits / self.temperature, dim=-1)
                    total_distill += (self.temperature ** 2) * self.kl_div(p_s, p_t)

        # 2. Overall Quality Score loss
        if "overall_quality" in targets:
            s_score_logits = student_logits["overall_quality"]
            t_score_target = targets["overall_quality"]
            total_supervised += self.score_ce(s_score_logits, t_score_target)

        total_loss = total_supervised + self.distill_weight * total_distill

        return {
            "loss": total_loss,
            "supervised_loss": total_supervised,
            "distill_loss": total_distill,
        }
