"""Four typed design-judgment heads for a future independently labeled model.

This is a training architecture only. It is deliberately absent from the serving
and ONNX export paths until blind human/evaluator labels and held-out calibration
prove that its judgments do not merely reward the maker's own design.
"""

import torch
import torch.nn as nn

from mesen.model.heads import ChoiceHead
from mesen.schema import DesignCriterion


class DesignAuditTrainingHeads(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                criterion.value: ChoiceHead(hidden_dim, dropout=dropout)
                for criterion in DesignCriterion
            }
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(features) for name, head in self.heads.items()}
