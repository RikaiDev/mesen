"""
Open-Jev inspired typed decision heads for multimodal UI reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChoiceHead(nn.Module):
    """
    Typed Choice head outputting calibrated probabilities over (yes, no, unknown).
    Index mapping: 0 -> 'yes', 1 -> 'no', 2 -> 'unknown'.
    """

    def __init__(self, hidden_dim: int, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        # Learnable temperature for post-training probability calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, hidden_dim]
        returns logits: [B, num_classes]
        """
        logits = self.mlp(x)
        # Bounded positive temperature to avoid division by zero or negative temp
        temp = torch.clamp(self.temperature, min=0.01, max=10.0)
        return logits / temp


class ScoreHead(nn.Module):
    """
    Typed Score head outputting calibrated probabilities over ordinal scores (0, 1, 2, 3).
    0 = Blocked or unsafe
    1 = Usable only with material confusion or workaround
    2 = Usable and clear
    3 = Excellent and resilient
    """

    def __init__(self, hidden_dim: int, num_scores: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_scores),
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, hidden_dim]
        returns logits: [B, num_scores]
        """
        logits = self.mlp(x)
        temp = torch.clamp(self.temperature, min=0.01, max=10.0)
        return logits / temp


class NoulHead(nn.Module):
    """
    Direct scalar probability head for binary decisions.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns probability in [0, 1]
        """
        return torch.sigmoid(self.mlp(x))
