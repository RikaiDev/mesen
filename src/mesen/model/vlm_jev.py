"""
VlmJevModel: Unified multimodal backbone + Open-Jev typed decision heads.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .heads import ChoiceHead, ScoreHead


class VlmJevModel(nn.Module):
    """
    Multimodal UI Decision Model.
    Fuses vision tokens from multiple viewport screenshots with witness state tokens,
    then evaluates all 6 atomic judge dimensions in a single forward pass.
    """

    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen3.5-2B-Base",
        hidden_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base_model_name = base_model_name_or_path
        self.hidden_dim = hidden_dim

        # Backbone: Qwen3.5 base or custom vision-language backbone
        # When exported to ONNX, we use frozen/fine-tuned projection layers
        self.config = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            base_model_name_or_path,
            config=self.config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        # Dimension adaptation layer
        backbone_dim = getattr(self.config, "hidden_size", hidden_dim)
        if backbone_dim != hidden_dim:
            self.proj = nn.Linear(backbone_dim, hidden_dim)
        else:
            self.proj = nn.Identity()

        # The 5 Atomic Choice Heads (yes=0, no=1, unknown=2)
        self.head_primary_action = ChoiceHead(hidden_dim, num_classes=3, dropout=dropout)
        self.head_visual_integrity = ChoiceHead(hidden_dim, num_classes=3, dropout=dropout)
        self.head_responsive_consistency = ChoiceHead(hidden_dim, num_classes=3, dropout=dropout)
        self.head_evidence_consistency = ChoiceHead(hidden_dim, num_classes=3, dropout=dropout)
        self.head_operator_clarity = ChoiceHead(hidden_dim, num_classes=3, dropout=dropout)

        # The Overall Quality Score Head (0..3)
        self.head_overall_quality = ScoreHead(hidden_dim, num_scores=4, dropout=dropout)

    def extract_features(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Extracts pooled latent representation across vision and text inputs.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Use last hidden state of the sequence (or mean pooling)
        last_hidden = outputs.last_hidden_state  # [B, SeqLen, Dim]
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
            sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = last_hidden.mean(dim=1)

        return self.proj(pooled)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        latent_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Single forward pass returning logits for all 6 dimensions.
        """
        if latent_features is None:
            features = self.extract_features(input_ids, attention_mask, pixel_values)
        else:
            features = latent_features

        return {
            "primary_action_reachable": self.head_primary_action(features),
            "visual_integrity": self.head_visual_integrity(features),
            "responsive_consistency": self.head_responsive_consistency(features),
            "evidence_consistency": self.head_evidence_consistency(features),
            "operator_clarity": self.head_operator_clarity(features),
            "overall_quality": self.head_overall_quality(features),
        }
