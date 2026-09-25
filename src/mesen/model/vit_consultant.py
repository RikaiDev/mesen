"""
Mesen Vision Transformer (ViT) Consultant Model.
End-to-End Multimodal Architecture:
1. Vision Backbone: Pretrained Vision Transformer (ViT-B/16)
2. Tri-Zone Spatial Patch ROI Extractor (14x14 spatial patch grid):
   - Center Content ROI patches (rows 2..11, cols 3..10): Captures central UI elements & optical core
   - Flanking Margin ROI patches (cols 0..2 and cols 11..13): Explicitly captures horizontal viewport margins,
     detecting aspect-ratio wastelands and responsiveness deficits
   - Global CLS Token: Captures holistic image-level semantics
3. Multimodal Fusion:
   - Projects [CLS (768), Center_ROI (768), Flank_Margin_ROI (768)] -> 1536
4. Consultant Heads:
   - 6 Atomic CI/CD Gate Heads
   - 17 Canonical Rule Violation Heads (WCAG 2.1, ISO 9241, Responsive UI)
   - Spatial BBox Regressor
"""

import torch
import torch.nn as nn
import torchvision.models as models

from mesen.model.consultant_model import MesenConsultantModel
from mesen.rules.registry import RULE_DEFINITIONS


class MesenViTConsultantModel(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone_epochs: int = 0,
        num_rules: int = len(RULE_DEFINITIONS),
    ):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.vit = models.vit_b_16(weights=weights)

        # Remove original ImageNet classifier head
        self.vit.heads = nn.Identity()

        # Dimension of ViT-B/16 token representation is 768
        self.vit_dim = 768

        # Spatial patch layout: 224 / 16 = 14 patches per row/col -> 196 total patches (+1 for CLS)
        # 1. Center Content Area (rows 2..11, cols 3..10) -> 10 * 8 = 80 patches
        center_indices = []
        for r in range(2, 12):
            for c in range(3, 11):
                center_indices.append(1 + r * 14 + c)
        self.register_buffer("center_patch_indices", torch.tensor(center_indices, dtype=torch.long))

        # 2. Flanking Margin Areas: Left 3 cols (0..2) + Right 3 cols (11..13) -> 14 * 6 = 84 patches
        flank_indices = []
        for r in range(14):
            for c in [0, 1, 2, 11, 12, 13]:
                flank_indices.append(1 + r * 14 + c)
        self.register_buffer("flank_patch_indices", torch.tensor(flank_indices, dtype=torch.long))

        # Project combined [CLS_token (768), Center_ROI_mean (768), Flank_Margin_mean (768)] -> 1536
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(self.vit_dim * 3, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Consultant Decision & Rule Heads
        self.consultant_heads = MesenConsultantModel(
            hidden_size=1536,
            num_rules=num_rules,
            dropout=0.1,
        )

    def extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts spatial tokens from ViT.
        images: (B, 3, 224, 224)
        Returns: (B, 1536) fused representation with tri-zone spatial awareness
        """
        # Step 1: Preprocess through ViT stem
        x = self.vit._process_input(images)
        n = x.shape[0]

        # Step 2: Add class token & positional embedding
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)  # (B, 197, 768)

        cls_token = x[:, 0]  # (B, 768)

        # Step 3: Tri-zone spatial patch pooling
        # Center content ROI
        center_patches = torch.index_select(x, 1, self.center_patch_indices)  # (B, 80, 768)
        center_roi = center_patches.mean(dim=1)  # (B, 768)

        # Flanking margins ROI (left & right borders)
        flank_patches = torch.index_select(x, 1, self.flank_patch_indices)  # (B, 84, 768)
        flank_roi = flank_patches.mean(dim=1)  # (B, 768)

        # Step 4: Tri-zone fusion: Holistic CLS + Center Core + Flanking Margins
        fused = torch.cat([cls_token, center_roi, flank_roi], dim=-1)  # (B, 2304)
        hidden_states = self.multimodal_fusion(fused)  # (B, 1536)

        return hidden_states

    def forward(
        self,
        images: torch.Tensor,
        atomic_labels: dict[str, torch.Tensor] | None = None,
        rule_targets: torch.Tensor | None = None,
        bbox_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        images: (B, 3, 224, 224) normalized input screenshots
        """
        hidden_states = self.extract_patch_features(images)
        out = self.consultant_heads(
            hidden_states=hidden_states,
            atomic_labels=atomic_labels,
            rule_targets=rule_targets,
            bbox_targets=bbox_targets,
        )
        out["hidden_states"] = hidden_states
        return out
