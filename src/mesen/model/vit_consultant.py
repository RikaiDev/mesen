"""
Mesen Vision Transformer (ViT) Consultant Model.
End-to-End Multimodal Architecture:
1. Vision Backbone: Pretrained Vision Transformer (ViT-B/16)
2. Spatial Patch ROI Extractor: Explicitly captures 14x14 spatial patch grid
   - Center ROI patches (rows 3..10, cols 3..10): Captures central 60% optical reflection clearance
   - Vignette patches (borders): Captures edge presence flowers & ambient illumination
3. Consultant Heads:
   - 6 Atomic CI/CD Gate Heads
   - 13 Canonical Rule Violation Heads (WCAG 2.1, ISO 9241, Ambient Mirror Protocol)
   - Spatial BBox Regressor
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from mesen.model.consultant_model import MesenConsultantModel
from mesen.rules.registry import RULE_DEFINITIONS, RULE_ID_LIST


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

        # Spatial patch layout: 224 / 16 = 14 patches per row/col -> 196 total patches
        # Central 60% corresponds to patch rows 3..10 (indices 3 to 10 inclusive), cols 3..10
        center_indices = []
        for r in range(3, 11):
            for c in range(3, 11):
                # +1 because index 0 is the [CLS] token
                center_indices.append(1 + r * 14 + c)
        self.register_buffer("center_patch_indices", torch.tensor(center_indices, dtype=torch.long))

        # Project combined [CLS_token (768), Center_ROI_mean (768)] -> 1536
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(self.vit_dim * 2, 1536),
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
        Returns: (B, 1536) fused representation
        """
        # Step 1: Preprocess through ViT stem
        x = self.vit._process_input(images)
        n = x.shape[0]

        # Step 2: Add class token & positional embedding
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)  # (B, 197, 768)

        cls_token = x[:, 0]  # (B, 768)

        # Step 3: Spatial Patch ROI Pooling for Optical Center Clearance
        center_patches = torch.index_select(x, 1, self.center_patch_indices)  # (B, 64, 768)
        center_roi_feature = center_patches.mean(dim=1)  # (B, 768)

        # Step 4: Multimodal fusion of global screen context + optical center region
        fused = torch.cat([cls_token, center_roi_feature], dim=-1)  # (B, 1536)
        hidden_states = self.multimodal_fusion(fused)  # (B, 1536)

        return hidden_states

    def forward(
        self,
        images: torch.Tensor,
        atomic_labels: Optional[Dict[str, torch.Tensor]] = None,
        rule_targets: Optional[torch.Tensor] = None,
        bbox_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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
