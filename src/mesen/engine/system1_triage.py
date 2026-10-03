"""System 1: nine-tile triage against measured targets.

The shipped judge resized a whole page to one 224px frame and answered from
heads that were never fit to anything. Both are wrong. Measured over 8
production viewports, a single squashed frame reaches R² 0.22 against real
WCAG contrast while nine tiles reach R² 0.40, and no head may speak without a
calibration receipt.

So System 1 does the one job its representation can do honestly: cut the page
into nine regions, score each for "does this tile contain text that fails
4.5:1", and name the tiles worth measuring. System 2 still measures every
element properly; this is a cheap pointer, not a verdict, and it never
overrides a measurement.
"""

import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

GRID = 3
TILE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# A tile is only worth sending to System 2 if the triage head is meaningfully
# above chance on it. Below this the honest answer is "no signal", not "clean".
TRIAGE_REPORT_MIN = 0.60


@dataclass(frozen=True)
class TileRegion:
    """One tile in page-relative coordinates, so a finding can be located."""

    index: int
    row: int
    col: int
    box: tuple[int, int, int, int]  # (x0, y0, x1, y1) in screenshot pixels


def page_regions(width: int, height: int, grid: int = GRID) -> list[TileRegion]:
    """Nine equal regions, top-left first, in screenshot pixel coordinates."""
    regions = []
    for row in range(grid):
        for col in range(grid):
            regions.append(
                TileRegion(
                    index=row * grid + col,
                    row=row,
                    col=col,
                    box=(
                        col * width // grid,
                        row * height // grid,
                        (col + 1) * width // grid,
                        (row + 1) * height // grid,
                    ),
                )
            )
    return regions


def normalized_box(box: tuple[int, int, int, int], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = box
    return [
        round(y0 / height, 4),
        round(x0 / width, 4),
        round(y1 / height, 4),
        round(x1 / width, 4),
    ]


def tile_batch(image_path: str, grid: int = GRID):
    """Nine tile crops as one normalized batch, plus the page size.

    Tiles are cropped at native resolution and only then resized, so a glyph
    stays legible instead of being averaged away by a page-wide downscale.
    """
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        width, height = image.size
        crops = []
        for region in page_regions(width, height, grid):
            crops.append(image.crop(region.box).resize((TILE, TILE), Image.LANCZOS))

    batch = np.stack([np.asarray(crop, dtype=np.float32) / 255.0 for crop in crops])
    batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
    return batch.transpose(0, 3, 1, 2), (width, height)


class TriageHead:
    """The fitted per-tile head, loaded from its checkpoint and receipt.

    Loads lazily and reports its own provenance, so a caller can tell a
    calibrated triage from a missing file rather than silently scoring zeros.
    """

    def __init__(self, checkpoint_path: str | None = None):
        self.checkpoint_path = checkpoint_path
        self._model = None
        self.receipt: dict = {}

    @property
    def available(self) -> bool:
        return bool(self.checkpoint_path) and os.path.exists(self.checkpoint_path)

    def load(self):
        if self._model is not None:
            return self._model
        if not self.available:
            return None
        import torch
        from torch import nn

        blob = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        trunk = nn.Sequential(nn.Linear(blob["input_dim"], 512), nn.LayerNorm(512), nn.GELU())
        trunk.load_state_dict(blob["trunk"])
        trunk.eval()
        fail_head = nn.Linear(512, 2)
        fail_head.load_state_dict(blob["fail_head"])
        fail_head.eval()
        contrast_head = nn.Linear(512, 1)
        contrast_head.load_state_dict(blob["contrast_head"])
        contrast_head.eval()
        receipt_path = os.path.join(os.path.dirname(self.checkpoint_path), "triage_receipt.json")
        if os.path.exists(receipt_path):
            with open(receipt_path, encoding="utf-8") as handle:
                self.receipt = json.load(handle)
        self._model = (trunk, fail_head, contrast_head, blob["feature_mean"], blob["feature_std"])
        return self._model

    def score(self, features: np.ndarray):
        """Per-tile (probability of failing text, predicted mean contrast)."""
        import torch

        loaded = self.load()
        if loaded is None:
            return None, None
        trunk, fail_head, contrast_head, mean, std = loaded
        with torch.no_grad():
            tensor = torch.from_numpy(features.astype(np.float32))
            tensor = (tensor - mean) / std
            hidden = trunk(tensor)
            prob = fail_head(hidden).softmax(1)[:, 1].numpy()
            contrast = contrast_head(hidden).squeeze(-1).exp().numpy()
        return prob, contrast


def triage_page(
    image_path: str,
    session,
    head: TriageHead,
    grid: int = GRID,
) -> dict:
    """Score every tile of a page.

    Returns per-tile probabilities, the predicted contrast, and the tiles worth
    measuring. `signal` is False when no head is loaded, which the caller must
    treat as "no opinion" rather than "no problem".
    """
    batch, (width, height) = tile_batch(image_path, grid)
    regions = page_regions(width, height, grid)
    outputs = session.run(None, {"screenshot": batch})
    names = [o.name for o in session.get_outputs()]
    features_name = next((n for n in names if n == "hidden_states"), None)
    if features_name is None:
        return {"signal": False, "reason": "graph does not expose hidden_states", "tiles": []}

    out_map = dict(zip(names, outputs, strict=True))
    # The exporter may keep or drop the trailing singleton; accept either rather
    # than pinning one shape and failing on the other.
    features = np.asarray(out_map[features_name])
    if features.ndim == 3:
        features = features[:, 0, :]
    if features.shape[0] != len(regions):
        return {
            "signal": False,
            "reason": f"graph returned {features.shape[0]} features for {len(regions)} tiles",
            "tiles": [],
        }
    prob, contrast = head.score(features)
    if prob is None:
        return {
            "signal": False,
            "reason": "triage head checkpoint not found; no trained head may speak",
            "tiles": [],
        }

    tiles = []
    for region, p, c in zip(regions, prob, contrast, strict=True):
        tiles.append(
            {
                "index": region.index,
                "row": region.row,
                "col": region.col,
                "p_failing_text": round(float(p), 4),
                "predicted_mean_contrast": round(float(c), 3),
                "box": list(region.box),
                "normalized_box": normalized_box(region.box, width, height),
                "worth_measuring": bool(p >= TRIAGE_REPORT_MIN),
            }
        )
    flagged = [t["index"] for t in tiles if t["worth_measuring"]]
    return {
        "signal": True,
        "page": {"width": width, "height": height, "grid": grid},
        "tiles": tiles,
        "flagged_tiles": flagged,
        "max_p_failing_text": round(float(max(t["p_failing_text"] for t in tiles)), 4),
        "mean_predicted_contrast": round(
            float(np.mean([t["predicted_mean_contrast"] for t in tiles])), 3
        ),
        "receipt": head.receipt,
        "role": (
            "Cheap pointer only. Every finding here is a hypothesis about where "
            "to measure; System 2's pixel measurements are the verdict."
        ),
    }
