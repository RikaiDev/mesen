# BBox calibration note (2026-10-02, yana storefront)

## Method

Ran `JevVlmEngine.judge_system1` on 4 screenshots (fixed PDP viewport
1440x900, listing viewport, pre-fix PDP viewport, fixed PDP full-page
1440x5497) and compared `pred_bboxes`, verdicts and rule-9 probabilities.
Viewed the fixed PDP screenshot to annotate true defects by eye.

## Finding

All four outputs are effectively constant:

- `pred_bboxes` ≈ [0.18, 0.31, 0.30, 0.56] on every image, all pages,
  all builds, all aspect ratios.
- `rule9` probability = 0.002 on every image (neural affordance rule never
  fires; threshold is 0.4).
- verdicts identical (`visual_integrity=no`, `overall_quality=1`) on every
  image, including a layout whose measured column ratio changed 4.97 → 0.92
  between two of the samples.

The predicted box lands on empty/title whitespace (breadcrumb-to-title gap
on the PDP). Annotated true-defect IoU for that box is 0: there is no
visible defect inside it on the fixed page.

## Decision

`pred_bboxes` must not direct System 2 measurement until a retrained head
demonstrates hit rate on human-annotated boxes (IoU > 0.5). Per the agreed
criterion (kill below 0.3), the bbox-directed proposal is rejected on
current weights.

## UICrit result (2026-10-03, RTX 4080)

1000 RICO screens, 10,286 human/both defect boxes, `scripts/bbox_hit_rate_uicrit.py`:

- **hit rate @0.5 = 0.001** (1/1000), 149s total.
- Best hits: 0.55, 0.28, 0.25. The mass of predictions sits at IoU 0.
- The regressor is effectively constant across pages, builds, aspect
  ratios AND datasets (ecommerce + mobile RICO alike). This is not
  domain shift; the localization output carries no signal on any
  tested distribution.
- `analyze_roi_contrast(img, x, y, w, h)` remains the directed interface
  for future retrained weights; nothing may call it with model boxes
  until the hit-rate gate passes.

`analyze_roi_contrast(img, x, y, w, h)` stays available as the directed
interface for that future; nothing calls it with model boxes today.

## Consequence

Dual-mode `overall_quality` is now bounded above by the measured
`consultation.summary_score` (see `jev_vlm_engine.py`): with neural heads
saturated, the only honest score is the one evidence supports.
`tests/test_dual_quality_bound.py` locks the property.
