# mesen System 1 — what the instrument can and cannot see (2026-10-03)

Two independent probes on 1000 RICO screens (RTX 4080). Both use ground truth
that needs no human annotation, so neither result depends on anyone's opinion.

## Probe 1 — can it read a page? (label-free ground truth)

`scripts/probe_page_visibility.py`, 300 screens. Target is what
EvidenceEngine measures on real pixels: mean WCAG contrast per page, and
whether the page contains any text element below 4.5:1.

| view | tiles | Spearman (mean contrast) | R² | AUC (any sub-4.5:1) |
|---|---|---|---|---|
| squash (shipped) | 1 | +0.504 | 0.224 | 0.810 |
| fold (top 9:16) | 1 | +0.504 | 0.224 | 0.810 |
| grid3x3 | 9 | **+0.596** | **0.400** | 0.782 |

**The frozen backbone reads the page.** AUC 0.810 against a baseline of 0.5
for a defect it was never trained to detect, and rank correlation 0.50 with
measured contrast. Seeing the page in nine tiles instead of one squashed
frame raises R² from 0.22 to 0.40.

`squash` and `fold` are identical because every RICO screen is 1080×1920 —
already 9:16, so the top band *is* the whole page. That coincidence is a
property of this corpus, not of the probe.

## Probe 2 — can it learn human taste? (UICrit)

`scripts/train_system1_heads.py train`. 1000 screens, frozen ViT, cached
features, held-out 200.

- val accuracy **0.535** vs **0.600** majority baseline
- val Spearman **0.043**

The label, not the pipeline, explains that. Each screen carries 3
independent ratings. Leave one out and predict it from the others:

- **human-vs-human Spearman = 0.033**
- binned exact agreement = 0.386
- mean within-screen spread = 1.58 points on a 9-point scale

Two people rating the same screenshot agree at chance. No image model can
exceed that, so UICrit cannot calibrate a quality head — and training on it
yields a head that reports a number no human would endorse.

## Root cause of the shipped weights

`mesen_jev_vlm.onnx` consultant heads sit at PyTorch default
initialization: uniform weights (excess kurtosis −1.17, matching
`nn.Linear` init to three decimals), `feature_proj` LayerNorm gain still
1.0000 ± 0.0017, while `vit.encoder.ln.weight` shows the +9.16 excess
kurtosis of a genuinely trained tensor. The ViT is real; the heads were
never fit to anything. They were trained against synthetic mutation labels
whose own labels were the generator's mutation type, and now against a
target humans themselves cannot agree on.

## Decision

System 1 is trainable, but only against targets that are facts:

- **measurable visual properties** — WCAG contrast, font size, text
  density, element count. AUC 0.81 proves the representation carries them.
- **not a subjective 0–3 quality number**, until experts label pages
  themselves and agree with each other more than chance.

`overall_quality` therefore stops being a model prediction. It becomes a
System 2 verdict over measured violations. System 1 keeps the job its
representation can actually do: fast triage of *where* a page has
measurable trouble, at nine tiles instead of one squashed frame.

Gate stays report-only until a human-labeled set exists.
## What was fixed, verified on production

`http://45.32.56.160`, 8 viewports, gate `report-only`:

| | before | after |
|---|---|---|
| violations | 283 | 184 |
| font-size violations (DPI artifact) | 108 | 9 |
| locators that were OCR strings | 175 | 0 |
| criticals citing a wrong string | 40 | 0 |
| icons judged by the 4.5:1 text rule | 3 | 0 |

The 9 remaining font findings are genuine 12sp caption text under the 14sp
mobile floor. Three graphic regions moved to `1.4.11` at 3:1, which is the
criterion that governs them.

## What System 1 does now

It abstains. All five choice heads, the score head, the rule classifier and
the bbox regressor lack a calibration receipt, so each returns `unknown`,
confidence 0.0, naming the receipt it waits for. The gate reads:

```
system1_vi=unknown system1_q=1 system2_verdict=conditional_pass system2_score=1
```

System 2's measurement is the quality answer, unopposed. Three fixes made
that true rather than nominal:

- the dual bound was `min(System1, System2)`, which only means something when
  System 1 scored the page; an untrained head emitting 0 pinned every page to
  0 and discarded the measurement
- the rule classifier was thresholded at 0.4 to emit a critical affordance
  finding — a probability from untrained weights is not evidence
- `evidence_consistency` stays registered as derived, since it is the
  cross-system check and not a neural prediction

## Still missing: a System 1 that is useful, not merely honest

The representation works (AUC 0.810 label-free, R² 0.400 with nine tiles vs
0.224 squashed). What is not built yet is a head trained on those
label-free targets — per-tile contrast and font-size risk — served from a
nine-tile forward pass instead of one squashed 224px frame. That is the
remaining work, and it needs no human labels.

## System 1 v2: calibrated, and validated out of distribution

One head now has a receipt. It predicts, per tile of nine, whether that tile
contains text failing 4.5:1 and what its mean measured contrast is. The
targets are EvidenceEngine measurements on each tile's own pixels, so no human
label is involved.

Training set: 2397 tiles from 385 RICO pages, split by page so no tile of a
validation page appears in training. Held out (593 tiles):

| | triage head | baseline |
|---|---|---|
| per-tile AUC | **0.745** | 0.500 |
| accuracy | **0.675** | 0.528 (majority) |
| Spearman vs measured contrast | **+0.687** | 0 |

The number that matters is the next one. On yana's own production pages, which
this head never saw, scored against System 2's independent pixel
measurements:

| | value |
|---|---|
| tiles compared | 57 (49 measured failing) |
| per-tile AUC | **0.663** |
| precision of a flagged tile | **0.925** (TP 37, FP 3) |
| recall | 0.755 (FN 12) |

AUC drops from 0.745 in-distribution to 0.663 on a different domain, which is
the expected cost. Precision holds at 0.925, so when System 1 says a tile is
worth measuring it is right nine times in ten.

It is a pointer. `system1_triage` is its own field in the verdict payload and
changes nothing System 2 measured; `head_calibration` still denies the score
head, the rule classifier and the bbox regressor a voice.

## Two defects the export caught

- Tracing with batch 1 lets ONNX specialize the ViT's internal token reshape,
  after which the graph rejects every other batch with
  `input_shape_size == requested_shape_size was false`. Tracing at batch 2 and
  verifying batches 1 and 9 before declaring the export good.
- `hidden_states` came back as `[batch, 1536]` from one exporter and
  `[batch, 1, 1536]` from another, so the consumer accepts both rather than
  pinning the shape that happened to be produced first.
