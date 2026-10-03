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