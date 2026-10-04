# System 1 is a Jev decision model, not a text generator

Written down because it was got wrong repeatedly. Every failure below was real
and cost hours.

## What Jev is

Jev is TypeSafe AI's "System One" model. It produces fast, typed decisions.
You describe the decision in plain words at request time, with your own
labels. It returns a probability and a confidence per answer.

The defining property, from the OpenJev README (an independent
implementation, `razorback16/openjev`):

> It reads the answers directly from the model's probabilities and parses no
> text, so an answer cannot go off-schema.

## The mechanism

Build a canvas where only the answer slots are masked, one token per question:

```
canvas in                 one read-only pass         answer out
  q1: [?]        ──►      P(yes) 0.001        ──►    noul  0.001
  q2: [?]                 P(A) 0.000                 choice "billing"
                          P(B) 0.999                 confidence 0.997
                          P(C) 0.000
  q3: [?]                 P(0) 0.000                 score 1.00
                          P(1) 0.996
                          P(2) 0.004
```

- One label per token: `yes`/`no` for a noul, `A`/`B`/`C` for a choice,
  `0`/`1`/`2` for a score.
- The model never writes into those slots. A single read-only forward pass
  yields the distribution, **and that distribution is the answer**.
- `confidence = 1 − H(p)/ln K`. It comes from the distribution, never from a
  number the model reports about itself.
- Three question types: `noul` (yes/no → `P(yes)`), `choice` (→ choice,
  probabilities, confidence), `score` (ordinal levels → `Σ i·pᵢ`).
- Up to 8 images per request, about 280 input tokens each.
- Verdict (another System One model) adds an "insufficient evidence" option to
  every question. **That is mesen's `unknown`.**

Related implementations, all reading label logits rather than generating text:

- **JevK5**: Qwen3.5-4B plus a distilled LoRA, options lettered A–P, answer is a
  softmax over those letters' next-token logits under one calibration
  temperature (1.532, from its `jevk5_config.json`).
- **CLM**: two small heads over a frozen Qwen3-8B; answer is the softmax over
  the scaled cosine of each option embedding with the state.
- **Laya / Verdict**: bidirectional encoders with classification heads.

## What mesen has instead, and why it fails

`mesen_jev_vlm.onnx` is ViT-B/16 plus linear heads at PyTorch default
initialization (uniform weights, excess kurtosis −1.17, LayerNorm gain still
1.0000) sitting on a genuinely pretrained backbone. It emits confident
nonsense: `visual_integrity=no` at 0.999 on every audited page.

Three mistakes were made while trying to fix it, all of them repeating the same
error:

1. **Asking a text model to generate the answer and regexing it.** 400 pages
   produced 0 parseable answers. Fixed three times — by disabling thinking
   mode, by prefilling `$\boxed{`, by shortening prompts. Every fix treated the
   symptom. Jev's entire reason for existing is that you never parse text.
2. **Choosing a target that cannot be learned.** UICrit's human ratings have a
   leave-one-out human-vs-human Spearman of 0.033 on the same screenshots, so
   no model can beat chance on them either.
3. **Giving System 1 the same job as System 2.** The triage head predicted
   WCAG contrast failures, which is what System 2 measures. On yana it
   saturated at ~1.0 on seven of eight pages and scored Spearman −0.024 against
   System 2's violation count, so `evidence_consistency` was a number
   confirming itself.

## What System 1 must be

```
screenshot + question, options lettered, `unknown` among them
  → one forward pass of a vision-language model
  → read the logits of only the option-letter tokens
  → softmax = per-option probability;  1 − H(p)/ln K = confidence
  → several questions scored in the same pass, no generation
```

System 2 stays as it is: measured pixels. The point of two systems is that
they disagree sometimes. System 1 must be able to see what System 2 cannot —
hierarchy, clarity, coherence, whether the page reads as designed. If both
measure the same thing, there is no cross-check, only an echo.

## Hardware

The image-capable reference implementation is DiffusionGemma 26B-A4B NVFP4
(~18 GB) through vLLM; it needs 24 GB. Available hardware:

- `a100` (`frankiris-a100`, ProxyJump `desktop-cudbf36-1`): 4 × A100-SXM4-40GB,
  1.1 TB disk. **Usually occupied by other tenants** — check
  `nvidia-smi --query-gpu=memory.free --format=csv,noheader` before planning
  around 40 GB.
- `desktop-cudbf36-1`: RTX 4080 16 GB. Not enough for the 26B reference model,
  enough for a small vision LM read the same way.

A vision LM of 2–3 B fits the 4080 and uses the identical readout, so the
architecture is not blocked by the reference model's size. What matters is the
label-logit readout, the entropy confidence, and a calibration temperature.

## Status

- System 2: measured, fixed, running on production. 184 real violations found
  on 8 viewports before the DPI and criterion fixes.
- System 1: **does not exist yet in any usable form.** Until it does, the
  shipped judge abstains, and that abstention is the correct behaviour.