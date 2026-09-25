# mesen (目線) — On-Prem Typed VLM UI Decision Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Architecture: vlm-jev](https://img.shields.io/badge/Architecture-vlm--jev-emerald.svg)]()
[![Model: Qwen3.5-2B](https://img.shields.io/badge/Backbone-Qwen3.5--2B-orange.svg)]()
[![Export: ONNX](https://img.shields.io/badge/Export-ONNX%20INT8-purple.svg)]()

> **Deterministic, typed, sub-second UX decisions powered by on-prem multimodal backbones.**  
> Moving away from slow, expensive, generative cloud VLM agent prompts to a lean, single-forward-pass typed decision engine.

---

## 1. What is the new mesen (vlm-jev)?

The original `mesen` explored dual-agent VLM simulations (Naive User + UX Expert) calling cloud LLM APIs. While expressive, cloud LLMs are slow (10–30s), expensive ($0.14–$0.42 per run), non-deterministic, and cannot run in private, air-gapped or HIPAA/GDPR-regulated environments.

**The new mesen completely replaces that architecture with `vlm-jev`**:
- **Non-Generative & Typed**: Uses Open-Jev typed decision heads (`Choice`, `Score`, `Noul`) directly predicting probabilities rather than generating long token streams.
- **Multimodal UI Reasoning**: One screenshot per forward pass, fused with structured DOM / accessibility / contract witness state. Run once per breakpoint (375, 768, 1024, 1440px) and compare verdicts.
- **Single Forward Pass**: Measured **~1.6s per screenshot** via ONNX Runtime on Apple Silicon CPU (2 threads). GPU serving targets < 200ms (unverified on this machine).
- **Zero Cloud API Dependencies**: 100% on-premises / edge, zero API fees, privacy-safe (zero PHI leak).
- **Zero Schema Failures**: Outputs fixed-shape probability tensors that map directly to strong types without JSON parsing errors.

---

## 2. Core Architecture

```
[ Multi-Viewport Screenshots (375, 768, 1024, 1440) + Structured DOM State ]
                                   │
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │                 Qwen3.5-2B-Base                      │
        │        (Native Vision Encoder + Multimodal Fusion)     │
        └──────────────────────────┬────────────────────────────┘
                                   │ (Latent Representation)
                                   ▼
        ┌───────────────────────────────────────────────────────┐
        │            Open-Jev Typed Decision Heads             │
        ├───────────────────────────────────────────────────────┤
        │ • Choice Head: primary_action_reachable (yes/no/unk)  │
        │ • Choice Head: visual_integrity         (yes/no/unk)  │
        │ • Choice Head: responsive_consistency   (yes/no/unk)  │
        │ • Choice Head: evidence_consistency     (yes/no/unk)  │
        │ • Choice Head: operator_clarity         (yes/no/unk)  │
        │ • Score Head:  overall_quality          (0 .. 3)      │
        └──────────────────────────┬────────────────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                ▼                                     ▼
     【 GPU Server Serving 】                  【 Local ONNX Runtime 】
     • RTX 4080 (WSL2 / Linux)                • INT8 / FP16 .onnx
     • FastAPI / gRPC Endpoint                • Python click CLI
     • Shared across dev team & CI            • Fully offline on laptops
```

---

## 3. Decision Contract Schema

The model's output strictly adheres to the atomic UI judge contract:

| Dimension | Type | Description |
|---|---|---|
| `primary_action_reachable` | `choice (yes/no/unknown)` | Are next primary actions and controls visible and reachable at all viewports? |
| `visual_integrity` | `choice (yes/no/unknown)` | Are controls free of clipping, overlapping, and obstruction? |
| `responsive_consistency` | `choice (yes/no/unknown)` | Does workflow meaning and layout remain consistent across breakpoints? |
| `evidence_consistency` | `choice (yes/no/unknown)` | Do screenshots agree with deterministic contract, accessibility & geometry facts? |
| `operator_clarity` | `choice (yes/no/unknown)` | Is the next operator action and its consequences clear from the UI? |
| `overall_quality` | `score (0..3)` | 0 = Blocked/Unsafe, 1 = Confusing/Workaround, 2 = Usable/Clear, 3 = Excellent |

---

## 4. Distillation & Training Lineage

To achieve high defect diagnosis accuracy without running huge models in production:
1. **Defect-Reasoning Teacher**: `afx-team/UI-UX` (4B, SOTA 79.63% on UXBench) provides defect reasoning and attention maps on occlusion, clipping, and modal blocks.
2. **Grounding Teacher**: `inclusionAI/UI-Venus-2-9B` provides element coordinate grounding and action-region boundaries.
3. **Student Model**: `Qwen3.5-2B-Base` equipped with Open-Jev classification heads, trained on A100 SXM4 (40GB) with multi-task focal loss and calibration penalties.

---

## 5. Deployment Options

### Option A: Centralized GPU Server (RTX 4080)
Run the fast HTTP daemon on the internal GPU server:
```bash
mesen serve --host 0.0.0.0 --port 8088 --checkpoint /mnt/model-cache/vlm-jev/latest.safetensors
```
Team members and CI runners call:
```bash
mesen judge --remote http://gpu-server:8088 --images v1.jpg --state state.json
```
(`judge` evaluates the first existing `--images` entry; pass one screenshot per invocation.)

### Option B: Local ONNX Runtime (CPU / CoreML)
Export to ONNX and run without any GPU:
```bash
mesen export --checkpoint /path/to/checkpoint --output mesen.onnx --quantize
mesen judge --model mesen.onnx --images v1.jpg --state state.json
```

---

## Development / QC

```bash
uv sync --extra dev   # install ruff + pytest
bash scripts/qc.sh    # ruff check + format check + pytest (CI runs the same)
```

Ruff config lives in `pyproject.toml` (`[tool.ruff]`); pytest in `[tool.pytest.ini_options]`.
Pydantic models stay snake_case — camelCase wire names survive only as field aliases.

---

## 6. License
MIT License.
