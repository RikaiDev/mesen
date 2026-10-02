# mesen (目線) — UI Evaluation Harness

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Architecture: evidence harness](https://img.shields.io/badge/Architecture-evidence%20harness-emerald.svg)]()
[![Local model: ViT-B/16](https://img.shields.io/badge/Local%20model-ViT--B%2F16-orange.svg)]()
[![Export: ONNX](https://img.shields.io/badge/Export-ONNX-purple.svg)]()

> Capture UI evidence, check measurable defects, and compare model judgments with the evidence. The bundled local judge is a ViT-B/16 ONNX classifier; Qwen and specialist UI/UX teacher paths are research code, not the default CLI inference path.

---

## 1. What does mesen run today?

The local `mesen judge` command loads `models/onnx/mesen_jev_vlm.onnx`. Its neural input is one screenshot resized to 224×224. A separate evidence evaluator consumes validated witness state and can reject a measured defect even when image logits disagree. Document-level horizontal overflow is a hard responsive defect; child containers that scroll horizontally remain informational. One screenshot cannot establish semantic consistency across breakpoints; without a measured document defect, the responsive answer abstains.

The harness contract is: capture screenshots and browser facts, validate the witness, apply measurable checks, then record the model answer and any disagreement. Its usefulness depends on real-browser holdouts and human review of false negatives and confidence, not on the backbone name.

---

## 2. Core Architecture

```
[ Browser screenshots + measured viewport/DOM facts ]
                         │
                         ▼
             [ Validated witness state ]
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      [ ViT-B/16 ONNX ]      [ Evidence evaluator ]
      one 224px image       geometry and rule registry
              └──────────┬──────────┘
                         ▼
          [ Typed answers + consultation ]
```

---

## 3. Decision Contract Schema

The harness returns these typed answers. Measured evidence can override or abstain from an image-only prediction:

| Dimension | Type | Description |
|---|---|---|
| `primary_action_reachable` | `choice (yes/no/unknown)` | Are next primary actions and controls visible and reachable at all viewports? |
| `visual_integrity` | `choice (yes/no/unknown)` | Are controls free of clipping, overlapping, and obstruction? |
| `responsive_consistency` | `choice (yes/no/unknown)` | Does workflow meaning and layout remain consistent across breakpoints? |
| `evidence_consistency` | `choice (yes/no/unknown)` | Do screenshots agree with deterministic contract, accessibility & geometry facts? |
| `operator_clarity` | `choice (yes/no/unknown)` | Is the next operator action and its consequences clear from the UI? |
| `overall_quality` | `score (0..3)` | 0 = Blocked/Unsafe, 1 = Confusing/Workaround, 2 = Usable/Clear, 3 = Excellent |

---

## 4. Research model paths

The repository contains Qwen3.5-2B training/export code and names `afx-team/UI-UX` and `inclusionAI/UI-Venus-2-9B` as candidate teachers. Those names do not establish which checkpoint produced a local judgment. Record the exact checkpoint, training manifest, validation receipt, and inference path before claiming teacher distillation or model performance.

---

## 5. Deployment Options

### Option A: Experimental GPU server
The server code exists, but `mesen judge --remote` is not implemented. Use a separately validated server client before treating this as a release gate:
```bash
mesen serve --host 0.0.0.0 --port 8088 --checkpoint /mnt/model-cache/vlm-jev/latest.safetensors
```
The local CLI rejects `--remote` explicitly.

### Option B: Local ONNX Runtime (CPU)
Use the bundled ONNX checkpoint with a captured witness and screenshot:
```bash
mesen judge --images 375.png --state state.json
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
