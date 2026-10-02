# Probe receipt: discrimination on synthetic val (2026-10-03, RTX 4080)

8 images (labels 0/1/2), dual mode, desktop_web persona. ONNX ran on CPU;
no GPU lock was needed for this stage.

| sample | label | system1 | system2 |
|---|---|---|---|
| occlusion x3 | 0 | 0 | 0 |
| responsive_break / overflow x3 | 1 | 1 | 1 |
| clean x2 | 2 | 1 | 1 then 3 |

Latency 0.3-0.6s per full dual judge (target: p95 < 2s).

## Reading

- System 1 discriminates 0 vs 1 HERE, on portal-style synthetic pages.
  The all-1 collapse seen on yana ecommerce pages is therefore
  distribution-specific, not total head death.
- Label-2 clean pages get system1=1 (conservative bias); the min-bound
  keeps 1 even where System 2 says 3. Safe direction for a gate
  (never passes bad pages; may underrate good ones).
- 7/8 system1==system2 agreement; the one split is conservative-side.
