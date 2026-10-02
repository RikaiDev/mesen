# Temperature calibration receipt (2026-10-03, RTX 4080)

## Setup

- Data: `data/synthetic/train.json` (72) fit, `val.json` (18) report.
- Method: single-temperature softmax scaling (`engine/temperature.py`),
  grid-searched on train NLL; Brier reported on held-out val.
- Script: `scripts/calibrate_temperature.py`. No retraining, no new deps.

## Result

- Fitted T = 5.834 (train NLL 1.2261).
- Val Brier 0.8346 -> 0.5998 (28% relative improvement).

## Reading

- T >> 1 confirms the heads are systematically overconfident, on
  synthetic data too — not just yana pages. Uniform guessing scores
  Brier 0.75 on 4 classes; the raw heads (0.83) were worse than uniform
  because they were confidently wrong. After scaling (0.60) they beat
  uniform but remain far from good (< 0.3).
- T is NOT wired into the judge path by this change. Applying it alters
  reported confidences for every existing user (portal/kiosk/mirror);
  that needs their validation first. Scaling never flips argmax, so
  decisions are unchanged either way.
- n=18 val is a pilot, not a calibration certificate. Do not quote T=5.83
  as a general constant; refit once the teacher-grade set (ESCI) lands.
