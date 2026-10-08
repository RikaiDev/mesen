# Real Clinical Benchmark Fixtures

This directory contains real clinical UI screenshots used for System 1 consultant head calibration and generalization benchmarking.

## Benchmark Splits and Provenance

1. **Held-Out Generalization Evaluation** (evaluated in `tests/test_system1_generalization.py`, never included in training):
   - `hicare_cancer_screening_preview.png`: Desktop preview of cancer screening clinical workflow. Ground truth: clean (`overall_quality >= 2`, `visual_integrity="yes"`).
   - `hi-care-preventive-care.png`: Preventive care modal dialog on clinical workstation. Ground truth: clean (`overall_quality >= 2`, `visual_integrity="yes"`).
   - `hicare_real_tablet.png`: Tablet viewport with horizontal table overflow layout blowout. Ground truth: defect (`responsive_consistency="no"`, `overall_quality <= 1`).

2. **Calibration Train / Validation Splits**:
   - `hi-care-care-plan.png`: Care plan overview (training split).
   - `hi-care-cases-list.png`: Clinical cases list (training split).
   - `hicare_real_tablet_scrolled.png`: Scrolled viewport of tablet overflow (training split).
   - `hi-care-icope-status.png`: ICOPE assessment status view (validation split).

## Reproducibility & Integrity Notes
- Fixtures are self-contained and tracked in git to ensure deterministic CI runs without skipping.
- Training set and generalization test assertions are strictly isolated to prevent dataset contamination.
- Viewport overlap caveat: `hicare_real_tablet_scrolled.png` (train) and `hicare_real_tablet.png` (test) represent the same clinical interface at different scroll offsets. While they test whether the model detects blowout regardless of viewport scroll, an independent responsive defect screen from a different route should be collected to provide complete out-of-domain layout evaluation.
