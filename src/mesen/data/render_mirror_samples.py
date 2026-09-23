"""
Renders real Ground-Truth PNG screenshots for the-mirror (Ambient Half-Mirror) domain.
Produces clean mirror states, center-intrusion defects, low-optical-luminance defects, and touch violations.
"""

import json
import os
from typing import Dict, List
from mesen.data.renderer import render_html_viewports


MIRROR_CLEAN_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body, html {
    margin: 0; padding: 0; width: 100vw; height: 100vh;
    background-color: #0b0e14;
    font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Microsoft JhengHei", sans-serif;
    overflow: hidden;
    color: #f1f5f9;
  }
  .mirror-frame {
    position: relative; width: 100%; height: 100%;
    display: flex; flex-direction: column; justify-content: space-between;
  }
  .top-bar {
    padding: 32px 48px; display: flex; justify-content: space-between; align-items: center;
  }
  .brand-heritage {
    font-size: 28px; font-weight: 700; color: #dc2626; letter-spacing: 2px;
  }
  .vignette-flower {
    position: absolute; width: 96px; height: 96px; border-radius: 50%;
    background: radial-gradient(circle, rgba(239,68,68,0.85) 0%, rgba(220,38,38,0.2) 70%, transparent 100%);
    box-shadow: 0 0 32px rgba(239,68,68,0.4);
  }
  .flower-tl { top: 24px; left: 24px; }
  .flower-tr { top: 24px; right: 24px; }
  .flower-bl { bottom: 24px; left: 24px; }
  .flower-br { bottom: 24px; right: 24px; }
  
  /* Ambient Side Orbs */
  .ambient-orb {
    position: absolute; top: 50%; width: 64px; height: 64px; border-radius: 50%;
    transform: translateY(-50%);
    background: radial-gradient(circle, rgba(245,158,11,0.9) 0%, rgba(217,119,6,0.3) 70%, transparent 100%);
    box-shadow: 0 0 28px rgba(245,158,11,0.5);
  }
  .orb-left { left: 40px; }
  .orb-right { right: 40px; }

  /* 100% CLEAR CENTER REFLECTION ZONE */
  .center-clearance-roi {
    position: absolute; top: 18%; left: 20%; width: 60%; height: 64%;
    border: 2px dashed rgba(255,255,255,0.04);
    pointer-events: none;
  }
  .bottom-guide {
    text-align: center; padding-bottom: 40px; font-size: 24px; color: #e2e8f0; font-weight: 500;
  }
</style>
</head>
<body>
<div class="mirror-frame">
  <div class="top-bar">
    <div class="brand-heritage">FANSEE 智慧魔鏡</div>
  </div>
  <div class="flower-tl vignette-flower"></div>
  <div class="flower-tr vignette-flower"></div>
  <div class="flower-bl vignette-flower"></div>
  <div class="flower-br vignette-flower"></div>
  <div class="orb-left ambient-orb"></div>
  <div class="orb-right ambient-orb"></div>
  <div class="center-clearance-roi"></div>
  <div class="bottom-guide">請自然注視鏡面，為您進行量測</div>
</div>
</body>
</html>
"""

MIRROR_CENTER_INTRUSION_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body, html {
    margin: 0; padding: 0; width: 100vw; height: 100vh;
    background-color: #0b0e14;
    font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", sans-serif;
    overflow: hidden;
  }
  /* FATAL DEFECT: Giant white dialog occupying the central 60% face ROI */
  .intrusion-modal {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    width: 520px; height: 380px; background-color: #ffffff;
    border-radius: 16px; padding: 36px; box-shadow: 0 20px 40px rgba(0,0,0,0.8);
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    text-align: center; z-index: 999;
  }
  .modal-title { font-size: 32px; font-weight: 700; color: #1e293b; margin-bottom: 20px; }
  .modal-desc { font-size: 18px; color: #475569; margin-bottom: 30px; }
  .modal-btn {
    padding: 14px 40px; background-color: #2563eb; color: #fff; font-size: 20px;
    border-radius: 8px; border: none; font-weight: 600;
  }
</style>
</head>
<body>
<div class="intrusion-modal">
  <div class="modal-title">量測準備就緒</div>
  <div class="modal-desc">請長者靠近鏡面並點選下方按鈕開始進行心率與血壓量測</div>
  <button class="modal-btn">立即開始</button>
</div>
</body>
</html>
"""

MIRROR_LOW_LUMINANCE_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body, html {
    margin: 0; padding: 0; width: 100vw; height: 100vh;
    background-color: #050505;
    font-family: -apple-system, sans-serif;
    overflow: hidden;
  }
  /* CRITICAL DEFECT: Low luminance dark gray text behind 50% transmittance mirror */
  .invisible-instructions {
    position: absolute; bottom: 30px; width: 100%; text-align: center;
    color: #262626; font-size: 14px;
  }
</style>
</head>
<body>
  <div class="invisible-instructions">
    此處文字對比度極低 (1.2:1)，在半面鏡與環境反射下完全不可見
  </div>
</body>
</html>
"""


def render_all_mirror_samples(output_dir: str = "data/synthetic/screenshots") -> List[Dict]:
    os.makedirs(output_dir, exist_ok=True)
    generated: List[Dict] = []

    configs = [
        ("mirror_clean", MIRROR_CLEAN_HTML, "clean", {"primary_action_reachable": "yes", "visual_integrity": "yes", "responsive_consistency": "yes", "evidence_consistency": "yes", "operator_clarity": "yes", "overall_quality": 3}),
        ("mirror_center_intrusion", MIRROR_CENTER_INTRUSION_HTML, "mirror_center_obstruction", {"primary_action_reachable": "yes", "visual_integrity": "no", "responsive_consistency": "yes", "evidence_consistency": "yes", "operator_clarity": "no", "overall_quality": 0}),
        ("mirror_low_luminance", MIRROR_LOW_LUMINANCE_HTML, "low_contrast", {"primary_action_reachable": "yes", "visual_integrity": "yes", "responsive_consistency": "yes", "evidence_consistency": "yes", "operator_clarity": "no", "overall_quality": 1}),
    ]

    for name, html, mutation, labels in configs:
        for var_idx in range(5):
            sample_id = f"{name}_var{var_idx}"
            paths = render_html_viewports(html, output_dir, prefix=sample_id)
            generated.append({
                "id": sample_id,
                "screenshots": paths,
                "mutation_type": mutation,
                "labels": labels,
                "state": {
                    "product": "the-mirror",
                    "route": "/mirror-kiosk",
                    "context": {
                        "cohort": "older_adult_65plus",
                        "modality": "ambient_mirror",
                        "interaction_mode": "zero_touch_vision_voice"
                    }
                }
            })

    print(f"Rendered {len(generated) * 4} mirror screenshots across all viewports into {output_dir}")
    return generated


if __name__ == "__main__":
    samples = render_all_mirror_samples()
    with open("data/synthetic/mirror_samples.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
