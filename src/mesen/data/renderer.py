"""
Multi-viewport screenshot renderer using Headless Chrome.
Captures screenshots across mobile, tablet, laptop, and desktop viewports.
"""

import os
import shutil
import subprocess
import tempfile

VIEWPORTS: dict[str, tuple[int, int]] = {
    "mobile": (375, 812),
    "tablet": (768, 1024),
    "laptop": (1024, 768),
    "desktop": (1440, 900),
}


def find_chrome_binary() -> str | None:
    """Finds available Chrome or Chromium binary across macOS and Linux."""
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def render_html_viewports(
    html_content: str,
    output_dir: str,
    prefix: str = "sample",
) -> list[str]:
    """
    Renders HTML string into 4 viewport screenshot PNGs.
    Returns:
      List of 4 screenshot file paths ordered: [mobile, tablet, laptop, desktop]
    """
    chrome_bin = find_chrome_binary()
    if not chrome_bin:
        raise RuntimeError("Google Chrome or Chromium binary not found for rendering.")

    os.makedirs(output_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(html_content)
        temp_html_path = f.name

    captured_paths: list[str] = []
    try:
        for vp_name in ["mobile", "tablet", "laptop", "desktop"]:
            width, height = VIEWPORTS[vp_name]
            out_png = os.path.join(output_dir, f"{prefix}_{vp_name}_{width}x{height}.png")

            cmd = [
                chrome_bin,
                "--headless",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--window-size={width},{height}",
                f"--screenshot={out_png}",
                temp_html_path,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            captured_paths.append(out_png)
    finally:
        if os.path.exists(temp_html_path):
            os.remove(temp_html_path)

    return captured_paths
