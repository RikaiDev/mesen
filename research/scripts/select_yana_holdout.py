"""Freeze unseen public Yana category routes before model selection."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
SITEMAP = ROOT / "artifacts" / "holdout" / "yana-sitemap.xml"
OUTPUT = ROOT / "artifacts" / "holdout" / "yana-route-selection.json"
SEED = "mesen-uiux-blind-v1-2026-09-28"


def main() -> None:
    """Select ten category routes by fixed hash rank without viewing pages."""
    sitemap_bytes = SITEMAP.read_bytes()
    routes = {
        urlsplit(element.text).path
        for element in ElementTree.fromstring(sitemap_bytes).iter()
        if element.tag.endswith("loc") and element.text
    }
    candidates = sorted(route for route in routes if route.startswith("/categories/"))
    if len(candidates) < 10:
        raise RuntimeError("Sitemap has fewer than ten category routes")
    selected = sorted(
        candidates,
        key=lambda route: hashlib.sha256(f"{SEED}:{route}".encode()).hexdigest(),
    )[:10]
    packet = {
        "schema": "mesen.route-selection.v1",
        "source": "http://45.32.56.160/sitemap.xml",
        "sitemap_sha256": hashlib.sha256(sitemap_bytes).hexdigest(),
        "selection_seed": SEED,
        "selection_method": "ten smallest SHA256(seed:route) among unique /categories/ paths",
        "excluded_from_training": True,
        "routes": selected,
        "selected_utc": datetime.now(timezone.utc).isoformat(),
    }
    if OUTPUT.exists():
        raise FileExistsError("Route selection is already sealed")
    OUTPUT.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"routes": len(selected), "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
