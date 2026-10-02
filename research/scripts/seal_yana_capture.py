"""Hash and inventory the preselected Yana browser captures without reviewing them."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "artifacts" / "holdout" / "yana-route-selection.json"
CAPTURE = ROOT / "artifacts" / "holdout" / "yana-sealed-capture-v2"
OUTPUT = ROOT / "artifacts" / "holdout" / "yana-sealed-manifest.json"
VIEWPORTS = (375, 768, 1024, 1440)


def digest(path: Path) -> str:
    """Hash one existing capture file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Require all forty screenshots and distinct route directories before sealing."""
    if OUTPUT.exists():
        raise FileExistsError("Capture manifest is already sealed")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection.get("schema") != "mesen.route-selection.v1":
        raise ValueError("Invalid route selection")
    records = []
    seen_images = set()
    for route in selection["routes"]:
        route_id = hashlib.sha256(route.encode()).hexdigest()[:12]
        route_dir = CAPTURE / f"route-{route_id}"
        states = list(route_dir.rglob("state.json"))
        if len(states) != 1:
            raise ValueError(f"Expected one witness for route {route_id}; found {len(states)}")
        state_path = states[0]
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("route") != route or state.get("httpFailed"):
            raise ValueError(f"Route or HTTP failure in witness {route_id}")
        facts = state.get("viewportFacts")
        if not isinstance(facts, list) or {fact.get("width") for fact in facts} != set(VIEWPORTS):
            raise ValueError(f"Incomplete viewport facts for {route_id}")
        by_width = {fact["width"]: fact for fact in facts}
        if len(by_width) != len(VIEWPORTS):
            raise ValueError(f"Repeated viewport fact for {route_id}")
        for width in VIEWPORTS:
            fact = by_width[width]
            if not isinstance(fact.get("docScrollWidth"), int):
                raise ValueError(f"No document-width measurement for {route_id}@{width}")
            image_path = (state_path.parent / fact["screenshot"]).resolve()
            if image_path in seen_images:
                raise ValueError(f"Screenshot path collision: {image_path}")
            seen_images.add(image_path)
            records.append(
                {
                    "id": f"{route_id}@{width}",
                    "site_family": "yana-public",
                    "route_family": route,
                    "viewport_width": width,
                    "document_scroll_width": fact["docScrollWidth"],
                    "screenshot": str(image_path.relative_to(OUTPUT.parent.resolve())),
                    "screenshot_sha256": digest(image_path),
                    "witness": str(state_path.resolve().relative_to(OUTPUT.parent.resolve())),
                    "witness_sha256": digest(state_path),
                    "console_error_count": len(state.get("consoleErrors") or []),
                    "broken_image_count": len(fact.get("brokenImages") or []),
                }
            )
    if len(records) != 40:
        raise ValueError(f"Expected forty capture cases; found {len(records)}")
    packet = {
        "schema": "mesen.raw-browser-capture.v1",
        "selection_sha256": digest(SELECTION),
        "source": selection["source"],
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "training_exclusion": "These images and labels must not be used for training, prompt selection, or checkpoint selection",
        "label_state": "unreviewed",
        "records": records,
    }
    OUTPUT.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
