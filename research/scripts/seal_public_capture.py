"""Hash and inventory a public-page capture without judging its design."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ROOT / "artifacts" / "holdout" / "mig-route-selection.json"
DEFAULT_CAPTURE = ROOT / "artifacts" / "holdout" / "mig-sealed-capture"
DEFAULT_OUTPUT = ROOT / "artifacts" / "holdout" / "mig-sealed-manifest.json"
VIEWPORTS = (375, 768, 1024, 1440)


def digest(path: Path) -> str:
    """Hash a required regular capture file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Seal exactly ten complete routes and forty distinct screenshot files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    selection_path = args.selection.resolve()
    capture_path = args.capture.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError("Public-page capture is already sealed")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema") != "mesen.public-route-selection.v1":
        raise ValueError("Invalid public-page route selection")
    records = []
    seen_images = set()
    for url in selection["routes"]:
        route_id = hashlib.sha256(url.encode()).hexdigest()[:12]
        directory = capture_path / f"route-{route_id}"
        witness = directory / "state.json"
        state = json.loads(witness.read_text(encoding="utf-8"))
        if state.get("url") != url or state.get("route") != urlsplit(url).path:
            raise ValueError(f"Wrong route witness: {route_id}")
        if state.get("product") != selection["site_family"]:
            raise ValueError(f"Wrong site family in witness: {route_id}")
        if state.get("httpFailed"):
            raise ValueError(f"HTTP errors in route witness: {route_id}")
        facts = state.get("viewportFacts")
        if not isinstance(facts, list) or len(facts) != 4:
            raise ValueError(f"Incomplete viewport facts: {route_id}")
        by_width = {fact.get("width"): fact for fact in facts}
        if set(by_width) != set(VIEWPORTS):
            raise ValueError(f"Missing or duplicate viewport facts: {route_id}")
        for width in VIEWPORTS:
            fact = by_width[width]
            if not isinstance(fact.get("docScrollWidth"), int):
                raise ValueError(f"Missing document width: {route_id}@{width}")
            image = (directory / fact.get("screenshot", "")).resolve()
            if image in seen_images:
                raise ValueError(f"Screenshot path reused: {image}")
            seen_images.add(image)
            records.append(
                {
                    "id": f"{route_id}@{width}",
                    "site_family": selection["site_family"],
                    "route_family": urlsplit(url).path,
                    "viewport_width": width,
                    "document_scroll_width": fact["docScrollWidth"],
                    "screenshot": str(image.relative_to(output_path.parent)),
                    "screenshot_sha256": digest(image),
                    "witness": str(witness.resolve().relative_to(output_path.parent)),
                    "witness_sha256": digest(witness),
                    "console_error_count": len(state.get("consoleErrors") or []),
                    "broken_image_count": len(fact.get("brokenImages") or []),
                }
            )
    if len(records) != 40:
        raise ValueError("Public-page capture must contain forty records")
    packet = {
        "schema": "mesen.raw-browser-capture.v1",
        "selection_sha256": digest(selection_path),
        "source": selection["source"],
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "training_exclusion": "These images and labels must not be used for training, prompt selection, or checkpoint selection",
        "label_state": "unreviewed",
        "records": records,
    }
    output_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "output": str(output_path)}))


if __name__ == "__main__":
    main()
