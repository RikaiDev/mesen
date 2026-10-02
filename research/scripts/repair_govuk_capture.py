"""Replace a redirect duplicate while preserving the original sealed capture."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "holdout"
ORIGINAL = ROOT / "govuk-sealed-manifest.json"
ORIGINAL_SELECTION = ROOT / "govuk-route-selection.json"
REPAIR = ROOT / "govuk-route-repair.json"
CAPTURE = ROOT / "govuk-repair-capture"
OUTPUT = ROOT / "govuk-final-manifest.json"
VIEWPORTS = (375, 768, 1024, 1440)


def digest(path: Path) -> str:
    """Hash a required regular file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Produce a new forty-case manifest with no reused screenshot bytes."""
    if OUTPUT.exists():
        raise FileExistsError("Repaired capture is already sealed")
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    repair = json.loads(REPAIR.read_text(encoding="utf-8"))
    if (
        repair.get("parent_capture_sha256") != digest(ORIGINAL)
        or repair.get("parent_selection_sha256") != digest(ORIGINAL_SELECTION)
        or repair.get("replaces") != "https://www.gov.uk/help/accessibility"
        or repair.get("routes") != ["https://www.gov.uk/help/beta"]
    ):
        raise ValueError("Replacement is not bound to the observed redirect duplicate")
    records = [
        record
        for record in original["records"]
        if record["route_family"] != urlsplit(repair["replaces"]).path
    ]
    if len(records) != 36:
        raise ValueError("Expected exactly four redirected records to remove")
    url = repair["routes"][0]
    route_id = hashlib.sha256(url.encode()).hexdigest()[:12]
    directory = CAPTURE / f"route-{route_id}"
    witness = directory / "state.json"
    state = json.loads(witness.read_text(encoding="utf-8"))
    if state.get("url") != url or state.get("route") != urlsplit(url).path:
        raise ValueError("Replacement witness route differs from registered URL")
    if state.get("httpFailed") or state.get("product") != "govuk-help":
        raise ValueError("Replacement witness has an HTTP error or wrong site family")
    facts = state.get("viewportFacts")
    if not isinstance(facts, list) or len(facts) != 4:
        raise ValueError("Replacement viewport facts are incomplete")
    by_width = {fact.get("width"): fact for fact in facts}
    if set(by_width) != set(VIEWPORTS):
        raise ValueError("Replacement viewport facts have missing or duplicate widths")
    for width in VIEWPORTS:
        fact = by_width[width]
        if not isinstance(fact.get("docScrollWidth"), int):
            raise ValueError("Replacement document width is absent")
        image = (directory / fact.get("screenshot", "")).resolve()
        records.append(
            {
                "id": f"{route_id}@{width}",
                "site_family": "govuk-help",
                "route_family": urlsplit(url).path,
                "viewport_width": width,
                "document_scroll_width": fact["docScrollWidth"],
                "screenshot": str(image.relative_to(ROOT.resolve())),
                "screenshot_sha256": digest(image),
                "witness": str(witness.resolve().relative_to(ROOT.resolve())),
                "witness_sha256": digest(witness),
                "console_error_count": len(state.get("consoleErrors") or []),
                "broken_image_count": len(fact.get("brokenImages") or []),
            }
        )
    if len(records) != 40 or len({record["screenshot_sha256"] for record in records}) != 40:
        raise ValueError("Repaired capture is not forty unique screenshot cases")
    packet = {
        "schema": "mesen.raw-browser-capture.v1",
        "selection_sha256": digest(ORIGINAL_SELECTION),
        "parent_capture_sha256": digest(ORIGINAL),
        "repair_sha256": digest(REPAIR),
        "source": original["source"],
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "training_exclusion": original["training_exclusion"],
        "label_state": "unreviewed",
        "records": records,
    }
    OUTPUT.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
