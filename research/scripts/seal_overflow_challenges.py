"""Seal measured clean/mutated browser pairs without subjective design labels."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "holdout"
SELECTION = ROOT / "overflow-challenge-selection.json"
CAPTURE = ROOT / "overflow-challenge-capture"
GOVUK_CAPTURE = ROOT / "overflow-challenge-capture-govuk-v2"
OUTPUT = ROOT / "overflow-challenge-manifest.json"


def digest(path: Path) -> str:
    """Hash one required regular artifact."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Require every selected mutation to have a clean baseline and measured overflow."""
    if OUTPUT.exists():
        raise FileExistsError("Overflow challenge manifest is already sealed")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection.get("schema") != "mesen.overflow-challenge-selection.v1":
        raise ValueError("Invalid challenge selection")
    records = []
    seen_image_hashes = set()
    for challenge in selection["cases"]:
        capture_root = GOVUK_CAPTURE if challenge["site_family"] == "govuk-help" else CAPTURE
        directory = capture_root / challenge["id"]
        witness = directory / "state.json"
        state = json.loads(witness.read_text(encoding="utf-8"))
        css = f"html, body {{ min-width: {challenge['injected_min_width']}px !important; }}"
        if (
            state.get("schema") != "mesen.paired-overflow-witness.v1"
            or state.get("id") != challenge["id"]
            or state.get("site_family") != challenge["site_family"]
            or state.get("route_family") != challenge["route_family"]
            or state.get("requested_url") != challenge["url"]
            or state.get("viewport_width") != challenge["viewport_width"]
            or state.get("injected_css") != css
            or state.get("http_errors")
        ):
            raise ValueError(f"Witness contradicts selected challenge {challenge['id']}")
        width = challenge["viewport_width"]
        before = state.get("baseline_document_scroll_width")
        after = state.get("mutated_document_scroll_width")
        if type(before) is not int or type(after) is not int or before > width or after <= width:
            raise ValueError(f"Unproven document overflow in {challenge['id']}")
        method = state.get("mutation_method", "local browser style tag")
        if challenge["site_family"] == "govuk-help":
            asset = state.get("style_asset")
            if (
                method != "same-origin stylesheet response append in local browser"
                or not isinstance(asset, dict)
                or not asset.get("url", "").startswith(
                    "https://www.gov.uk/assets/frontend/application-"
                )
                or asset.get("original_sha256") == asset.get("modified_sha256")
            ):
                raise ValueError(f"GOV.UK CSP-respecting mutation is unproven: {challenge['id']}")
        elif method != "local browser style tag":
            raise ValueError(f"Unexpected mutation method: {challenge['id']}")
        clean = directory / "clean.png"
        mutated = directory / "mutated.png"
        clean_hash = digest(clean)
        mutated_hash = digest(mutated)
        if (
            clean_hash == mutated_hash
            or clean_hash in seen_image_hashes
            or mutated_hash in seen_image_hashes
        ):
            raise ValueError(f"Duplicate or unchanged challenge image: {challenge['id']}")
        seen_image_hashes.update((clean_hash, mutated_hash))
        records.append(
            {
                "id": challenge["id"],
                "site_family": challenge["site_family"],
                "route_family": challenge["route_family"],
                "viewport_width": width,
                "baseline_document_scroll_width": before,
                "mutated_document_scroll_width": after,
                "injected_css": css,
                "mutation_method": method,
                "style_asset": state.get("style_asset"),
                "clean_screenshot": str(clean.relative_to(ROOT)),
                "clean_screenshot_sha256": clean_hash,
                "mutated_screenshot": str(mutated.relative_to(ROOT)),
                "mutated_screenshot_sha256": mutated_hash,
                "witness": str(witness.relative_to(ROOT)),
                "witness_sha256": digest(witness),
            }
        )
    if len(records) != 40:
        raise ValueError(f"Expected forty measured challenges, found {len(records)}")
    packet = {
        "schema": "mesen.paired-overflow-manifest.v1",
        "selection_sha256": digest(SELECTION),
        "measured_defect": "document_scroll_width > viewport_width after browser-only CSS mutation",
        "label_state": "objective geometry only; four-axis independent review pending",
        "excluded_from_training": True,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    OUTPUT.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
