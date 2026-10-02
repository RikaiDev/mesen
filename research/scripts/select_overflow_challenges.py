"""Preselect paired browser-only overflow defects across three sealed sites."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "holdout"
SOURCES = {
    "yana-public": ("yana-sealed-manifest.json", 7),
    "mig-guidelines": ("mig-sealed-manifest.json", 7),
    "govuk-help": ("govuk-final-manifest.json", 6),
}
OUTPUT = ROOT / "overflow-challenge-selection.json"
SEED = "mesen-blind-overflow-v1-2026-09-28"


def digest(path: Path) -> str:
    """Hash a sealed manifest or witness file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Freeze twenty route families and two viewport mutations per route."""
    if OUTPUT.exists():
        raise FileExistsError("Overflow challenge selection is already frozen")
    selected = []
    source_hashes = {}
    for site, (filename, count) in SOURCES.items():
        manifest_path = ROOT / filename
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("label_state") != "unreviewed":
            raise ValueError(f"Unexpected label state in {filename}")
        source_hashes[filename] = digest(manifest_path)
        records = [record for record in manifest["records"] if record["viewport_width"] == 375]
        if len(records) < count or len({record["route_family"] for record in records}) != len(
            records
        ):
            raise ValueError(f"Insufficient unique routes for {site}")
        ranked = sorted(
            records,
            key=lambda record: hashlib.sha256(
                f"{SEED}:{site}:{record['route_family']}".encode()
            ).hexdigest(),
        )[:count]
        for record in ranked:
            witness_path = ROOT / record["witness"]
            if digest(witness_path) != record["witness_sha256"]:
                raise ValueError("Raw witness changed after sealing")
            witness = json.loads(witness_path.read_text(encoding="utf-8"))
            if witness.get("route") != record["route_family"]:
                raise ValueError("Raw witness route differs from sealed manifest")
            for width, min_width in ((375, 768), (768, 1024)):
                selected.append(
                    {
                        "id": hashlib.sha256(
                            f"{site}:{record['route_family']}:{width}".encode()
                        ).hexdigest()[:16],
                        "site_family": site,
                        "route_family": record["route_family"],
                        "url": witness["url"],
                        "viewport_width": width,
                        "injected_min_width": min_width,
                        "mutation": "html, body { min-width: Npx !important; } in local browser only",
                    }
                )
    if len(selected) != 40:
        raise ValueError("Expected exactly forty preselected challenges")
    packet = {
        "schema": "mesen.overflow-challenge-selection.v1",
        "source_manifests": source_hashes,
        "selection_seed": SEED,
        "selection_method": "seven Yana, seven MIG, six GOV.UK routes by smallest SHA256(seed:site:route), then 375/768px mutations",
        "excluded_from_training": True,
        "cases": selected,
    }
    OUTPUT.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(selected), "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
