"""Fetch a pinned, licensed UI-UX teacher into Mesen's research workspace."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import model_info, snapshot_download

MODEL_ID = "afx-team/UI-UX"
REVISION = "846401e482f7b6fb8d61ab9fb7bc30d4e90e1c4b"
RESEARCH_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = RESEARCH_ROOT / "models" / f"ui-ux-{REVISION[:12]}"
RECEIPT_PATH = RESEARCH_ROOT / "artifacts" / "ui-ux-download.json"


def main() -> None:
    """Download one fixed teacher revision and record the resolved artifact path."""
    os.environ["HF_HOME"] = str(RESEARCH_ROOT / "cache")
    info = model_info(MODEL_ID, revision=REVISION)
    license_name = (info.card_data or {}).get("license")
    if info.sha != REVISION or license_name != "mit":
        raise RuntimeError(f"Unexpected model identity or license: {info.sha}, {license_name}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=MODEL_DIR)
    weights = sorted(MODEL_DIR.glob("*.safetensors"))
    if len(weights) != 2:
        raise RuntimeError(f"Expected two safetensors shards; found {len(weights)}")
    receipt = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "license": license_name,
        "model_dir": str(MODEL_DIR),
        "weight_files": [{"name": file.name, "bytes": file.stat().st_size} for file in weights],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
