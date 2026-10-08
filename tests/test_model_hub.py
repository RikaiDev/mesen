"""Tests for mesen.hub model discovery and verification."""

import os
import tempfile
from pathlib import Path

from mesen.hub import (
    ARTIFACT_PINS,
    compute_sha256,
    ensure_keys_dict,
    get_default_models_dir,
)


def test_artifact_pins_match_fetch_script():
    """Verify that Python hub and bash fetch_models.sh pin identical artifact hashes."""
    fetch_script = Path(__file__).resolve().parent.parent / "scripts" / "fetch_models.sh"
    assert fetch_script.exists(), "scripts/fetch_models.sh must exist"
    content = fetch_script.read_text(encoding="utf-8")
    for name, expected_hash in ARTIFACT_PINS.items():
        assert f"{name}:{expected_hash}" in content, f"{name} pin mismatch in fetch_models.sh"


def test_get_default_models_dir_env_override(monkeypatch):
    custom_dir = "/tmp/custom_models_dir"
    monkeypatch.setenv("MESEN_MODELS_DIR", custom_dir)
    assert get_default_models_dir() == os.path.abspath(custom_dir)


def test_ensure_keys_dict():
    with tempfile.TemporaryDirectory() as tmpdir:
        keys_path = ensure_keys_dict(tmpdir)
        assert os.path.exists(keys_path)
        assert os.path.getsize(keys_path) > 1000
        digest = compute_sha256(keys_path)
        assert len(digest) == 64
        with open(keys_path, encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) > 100
