"""Tests for mesen.hub model discovery and verification."""

import os
import tempfile

from mesen.hub import (
    ARTIFACT_PINS,
    compute_sha256,
    ensure_keys_dict,
    get_default_models_dir,
)


def test_artifact_pins_are_complete_and_well_formed():
    """Every pin is a full SHA-256, and no artifact is pinned twice."""
    for name, expected_hash in ARTIFACT_PINS.items():
        assert name.endswith(".onnx"), f"{name} is not an ONNX artifact"
        assert len(expected_hash) == 64, f"{name} pin is not a SHA-256"
        int(expected_hash, 16)  # raises if it is not hex


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
