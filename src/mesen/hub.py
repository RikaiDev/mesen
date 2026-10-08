"""
AI-Native Model Hub and Zero-Friction Bootstrapping for Mesen.
Automatically discovers, downloads, and verifies ONNX model artifacts and dictionaries.
Runs completely on local CPU without manual pre-download steps.
"""

import hashlib
import os
import shutil
import urllib.request

MODELS_RELEASE_BASE_URL = "https://github.com/RikaiDev/mesen/releases/download/models-v1"

# Pinned SHA256 hashes: single canonical source of truth for artifact integrity
ARTIFACT_PINS: dict[str, str] = {
    "mesen_jev_vlm.onnx": "2fe3cd2b75637aaaf816e430f722b7755be1403e8df3b84ddcfa68177765699f",
    "ch_PP-OCRv4_det.onnx": "c255248806ccdf52d6af1e45e362e6b27dcb770c6e2b92707459ee9a20f54587",
    "ch_PP-OCRv4_rec.onnx": "8cd07d8689f3a0ba58741c97eea1bc4964bc60f005ef1802ba54c8cd4abd28c3",
}


def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_default_models_dir() -> str:
    """Resolve models directory.

    Priority:
    1. MESEN_MODELS_DIR environment variable
    2. Local checkout models/onnx if it exists and contains mesen_jev_vlm.onnx
    3. User cache directory ~/.cache/mesen/models/onnx (or XDG_CACHE_HOME)
    """
    if "MESEN_MODELS_DIR" in os.environ:
        return os.path.abspath(os.environ["MESEN_MODELS_DIR"])

    # Check if running inside repository checkout
    repo_models_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "onnx")
    )
    if os.path.isdir(repo_models_dir) and os.path.exists(
        os.path.join(repo_models_dir, "mesen_jev_vlm.onnx")
    ):
        return repo_models_dir

    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return os.path.join(cache_home, "mesen", "models", "onnx")


def ensure_keys_dict(target_dir: str) -> str:
    """Ensure ppocr_keys_v1.txt is present in the target directory."""
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "ppocr_keys_v1.txt")
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return target_path

    # Try copying from package resources
    try:
        from importlib.resources import files

        res = files("mesen.resources").joinpath("ppocr_keys_v1.txt")
        if res.is_file():
            with res.open("rb") as src_f, open(target_path, "wb") as dst_f:
                shutil.copyfileobj(src_f, dst_f)
            return target_path
    except Exception:
        pass

    # Fallback to local checkout path if available
    repo_keys = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "onnx", "ppocr_keys_v1.txt")
    )
    if os.path.exists(repo_keys):
        shutil.copyfile(repo_keys, target_path)
        return target_path

    # Last resort fallback: fetch from GitHub release or raw repo
    raw_url = "https://raw.githubusercontent.com/RikaiDev/mesen/main/models/onnx/ppocr_keys_v1.txt"
    try:
        req = urllib.request.Request(raw_url, headers={"User-Agent": "mesen-vlm/0.2.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, "wb") as f:
            f.write(resp.read())
        return target_path
    except Exception as e:
        raise FileNotFoundError(
            f"Could not locate or fetch ppocr_keys_v1.txt for OCR engine: {e}"
        ) from e


def fetch_artifact(name: str, target_dir: str, verbose: bool = True) -> str:
    """Download an artifact from GitHub Release models-v1 with SHA256 integrity verification."""
    expected_hash = ARTIFACT_PINS.get(name)
    if not expected_hash:
        raise ValueError(f"Unknown artifact name: {name}")

    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, name)
    part_path = f"{target_path}.part"

    if os.path.exists(target_path):
        if compute_sha256(target_path) == expected_hash:
            return target_path
        # Hash mismatch, remove corrupted file
        os.remove(target_path)

    # Check if a completed .part exists
    if os.path.exists(part_path) and compute_sha256(part_path) == expected_hash:
        os.rename(part_path, target_path)
        return target_path

    url = f"{MODELS_RELEASE_BASE_URL}/{name}"
    if verbose:
        print(f"[mesen] Auto-fetching model '{name}' from GitHub Releases...")

    req = urllib.request.Request(url, headers={"User-Agent": "mesen-vlm/0.2.0"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp, open(part_path, "wb") as out_f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            last_reported = -1
            while True:
                chunk = resp.read(1024 * 512)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and verbose:
                    pct = int(downloaded * 100 / total)
                    if pct != last_reported and pct % 25 == 0:
                        mb_done = round(downloaded / (1024 * 1024), 1)
                        mb_total = round(total / (1024 * 1024), 1)
                        print(f"[mesen]   -> {name}: {pct}% ({mb_done}MB / {mb_total}MB)")
                        last_reported = pct
    except Exception as e:
        if os.path.exists(part_path) and os.path.getsize(part_path) == 0:
            os.remove(part_path)
        raise RuntimeError(f"Failed to download {name} from {url}: {e}") from e

    actual_hash = compute_sha256(part_path)
    if actual_hash != expected_hash:
        if os.path.exists(part_path):
            os.remove(part_path)
        raise RuntimeError(
            f"SHA256 integrity verification failed for {name}: expected {expected_hash}, got {actual_hash}"
        )

    os.rename(part_path, target_path)
    if verbose:
        print(f"[mesen] Verified and staged {name} -> {target_path}")
    return target_path


def ensure_models(models_dir: str | None = None, verbose: bool = True) -> str:
    """Ensure all required ONNX models and keys dictionary are staged and valid.

    Returns the directory path where models are staged.
    """
    target_dir = models_dir or get_default_models_dir()
    os.makedirs(target_dir, exist_ok=True)
    ensure_keys_dict(target_dir)
    for name in ARTIFACT_PINS:
        fetch_artifact(name, target_dir, verbose=verbose)
    return target_dir
