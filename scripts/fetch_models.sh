#!/usr/bin/env bash
# Stage the ONNX artifacts that scripts/qc.sh needs. The binaries are
# gitignored (*.onnx), so a clean checkout has no judge and the test suite
# cannot load it. This script is the single owner of which artifacts exist and
# what they must hash to; everything else points here.
#
# Usage: bash scripts/fetch_models.sh
# Env:   MESEN_MODELS_DIR  destination directory (default: <repo>/models/onnx)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MESEN_MODELS_DIR:-$ROOT/models/onnx}"
BASE_URL="https://github.com/RikaiDev/mesen/releases/download/models-v1"

# name:sha256 — the pin is the contract, not the file name.
ARTIFACTS=(
	"mesen_jev_vlm.onnx:8325223078a0e442a654a5d3d0c67db826d9aa4e14b10f1ae8b1f4d59444f1b6"
	"ch_PP-OCRv4_det.onnx:c255248806ccdf52d6af1e45e362e6b27dcb770c6e2b92707459ee9a20f54587"
	"ch_PP-OCRv4_rec.onnx:8cd07d8689f3a0ba58741c97eea1bc4964bc60f005ef1802ba54c8cd4abd28c3"
)

if command -v sha256sum >/dev/null 2>&1; then
	sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
	sha256_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
	echo "fetch_models: no sha256sum or shasum available" >&2
	exit 1
fi

mkdir -p "$MODELS_DIR" || exit 1

status=0
for entry in "${ARTIFACTS[@]}"; do
	name="${entry%%:*}"
	want="${entry##*:}"
	dest="$MODELS_DIR/$name"

	if [ -f "$dest" ] && [ "$(sha256_of "$dest")" = "$want" ]; then
		echo "fetch_models: $name already staged"
		continue
	fi

	# A complete .part means a previous run was interrupted between the
	# download and the rename; adopt it instead of re-fetching 375MB.
	if [ -f "$dest.part" ] && [ "$(sha256_of "$dest.part")" = "$want" ]; then
		mv "$dest.part" "$dest"
		echo "fetch_models: staged $name (from a completed partial)"
		continue
	fi

	echo "fetch_models: downloading $name"
	# --continue-at resumes a partial transfer, and --speed-time/--speed-limit
	# aborts a stalled one instead of hanging; the judge is 375MB, so a
	# restart from zero on every retry is not affordable.
	if ! curl --fail --location --retry 3 --connect-timeout 30 \
		--speed-limit 10240 --speed-time 120 \
		--continue-at - --output "$dest.part" "$BASE_URL/$name"; then
		echo "fetch_models: download failed for $name" >&2
		echo "  a partial transfer may remain at $dest.part; re-run to resume" >&2
		status=1
		continue
	fi

	got="$(sha256_of "$dest.part")"
	if [ "$got" != "$want" ]; then
		echo "fetch_models: hash mismatch for $name" >&2
		echo "  expected $want" >&2
		echo "  actual   $got" >&2
		rm -f "$dest.part"
		status=1
		continue
	fi

	mv "$dest.part" "$dest"
	echo "fetch_models: staged $name"
done

[ "$status" -eq 0 ] || exit "$status"
echo "fetch_models: PASS"
