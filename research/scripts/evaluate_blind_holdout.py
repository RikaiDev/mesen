"""Emit a reproducible blind-holdout acceptance receipt."""

import argparse
import json
from pathlib import Path

from mesen.evaluation.blind_holdout import evaluate_blind_holdout


def main() -> None:
    """Write all metrics and fail the process if any predeclared gate is unmet."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_blind_holdout(args.manifest, args.predictions, args.contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"accepted": report["accepted"], "failures": report["failures"]}))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
