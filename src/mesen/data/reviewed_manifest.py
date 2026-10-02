"""Verify image-grounded, independently reviewed UI/UX training records."""

import json
from hashlib import sha256
from pathlib import Path

from mesen.schema import ChoiceValue, DesignAuditLabel, DesignCriterion


def file_sha256(file_path: Path) -> str:
    """Hash a regular file; missing inputs fail before any optimizer is created."""
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    digest = sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_ref(manifest_path: Path, ref: dict) -> tuple[Path, str]:
    """Resolve and hash one manifest-owned image, brief, witness, or review."""
    if not isinstance(ref, dict) or not ref.get("path") or not ref.get("sha256"):
        raise ValueError("Every file reference needs path and sha256")
    path = (manifest_path.parent / ref["path"]).resolve()
    expected = ref["sha256"]
    if file_sha256(path) != expected:
        raise ValueError(f"File hash mismatch: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Empty evidence file: {path}")
    return path, expected


def _artifact_sha256(image_hashes: list[str], witness_hash: str, brief_hash: str) -> str:
    packet = json.dumps(
        {"images": image_hashes, "witness": witness_hash, "brief": brief_hash},
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(packet.encode()).hexdigest()


def load_reviewed_manifest(manifest_path: str | Path) -> tuple[list[dict], dict]:
    """Return only complete records whose labels bind the exact visual packet."""
    path = Path(manifest_path).resolve()
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema") != "mesen.reviewed-uiux.v1":
        raise ValueError("Unsupported reviewed manifest schema")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Reviewed manifest needs nonempty records")

    verified = []
    seen_ids = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every reviewed sample must be an object")
        sample_id = record.get("id")
        site = record.get("site_family")
        route = record.get("route_family")
        if not all(isinstance(value, str) and value.strip() for value in (sample_id, site, route)):
            raise ValueError("Every sample needs id, site_family, and route_family")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample id: {sample_id}")
        seen_ids.add(sample_id)

        image_refs = record.get("images")
        if not isinstance(image_refs, list) or not image_refs:
            raise ValueError(f"Sample {sample_id} has no screenshots")
        images = [_verified_ref(path, image_ref) for image_ref in image_refs]
        witness = _verified_ref(path, record.get("witness"))
        brief = _verified_ref(path, record.get("brief"))
        review = _verified_ref(path, record.get("evaluator_receipt"))
        label = DesignAuditLabel.model_validate(record.get("label"))
        critical_axes = record.get("critical_axes", [])
        clean_control = record.get("clean_control", False)
        if not isinstance(critical_axes, list) or any(
            not isinstance(axis, str) or axis not in {item.value for item in DesignCriterion}
            for axis in critical_axes
        ):
            raise ValueError(f"Invalid critical_axes: {sample_id}")
        if len(critical_axes) != len(set(critical_axes)):
            raise ValueError(f"Repeated critical axis: {sample_id}")
        if type(clean_control) is not bool or (clean_control and critical_axes):
            raise ValueError(f"Invalid clean_control: {sample_id}")
        if any(
            label.criteria[DesignCriterion(axis)].choice != ChoiceValue.NO for axis in critical_axes
        ):
            raise ValueError(f"Critical axis is not independently labeled no: {sample_id}")
        if clean_control and any(item.choice == ChoiceValue.NO for item in label.criteria.values()):
            raise ValueError(f"Clean control contains a failed axis: {sample_id}")
        if label.artifact_sha256 != _artifact_sha256(
            [digest for _, digest in images], witness[1], brief[1]
        ):
            raise ValueError(f"Label is not bound to sample pixels and evidence: {sample_id}")
        if (path.parent / label.evaluator_receipt_ref).resolve() != review[0]:
            raise ValueError(f"Label review reference differs from verified receipt: {sample_id}")
        if (path.parent / label.brief_ref).resolve() != brief[0]:
            raise ValueError(f"Label brief reference differs from verified brief: {sample_id}")
        review_data = json.loads(review[0].read_text(encoding="utf-8"))
        if (
            review_data.get("schema") != "mesen.evaluator-receipt.v1"
            or review_data.get("evaluator_id") != label.evaluator_id
            or review_data.get("artifact_sha256") != label.artifact_sha256
            or review_data.get("criteria")
            != {
                criterion.value: result.model_dump(mode="json")
                for criterion, result in label.criteria.items()
            }
            or review_data.get("critical_axes", []) != critical_axes
        ):
            raise ValueError(f"Evaluator receipt contradicts label: {sample_id}")
        verified.append(
            {
                "id": sample_id,
                "group": (site, route),
                "images": [str(image_path) for image_path, _ in images],
                "witness": str(witness[0]),
                "brief": str(brief[0]),
                "label": label,
                "critical_axes": critical_axes,
                "clean_control": clean_control,
            }
        )
    receipt = {
        "manifest": str(path),
        "manifest_sha256": sha256(raw).hexdigest(),
        "records_verified": len(verified),
        "images_verified": sum(len(record["images"]) for record in verified),
        "groups": sorted(
            {f"{site}:{route}" for record in verified for site, route in [record["group"]]}
        ),
        "missing_or_skipped": 0,
    }
    return verified, receipt


def load_reviewed_splits(
    train_path: str | Path, validation_path: str | Path, holdout_path: str | Path
) -> tuple[list[dict], list[dict], dict]:
    """Validate all split groups while withholding blind records from training."""
    train, train_receipt = load_reviewed_manifest(train_path)
    validation, validation_receipt = load_reviewed_manifest(validation_path)
    holdout, holdout_receipt = load_reviewed_manifest(holdout_path)
    groups = [{record["group"] for record in split} for split in (train, validation, holdout)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Site and route family overlap between data splits")
    ids = [{record["id"] for record in split} for split in (train, validation, holdout)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("Sample id overlap between data splits")
    return (
        train,
        validation,
        {
            "train": train_receipt,
            "validation": validation_receipt,
            "sealed_holdout": holdout_receipt,
        },
    )
