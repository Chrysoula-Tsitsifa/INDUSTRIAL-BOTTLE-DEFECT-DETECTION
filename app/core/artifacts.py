from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ArtifactIntegrityError(RuntimeError):
    """Raised when a frozen deployment artifact fails integrity validation."""


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "final_artifacts"
MANIFEST_PATH = ARTIFACT_ROOT / "artifact_manifest.json"


def _read_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise ArtifactIntegrityError(
            f"Artifact manifest not found: {MANIFEST_PATH}"
        )

    try:
        with MANIFEST_PATH.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except json.JSONDecodeError as exc:
        raise ArtifactIntegrityError(
            f"Invalid artifact manifest JSON: {MANIFEST_PATH}"
        ) from exc

    if not isinstance(manifest, dict):
        raise ArtifactIntegrityError(
            "Artifact manifest root must be a JSON object."
        )

    files = manifest.get("files")

    if not isinstance(files, list) or not files:
        raise ArtifactIntegrityError(
            "Artifact manifest contains no valid file inventory."
        )

    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_artifact_integrity() -> dict[str, Any]:
    """
    Verify all canonical artifacts against the frozen integrity manifest.

    Validation includes:
    - path containment inside final_artifacts
    - file existence
    - exact byte size
    - SHA-256 equality
    """
    manifest = _read_manifest()
    artifact_root_resolved = ARTIFACT_ROOT.resolve()

    verified = 0

    for record in manifest["files"]:
        if not isinstance(record, dict):
            raise ArtifactIntegrityError(
                "Invalid file record in artifact manifest."
            )

        relative_path = record.get("path")
        expected_size = record.get("size_bytes")
        expected_sha256 = record.get("sha256")

        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ArtifactIntegrityError(
                "Manifest contains an invalid artifact path."
            )

        if not isinstance(expected_size, int) or expected_size < 0:
            raise ArtifactIntegrityError(
                f"Invalid size metadata for: {relative_path}"
            )

        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
        ):
            raise ArtifactIntegrityError(
                f"Invalid SHA-256 metadata for: {relative_path}"
            )

        artifact_path = (ARTIFACT_ROOT / relative_path).resolve()

        try:
            artifact_path.relative_to(artifact_root_resolved)
        except ValueError as exc:
            raise ArtifactIntegrityError(
                f"Artifact path escapes final_artifacts: {relative_path}"
            ) from exc

        if not artifact_path.is_file():
            raise ArtifactIntegrityError(
                f"Artifact is missing: {relative_path}"
            )

        actual_size = artifact_path.stat().st_size

        if actual_size != expected_size:
            raise ArtifactIntegrityError(
                f"Artifact size mismatch for {relative_path}: "
                f"expected {expected_size}, found {actual_size}"
            )

        actual_sha256 = _sha256(artifact_path)

        if actual_sha256.lower() != expected_sha256.lower():
            raise ArtifactIntegrityError(
                f"SHA-256 mismatch for: {relative_path}"
            )

        verified += 1

    return {
        "status": "verified",
        "verified_files": verified,
        "manifest_schema_version": manifest.get("schema_version"),
    }