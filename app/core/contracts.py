from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    """Raised when a deployment contract is missing, invalid, or inconsistent."""


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "final_artifacts"

CONTRACT_FILES = {
    "notebook1": "baseline_metrics.json",
    "notebook2": "notebook2_metrics.json",
    "notebook3": "notebook3_metrics.json",
    "notebook4": "notebook4_metrics.json",
}

EXPECTED_SCHEMA_VERSIONS = {
    "notebook1": "2.1",
    "notebook2": "2.1",
    "notebook3": "1.0",
    "notebook4": "1.0",
}

REQUIRED_TOP_LEVEL_KEYS = {
    "notebook1": {
        "schema_version",
        "artifact_path_policy",
        "artifacts",
        "img_size",
        "label_contract",
        "score_contract",
        "threshold_policy",
    },
    "notebook2": {
        "schema_version",
        "artifact_path_policy",
        "artifacts",
        "img_size",
        "label_contract",
        "score_contract",
        "threshold_policy",
    },
    "notebook3": {
        "schema_version",
        "artifact_path_policy",
        "artifacts",
        "img_size",
        "label_contract",
        "score_contract",
        "threshold_policy",
        "upstream_notebook2",
    },
    "notebook4": {
        "schema_version",
        "artifact_path_policy",
        "artifacts",
        "img_size",
        "label_contract",
        "score_contract",
        "decision_contract",
        "probability_calibration",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"Contract file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ContractError(
            f"Invalid JSON contract: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise ContractError(
            f"Contract root must be a JSON object: {path}"
        )

    return payload


def _validate_required_keys(
    notebook: str,
    contract: dict[str, Any],
) -> None:
    required = REQUIRED_TOP_LEVEL_KEYS[notebook]
    missing = sorted(required.difference(contract))

    if missing:
        raise ContractError(
            f"{notebook} contract is missing required keys: "
            f"{', '.join(missing)}"
        )


def _validate_schema_version(
    notebook: str,
    contract: dict[str, Any],
) -> None:
    expected = EXPECTED_SCHEMA_VERSIONS[notebook]
    actual = str(contract.get("schema_version"))

    if actual != expected:
        raise ContractError(
            f"{notebook} schema mismatch: "
            f"expected {expected}, found {actual}"
        )


def _validate_artifact_paths(
    contract_path: Path,
    contract: dict[str, Any],
) -> None:
    artifacts = contract.get("artifacts")

    if not isinstance(artifacts, dict):
        raise ContractError(
            f"'artifacts' must be an object in {contract_path.name}"
        )

    artifact_root_resolved = ARTIFACT_ROOT.resolve()

    for artifact_name, relative_path in artifacts.items():
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ContractError(
                f"Invalid artifact path for '{artifact_name}' "
                f"in {contract_path.name}"
            )

        candidate = (contract_path.parent / relative_path).resolve()

        try:
            candidate.relative_to(artifact_root_resolved)
        except ValueError as exc:
            raise ContractError(
                f"Artifact path escapes final_artifacts: {relative_path}"
            ) from exc

        if not candidate.is_file():
            raise ContractError(
                f"Declared artifact does not exist: {relative_path}"
            )


def load_contract(notebook: str) -> dict[str, Any]:
    """
    Load and validate one frozen notebook contract.

    Parameters
    ----------
    notebook:
        One of: notebook1, notebook2, notebook3, notebook4.
    """
    if notebook not in CONTRACT_FILES:
        valid = ", ".join(CONTRACT_FILES)
        raise ContractError(
            f"Unknown contract '{notebook}'. Valid values: {valid}"
        )

    contract_path = ARTIFACT_ROOT / CONTRACT_FILES[notebook]
    contract = _read_json(contract_path)

    _validate_required_keys(notebook, contract)
    _validate_schema_version(notebook, contract)
    _validate_artifact_paths(contract_path, contract)

    return contract


def load_all_contracts() -> dict[str, dict[str, Any]]:
    """Load and validate all frozen notebook contracts."""
    return {
        notebook: load_contract(notebook)
        for notebook in CONTRACT_FILES
    }