from __future__ import annotations

import unittest

from app.core.artifacts import verify_artifact_integrity
from app.core.contracts import CONTRACT_FILES, load_all_contracts


class DeploymentIntegrityTests(unittest.TestCase):
    """Smoke tests for the frozen deployment contracts and artifacts."""

    def test_all_contracts_load_and_validate(self) -> None:
        contracts = load_all_contracts()

        self.assertEqual(
            set(contracts),
            set(CONTRACT_FILES),
        )

    def test_all_canonical_artifacts_match_manifest(self) -> None:
        result = verify_artifact_integrity()

        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["verified_files"], 21)
        self.assertEqual(result["manifest_schema_version"], "1.0")


if __name__ == "__main__":
    unittest.main()