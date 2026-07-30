from __future__ import annotations

import json
from pathlib import Path

from eigentruth.registry import load_and_verify_artifact_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pre_generation_probe_comparison_baseline_manifest_verifies():
    manifest_path = (
        REPO_ROOT
        / "artifacts"
        / "baselines"
        / "pre_generation_probe_comparison"
        / "artifact-manifest.json"
    )

    verification = load_and_verify_artifact_manifest(manifest_path)

    assert verification.passed is True
    assert verification.checked == 1


def test_belief_revision_real_model_kill_test_manifest_verifies():
    manifest_path = (
        REPO_ROOT
        / "artifacts"
        / "baselines"
        / "belief_revision_text"
        / "kill-test-v1"
        / "real-model-results"
        / "artifact-manifest.json"
    )

    verification = load_and_verify_artifact_manifest(manifest_path, root=REPO_ROOT)

    assert verification.passed is True
    assert verification.checked == 5


def test_default_product_handoff_manifests_do_not_require_runtime_evidence():
    manifest_paths = (
        REPO_ROOT
        / "artifacts"
        / "smollm2_product_promotion_evidence_handoff_v1_9_frontier_v7"
        / "artifact-manifest.json",
        REPO_ROOT
        / "artifacts"
        / "smollm2_product_promotion_contract_v1_9"
        / "evidence-handoff-artifact-manifest.json",
    )

    for manifest_path in manifest_paths:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = [
            str(entry.get("path", ""))
            for entry in payload.get("artifacts", {}).values()
            if isinstance(entry, dict)
        ]
        assert not any("runtime_evidence" in path for path in paths)
        assert any("baselines/pre_generation_probe_comparison" in path for path in paths)
