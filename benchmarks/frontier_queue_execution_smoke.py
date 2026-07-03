"""No-model smoke fixture for frontier queue execution control-plane staging."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.bind_frontier_research_queue_command_plan import (  # noqa: E402
    build_frontier_research_queue_bound_command_plan,
)
from benchmarks.bind_frontier_research_queue_seed_inputs import (  # noqa: E402
    bind_frontier_research_queue_seed_inputs,
)
from benchmarks.plan_frontier_research_queue_commands import (  # noqa: E402
    build_frontier_research_queue_command_plan,
)
from benchmarks.review_frontier_research_queue_command_bindings import (  # noqa: E402
    review_frontier_research_queue_command_bindings,
)
from benchmarks.run_frontier_research_queue_bound_command_plan import (  # noqa: E402
    run_frontier_research_queue_bound_command_plan,
)
from benchmarks.scaffold_frontier_research_queue_bindings import (  # noqa: E402
    scaffold_frontier_research_queue_bindings,
)
from benchmarks.stage_frontier_research_queue_binding_suggestions import (  # noqa: E402
    stage_frontier_research_queue_binding_suggestions,
)
from benchmarks.summarize_unresolved_frontier_evidence import (  # noqa: E402
    run as summarize_unresolved_frontier_evidence,
)
from eigentruth.json_utils import strict_json_dumps  # noqa: E402
from eigentruth.registry import (  # noqa: E402
    ArtifactRegistry,
    build_artifact_manifest,
    load_and_verify_artifact_manifest,
)

SMOKE_NAME = "frontier-queue-execution-smoke"
SMOKE_VERSION = "0.1"
SMOKE_RECORD_KEY = f"report:{SMOKE_NAME}:{SMOKE_VERSION}"
SMOKE_MANIFEST_RECORD_KEY = f"benchmark_manifest:{SMOKE_NAME}:{SMOKE_VERSION}"


def build_frontier_queue_execution_smoke(
    output_dir: Path,
    *,
    name: str = SMOKE_NAME,
    version: str = SMOKE_VERSION,
    compact_json: bool = False,
) -> dict[str, Any]:
    """Build a reproducible dry-run fixture for frontier queue execution.

    The smoke creates synthetic local control-plane artifacts, plans the
    ``execute_reviewed_frontier_queue_command_plan`` action from an unresolved
    summary, stages same-action upstream outputs, applies the command-binding
    review gate, and finally dry-runs the approved execution plan. It also
    exercises the source-family URL seed handoff from input-binding audit to
    command-plan action and seed-binding staging. It never runs child frontier
    commands or treats any fixture artifact as verifier evidence.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path = output_dir / "registry.json"
    paths = _paths(output_dir)
    _write_source_fixture(paths, compact=compact_json)

    seed_summary = summarize_unresolved_frontier_evidence(
        input_binding_audit_paths=(paths["source_input_binding_audit"],),
        frontier_command_binding_paths=(paths["source_seed_base_bindings"],),
        json_path=paths["seed_unresolved_summary"],
        artifact_manifest_path=paths["seed_unresolved_summary_manifest"],
        compact_json=compact_json,
        metadata={"smoke": name, "not_verifier_evidence": True},
    )
    seed_command_plan = build_frontier_research_queue_command_plan(
        source=paths["seed_unresolved_summary"],
        json_path=paths["seed_command_plan"],
        artifact_manifest_path=paths["seed_command_plan_manifest"],
        output_dir=output_dir / "seed-command-outputs",
        include_action_ids=("stage_frontier_queue_seed_inputs",),
        compact_json=compact_json,
        metadata={"smoke": name, "not_verifier_evidence": True},
    )
    seed_binding_stage = bind_frontier_research_queue_seed_inputs(
        input_binding_audit=paths["source_input_binding_audit"],
        base_bindings=paths["source_seed_base_bindings"],
        output_dir=paths["seed_binding_stage_dir"],
        json_path=paths["seed_binding_stage_report"],
        bindings_json_path=paths["seed_staged_bindings"],
        records_jsonl_path=paths["seed_binding_records"],
        artifact_manifest_path=paths["seed_binding_stage_manifest"],
        compact_json=compact_json,
        metadata={"smoke": name, "not_verifier_evidence": True},
    )

    build_frontier_research_queue_command_plan(
        source=paths["unresolved_summary"],
        json_path=paths["command_plan"],
        artifact_manifest_path=paths["command_plan_manifest"],
        output_dir=output_dir / "planned-command-outputs",
        compact_json=compact_json,
        metadata={"smoke": name, "not_verifier_evidence": True},
    )
    scaffold_frontier_research_queue_bindings(
        command_plan=paths["command_plan"],
        json_path=paths["binding_scaffold"],
        bindings_json_path=paths["binding_skeleton"],
        artifact_manifest_path=paths["binding_scaffold_manifest"],
        registry_output_path=registry_path,
        compact_json=compact_json,
        metadata={"smoke": name, "not_verifier_evidence": True},
    )
    staged = stage_frontier_research_queue_binding_suggestions(
        scaffold=paths["binding_scaffold"],
        bindings_json_path=paths["staged_bindings"],
        artifact_manifest_path=paths["staged_bindings_manifest"],
        stage_upstream_outputs=True,
        compact_json=compact_json,
        metadata={"smoke": name, "not_verifier_evidence": True},
    )
    rebound = build_frontier_research_queue_bound_command_plan(
        command_plan=paths["command_plan"],
        bindings=paths["staged_bindings"],
        json_path=paths["bound_command_plan"],
        artifact_manifest_path=paths["bound_command_plan_manifest"],
        compact_json=compact_json,
        metadata={"smoke": name, "not_verifier_evidence": True},
    )
    _write_jsonl(
        paths["review_decisions"],
        (
            {
                "action_id": "execute_reviewed_frontier_queue_command_plan",
                "decision": "approved",
                "reviewer": "frontier-queue-execution-smoke",
                "reviewed_at": "2026-07-02T00:00:00Z",
                "not_verifier_evidence": True,
            },
        ),
        compact=compact_json,
    )
    review = review_frontier_research_queue_command_bindings(
        bound_command_plan=paths["bound_command_plan"],
        base_bindings=paths["staged_bindings"],
        review_decisions=paths["review_decisions"],
        output_dir=paths["review_dir"],
        approved_bindings_path=paths["approved_bindings"],
        compact_json=compact_json,
        metadata={"smoke": name, "not_verifier_evidence": True},
    )
    approved_rebound = build_frontier_research_queue_bound_command_plan(
        command_plan=paths["command_plan"],
        bindings=paths["approved_bindings"],
        json_path=paths["approved_bound_command_plan"],
        artifact_manifest_path=paths["approved_bound_command_plan_manifest"],
        compact_json=compact_json,
        metadata={"smoke": name, "not_verifier_evidence": True},
    )
    dry_run = run_frontier_research_queue_bound_command_plan(
        bound_command_plan=paths["approved_bound_command_plan"],
        json_path=paths["dry_run_report"],
        artifact_manifest_path=paths["dry_run_manifest"],
        dry_run=True,
        require_reviewed_bindings=True,
        compact_json=compact_json,
        metadata={"smoke": name, "not_verifier_evidence": True},
    )

    _assert_smoke_chain(
        seed_summary=seed_summary,
        seed_command_plan=seed_command_plan,
        seed_binding_stage=seed_binding_stage,
        staged=staged,
        rebound=rebound,
        review=review,
        approved_rebound=approved_rebound,
        dry_run=dry_run,
    )
    report = _smoke_report(
        paths,
        seed_summary=seed_summary,
        seed_command_plan=seed_command_plan,
        seed_binding_stage=seed_binding_stage,
        staged=staged,
        rebound=rebound,
        review=review,
        approved_rebound=approved_rebound,
        dry_run=dry_run,
    )
    _write_json(paths["smoke_report"], report, compact=compact_json)
    manifest = build_artifact_manifest(
        _manifest_artifacts(paths, review=review),
        root=output_dir,
        metadata={
            "runner": "frontier_queue_execution_smoke",
            "workflow": "frontier_queue_execution_smoke",
            "status": report["status"],
            "dry_run_count": _nested(dry_run, "summary", "dry_run_count"),
            "seed_applied_placeholder_count": _nested(
                seed_binding_stage,
                "summary",
                "applied_placeholder_count",
            ),
            "staged_upstream_output_count": _nested(
                staged,
                "staging_summary",
                "staged_upstream_output_count",
            ),
            "not_verifier_evidence": True,
        },
    )
    _write_json(paths["artifact_manifest"], manifest, compact=compact_json)
    verification = load_and_verify_artifact_manifest(paths["artifact_manifest"], recursive=True)
    if verification.passed is not True:
        raise AssertionError("frontier queue execution smoke artifact manifest failed verification.")

    registry = ArtifactRegistry.load_json(registry_path)
    registry.record_report(
        name=name,
        version=version,
        path=paths["smoke_report"],
        metadata={
            "workflow": "frontier_queue_execution_smoke",
            "status": report["status"],
            "artifact_manifest": str(paths["artifact_manifest"]),
            "dry_run_count": _nested(dry_run, "summary", "dry_run_count"),
            "seed_applied_placeholder_count": _nested(
                seed_binding_stage,
                "summary",
                "applied_placeholder_count",
            ),
            "binding_not_reviewed_count": _nested(
                dry_run,
                "summary",
                "binding_not_reviewed_count",
            ),
            "manifest_passed": True,
            "not_verifier_evidence": True,
        },
    ).record_benchmark_manifest(
        name=name,
        version=version,
        path=paths["artifact_manifest"],
        metadata={
            "workflow": "frontier_queue_execution_smoke",
            "status": report["status"],
            "manifest_summary": manifest.get("summary", {}),
            "not_verifier_evidence": True,
        },
    ).save_json()
    return {
        **report,
        "manifest_verification": verification.to_dict(),
        "registry": str(registry_path),
        "registry_record": f"report:{name}:{version}",
        "registry_manifest_record": f"benchmark_manifest:{name}:{version}",
    }


def _paths(output_dir: Path) -> dict[str, Path]:
    fixture_dir = output_dir / "fixture"
    review_dir = output_dir / "command-binding-review"
    return {
        "fixture_dir": fixture_dir,
        "source_command_plan": fixture_dir / "source-frontier-command-plan.json",
        "source_approved_bindings": fixture_dir / "source-approved-command-bindings.json",
        "source_bound_command_plan": fixture_dir / "source-bound-command-plan.json",
        "source_binding_review": fixture_dir / "source-command-binding-review.json",
        "source_seed_sidecar": fixture_dir / "source-family-url-seeds.jsonl",
        "source_input_binding_audit": fixture_dir / "source-input-binding-audit.json",
        "source_seed_base_bindings": fixture_dir / "source-seed-base-bindings.json",
        "seed_unresolved_summary": output_dir / "seed-unresolved-frontier-summary.json",
        "seed_unresolved_summary_manifest": (
            output_dir / "seed-unresolved-frontier-summary-manifest.json"
        ),
        "seed_command_plan": output_dir / "seed-frontier-command-plan.json",
        "seed_command_plan_manifest": output_dir / "seed-frontier-command-plan-manifest.json",
        "seed_binding_stage_dir": output_dir / "seed-binding-stage",
        "seed_binding_stage_report": output_dir / "seed-binding-stage" / "seed-binding-stage.json",
        "seed_staged_bindings": output_dir / "seed-binding-stage" / "seed-staged-bindings.json",
        "seed_binding_records": output_dir / "seed-binding-stage" / "seed-binding-records.jsonl",
        "seed_binding_stage_manifest": output_dir / "seed-binding-stage" / "artifact-manifest.json",
        "unresolved_summary": fixture_dir / "unresolved-frontier-summary.json",
        "command_plan": output_dir / "frontier-queue-execute-command-plan.json",
        "command_plan_manifest": output_dir / "frontier-queue-execute-command-plan-manifest.json",
        "binding_scaffold": output_dir / "frontier-queue-execute-binding-scaffold.json",
        "binding_skeleton": output_dir / "frontier-queue-execute-binding-skeleton.json",
        "binding_scaffold_manifest": output_dir / "frontier-queue-execute-binding-scaffold-manifest.json",
        "staged_bindings": output_dir / "frontier-queue-execute-staged-bindings.json",
        "staged_bindings_manifest": output_dir / "frontier-queue-execute-staged-bindings-manifest.json",
        "bound_command_plan": output_dir / "frontier-queue-execute-bound-command-plan.json",
        "bound_command_plan_manifest": output_dir / "frontier-queue-execute-bound-command-plan-manifest.json",
        "review_decisions": output_dir / "frontier-queue-execute-review-decisions.jsonl",
        "review_dir": review_dir,
        "approved_bindings": review_dir / "frontier-queue-execute-approved-bindings.json",
        "approved_bound_command_plan": output_dir / "frontier-queue-execute-approved-bound-command-plan.json",
        "approved_bound_command_plan_manifest": (
            output_dir / "frontier-queue-execute-approved-bound-command-plan-manifest.json"
        ),
        "dry_run_report": output_dir / "frontier-queue-execution-dry-run.json",
        "dry_run_manifest": output_dir / "frontier-queue-execution-dry-run-manifest.json",
        "smoke_report": output_dir / "frontier-queue-execution-smoke.json",
        "artifact_manifest": output_dir / "artifact-manifest.json",
    }


def _write_source_fixture(paths: Mapping[str, Path], *, compact: bool) -> None:
    fixture_dir = paths["fixture_dir"]
    fixture_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        paths["source_command_plan"],
        {
            "schema_version": 1,
            "workflow": "frontier_research_queue_command_plan",
            "status": "ready",
            "entries": (),
            "metadata": {"fixture": "frontier_queue_execution_smoke"},
        },
        compact=compact,
    )
    _write_json(
        paths["source_approved_bindings"],
        {
            "schema_version": 1,
            "workflow": "frontier_research_queue_command_bindings",
            "status": "approved",
            "bindings": {},
            "metadata": {"fixture": "frontier_queue_execution_smoke"},
        },
        compact=compact,
    )
    _write_json(
        paths["source_bound_command_plan"],
        {
            "schema_version": 1,
            "workflow": "frontier_research_queue_bound_command_plan",
            "status": "ready",
            "source": {"command_plan": str(paths["source_command_plan"])},
            "entries": (),
            "metadata": {"fixture": "frontier_queue_execution_smoke"},
        },
        compact=compact,
    )
    _write_json(
        paths["source_binding_review"],
        {
            "schema_version": 1,
            "workflow": "frontier_research_queue_command_binding_review",
            "status": "ready_for_execution",
            "source": {"bound_command_plan": str(paths["source_bound_command_plan"])},
            "paths": {"approved_bindings": str(paths["source_approved_bindings"])},
            "metadata": {
                "fixture": "frontier_queue_execution_smoke",
                "not_verifier_evidence": True,
            },
        },
        compact=compact,
    )
    _write_jsonl(
        paths["source_seed_sidecar"],
        (
            {
                "task_id": "catalog-news-alpha",
                "collection_task_id": "catalog-news-alpha",
                "action_id": "run_source_family_catalog_adapters",
                "input_name": "source_family_url_seeds",
                "source_family": "news",
                "provider": "seeded_news",
                "seed_key": "news:catalog-news-alpha",
                "url": "https://example.com/news-alpha",
                "review_status": "approved",
                "not_verifier_evidence": True,
            },
        ),
        compact=compact,
    )
    _write_json(
        paths["source_input_binding_audit"],
        {
            "schema_version": 1,
            "workflow": "frontier_research_queue_input_binding_audit",
            "status": "ready",
            "summary": {
                "sidecar_counts": {"source_family_url_seeds": 1},
                "sidecar_status_counts": {
                    "source_family_url_seeds": {"ready": 1, "blocked": 0},
                },
                "ready_by_sidecar": {"source_family_url_seeds": 1},
                "blocked_by_sidecar": {},
            },
            "paths": {"source_family_url_seeds": str(paths["source_seed_sidecar"])},
            "metadata": {
                "fixture": "frontier_queue_execution_smoke",
                "not_verifier_evidence": True,
            },
        },
        compact=compact,
    )
    _write_json(
        paths["source_seed_base_bindings"],
        {
            "schema_version": 1,
            "workflow": "frontier_research_queue_command_bindings",
            "status": "needs_review",
            "inputs": {},
            "bindings": {
                "run_source_family_catalog_adapters": {
                    "bound_commands": (
                        "python benchmarks/run_seeded_url_source_family_catalog_adapter.py "
                        "--tasks tasks.jsonl --source-family news --seeds ... "
                        "--output seeded.jsonl --report-json seeded.json "
                        "--artifact-manifest manifest.json",
                    ),
                    "review_status": "needs_review",
                    "required_inputs": ("source_family_url_seeds",),
                },
            },
            "metadata": {
                "fixture": "frontier_queue_execution_smoke",
                "not_verifier_evidence": True,
            },
        },
        compact=compact,
    )
    _write_json(
        paths["unresolved_summary"],
        {
            "schema_version": 1,
            "workflow": "unresolved_frontier_evidence_summary",
            "status": "needs_evidence",
            "paths": {
                "frontier_command_binding_reviews": [str(paths["source_binding_review"])]
            },
            "next_actions": (
                {
                    "action_id": "execute_reviewed_frontier_queue_command_plan",
                    "lane": "frontier_queue_execution",
                    "priority": 83,
                    "reason": "smoke fixture has a reviewed command plan awaiting execution",
                    "ready_review_count": 1,
                    "dry_run_report_count": 0,
                    "command_count": 2,
                },
            ),
            "metadata": {
                "fixture": "frontier_queue_execution_smoke",
                "not_verifier_evidence": True,
            },
        },
        compact=compact,
    )


def _assert_smoke_chain(
    *,
    seed_summary: Mapping[str, Any],
    seed_command_plan: Mapping[str, Any],
    seed_binding_stage: Mapping[str, Any],
    staged: Mapping[str, Any],
    rebound: Mapping[str, Any],
    review: Mapping[str, Any],
    approved_rebound: Mapping[str, Any],
    dry_run: Mapping[str, Any],
) -> None:
    seed_action_ids = tuple(
        str(action.get("action_id") or "")
        for action in seed_summary.get("next_actions", ())
        if isinstance(action, Mapping)
    )
    if "stage_frontier_queue_seed_inputs" not in seed_action_ids:
        raise AssertionError("frontier queue smoke did not surface seed input staging.")
    seed_entries = tuple(
        item for item in seed_command_plan.get("entries", ()) if isinstance(item, Mapping)
    )
    if _nested(seed_command_plan, "summary", "entry_count") != 1:
        raise AssertionError("frontier queue smoke seed command plan did not isolate one action.")
    if not seed_entries or seed_entries[0].get("action_id") != "stage_frontier_queue_seed_inputs":
        raise AssertionError("frontier queue smoke seed command plan selected the wrong action.")
    if seed_binding_stage.get("status") != "ready_for_binding_review":
        raise AssertionError("frontier queue smoke seed binding stage is not ready for review.")
    if _nested(seed_binding_stage, "summary", "applied_placeholder_count") != 1:
        raise AssertionError("frontier queue smoke seed binding stage did not bind the seed placeholder.")
    if _nested(seed_binding_stage, "label_usage", "stage_executes_commands") is not False:
        raise AssertionError("frontier queue smoke seed binding stage unexpectedly executes commands.")
    if _nested(staged, "staging_summary", "staged_upstream_output_count") != 1:
        raise AssertionError("frontier queue smoke did not stage the upstream bound-plan output.")
    if _nested(staged, "staging_summary", "remaining_placeholder_count") != 0:
        raise AssertionError("frontier queue smoke left command placeholders unbound.")
    if rebound.get("status") != "ready":
        raise AssertionError("frontier queue smoke rebound plan is not ready.")
    if _nested(rebound, "summary", "unbound_placeholder_count") != 0:
        raise AssertionError("frontier queue smoke rebound plan has unbound placeholders.")
    if review.get("status") != "ready_for_execution":
        raise AssertionError("frontier queue smoke command-binding review did not approve execution.")
    if _nested(approved_rebound, "summary", "review_required_entry_count") != 0:
        raise AssertionError("frontier queue smoke approved plan still requires review.")
    if dry_run.get("status") != "dry_run":
        raise AssertionError("frontier queue smoke run report is not a dry-run report.")
    if _nested(dry_run, "summary", "dry_run_count") != 2:
        raise AssertionError("frontier queue smoke did not dry-run both child commands.")
    if _nested(dry_run, "summary", "binding_not_reviewed_count") != 0:
        raise AssertionError("frontier queue smoke dry-run was blocked by binding review.")


def _smoke_report(
    paths: Mapping[str, Path],
    *,
    seed_summary: Mapping[str, Any],
    seed_command_plan: Mapping[str, Any],
    seed_binding_stage: Mapping[str, Any],
    staged: Mapping[str, Any],
    rebound: Mapping[str, Any],
    review: Mapping[str, Any],
    approved_rebound: Mapping[str, Any],
    dry_run: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "workflow": "frontier_queue_execution_smoke",
        "status": "pass",
        "scope": (
            "Synthetic no-model fixture for frontier queue command execution. "
            "It validates source-family seed staging, command-plan staging, review "
            "gating, and dry-run execution without executing child frontier workflows "
            "or creating verifier evidence."
        ),
        "paths": {
            "seed_unresolved_summary": str(paths["seed_unresolved_summary"]),
            "seed_command_plan": str(paths["seed_command_plan"]),
            "seed_binding_stage_report": str(paths["seed_binding_stage_report"]),
            "seed_staged_bindings": str(paths["seed_staged_bindings"]),
            "unresolved_summary": str(paths["unresolved_summary"]),
            "command_plan": str(paths["command_plan"]),
            "binding_scaffold": str(paths["binding_scaffold"]),
            "staged_bindings": str(paths["staged_bindings"]),
            "bound_command_plan": str(paths["bound_command_plan"]),
            "review_report": str(_nested(review, "paths", "report")),
            "approved_bindings": str(paths["approved_bindings"]),
            "approved_bound_command_plan": str(paths["approved_bound_command_plan"]),
            "dry_run_report": str(paths["dry_run_report"]),
            "artifact_manifest": str(paths["artifact_manifest"]),
        },
        "summary": {
            "seed_next_action_count": len(tuple(seed_summary.get("next_actions", ()))),
            "seed_command_plan_entry_count": _nested(
                seed_command_plan,
                "summary",
                "entry_count",
            ),
            "seed_binding_status": seed_binding_stage.get("status"),
            "seed_applied_input_count": _nested(
                seed_binding_stage,
                "summary",
                "applied_input_count",
            ),
            "seed_applied_placeholder_count": _nested(
                seed_binding_stage,
                "summary",
                "applied_placeholder_count",
            ),
            "staged_upstream_output_count": _nested(
                staged,
                "staging_summary",
                "staged_upstream_output_count",
            ),
            "remaining_placeholder_count": _nested(
                staged,
                "staging_summary",
                "remaining_placeholder_count",
            ),
            "bound_ready_entry_count": _nested(rebound, "summary", "ready_entry_count"),
            "approved_ready_entry_count": _nested(
                approved_rebound,
                "summary",
                "ready_entry_count",
            ),
            "review_approved_entry_count": _nested(
                review,
                "summary",
                "approved_entry_count",
            ),
            "dry_run_count": _nested(dry_run, "summary", "dry_run_count"),
            "binding_not_reviewed_count": _nested(
                dry_run,
                "summary",
                "binding_not_reviewed_count",
            ),
        },
        "label_usage": {
            "labels_used": False,
            "model_answers_used": False,
            "artifacts_are_verifier_evidence": False,
            "executes_child_commands": False,
        },
    }


def _manifest_artifacts(
    paths: Mapping[str, Path],
    *,
    review: Mapping[str, Any],
) -> dict[str, Path | str | None]:
    return {
        "smoke_report": paths["smoke_report"],
        "unresolved_summary": paths["unresolved_summary"],
        "source_command_plan": paths["source_command_plan"],
        "source_approved_bindings": paths["source_approved_bindings"],
        "source_bound_command_plan": paths["source_bound_command_plan"],
        "source_binding_review": paths["source_binding_review"],
        "source_family_url_seeds": paths["source_seed_sidecar"],
        "source_input_binding_audit": paths["source_input_binding_audit"],
        "source_seed_base_bindings": paths["source_seed_base_bindings"],
        "seed_unresolved_summary": paths["seed_unresolved_summary"],
        "seed_unresolved_summary_manifest": paths["seed_unresolved_summary_manifest"],
        "seed_command_plan": paths["seed_command_plan"],
        "seed_command_plan_manifest": paths["seed_command_plan_manifest"],
        "seed_binding_stage_report": paths["seed_binding_stage_report"],
        "seed_staged_bindings": paths["seed_staged_bindings"],
        "seed_binding_records": paths["seed_binding_records"],
        "seed_binding_stage_manifest": paths["seed_binding_stage_manifest"],
        "command_plan": paths["command_plan"],
        "command_plan_manifest": paths["command_plan_manifest"],
        "binding_scaffold": paths["binding_scaffold"],
        "binding_skeleton": paths["binding_skeleton"],
        "binding_scaffold_manifest": paths["binding_scaffold_manifest"],
        "staged_bindings": paths["staged_bindings"],
        "staged_bindings_manifest": paths["staged_bindings_manifest"],
        "bound_command_plan": paths["bound_command_plan"],
        "bound_command_plan_manifest": paths["bound_command_plan_manifest"],
        "review_decisions": paths["review_decisions"],
        "binding_review_report": _nested(review, "paths", "report"),
        "binding_review_records": _nested(review, "paths", "review_records"),
        "binding_review_template": _nested(review, "paths", "review_template"),
        "binding_review_manifest": _nested(review, "paths", "artifact_manifest"),
        "approved_bindings": paths["approved_bindings"],
        "approved_bound_command_plan": paths["approved_bound_command_plan"],
        "approved_bound_command_plan_manifest": paths["approved_bound_command_plan_manifest"],
        "dry_run_report": paths["dry_run_report"],
        "dry_run_manifest": paths["dry_run_manifest"],
    }


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _write_json(path: str | Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = (
        strict_json_dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        if compact
        else strict_json_dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    output.write_text(text, encoding="utf-8")


def _write_jsonl(path: str | Path, rows: tuple[Mapping[str, Any], ...], *, compact: bool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = "".join(
            strict_json_dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    else:
        text = "".join(strict_json_dumps(row, sort_keys=True) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the no-model frontier queue execution smoke fixture"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="optional output directory; defaults to a temporary directory",
    )
    parser.add_argument("--name", default=SMOKE_NAME, help="registry record name")
    parser.add_argument("--version", default=SMOKE_VERSION, help="registry record version")
    parser.add_argument("--compact-json", action="store_true", help="write compact JSON")
    args = parser.parse_args(argv)
    if args.output_dir is not None:
        report = build_frontier_queue_execution_smoke(
            Path(args.output_dir),
            name=args.name,
            version=args.version,
            compact_json=bool(args.compact_json),
        )
        _print_report(report)
        return
    with tempfile.TemporaryDirectory(prefix="eigentruth-frontier-queue-execution-smoke-") as tmpdir:
        report = build_frontier_queue_execution_smoke(
            Path(tmpdir),
            name=args.name,
            version=args.version,
            compact_json=bool(args.compact_json),
        )
        _print_report(report)


def _print_report(report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    print(
        "frontier_queue_execution_smoke_ok "
        f"status={report['status']} "
        f"seed_bound={summary['seed_applied_placeholder_count']} "
        f"staged_upstream={summary['staged_upstream_output_count']} "
        f"dry_run={summary['dry_run_count']} "
        f"binding_not_reviewed={summary['binding_not_reviewed_count']} "
        f"record={report['registry_record']}"
    )


if __name__ == "__main__":
    main()
