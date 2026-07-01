"""Shared CLI requirement checks for frontier research command handoffs."""

from __future__ import annotations

import shlex
from typing import Any, Mapping, Sequence

REQUIRED_COMMAND_FLAGS = {
    "benchmarks/compare_frontier_release_evidence.py": (
        "--verifier-stability-report",
        "--abstention-stability-report",
        "--json",
    ),
    "benchmarks/compare_product_runtime_baselines.py": ("--current", "--json"),
    "benchmarks/eval_abstention_stability.py": ("--scores", "--signals", "--json"),
    "benchmarks/export_product_promotion_contract_evidence_handoff.py": (
        "--contract",
        "--json",
        "--audit-json",
    ),
    "benchmarks/plan_frontier_abstention_evidence_reruns.py": ("--source",),
    "benchmarks/run_product_runtime_baseline.py": ("--trace", "--json"),
    "benchmarks/rollup_frontier_abstention_evidence_reruns.py": ("--queue", "--json"),
}

REQUIRED_INPUT_FLAGS = {
    "benchmarks/compare_frontier_release_evidence.py": {
        "verifier_stability_report": "--verifier-stability-report",
        "abstention_stability_report": "--abstention-stability-report",
    },
    "benchmarks/compare_product_runtime_baselines.py": {
        "baseline_product_runtime_report": "--baseline",
    },
    "benchmarks/export_product_promotion_contract_evidence_handoff.py": {
        "product_promotion_contract_source": "--contract",
    },
    "benchmarks/plan_frontier_abstention_evidence_reruns.py": {
        "frontier_release_report_or_evidence_gap_plan": "--source",
        "abstention_score_dump_paths": "--scores",
        "abstention_signal_groups": "--signal-groups",
    },
    "benchmarks/run_product_runtime_baseline.py": {
        "product_trace_corpus": "--trace",
        "product_promotion_contract_source": "--promotion-contract",
    },
}


def frontier_command_requirement_summary(
    command: str,
    *,
    index: int,
    required_inputs: Sequence[str] = (),
) -> dict[str, Any]:
    """Return known CLI requirements for a frontier command template or command."""
    try:
        argv = tuple(shlex.split(command))
    except ValueError as exc:
        return {
            "command_index": index,
            "script": None,
            "known_command": False,
            "parse_error": str(exc),
            "required_flags": (),
            "missing_required_flags": (),
            "required_input_flags": (),
            "missing_required_input_flags": (),
            "status": "invalid_command",
        }
    script = frontier_command_script(argv)
    required_flags = REQUIRED_COMMAND_FLAGS.get(script or "", ())
    required_input_flags = _required_input_flags(
        script,
        required_inputs=required_inputs,
    )
    missing_required_flags = tuple(flag for flag in required_flags if flag not in argv)
    missing_required_input_flags = tuple(
        item for item in required_input_flags if item["flag"] not in argv
    )
    known_command = bool(required_flags or required_input_flags)
    if missing_required_flags or missing_required_input_flags:
        status = "needs_review"
    elif known_command:
        status = "ready"
    else:
        status = "unknown"
    return {
        "command_index": index,
        "script": script,
        "known_command": known_command,
        "required_flags": required_flags,
        "missing_required_flags": missing_required_flags,
        "required_input_flags": required_input_flags,
        "missing_required_input_flags": missing_required_input_flags,
        "status": status,
    }


def frontier_command_validation_issues(
    command: str,
    *,
    index: int,
    required_inputs: Sequence[str] = (),
    ignore_placeholders: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Return fail-closed validation issues for a concrete bound command."""
    if ignore_placeholders and "..." in command:
        return ()
    summary = frontier_command_requirement_summary(
        command,
        index=index,
        required_inputs=required_inputs,
    )
    script = summary.get("script")
    if summary.get("parse_error") or script is None:
        return ()
    issues = []
    missing_required_flags = _string_tuple(summary.get("missing_required_flags"))
    if missing_required_flags:
        issues.append({
            "command_index": index,
            "script": script,
            "issue": "missing_required_cli_flags",
            "missing_flags": missing_required_flags,
        })
    missing_required_input_flags = tuple(
        item
        for item in _mapping_sequence(summary.get("missing_required_input_flags"))
        if item.get("flag")
    )
    if missing_required_input_flags:
        issues.append({
            "command_index": index,
            "script": script,
            "issue": "required_input_not_bound_to_command_flags",
            "required_inputs": tuple(str(item["input"]) for item in missing_required_input_flags),
            "missing_flags": tuple(str(item["flag"]) for item in missing_required_input_flags),
        })
    return tuple(issues)


def validate_frontier_bound_commands(
    commands: Sequence[str],
    *,
    required_inputs: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate concrete bound frontier commands against known script requirements."""
    issues = []
    for index, command in enumerate(commands, start=1):
        issues.extend(
            frontier_command_validation_issues(
                command,
                index=index,
                required_inputs=required_inputs,
            )
        )
    return {
        "issue_count": len(issues),
        "issues": tuple(issues),
    }


def frontier_command_script(argv: Sequence[str]) -> str | None:
    """Return a normalized benchmark script path from a command argv."""
    for item in argv:
        text = str(item)
        if text.endswith(".py"):
            return _normalize_script_path(text)
    return None


def _required_input_flags(
    script: str | None,
    *,
    required_inputs: Sequence[str],
) -> tuple[Mapping[str, str], ...]:
    if not script or not required_inputs:
        return ()
    configured = REQUIRED_INPUT_FLAGS.get(script, {})
    required = {str(item) for item in required_inputs if str(item)}
    return tuple(
        {"input": input_name, "flag": flag}
        for input_name, flag in configured.items()
        if input_name in required
    )


def _normalize_script_path(path: str) -> str:
    text = str(path).replace("\\", "/")
    if "/benchmarks/" in text:
        return "benchmarks/" + text.rsplit("/benchmarks/", 1)[1]
    while text.startswith("./"):
        text = text[2:]
    return text


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return (str(value),) if str(value) else ()
    return tuple(str(item) for item in value if str(item))
