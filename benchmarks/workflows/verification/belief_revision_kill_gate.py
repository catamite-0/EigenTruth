"""Aggregate real-model belief-revision reports into a strict go/pause decision."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_METHODS = (
    "baseline_prompt",
    "self_correction_prompt",
    "rag_evidence_only",
    "eigentruth_revision_loop",
)


@dataclass(frozen=True)
class KillGatePolicy:
    minimum_models: int = 2
    minimum_model_families: int = 2
    minimum_examples_per_model: int = 20
    minimum_stubbornness_reduction: float = 0.10
    minimum_correction_gain: float = 0.10
    maximum_regressed_models: int = 0
    require_qwen_and_non_qwen: bool = True


def evaluate_kill_gate(
    reports: Sequence[Mapping[str, Any]],
    *,
    policy: KillGatePolicy = KillGatePolicy(),
) -> dict[str, Any]:
    model_rows = tuple(_model_row(report) for report in reports)
    failures: list[str] = []
    if len(model_rows) < policy.minimum_models:
        failures.append(f"need at least {policy.minimum_models} distinct model reports")
    model_ids = {row["model_id"] for row in model_rows}
    if len(model_ids) != len(model_rows):
        failures.append("model reports must use distinct model_id values")
    model_families = {row["model_family"] for row in model_rows if row["model_family"]}
    if len(model_families) < policy.minimum_model_families:
        failures.append(
            f"need at least {policy.minimum_model_families} distinct model families"
        )
    if policy.require_qwen_and_non_qwen and model_rows:
        has_qwen = any(row["model_family"] == "qwen" for row in model_rows)
        has_non_qwen = any(row["model_family"] != "qwen" for row in model_rows)
        if not has_qwen or not has_non_qwen:
            failures.append("reports must include one Qwen and one non-Qwen model")

    dataset_fingerprints = {
        (
            row["runtime_examples_sha256"],
            row["scoring_labels_sha256"],
            row["example_ids_sha256"],
        )
        for row in model_rows
        if row["runtime_examples_sha256"]
        and row["scoring_labels_sha256"]
        and row["example_ids_sha256"]
    }
    if len(dataset_fingerprints) > 1:
        failures.append("model reports must use the same fingerprinted dataset split")
    prompt_fingerprints = {
        tuple(sorted(row["prompt_template_sha256"].items()))
        for row in model_rows
        if row["prompt_template_sha256"]
    }
    if len(prompt_fingerprints) > 1:
        failures.append("model reports must use the same prompt templates")
    decoding_configs = {
        tuple(sorted(row["decoding_config"].items()))
        for row in model_rows
        if row["decoding_config"]
    }
    if len(decoding_configs) > 1:
        failures.append("model reports must use the same decoding parameters")

    eligible = []
    for row in model_rows:
        failures.extend(
            f"{row['model_id']}: {failure}" for failure in row["integrity_failures"]
        )
        if row["example_count"] < policy.minimum_examples_per_model:
            failures.append(
                f"{row['model_id']}: need at least {policy.minimum_examples_per_model} examples"
            )
        if row["missing_methods"]:
            failures.append(
                f"{row['model_id']}: missing methods {', '.join(row['missing_methods'])}"
            )
        if (
            not row["missing_methods"]
            and not row["integrity_failures"]
            and row["example_count"] >= policy.minimum_examples_per_model
        ):
            eligible.append(row)

    if failures:
        decision = "INSUFFICIENT_EVIDENCE"
    else:
        regressed = sum(1 for row in eligible if row["stubbornness_reduction"] < 0)
        passes = all(
            row["stubbornness_reduction"] >= policy.minimum_stubbornness_reduction
            and row["correction_gain"] >= policy.minimum_correction_gain
            for row in eligible
        )
        decision = (
            "CONTINUE_0_3"
            if passes and regressed <= policy.maximum_regressed_models
            else "PAUSE_PROJECT"
        )

    return {
        "schema_version": 1,
        "workflow": "belief_revision_kill_gate",
        "decision": decision,
        "policy": {
            "minimum_models": policy.minimum_models,
            "minimum_model_families": policy.minimum_model_families,
            "minimum_examples_per_model": policy.minimum_examples_per_model,
            "minimum_stubbornness_reduction": policy.minimum_stubbornness_reduction,
            "minimum_correction_gain": policy.minimum_correction_gain,
            "maximum_regressed_models": policy.maximum_regressed_models,
            "require_qwen_and_non_qwen": policy.require_qwen_and_non_qwen,
        },
        "models": model_rows,
        "failures": failures,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--minimum-models", type=int, default=2)
    parser.add_argument("--minimum-model-families", type=int, default=2)
    parser.add_argument("--minimum-examples", type=int, default=20)
    parser.add_argument("--minimum-stubbornness-reduction", type=float, default=0.10)
    parser.add_argument("--minimum-correction-gain", type=float, default=0.10)
    args = parser.parse_args(argv)

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    result = evaluate_kill_gate(
        reports,
        policy=KillGatePolicy(
            minimum_models=args.minimum_models,
            minimum_model_families=args.minimum_model_families,
            minimum_examples_per_model=args.minimum_examples,
            minimum_stubbornness_reduction=args.minimum_stubbornness_reduction,
            minimum_correction_gain=args.minimum_correction_gain,
        ),
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["decision"] == "CONTINUE_0_3" else 2


def _model_row(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = _mapping(report.get("summary"))
    by_method = _mapping(summary.get("by_method"))
    missing = tuple(method for method in REQUIRED_METHODS if method not in by_method)
    self_correction = _mapping(by_method.get("self_correction_prompt"))
    eigentruth = _mapping(by_method.get("eigentruth_revision_loop"))
    example_count = int(summary.get("example_count", 0))
    generation = _mapping(report.get("generation"))
    generation_config = _mapping(generation.get("config"))
    dataset = _mapping(report.get("dataset"))
    protocol = _mapping(report.get("protocol"))
    prompt_template_sha256 = _mapping(report.get("prompt_template_sha256"))
    integrity_failures = _report_integrity_failures(
        report,
        example_count=example_count,
        by_method=by_method,
        generation=generation,
        dataset=dataset,
        protocol=protocol,
        prompt_template_sha256=prompt_template_sha256,
    )
    return {
        "model_id": str(report.get("model_id", "")).strip() or "unknown-model",
        "model_family": str(report.get("model_family", "")).strip().lower(),
        "model_revision": str(report.get("model_revision", "")).strip(),
        "example_count": example_count,
        "missing_methods": missing,
        "integrity_failures": integrity_failures,
        "runtime_examples_sha256": str(dataset.get("runtime_examples_sha256", "")),
        "scoring_labels_sha256": str(dataset.get("scoring_labels_sha256", "")),
        "example_ids_sha256": str(dataset.get("example_ids_sha256", "")),
        "prompt_template_sha256": dict(prompt_template_sha256),
        "decoding_config": {
            key: generation_config.get(key)
            for key in ("do_sample", "max_new_tokens", "seed", "temperature", "top_p")
            if key in generation_config
        },
        "self_correction_stubbornness_rate": float(
            self_correction.get("stubbornness_rate", 0.0)
        ),
        "eigentruth_stubbornness_rate": float(eigentruth.get("stubbornness_rate", 0.0)),
        "stubbornness_reduction": float(self_correction.get("stubbornness_rate", 0.0))
        - float(eigentruth.get("stubbornness_rate", 0.0)),
        "self_correction_success_rate": float(
            self_correction.get("correction_success_rate", 0.0)
        ),
        "eigentruth_correction_success_rate": float(
            eigentruth.get("correction_success_rate", 0.0)
        ),
        "correction_gain": float(eigentruth.get("correction_success_rate", 0.0))
        - float(self_correction.get("correction_success_rate", 0.0)),
    }


def _report_integrity_failures(
    report: Mapping[str, Any],
    *,
    example_count: int,
    by_method: Mapping[str, Any],
    generation: Mapping[str, Any],
    dataset: Mapping[str, Any],
    protocol: Mapping[str, Any],
    prompt_template_sha256: Mapping[str, Any],
) -> tuple[str, ...]:
    failures: list[str] = []
    if int(report.get("schema_version", 0)) < 2:
        failures.append("report schema_version must be at least 2")
    if report.get("workflow") != "belief_revision_real_model_eval":
        failures.append("report must come from belief_revision_real_model_eval")
    model_id = str(report.get("model_id", "")).strip()
    model_family = str(report.get("model_family", "")).strip()
    if not model_id:
        failures.append("model_id is required")
    if not model_family:
        failures.append("model_family is required")
    model_revision = str(report.get("model_revision", "")).strip()
    if not model_revision:
        failures.append("model_revision is required")
    if generation.get("backend") != "transformers" or generation.get("is_real_model") is not True:
        failures.append("generation must come from a real transformers model")
    if str(generation.get("model_id", "")).strip() != model_id:
        failures.append("generation model_id must match report model_id")
    if str(generation.get("model_revision", "")).strip() != model_revision:
        failures.append("generation model_revision must match report model_revision")
    generation_config = _mapping(generation.get("config"))
    for key in ("do_sample", "max_new_tokens", "seed", "temperature", "top_p"):
        if key not in generation_config:
            failures.append(f"generation config is missing {key}")
    if dataset.get("split_name") != "kill-test-v1":
        failures.append("dataset split_name must be kill-test-v1")
    if int(dataset.get("example_count", 0)) != example_count:
        failures.append("dataset and summary example counts differ")
    for key in (
        "runtime_examples_sha256",
        "scoring_labels_sha256",
        "example_ids_sha256",
    ):
        value = str(dataset.get(key, ""))
        if len(value) != 64:
            failures.append(f"dataset is missing a valid {key}")
    if dataset.get("evaluation_held_out_from_prompt_development") is not True:
        failures.append("dataset must be held out from prompt development")
    if protocol.get("labels_separated_from_generation_inputs") is not True:
        failures.append("scoring labels must be separated from generation inputs")
    if protocol.get("labels_passed_to_generator") is not False:
        failures.append("labels_passed_to_generator must be false")
    if protocol.get("runtime_validation_passed") is not True:
        failures.append("runtime leakage validation must pass")
    if protocol.get("all_methods_generated") is not True:
        failures.append("all four methods must be model-generated")
    if set(prompt_template_sha256) != set(REQUIRED_METHODS):
        failures.append("prompt template hashes must cover all required methods")

    results = tuple(
        row for row in _sequence(report.get("results")) if isinstance(row, Mapping)
    )
    if len(results) != example_count * len(REQUIRED_METHODS):
        failures.append("result count must equal examples multiplied by methods")
    for method in REQUIRED_METHODS:
        method_summary = _mapping(by_method.get(method))
        if int(method_summary.get("count", -1)) != example_count:
            failures.append(f"{method}: summary count does not match example count")
        rows = [row for row in results if row.get("method") == method]
        ids = [str(row.get("example_id", "")) for row in rows]
        if len(rows) != example_count:
            failures.append(f"{method}: result count does not match example count")
        if len(set(ids)) != len(ids) or any(not item for item in ids):
            failures.append(f"{method}: example ids must be non-empty and unique")
        if any(not str(row.get("revision_answer", "")).strip() for row in rows):
            failures.append(f"{method}: generated answers must be non-empty")
        if any(len(str(row.get("input_prompt_sha256", ""))) != 64 for row in rows):
            failures.append(f"{method}: prompt fingerprints are missing")
        if any(len(str(row.get("output_sha256", ""))) != 64 for row in rows):
            failures.append(f"{method}: output fingerprints are missing")
    return tuple(dict.fromkeys(failures))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


if __name__ == "__main__":
    raise SystemExit(main())
