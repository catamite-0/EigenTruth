"""Run controlled prompt ablations for evidence-conditioned belief revision."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.workflows.verification.belief_revision_real_model_eval import (
    DEFAULT_LABELS,
    DEFAULT_RUNTIME_EXAMPLES,
    HFTextGenerator,
    RuntimeExample,
    ScoringLabel,
    TextGenerator,
    _infer_model_family,
    _method_summary,
    _sha256_file,
    _sha256_text,
    load_runtime_examples,
    load_scoring_labels,
    score_generated_answer,
)
from benchmarks.workflows.verification.build_belief_revision_kill_test import (
    _strict_json_dumps,
)


@dataclass(frozen=True)
class AblationArm:
    name: str
    evidence_available: bool
    untrusted_draft_present: bool
    explicit_stance_instruction: bool
    template: str


ABLATION_ARMS = (
    AblationArm(
        name="closed_book_draft",
        evidence_available=False,
        untrusted_draft_present=True,
        explicit_stance_instruction=False,
        template="""Question:
{question}

Draft answer:
{initial_answer}

Return the final answer only.""",
    ),
    AblationArm(
        name="closed_book_explicit_review",
        evidence_available=False,
        untrusted_draft_present=True,
        explicit_stance_instruction=True,
        template="""Question:
{question}

Draft answer:
{initial_answer}

Using only your own knowledge, decide whether the draft is correct, incorrect, or uncertain.
Revise it if incorrect. Return the final answer only.""",
    ),
    AblationArm(
        name="evidence_only",
        evidence_available=True,
        untrusted_draft_present=False,
        explicit_stance_instruction=False,
        template="""Use only the supplied evidence to answer the question.
If it is insufficient, reply exactly: INSUFFICIENT EVIDENCE

Question:
{question}

Evidence:
{evidence}

Return the final answer only.""",
    ),
    AblationArm(
        name="evidence_with_draft",
        evidence_available=True,
        untrusted_draft_present=True,
        explicit_stance_instruction=False,
        template="""Use only the supplied evidence to answer the question. The draft may be useful or wrong.
If the evidence is insufficient, reply exactly: INSUFFICIENT EVIDENCE

Question:
{question}

Draft answer:
{initial_answer}

Evidence:
{evidence}

Return the final answer only.""",
    ),
    AblationArm(
        name="evidence_only_explicit_stance",
        evidence_available=True,
        untrusted_draft_present=False,
        explicit_stance_instruction=True,
        template="""Use only the supplied evidence. First determine internally whether it answers the question
or is insufficient. If insufficient, reply exactly: INSUFFICIENT EVIDENCE

Question:
{question}

Evidence:
{evidence}

Return the final answer only.""",
    ),
    AblationArm(
        name="evidence_draft_explicit_stance",
        evidence_available=True,
        untrusted_draft_present=True,
        explicit_stance_instruction=True,
        template="""Treat the draft as untrusted. Audit it against the supplied evidence.
Decide internally whether the evidence supports the draft, contradicts it, or is insufficient.
If contradicted, replace the incorrect answer with the evidence-grounded answer.
If insufficient, reply exactly: INSUFFICIENT EVIDENCE

Question:
{question}

Draft answer:
{initial_answer}

Evidence:
{evidence}

Return the revised final answer only.""",
    ),
)
ARM_BY_NAME = {arm.name: arm for arm in ABLATION_ARMS}

CONTRASTS = {
    "evidence_effect_without_explicit_stance": (
        "evidence_with_draft",
        "closed_book_draft",
    ),
    "evidence_effect_with_explicit_stance": (
        "evidence_draft_explicit_stance",
        "closed_book_explicit_review",
    ),
    "draft_anchoring_without_explicit_stance": (
        "evidence_with_draft",
        "evidence_only",
    ),
    "draft_anchoring_with_explicit_stance": (
        "evidence_draft_explicit_stance",
        "evidence_only_explicit_stance",
    ),
    "stance_effect_without_draft": (
        "evidence_only_explicit_stance",
        "evidence_only",
    ),
    "stance_effect_with_draft": (
        "evidence_draft_explicit_stance",
        "evidence_with_draft",
    ),
}


def build_ablation_prompt(example: RuntimeExample, arm_name: str) -> str:
    try:
        arm = ARM_BY_NAME[arm_name]
    except KeyError as exc:
        raise ValueError(f"unknown ablation arm: {arm_name}") from exc
    evidence = "\n".join(
        f"- {str(document.get('evidence_text', '')).strip()}"
        for document in example.evidence_docs
    )
    return arm.template.format(
        question=example.prompt,
        initial_answer=example.initial_answer,
        evidence=evidence,
    )


def build_ablation_report(
    *,
    examples: Sequence[RuntimeExample],
    labels: Sequence[ScoringLabel],
    generator: TextGenerator,
    model_id: str,
    model_family: str | None = None,
    runtime_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    split_name: str = "kill-test-v1",
) -> dict[str, Any]:
    """Generate every arm before labels are joined for alias scoring."""
    label_by_id = {label.example_id: label for label in labels}
    example_ids = tuple(example.example_id for example in examples)
    if set(example_ids) != set(label_by_id):
        raise ValueError("runtime and scoring-label example ids must match")

    generated: list[dict[str, Any]] = []
    for example in examples:
        for arm in ABLATION_ARMS:
            prompt = build_ablation_prompt(example, arm.name)
            answer = generator.generate(
                prompt,
                example_id=example.example_id,
                method=arm.name,
            ).strip()
            if not answer:
                raise ValueError(
                    f"model returned an empty answer for {example.example_id}/{arm.name}"
                )
            generated.append(
                {
                    "example_id": example.example_id,
                    "arm": arm.name,
                    "baseline_answer": example.initial_answer,
                    "revision_answer": answer,
                    "input_prompt_sha256": _sha256_text(prompt),
                    "output_sha256": _sha256_text(answer),
                }
            )

    results = [
        {
            **row,
            **score_generated_answer(
                answer=str(row["revision_answer"]),
                initial_answer=str(row["baseline_answer"]),
                label=label_by_id[str(row["example_id"])],
            ),
        }
        for row in generated
    ]
    summaries = {
        arm.name: _method_summary(
            [row for row in results if row.get("arm") == arm.name]
        )
        for arm in ABLATION_ARMS
    }
    metadata = dict(generator.metadata)
    report = {
        "schema_version": 1,
        "workflow": "belief_revision_mechanism_ablation",
        "model_id": model_id,
        "model_family": model_family or _infer_model_family(model_id),
        "model_revision": str(metadata.get("model_revision", "")).strip(),
        "arms": [
            {
                "name": arm.name,
                "evidence_available": arm.evidence_available,
                "untrusted_draft_present": arm.untrusted_draft_present,
                "explicit_stance_instruction": arm.explicit_stance_instruction,
            }
            for arm in ABLATION_ARMS
        ],
        "contrasts": _compute_contrasts(summaries, results),
        "summary": {
            "example_count": len(examples),
            "result_count": len(results),
            "by_arm": summaries,
            "by_arm_and_case_type": _summarize_by_arm_and_case_type(results),
        },
        "results": results,
        "generation": metadata,
        "dataset": {
            "split_name": split_name,
            "example_count": len(examples),
            "runtime_examples_sha256": (
                _sha256_file(runtime_path) if runtime_path is not None else None
            ),
            "scoring_labels_sha256": (
                _sha256_file(labels_path) if labels_path is not None else None
            ),
            "example_ids_sha256": _sha256_text("\n".join(example_ids) + "\n"),
        },
        "protocol": {
            "labels_separated_from_generation_inputs": True,
            "labels_passed_to_generator": False,
            "all_arms_generated": True,
            "semantic_adjudication_required_for_claim": True,
        },
        "prompt_template_sha256": {
            arm.name: _sha256_text(arm.template) for arm in ABLATION_ARMS
        },
    }
    return report


def build_blind_adjudication_packet(
    *,
    reports: Sequence[Mapping[str, Any]],
    examples: Sequence[RuntimeExample],
    salt: str,
) -> dict[str, Any]:
    """Create a deterministic packet with model, method, and alias scores hidden."""
    packet, _key = build_blind_adjudication_bundle(
        reports=reports,
        examples=examples,
        salt=salt,
    )
    return packet


def build_blind_adjudication_bundle(
    *,
    reports: Sequence[Mapping[str, Any]],
    examples: Sequence[RuntimeExample],
    salt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a public blind packet and a separate private unblinding key."""
    if not salt.strip():
        raise ValueError("a non-empty private blinding salt is required")
    example_by_id = {example.example_id: example for example in examples}
    items: list[dict[str, Any]] = []
    key_items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for report in reports:
        model_id = str(report.get("model_id", ""))
        model_revision = str(report.get("model_revision", ""))
        for row in report.get("results", ()):
            if not isinstance(row, Mapping):
                continue
            example_id = str(row.get("example_id", ""))
            arm = str(row.get("arm", ""))
            example = example_by_id.get(example_id)
            if example is None or arm not in ARM_BY_NAME:
                raise ValueError(f"unknown ablation result identity: {example_id}/{arm}")
            opaque_id = hashlib.sha256(
                f"{salt}\0{model_id}\0{example_id}\0{arm}".encode()
            ).hexdigest()[:24]
            if opaque_id in seen_ids:
                raise ValueError(f"duplicate adjudication id: {opaque_id}")
            seen_ids.add(opaque_id)
            items.append(
                {
                    "adjudication_id": opaque_id,
                    "question": example.prompt,
                    "draft_answer": example.initial_answer,
                    "evidence": [
                        str(document.get("evidence_text", "")).strip()
                        for document in example.evidence_docs
                    ],
                    "model_response": str(row.get("revision_answer", "")),
                }
            )
            key_items.append(
                {
                    "adjudication_id": opaque_id,
                    "model_id": model_id,
                    "model_revision": model_revision,
                    "example_id": example_id,
                    "arm": arm,
                    "alias_correction_success": bool(
                        row.get("correction_success", False)
                    ),
                    "alias_stubbornness": bool(row.get("stubbornness", False)),
                }
            )
    items.sort(key=lambda item: str(item["adjudication_id"]))
    key_items.sort(key=lambda item: str(item["adjudication_id"]))
    packet = {
        "schema_version": 1,
        "workflow": "belief_revision_blind_semantic_adjudication",
        "blinding": {
            "model_hidden": True,
            "arm_hidden": True,
            "alias_score_hidden": True,
            "source_example_id_hidden": True,
        },
        "rubric": {
            "semantic_verdict": ("correct", "incorrect", "appropriate_abstention", "unclear"),
            "draft_persistence": ("yes", "no", "unclear"),
            "required_adjudicators": 2,
            "adjudicators_must_work_independently": True,
        },
        "items": items,
    }
    key = {
        "schema_version": 1,
        "workflow": "belief_revision_blind_semantic_adjudication_key",
        "private_do_not_commit": True,
        "items": key_items,
    }
    return packet, key


def reconcile_blind_adjudications(
    *,
    annotation_sets: Sequence[Mapping[str, Any]],
    private_key: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate two independent annotation sets and report agreement before unblinding."""
    if len(annotation_sets) != 2:
        raise ValueError("exactly two independent annotation sets are required")
    key_by_id = {
        str(item.get("adjudication_id", "")): item
        for item in private_key.get("items", ())
        if isinstance(item, Mapping)
    }
    required_ids = set(key_by_id)
    if not required_ids:
        raise ValueError("private key contains no adjudication identities")
    normalized_sets: list[dict[str, Mapping[str, Any]]] = []
    adjudicators: list[str] = []
    valid_verdicts = {"correct", "incorrect", "appropriate_abstention", "unclear"}
    valid_persistence = {"yes", "no", "unclear"}
    for annotation_set in annotation_sets:
        adjudicator = str(annotation_set.get("adjudicator_id", "")).strip()
        if not adjudicator:
            raise ValueError("each annotation set requires an adjudicator_id")
        adjudicators.append(adjudicator)
        by_id: dict[str, Mapping[str, Any]] = {}
        for item in annotation_set.get("annotations", ()):
            if not isinstance(item, Mapping):
                raise ValueError("annotations must be objects")
            adjudication_id = str(item.get("adjudication_id", ""))
            if adjudication_id in by_id:
                raise ValueError(f"duplicate annotation id: {adjudication_id}")
            if item.get("semantic_verdict") not in valid_verdicts:
                raise ValueError(f"invalid semantic verdict for {adjudication_id}")
            if item.get("draft_persistence") not in valid_persistence:
                raise ValueError(f"invalid draft-persistence verdict for {adjudication_id}")
            by_id[adjudication_id] = item
        if set(by_id) != required_ids:
            raise ValueError("annotation ids must exactly match the private key")
        normalized_sets.append(by_id)
    if adjudicators[0] == adjudicators[1]:
        raise ValueError("adjudicators must be independent")

    rows: list[dict[str, Any]] = []
    for adjudication_id in sorted(required_ids):
        first = normalized_sets[0][adjudication_id]
        second = normalized_sets[1][adjudication_id]
        verdict_agreement = first["semantic_verdict"] == second["semantic_verdict"]
        persistence_agreement = (
            first["draft_persistence"] == second["draft_persistence"]
        )
        verdict = first["semantic_verdict"] if verdict_agreement else "disagreement"
        semantic_success = verdict in {"correct", "appropriate_abstention"}
        identity = key_by_id[adjudication_id]
        rows.append(
            {
                **identity,
                "semantic_verdict": verdict,
                "draft_persistence": (
                    first["draft_persistence"]
                    if persistence_agreement
                    else "disagreement"
                ),
                "semantic_correction_success": semantic_success,
                "verdict_agreement": verdict_agreement,
                "draft_persistence_agreement": persistence_agreement,
                "alias_semantic_disagreement": (
                    verdict_agreement
                    and bool(identity.get("alias_correction_success"))
                    != semantic_success
                ),
            }
        )
    return {
        "schema_version": 1,
        "workflow": "belief_revision_semantic_adjudication_reconciliation",
        "adjudicators": adjudicators,
        "item_count": len(rows),
        "verdict_agreement_rate": sum(
            bool(row["verdict_agreement"]) for row in rows
        )
        / len(rows),
        "draft_persistence_agreement_rate": sum(
            bool(row["draft_persistence_agreement"]) for row in rows
        )
        / len(rows),
        "unresolved_count": sum(
            not bool(row["verdict_agreement"])
            or not bool(row["draft_persistence_agreement"])
            for row in rows
        ),
        "alias_semantic_disagreement_count": sum(
            bool(row["alias_semantic_disagreement"]) for row in rows
        ),
        "results": rows,
    }


def evaluate_ablation_gate(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Decide whether the draft-bearing revision loop merits deeper intervention."""
    failures: list[str] = []
    if len(reports) < 2:
        failures.append("at least two real-model reports are required")
    families = {str(report.get("model_family", "")) for report in reports}
    if len(families) < 2:
        failures.append("at least two model families are required")
    dataset_fingerprints = {
        (
            str(report.get("dataset", {}).get("runtime_examples_sha256", "")),
            str(report.get("dataset", {}).get("scoring_labels_sha256", "")),
            str(report.get("dataset", {}).get("example_ids_sha256", "")),
        )
        for report in reports
    }
    if len(dataset_fingerprints) != 1 or any(
        not value for fingerprint in dataset_fingerprints for value in fingerprint
    ):
        failures.append("all reports must use the same fingerprinted dataset")
    generation_configs = {
        _strict_json_dumps(report.get("generation", {}).get("config", {}))
        for report in reports
    }
    if len(generation_configs) != 1:
        failures.append("all reports must use the same generation configuration")

    model_results: list[dict[str, Any]] = []
    for report in reports:
        model_id = str(report.get("model_id", ""))
        generation = report.get("generation", {})
        if (
            report.get("workflow") != "belief_revision_mechanism_ablation"
            or not generation.get("is_real_model")
            or generation.get("backend") != "transformers"
        ):
            failures.append(f"{model_id or '<unknown>'} is not a real-model report")
        if not str(report.get("model_revision", "")).strip():
            failures.append(f"{model_id or '<unknown>'} has no pinned model revision")
        arm_names = {
            str(arm.get("name", ""))
            for arm in report.get("arms", ())
            if isinstance(arm, Mapping)
        }
        if arm_names != set(ARM_BY_NAME):
            failures.append(f"{model_id or '<unknown>'} does not contain all ablation arms")
        primary = report.get("contrasts", {}).get(
            "draft_anchoring_with_explicit_stance", {}
        )
        model_results.append(
            {
                "model_id": model_id,
                "model_family": str(report.get("model_family", "")),
                "revision_loop_minus_no_draft_success": primary.get(
                    "correction_success_difference"
                ),
                "paired_bootstrap_95ci": primary.get(
                    "correction_success_paired_bootstrap_95ci"
                ),
                "paired_outcomes": primary.get("paired_outcomes"),
            }
        )
    if failures:
        decision = "INSUFFICIENT_EVIDENCE"
    elif any(
        float(result["revision_loop_minus_no_draft_success"]) <= 0.0
        for result in model_results
    ):
        decision = "PAUSE_DISTINCT_REVISION_LOOP"
    else:
        decision = "AWAIT_SEMANTIC_ADJUDICATION"
    return {
        "schema_version": 1,
        "workflow": "belief_revision_mechanism_ablation_gate",
        "decision": decision,
        "primary_comparison": (
            "evidence_draft_explicit_stance minus evidence_only_explicit_stance"
        ),
        "failures": failures,
        "models": model_results,
        "interpretation": (
            "This gate concerns the draft-bearing revision-loop mechanism only. "
            "It does not pause evidence-conditioned correction research."
        ),
    }


def write_ablation_report(
    *,
    generator: TextGenerator,
    model_id: str,
    output_path: str | Path,
    runtime_path: str | Path = DEFAULT_RUNTIME_EXAMPLES,
    labels_path: str | Path = DEFAULT_LABELS,
    model_family: str | None = None,
) -> dict[str, Any]:
    report = build_ablation_report(
        examples=load_runtime_examples(runtime_path),
        labels=load_scoring_labels(labels_path),
        generator=generator,
        model_id=model_id,
        model_family=model_family,
        runtime_path=runtime_path,
        labels_path=labels_path,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _strict_json_dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _compute_contrasts(
    summaries: Mapping[str, Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    outcome_by_arm_and_id = {
        (str(row.get("arm")), str(row.get("example_id"))): bool(
            row.get("correction_success", False)
        )
        for row in results
    }
    output: dict[str, Any] = {}
    for name, (treatment, control) in CONTRASTS.items():
        treatment_summary = summaries[treatment]
        control_summary = summaries[control]
        example_ids = sorted(
            example_id
            for arm, example_id in outcome_by_arm_and_id
            if arm == treatment
        )
        paired_differences = [
            int(outcome_by_arm_and_id[(treatment, example_id)])
            - int(outcome_by_arm_and_id[(control, example_id)])
            for example_id in example_ids
        ]
        output[name] = {
            "treatment": treatment,
            "control": control,
            "correction_success_difference": round(
                float(treatment_summary["correction_success_rate"])
                - float(control_summary["correction_success_rate"]),
                6,
            ),
            "stubbornness_difference": round(
                float(treatment_summary["stubbornness_rate"])
                - float(control_summary["stubbornness_rate"]),
                6,
            ),
            "paired_outcomes": {
                "treatment_wins": paired_differences.count(1),
                "ties": paired_differences.count(0),
                "treatment_losses": paired_differences.count(-1),
            },
            "correction_success_paired_bootstrap_95ci": _paired_bootstrap_interval(
                paired_differences
            ),
        }
    return output


def _paired_bootstrap_interval(
    differences: Sequence[int],
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> list[float]:
    if not differences:
        return [0.0, 0.0]
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    )
    lower_index = int(samples * 0.025)
    upper_index = min(samples - 1, int(samples * 0.975) - 1)
    return [round(estimates[lower_index], 6), round(estimates[upper_index], 6)]


def _summarize_by_arm_and_case_type(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    case_types = sorted(
        {str(row.get("case_type", "")) for row in results if row.get("case_type")}
    )
    return {
        arm.name: {
            case_type: _method_summary(
                [
                    row
                    for row in results
                    if row.get("arm") == arm.name
                    and row.get("case_type") == case_type
                ]
            )
            for case_type in case_types
        }
        for arm in ABLATION_ARMS
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-examples", type=Path, default=DEFAULT_RUNTIME_EXAMPLES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--model-family")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)
    generator = HFTextGenerator(
        args.model_id,
        revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
    )
    report = write_ablation_report(
        generator=generator,
        model_id=args.model_id,
        model_family=args.model_family,
        runtime_path=args.runtime_examples,
        labels_path=args.labels,
        output_path=args.json,
    )
    print(
        "belief_revision_mechanism_ablation_ok "
        f"model={args.model_id} examples={report['summary']['example_count']} "
        f"results={report['summary']['result_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
