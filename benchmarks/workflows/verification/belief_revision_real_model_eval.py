"""Run the label-separated belief-revision kill-test with a real HF model."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.lib.paths import ensure_repo_root_on_path
from benchmarks.workflows.verification.build_belief_revision_kill_test import (
    DEFAULT_LABELS,
    DEFAULT_RUNTIME_EXAMPLES,
    FORBIDDEN_RUNTIME_FIELDS,
    _build_file_artifact_manifest,
    _find_forbidden_fields,
    _strict_json_dumps,
)

REPO_ROOT = ensure_repo_root_on_path()

METHODS = (
    "baseline_prompt",
    "self_correction_prompt",
    "rag_evidence_only",
    "eigentruth_revision_loop",
)
ABSTENTION_MARKERS = (
    "insufficient evidence",
    "cannot determine",
    "cannot be determined",
    "not enough evidence",
    "unable to determine",
)
PROMPT_TEMPLATES = {
    "baseline_prompt": """Question:
{question}

Draft answer:
{initial_answer}

Return the final answer only.""",
    "self_correction_prompt": """Question:
{question}

Draft answer:
{initial_answer}

Review the draft for factual errors using only your own knowledge. Revise it if needed.
Return the final answer only.""",
    "rag_evidence_only": """Use only the supplied evidence to answer the question. Do not use a draft answer.
If the evidence does not answer the question, reply exactly: INSUFFICIENT EVIDENCE

Question:
{question}

Evidence:
{evidence}

Return the final answer only.""",
    "eigentruth_revision_loop": """Treat the draft as untrusted. Audit it against the supplied evidence.
Decide whether the evidence supports the draft, contradicts it, or is insufficient.
If contradicted, replace the incorrect answer with the evidence-grounded answer.
If insufficient, reply exactly: INSUFFICIENT EVIDENCE

Question:
{question}

Draft answer:
{initial_answer}

Evidence:
{evidence}

Return the revised final answer only.""",
}


@dataclass(frozen=True)
class RuntimeExample:
    example_id: str
    prompt: str
    initial_answer: str
    claims: Sequence[str]
    evidence_docs: Sequence[Mapping[str, Any]]
    language: str = "en"
    risk_category: str = "factual_conflict"
    source_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("RuntimeExample.example_id must be non-empty.")
        if not self.prompt.strip():
            raise ValueError("RuntimeExample.prompt must be non-empty.")
        if not self.initial_answer.strip():
            raise ValueError("RuntimeExample.initial_answer must be non-empty.")
        if not self.evidence_docs:
            raise ValueError("RuntimeExample.evidence_docs must be non-empty.")
        object.__setattr__(self, "claims", tuple(str(item) for item in self.claims))
        object.__setattr__(self, "evidence_docs", tuple(dict(item) for item in self.evidence_docs))
        object.__setattr__(self, "source_provenance", dict(self.source_provenance))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeExample":
        forbidden = _find_forbidden_fields(payload)
        if forbidden:
            raise ValueError(
                "runtime example contains scoring-only fields: "
                + ", ".join(sorted(forbidden))
            )
        return cls(
            example_id=str(payload.get("example_id", "")),
            prompt=str(payload.get("prompt", "")),
            initial_answer=str(payload.get("initial_answer", "")),
            claims=tuple(str(item) for item in _sequence(payload.get("claims"))),
            evidence_docs=tuple(
                dict(item)
                for item in _sequence(payload.get("evidence_docs"))
                if isinstance(item, Mapping)
            ),
            language=str(payload.get("language", "en")),
            risk_category=str(payload.get("risk_category", "factual_conflict")),
            source_provenance=_mapping(payload.get("source_provenance")),
        )


@dataclass(frozen=True)
class ScoringLabel:
    example_id: str
    case_type: str
    expected_action: str
    expected_revision: str
    accepted_answers: Sequence[str]
    rejected_answers: Sequence[str]
    risk_category: str

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("ScoringLabel.example_id must be non-empty.")
        if self.expected_action not in {"accept", "revise", "abstain"}:
            raise ValueError(f"unsupported expected_action: {self.expected_action}")
        if not self.accepted_answers:
            raise ValueError("ScoringLabel.accepted_answers must be non-empty.")
        object.__setattr__(
            self,
            "accepted_answers",
            tuple(str(item) for item in self.accepted_answers if str(item).strip()),
        )
        object.__setattr__(
            self,
            "rejected_answers",
            tuple(str(item) for item in self.rejected_answers if str(item).strip()),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScoringLabel":
        return cls(
            example_id=str(payload.get("example_id", "")),
            case_type=str(payload.get("case_type", "")),
            expected_action=str(payload.get("expected_action", "")),
            expected_revision=str(payload.get("expected_revision", "")),
            accepted_answers=tuple(
                str(item) for item in _sequence(payload.get("accepted_answers"))
            ),
            rejected_answers=tuple(
                str(item) for item in _sequence(payload.get("rejected_answers"))
            ),
            risk_category=str(payload.get("risk_category", "factual_conflict")),
        )


class TextGenerator(Protocol):
    metadata: Mapping[str, Any]

    def generate(self, prompt: str, *, example_id: str, method: str) -> str:
        """Generate one answer from one sanitized prompt."""


class CallableTextGenerator:
    """Small deterministic adapter used by unit tests and offline plumbing checks."""

    def __init__(
        self,
        callback: Callable[[str, str, str], str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.callback = callback
        self.metadata = {
            "backend": "test-double",
            "is_real_model": False,
            **({} if metadata is None else dict(metadata)),
        }

    def generate(self, prompt: str, *, example_id: str, method: str) -> str:
        return str(self.callback(prompt, example_id, method))


class HFTextGenerator:
    """Optional Hugging Face causal-LM generation adapter."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "float32",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional real-model environment.
            raise RuntimeError(
                "Real-model evaluation requires torch and transformers; "
                "install the eigentruth[hf] optional dependencies."
            ) from exc
        self._torch = torch
        self.model_id = model_id
        self.seed = int(seed)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        tokenizer_kwargs: dict[str, Any] = {}
        model_kwargs: dict[str, Any] = {}
        if revision:
            tokenizer_kwargs["revision"] = revision
            model_kwargs["revision"] = revision
        if trust_remote_code:
            tokenizer_kwargs["trust_remote_code"] = True
            model_kwargs["trust_remote_code"] = True
        if dtype != "auto":
            model_kwargs["dtype"] = _torch_dtype(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.device = _resolve_device(torch, device)
        self.model.to(self.device).eval()
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        resolved_revision = (
            getattr(getattr(self.model, "config", None), "_commit_hash", None)
            or revision
            or ""
        )
        self.metadata = {
            "backend": "transformers",
            "is_real_model": True,
            "model_id": model_id,
            "model_revision": str(resolved_revision),
            "tokenizer_class": type(self.tokenizer).__name__,
            "model_class": type(self.model).__name__,
            "device": str(self.device),
            "dtype": dtype,
            "transformers_version": _module_version("transformers"),
            "torch_version": _module_version("torch"),
            "config": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
                "do_sample": self.temperature > 0.0,
                "trust_remote_code": bool(trust_remote_code),
            },
        }

    def generate(self, prompt: str, *, example_id: str, method: str) -> str:
        rendered = _render_chat_prompt(self.tokenizer, prompt)
        encoded = self.tokenizer(rendered, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_width = int(encoded["input_ids"].shape[-1])
        generation_seed = _generation_seed(self.seed, example_id, method)
        random.seed(generation_seed)
        self._torch.manual_seed(generation_seed)
        if self._torch.cuda.is_available():
            self._torch.cuda.manual_seed_all(generation_seed)
        kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0.0:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        with self._torch.no_grad():
            output = self.model.generate(**kwargs)
        generated_ids = output[0, input_width:]
        return str(self.tokenizer.decode(generated_ids, skip_special_tokens=True)).strip()


def load_runtime_examples(path: str | Path) -> tuple[RuntimeExample, ...]:
    rows = _read_jsonl(path)
    examples = tuple(RuntimeExample.from_dict(row) for row in rows)
    _require_unique_ids((item.example_id for item in examples), source="runtime examples")
    return examples


def load_scoring_labels(path: str | Path) -> tuple[ScoringLabel, ...]:
    rows = _read_jsonl(path)
    labels = tuple(ScoringLabel.from_dict(row) for row in rows)
    _require_unique_ids((item.example_id for item in labels), source="scoring labels")
    return labels


def build_real_model_report(
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
    """Generate first from sanitized inputs, then join scoring labels by id."""
    label_by_id = {label.example_id: label for label in labels}
    example_ids = tuple(example.example_id for example in examples)
    if set(example_ids) != set(label_by_id):
        missing_labels = sorted(set(example_ids) - set(label_by_id))
        missing_examples = sorted(set(label_by_id) - set(example_ids))
        raise ValueError(
            f"runtime/label id mismatch: missing_labels={missing_labels}, "
            f"missing_examples={missing_examples}"
        )

    generated: list[dict[str, Any]] = []
    for example in examples:
        for method in METHODS:
            prompt = build_method_prompt(example, method)
            answer = generator.generate(
                prompt,
                example_id=example.example_id,
                method=method,
            ).strip()
            if not answer:
                raise ValueError(
                    f"model returned an empty answer for {example.example_id}/{method}"
                )
            generated.append(
                {
                    "example_id": example.example_id,
                    "method": method,
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
    summary = _summarize_results(results)
    generator_metadata = dict(generator.metadata)
    model_revision = str(generator_metadata.get("model_revision", "")).strip()
    prompt_hashes = {
        method: _sha256_text(PROMPT_TEMPLATES[method]) for method in METHODS
    }
    report = {
        "schema_version": 2,
        "workflow": "belief_revision_real_model_eval",
        "model_id": model_id,
        "model_family": model_family or _infer_model_family(model_id),
        "model_revision": model_revision,
        "methods": METHODS,
        "summary": summary,
        "results": results,
        "generation": generator_metadata,
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
            "evaluation_held_out_from_prompt_development": True,
            "pretraining_exclusion_claimed": False,
        },
        "protocol": {
            "labels_separated_from_generation_inputs": True,
            "labels_passed_to_generator": False,
            "runtime_validation_passed": True,
            "generation_input_fields": (
                "prompt",
                "initial_answer",
                "evidence_docs[].evidence_text",
            ),
            "forbidden_runtime_fields": tuple(sorted(FORBIDDEN_RUNTIME_FIELDS)),
            "all_methods_generated": True,
        },
        "prompt_template_sha256": prompt_hashes,
    }
    return report


def write_real_model_report(
    *,
    generator: TextGenerator,
    model_id: str,
    runtime_path: str | Path = DEFAULT_RUNTIME_EXAMPLES,
    labels_path: str | Path = DEFAULT_LABELS,
    output_path: str | Path,
    artifact_manifest_path: str | Path | None = None,
    model_family: str | None = None,
) -> dict[str, Any]:
    examples = load_runtime_examples(runtime_path)
    labels = load_scoring_labels(labels_path)
    report = build_real_model_report(
        examples=examples,
        labels=labels,
        generator=generator,
        model_id=model_id,
        model_family=model_family,
        runtime_path=runtime_path,
        labels_path=labels_path,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _strict_json_dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if artifact_manifest_path is not None:
        manifest = _build_file_artifact_manifest(
            {
                "runtime_examples": Path(runtime_path),
                "scoring_labels": Path(labels_path),
                "real_model_report": output_path,
            },
            root=REPO_ROOT,
            metadata={
                "workflow": report["workflow"],
                "model_id": model_id,
                "model_revision": report["model_revision"],
                "example_count": report["summary"]["example_count"],
                "labels_separated_from_generation_inputs": True,
            },
        )
        manifest_path = Path(artifact_manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            _strict_json_dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def build_method_prompt(example: RuntimeExample, method: str) -> str:
    if method not in PROMPT_TEMPLATES:
        raise ValueError(f"unknown belief-revision method: {method}")
    evidence = "\n".join(
        f"- {str(document.get('evidence_text', '')).strip()}"
        for document in example.evidence_docs
    )
    return PROMPT_TEMPLATES[method].format(
        question=example.prompt,
        initial_answer=example.initial_answer,
        evidence=evidence,
    )


def score_generated_answer(
    *,
    answer: str,
    initial_answer: str,
    label: ScoringLabel,
) -> dict[str, Any]:
    accepted = any(_contains_alias(answer, item) for item in label.accepted_answers)
    rejected = any(_contains_alias(answer, item) for item in label.rejected_answers)
    abstained = any(marker in _normalize_text(answer) for marker in ABSTENTION_MARKERS)
    if label.expected_action == "abstain":
        correction_success = accepted or (abstained and not rejected)
        evidence_uptake = correction_success
    else:
        correction_success = accepted and not rejected
        evidence_uptake = accepted
    stubbornness = label.expected_action in {"revise", "abstain"} and (
        _normalize_text(answer) == _normalize_text(initial_answer) or rejected
    )
    return {
        "stubbornness": stubbornness,
        "unsupported_persistence": rejected,
        "evidence_uptake": evidence_uptake,
        "correction_success": correction_success,
        "abstention_quality": (
            "appropriate"
            if label.expected_action == "abstain" and correction_success
            else "excessive"
            if label.expected_action != "abstain" and abstained
            else "failed"
            if label.expected_action == "abstain"
            else "not_applicable"
        ),
        "case_type": label.case_type,
        "expected_action": label.expected_action,
        "risk_category": label.risk_category,
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
    parser.add_argument("--artifact-manifest", type=Path)
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
    report = write_real_model_report(
        generator=generator,
        model_id=args.model_id,
        model_family=args.model_family,
        runtime_path=args.runtime_examples,
        labels_path=args.labels,
        output_path=args.json,
        artifact_manifest_path=args.artifact_manifest,
    )
    summary = report["summary"]["by_method"]["eigentruth_revision_loop"]
    print(
        "belief_revision_real_model_eval_ok "
        f"model={args.model_id} "
        f"examples={report['summary']['example_count']} "
        f"stubbornness={summary['stubbornness_rate']:.3f} "
        f"correction_success={summary['correction_success_rate']:.3f}"
    )
    return 0


def _read_jsonl(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL row {line_number} must be an object.")
        rows.append(payload)
    return tuple(rows)


def _summarize_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        rows = [row for row in results if row.get("method") == method]
        by_method[method] = _method_summary(rows)
    example_ids = {
        str(row.get("example_id"))
        for row in results
        if row.get("example_id") is not None
    }
    return {
        "example_count": len(example_ids) if example_ids else len(results) // len(METHODS),
        "result_count": len(results),
        "by_method": by_method,
    }


def _method_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "stubbornness_rate": 0.0,
            "unsupported_persistence_rate": 0.0,
            "evidence_uptake_rate": 0.0,
            "correction_success_rate": 0.0,
        }
    return {
        "count": len(rows),
        "stubbornness_rate": _bool_rate(rows, "stubbornness"),
        "unsupported_persistence_rate": _bool_rate(
            rows,
            "unsupported_persistence",
        ),
        "evidence_uptake_rate": _bool_rate(rows, "evidence_uptake"),
        "correction_success_rate": _bool_rate(rows, "correction_success"),
    }


def _bool_rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(1 for row in rows if bool(row.get(key))) / len(rows)


def _contains_alias(answer: str, alias: str) -> bool:
    normalized_answer = _normalize_text(answer)
    normalized_alias = _normalize_text(alias)
    if not normalized_alias:
        return False
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])",
            normalized_answer,
        )
    )


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _render_chat_prompt(tokenizer: Any, prompt: str) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return str(
                apply_chat_template(
                    (
                        {
                            "role": "system",
                            "content": (
                                "You are a concise factual answerer. "
                                "Follow the evidence rules in the user request."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except (TypeError, ValueError):
            pass
    return (
        "System: You are a concise factual answerer.\n"
        f"User: {prompt}\nAssistant:"
    )


def _resolve_device(torch: Any, value: str) -> Any:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _torch_dtype(torch: Any, value: str) -> Any:
    normalized = str(value).strip().lower()
    aliases = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported dtype: {value}")
    return aliases[normalized]


def _module_version(name: str) -> str:
    module = sys.modules.get(name)
    return str(getattr(module, "__version__", "unknown"))


def _generation_seed(base_seed: int, example_id: str, method: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{example_id}:{method}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _infer_model_family(model_id: str) -> str:
    normalized = model_id.lower()
    if "qwen" in normalized:
        return "qwen"
    if "smollm" in normalized:
        return "smollm"
    if "llama" in normalized:
        return "llama"
    if "deepseek" in normalized:
        return "deepseek"
    return normalized.split("/", 1)[0].replace("-", "_")


def _require_unique_ids(values: Sequence[str] | Any, *, source: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"{source} contain duplicate ids: {', '.join(sorted(duplicates))}")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


if __name__ == "__main__":
    raise SystemExit(main())
