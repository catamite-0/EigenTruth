"""Build paired context-sensitivity token logprobs with an optional HF model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch

DEFAULT_BASELINE_TEMPLATE = "{prompt}"
DEFAULT_CONTEXT_TEMPLATE = "Evidence:\n{evidence}\n\n{prompt}"


@dataclass(frozen=True)
class TokenLogprob:
    """One generated token and its conditional logprob."""

    token: str
    logprob: float
    token_id: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        logprob = _log_probability(self.logprob, name="logprob")
        token_id = _optional_non_negative_int(self.token_id, name="token_id")
        object.__setattr__(self, "token", str(self.token))
        object.__setattr__(self, "logprob", logprob)
        object.__setattr__(self, "token_id", token_id)
        object.__setattr__(self, "metadata", dict(self.metadata))


class CompletionLogprobScorer(Protocol):
    """Protocol for scoring completion tokens under a prompt."""

    def score(self, prompt: str, completion: str) -> Sequence[TokenLogprob | Mapping[str, Any]]:
        """Return one logprob per completion token."""


@dataclass(frozen=True)
class PreparedLogprobRecord:
    """One record prepared for no-context/evidence-context scoring."""

    run: str
    record_index: int
    completion: str
    baseline_prompt: str
    context_prompt: str
    claim_id: str | None = None
    evidence_texts: Sequence[str] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        run = str(self.run).strip()
        if not run:
            raise ValueError("run must be non-empty.")
        record_index = _non_negative_int(self.record_index, name="record_index")
        completion = str(self.completion)
        if not completion.strip():
            raise ValueError("completion must be non-empty.")
        baseline_prompt = str(self.baseline_prompt)
        context_prompt = str(self.context_prompt)
        if not baseline_prompt:
            raise ValueError("baseline_prompt must be non-empty.")
        if not context_prompt:
            raise ValueError("context_prompt must be non-empty.")
        object.__setattr__(self, "run", run)
        object.__setattr__(self, "record_index", record_index)
        object.__setattr__(self, "completion", completion)
        object.__setattr__(self, "baseline_prompt", baseline_prompt)
        object.__setattr__(self, "context_prompt", context_prompt)
        object.__setattr__(self, "claim_id", None if self.claim_id is None else str(self.claim_id))
        object.__setattr__(self, "evidence_texts", tuple(str(text) for text in self.evidence_texts if str(text)))
        object.__setattr__(self, "metadata", dict(self.metadata))


class HFCompletionLogprobScorer:
    """Hugging Face causal-LM scorer for completion token logprobs."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str | torch.device | None = None,
        max_length: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = _resolve_device(device, model=model)
        self.max_length = _resolve_max_length(max_length, model=model, tokenizer=tokenizer)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str | torch.device | None = None,
        dtype: str = "float32",
        revision: str | None = None,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
        max_length: int | None = None,
    ) -> "HFCompletionLogprobScorer":
        """Load an optional Hugging Face causal LM."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without optional deps
            raise RuntimeError("transformers is required; install eigentruth[hf] to use this benchmark.") from exc

        tokenizer_kwargs: dict[str, Any] = {}
        if revision:
            tokenizer_kwargs["revision"] = revision
        if trust_remote_code:
            tokenizer_kwargs["trust_remote_code"] = True
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)

        load_kwargs: dict[str, Any] = {
            "dtype": _torch_dtype(dtype),
            "low_cpu_mem_usage": True,
        }
        if revision:
            load_kwargs["revision"] = revision
        if trust_remote_code:
            load_kwargs["trust_remote_code"] = True
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        except TypeError:
            if not attn_implementation:
                raise
            load_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        resolved_device = _resolve_device(device, model=model)
        model.to(resolved_device).eval()
        return cls(model, tokenizer, device=resolved_device, max_length=max_length)

    def score(self, prompt: str, completion: str) -> tuple[TokenLogprob, ...]:
        """Return completion-token logprobs under ``prompt``."""
        prompt_ids = _encode_ids(self.tokenizer, prompt, add_special_tokens=True)
        completion_ids = _encode_ids(self.tokenizer, completion, add_special_tokens=False)
        if not completion_ids:
            raise ValueError("completion produced no tokenizer ids.")
        if not prompt_ids:
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_id is None:
                raise ValueError("prompt produced no tokenizer ids and tokenizer has no bos_token_id.")
            prompt_ids = [int(bos_id)]
        prompt_ids = _trim_prompt_ids(prompt_ids, completion_ids, max_length=self.max_length)
        input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits[0]
        if logits.shape[0] < 2:
            raise ValueError("model output is too short to score completion tokens.")
        log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
        prompt_width = len(prompt_ids)
        scores: list[TokenLogprob] = []
        for token_offset, token_id in enumerate(completion_ids):
            position = prompt_width + token_offset
            if position <= 0:
                raise ValueError("cannot score first completion token without preceding context.")
            logprob = float(log_probs[position - 1, int(token_id)].detach().cpu().item())
            scores.append(TokenLogprob(
                token=_decode_token(self.tokenizer, int(token_id)),
                token_id=int(token_id),
                logprob=logprob,
            ))
        return tuple(scores)


def load_json_records(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load JSON/JSONL records from a generic sidecar or fixture file."""
    source = Path(path)
    if source.suffix == ".jsonl":
        records = []
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"line {line_number} must be a JSON object.")
                records.append(dict(payload))
        if not records:
            raise ValueError("JSONL input must contain at least one record.")
        return tuple(records)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping):
        records = _first_sequence(
            payload.get("records"),
            payload.get("verified_records"),
            payload.get("items"),
            payload.get("examples"),
        )
        if records is None:
            return (dict(payload),)
        return _records_from_sequence(records)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return _records_from_sequence(payload)
    raise ValueError("input JSON must be an object, array of objects, or records object.")


def prepare_logprob_records(
    records: Sequence[Mapping[str, Any]],
    *,
    run_name: str | None = None,
    baseline_template: str = DEFAULT_BASELINE_TEMPLATE,
    context_template: str = DEFAULT_CONTEXT_TEMPLATE,
    require_evidence: bool = False,
    limit: int | None = None,
) -> tuple[PreparedLogprobRecord, ...]:
    """Prepare generic JSON records for paired logprob scoring."""
    if limit is not None:
        limit = _positive_int(limit, name="limit")
    selected = records if limit is None else records[:limit]
    default_run = _default_run(selected, run_name=run_name)
    prepared: list[PreparedLogprobRecord] = []
    for offset, record in enumerate(selected):
        run = str(record.get("run") or default_run or "default")
        record_index = _record_index(record, fallback=offset)
        nested = _mapping(record.get("record"))
        claim = _mapping(_first_mapping(record.get("claim"), nested.get("claim")))
        claim_id = _optional_text(record.get("claim_id"), nested.get("claim_id"), claim.get("claim_id"))
        completion = _completion_text(record, nested=nested, claim=claim)
        prompt = _prompt_text(record, nested=nested, claim=claim)
        evidence_texts = _evidence_texts(record, nested=nested)
        if require_evidence and not evidence_texts:
            raise ValueError(f"record run={run!r} record_index={record_index} has no evidence text.")
        baseline_prompt = baseline_template.format(
            prompt=prompt,
            completion=completion,
            claim=completion,
            evidence="",
            claim_id="" if claim_id is None else claim_id,
        )
        if evidence_texts:
            context_prompt = context_template.format(
                prompt=prompt,
                completion=completion,
                claim=completion,
                evidence="\n".join(evidence_texts),
                claim_id="" if claim_id is None else claim_id,
            )
        else:
            context_prompt = baseline_prompt
        prepared.append(PreparedLogprobRecord(
            run=run,
            record_index=record_index,
            claim_id=claim_id,
            completion=completion,
            baseline_prompt=baseline_prompt,
            context_prompt=context_prompt,
            evidence_texts=evidence_texts,
            metadata={
                "source_workflow": record.get("workflow"),
                "missing_evidence": not bool(evidence_texts),
                "prompt_source": _prompt_source(record, nested=nested),
                "completion_source": _completion_source(record, nested=nested, claim=claim),
            },
        ))
    if not prepared:
        raise ValueError("no records were prepared for logprob extraction.")
    return tuple(prepared)


def build_paired_logprob_records(
    records: Sequence[PreparedLogprobRecord],
    scorer: CompletionLogprobScorer,
    *,
    model_id: str | None = None,
    source_path: str | Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Build paired no-context/evidence-context token logprob records."""
    outputs: list[dict[str, Any]] = []
    for record in records:
        baseline_scores = _token_logprob_tuple(scorer.score(record.baseline_prompt, record.completion))
        context_scores = _token_logprob_tuple(scorer.score(record.context_prompt, record.completion))
        tokens = _paired_tokens(baseline_scores, context_scores, claim_id=record.claim_id)
        outputs.append({
            "schema_version": 1,
            "workflow": "context_sensitivity_paired_logprobs",
            "run": record.run,
            "record_index": record.record_index,
            "claim_id": record.claim_id,
            "tokens": tokens,
            "metadata": {
                **dict(record.metadata),
                "model_id": model_id,
                "source_path": None if source_path is None else str(source_path),
                "completion_sha256": _sha256_text(record.completion),
                "baseline_prompt_sha256": _sha256_text(record.baseline_prompt),
                "context_prompt_sha256": _sha256_text(record.context_prompt),
                "completion_chars": len(record.completion),
                "baseline_prompt_chars": len(record.baseline_prompt),
                "context_prompt_chars": len(record.context_prompt),
                "evidence_count": len(record.evidence_texts),
                "token_count": len(tokens),
            },
        })
    return tuple(outputs)


def write_paired_logprob_jsonl(records: Sequence[Mapping[str, Any]], output_path: str | Path) -> None:
    """Write paired logprob records as compact JSONL."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def build_report(
    *,
    records_path: str | Path,
    output: str | Path,
    scorer: CompletionLogprobScorer,
    model_id: str | None = None,
    run_name: str | None = None,
    baseline_template: str = DEFAULT_BASELINE_TEMPLATE,
    context_template: str = DEFAULT_CONTEXT_TEMPLATE,
    require_evidence: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build paired logprobs from input records and return a compact report."""
    raw_records = load_json_records(records_path)
    prepared = prepare_logprob_records(
        raw_records,
        run_name=run_name,
        baseline_template=baseline_template,
        context_template=context_template,
        require_evidence=require_evidence,
        limit=limit,
    )
    paired = build_paired_logprob_records(
        prepared,
        scorer,
        model_id=model_id,
        source_path=records_path,
    )
    write_paired_logprob_jsonl(paired, output)
    token_counts = [
        _non_negative_int(_mapping(record.get("metadata")).get("token_count", 0), name="metadata.token_count")
        for record in paired
    ]
    missing_evidence_count = sum(1 for record in prepared if not record.evidence_texts)
    return {
        "schema_version": 1,
        "workflow": "context_sensitivity_paired_logprob_extraction",
        "records_path": str(records_path),
        "output": str(output),
        "model_id": model_id,
        "run_name": run_name,
        "input_record_count": len(raw_records),
        "prepared_record_count": len(prepared),
        "paired_logprob_record_count": len(paired),
        "missing_evidence_count": missing_evidence_count,
        "total_token_count": sum(token_counts),
        "mean_token_count": _mean(token_counts),
        "max_token_count": max(token_counts) if token_counts else 0,
        "baseline_template": baseline_template,
        "context_template": context_template,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """CLI entrypoint helper."""
    scorer = HFCompletionLogprobScorer.from_pretrained(
        args.model_id,
        device=args.device,
        dtype=args.dtype,
        revision=args.model_revision,
        trust_remote_code=bool(args.trust_remote_code),
        attn_implementation=args.attn_implementation,
        max_length=args.max_length,
    )
    report = build_report(
        records_path=args.records,
        output=args.output,
        scorer=scorer,
        model_id=args.model_id,
        run_name=args.run_name,
        baseline_template=args.baseline_template,
        context_template=args.context_template,
        require_evidence=bool(args.require_evidence),
        limit=args.limit,
    )
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote paired context-sensitivity logprobs to {args.output}")
    return report


def _paired_tokens(
    baseline_scores: Sequence[TokenLogprob],
    context_scores: Sequence[TokenLogprob],
    *,
    claim_id: str | None,
) -> list[dict[str, Any]]:
    if len(baseline_scores) != len(context_scores):
        raise ValueError("baseline and context scorers returned different token counts.")
    tokens: list[dict[str, Any]] = []
    for index, (baseline, context) in enumerate(zip(baseline_scores, context_scores)):
        if baseline.token_id is not None and context.token_id is not None and baseline.token_id != context.token_id:
            raise ValueError(f"token_id mismatch at completion token {index}.")
        if baseline.token != context.token:
            raise ValueError(f"token text mismatch at completion token {index}.")
        token_payload = {
            "token": baseline.token,
            "baseline_logprob": baseline.logprob,
            "context_logprob": context.logprob,
            "claim_id": claim_id,
            "metadata": {
                "token_index": index,
                "baseline_metadata": dict(baseline.metadata),
                "context_metadata": dict(context.metadata),
            },
        }
        if baseline.token_id is not None:
            token_payload["token_id"] = baseline.token_id
        tokens.append(token_payload)
    if not tokens:
        raise ValueError("completion produced no scored tokens.")
    return tokens


def _token_logprob_tuple(values: Sequence[TokenLogprob | Mapping[str, Any]]) -> tuple[TokenLogprob, ...]:
    scores = []
    for value in values:
        if isinstance(value, TokenLogprob):
            scores.append(value)
        elif isinstance(value, Mapping):
            scores.append(TokenLogprob(
                token=str(value.get("token", "")),
                logprob=value.get("logprob", value.get("token_logprob")),
                token_id=value.get("token_id"),
                metadata=dict(value.get("metadata", {})),
            ))
        else:
            raise ValueError("scorer outputs must be TokenLogprob objects or mappings.")
    if not scores:
        raise ValueError("scorer returned no token logprobs.")
    return tuple(scores)


def _completion_text(
    record: Mapping[str, Any],
    *,
    nested: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> str:
    completion = _optional_text(
        record.get("completion"),
        record.get("answer"),
        record.get("generated_text"),
        record.get("target"),
        record.get("text"),
        record.get("claim"),
        nested.get("completion"),
        nested.get("answer"),
        nested.get("generated_text"),
        nested.get("target"),
        nested.get("text"),
        nested.get("claim"),
        claim.get("text"),
    )
    if completion is None:
        raise ValueError("record is missing completion/answer/claim text.")
    return completion


def _completion_source(
    record: Mapping[str, Any],
    *,
    nested: Mapping[str, Any],
    claim: Mapping[str, Any],
) -> str:
    keys = (
        ("completion", record),
        ("answer", record),
        ("generated_text", record),
        ("target", record),
        ("text", record),
        ("claim", record),
        ("completion", nested),
        ("answer", nested),
        ("generated_text", nested),
        ("target", nested),
        ("text", nested),
        ("claim", nested),
        ("text", claim),
    )
    for key, mapping in keys:
        if _optional_text(mapping.get(key)) is not None:
            return key
    return "unknown"


def _prompt_text(record: Mapping[str, Any], *, nested: Mapping[str, Any], claim: Mapping[str, Any]) -> str:
    explicit_prompt = _optional_text(
        record.get("baseline_prompt"),
        record.get("no_context_prompt"),
        nested.get("baseline_prompt"),
        nested.get("no_context_prompt"),
    )
    if explicit_prompt is not None:
        return explicit_prompt
    prompt = _optional_text(
        record.get("prompt"),
        record.get("question"),
        record.get("input"),
        nested.get("prompt"),
        nested.get("question"),
        nested.get("input"),
    )
    if prompt is not None:
        return _ensure_prompt_suffix(prompt)
    claim_text = _optional_text(claim.get("text"), nested.get("claim"), record.get("claim"), record.get("text"))
    if claim_text is not None:
        return "Claim: "
    return "Answer: "


def _prompt_source(record: Mapping[str, Any], *, nested: Mapping[str, Any]) -> str:
    for key, mapping in (
        ("baseline_prompt", record),
        ("no_context_prompt", record),
        ("baseline_prompt", nested),
        ("no_context_prompt", nested),
        ("prompt", record),
        ("question", record),
        ("input", record),
        ("prompt", nested),
        ("question", nested),
        ("input", nested),
    ):
        if _optional_text(mapping.get(key)) is not None:
            return key
    return "claim_default"


def _ensure_prompt_suffix(prompt: str) -> str:
    stripped = prompt.rstrip()
    if not stripped:
        return "Answer: "
    if stripped.endswith((":", "\n")):
        return stripped + " "
    return stripped + "\nAnswer: "


def _evidence_texts(record: Mapping[str, Any], *, nested: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for payload in (
        record.get("evidence"),
        record.get("evidence_text"),
        record.get("context"),
        record.get("retrieval_hits"),
        record.get("retrieval_documents"),
        record.get("initial_evidence"),
        nested.get("evidence"),
        nested.get("evidence_text"),
        nested.get("context"),
        nested.get("retrieval_hits"),
        nested.get("retrieval_documents"),
        nested.get("initial_evidence"),
        _mapping(nested.get("final")).get("evidence"),
        _mapping(nested.get("initial")).get("evidence"),
    ):
        values.extend(_text_items(payload))
    return tuple(_dedupe_texts(values))


def _text_items(payload: Any) -> tuple[str, ...]:
    if payload is None:
        return ()
    if isinstance(payload, str):
        text = payload.strip()
        return (text,) if text else ()
    if isinstance(payload, Mapping):
        text = _optional_text(
            payload.get("text"),
            payload.get("content"),
            payload.get("snippet"),
            payload.get("body"),
            payload.get("document"),
            payload.get("evidence"),
            payload.get("passage"),
        )
        return () if text is None else (text,)
    if isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray)):
        values: list[str] = []
        for item in payload:
            values.extend(_text_items(item))
        return tuple(values)
    return ()


def _dedupe_texts(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output = []
    for value in values:
        text = value.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return tuple(output)


def _default_run(records: Sequence[Mapping[str, Any]], *, run_name: str | None) -> str | None:
    if run_name:
        return str(run_name)
    runs = {str(record.get("run", "")).strip() for record in records if str(record.get("run", "")).strip()}
    if len(runs) == 1:
        return next(iter(runs))
    return None


def _record_index(record: Mapping[str, Any], *, fallback: int) -> int:
    value = record.get("record_index", fallback)
    if isinstance(value, bool) or value is None:
        raise ValueError("record_index must be an integer.")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("record_index must be an integer.") from exc
    return _non_negative_int(numeric, name="record_index")


def _resolve_device(device: str | torch.device | None, *, model: Any) -> torch.device:
    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _resolve_max_length(max_length: int | None, *, model: Any, tokenizer: Any) -> int | None:
    if max_length is not None:
        return _positive_int(max_length, name="max_length")
    config = getattr(model, "config", None)
    for value in (
        getattr(config, "max_position_embeddings", None),
        getattr(config, "n_positions", None),
        getattr(tokenizer, "model_max_length", None),
    ):
        if isinstance(value, int) and 0 < value < 1_000_000_000:
            return value
    return None


def _trim_prompt_ids(prompt_ids: Sequence[int], completion_ids: Sequence[int], *, max_length: int | None) -> list[int]:
    prompt = [int(token_id) for token_id in prompt_ids]
    completion = [int(token_id) for token_id in completion_ids]
    if max_length is None or len(prompt) + len(completion) <= max_length:
        return prompt
    if len(completion) >= max_length:
        raise ValueError("completion token count must be smaller than max_length.")
    keep_prompt = max_length - len(completion)
    return prompt[-keep_prompt:]


def _encode_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens, return_attention_mask=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else getattr(encoded, "input_ids")
    if input_ids and isinstance(input_ids[0], Sequence):
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([token_id], clean_up_tokenization_spaces=False))
    except TypeError:
        return str(tokenizer.decode([token_id]))


def _torch_dtype(value: str) -> torch.dtype:
    dtypes = {
        "float32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    key = str(value).lower()
    if key not in dtypes:
        raise ValueError(f"unsupported dtype {value!r}; expected one of {sorted(dtypes)}.")
    return dtypes[key]


def _records_from_sequence(values: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    records = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ValueError(f"record {index} must be a JSON object.")
        records.append(dict(value))
    if not records:
        raise ValueError("records must contain at least one record.")
    return tuple(records)


def _first_sequence(*values: Any) -> Sequence[Any] | None:
    for value in values:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return value
    return None


def _first_mapping(*values: Any) -> Mapping[str, Any] | None:
    for value in values:
        if isinstance(value, Mapping):
            return value
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _positive_int(value: Any, *, name: str) -> int:
    numeric = _non_negative_int(value, name=name)
    if numeric <= 0:
        raise ValueError(f"{name} must be positive.")
    return numeric


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool.")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative.")
    return numeric


def _optional_non_negative_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, name=name)


def _log_probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-positive number, not bool.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-positive number.") from exc
    if not math.isfinite(numeric) or numeric > 0.0:
        raise ValueError(f"{name} must be a finite non-positive number.")
    return numeric


def _mean(values: Sequence[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paired no-context/evidence-context token logprobs for context-sensitivity scoring."
    )
    parser.add_argument("--records", required=True, help="input verified-record sidecar or generic JSON/JSONL records")
    parser.add_argument("--output", required=True, help="output paired-logprob JSONL")
    parser.add_argument("--model-id", required=True, help="Hugging Face causal LM id or local path")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float", "bfloat16", "bf16", "float16", "fp16"),
    )
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--baseline-template", default=DEFAULT_BASELINE_TEMPLATE)
    parser.add_argument("--context-template", default=DEFAULT_CONTEXT_TEMPLATE)
    parser.add_argument("--require-evidence", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", default=None, help="optional summary report path")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
