"""Counterfactual robustness audits for claim verifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from eigentruth.json_utils import to_jsonable
from eigentruth.verify.claims import claim_entity_candidates
from eigentruth.verify.protocols import Claim, VerificationResult, VerificationStatus, Verifier


@dataclass(frozen=True)
class CounterfactualProbe:
    """One original claim and one counterfactual perturbation to verify."""

    original: Claim | Mapping[str, Any]
    counterfactual: Claim | Mapping[str, Any]
    probe_id: str | None = None
    probe_type: str = "counterfactual"
    expected_original_status: VerificationStatus | str | None = VerificationStatus.SUPPORTED
    expected_counterfactual_status: VerificationStatus | str | None = VerificationStatus.REFUTED
    expected_flip: bool | str | int = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        original = _coerce_claim(self.original, fallback_id="original")
        counterfactual = _coerce_claim(self.counterfactual, fallback_id="counterfactual")
        probe_type = str(self.probe_type).strip().casefold().replace("-", "_")
        if not probe_type:
            raise ValueError("probe_type must be non-empty.")
        probe_id = None if self.probe_id is None else str(self.probe_id).strip()
        object.__setattr__(self, "original", original)
        object.__setattr__(self, "counterfactual", counterfactual)
        object.__setattr__(self, "probe_id", probe_id or None)
        object.__setattr__(self, "probe_type", probe_type)
        object.__setattr__(
            self,
            "expected_original_status",
            _coerce_optional_status(self.expected_original_status, field_name="expected_original_status"),
        )
        object.__setattr__(
            self,
            "expected_counterfactual_status",
            _coerce_optional_status(
                self.expected_counterfactual_status,
                field_name="expected_counterfactual_status",
            ),
        )
        object.__setattr__(
            self,
            "expected_flip",
            _coerce_bool(self.expected_flip, field_name="expected_flip"),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "probe_id": self.probe_id,
            "probe_type": self.probe_type,
            "original": _claim_to_dict(self.original),
            "counterfactual": _claim_to_dict(self.counterfactual),
            "expected_original_status": _status_value(self.expected_original_status),
            "expected_counterfactual_status": _status_value(self.expected_counterfactual_status),
            "expected_flip": self.expected_flip,
            "metadata": to_jsonable(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CounterfactualProbe":
        """Build a probe from a JSON-like mapping."""
        original = _first_present_mapping(data, ("original", "source", "claim"))
        counterfactual = _first_present_mapping(data, ("counterfactual", "perturbed", "variant"))
        if original is None:
            text = data.get("original_text", data.get("text"))
            if text is not None:
                original = {
                    "text": text,
                    "claim_id": data.get("original_claim_id", data.get("claim_id")),
                    "metadata": data.get("original_metadata", {}),
                }
        if counterfactual is None:
            text = data.get("counterfactual_text", data.get("perturbed_text", data.get("variant_text")))
            if text is not None:
                counterfactual = {
                    "text": text,
                    "claim_id": data.get("counterfactual_claim_id"),
                    "metadata": data.get("counterfactual_metadata", {}),
                }
        if original is None or counterfactual is None:
            raise ValueError("counterfactual probe records must contain original and counterfactual claims.")
        raw_probe_id = data.get("probe_id", data.get("id"))
        return cls(
            original=original,
            counterfactual=counterfactual,
            probe_id=None if raw_probe_id is None else str(raw_probe_id),
            probe_type=str(data.get("probe_type", data.get("perturbation_type", "counterfactual"))),
            expected_original_status=data.get("expected_original_status", VerificationStatus.SUPPORTED),
            expected_counterfactual_status=data.get(
                "expected_counterfactual_status",
                data.get("expected_variant_status", VerificationStatus.REFUTED),
            ),
            expected_flip=_coerce_bool(data.get("expected_flip", True), field_name="expected_flip"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class CounterfactualProbeResult:
    """Verifier outputs and pass/fail status for one counterfactual probe."""

    probe: CounterfactualProbe
    original_result: VerificationResult
    counterfactual_result: VerificationResult
    passed: bool
    failure_reason: str | None = None

    @property
    def status_changed(self) -> bool:
        """Return whether verifier status changed across the perturbation."""
        return self.original_result.status != self.counterfactual_result.status

    @property
    def original_matches_expected(self) -> bool | None:
        """Return original expected-status match, or None when not configured."""
        expected = self.probe.expected_original_status
        if expected is None:
            return None
        return self.original_result.status == expected

    @property
    def counterfactual_matches_expected(self) -> bool | None:
        """Return counterfactual expected-status match, or None when not configured."""
        expected = self.probe.expected_counterfactual_status
        if expected is None:
            return None
        return self.counterfactual_result.status == expected

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "probe": self.probe.to_dict(),
            "original_result": _verification_result_to_dict(self.original_result),
            "counterfactual_result": _verification_result_to_dict(self.counterfactual_result),
            "status_changed": self.status_changed,
            "original_matches_expected": self.original_matches_expected,
            "counterfactual_matches_expected": self.counterfactual_matches_expected,
            "passed": self.passed,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class CounterfactualVerificationReport:
    """Aggregate counterfactual verifier robustness metrics."""

    results: Sequence[CounterfactualProbeResult]
    max_examples: int = 20
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        max_examples = int(self.max_examples)
        if max_examples < 0:
            raise ValueError("max_examples must be non-negative.")
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "max_examples", max_examples)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def summary(self) -> dict[str, Any]:
        """Return aggregate robustness metrics."""
        results = tuple(self.results)
        record_count = len(results)
        passed_count = sum(1 for result in results if result.passed)
        expected_flip = tuple(result for result in results if result.probe.expected_flip)
        expected_stable = tuple(result for result in results if not result.probe.expected_flip)
        flip_success = sum(1 for result in expected_flip if result.status_changed)
        false_invariance = len(expected_flip) - flip_success
        stable_success = sum(1 for result in expected_stable if not result.status_changed)
        unexpected_flip = len(expected_stable) - stable_success
        original_expected = tuple(
            result for result in results if result.original_matches_expected is not None
        )
        counterfactual_expected = tuple(
            result for result in results if result.counterfactual_matches_expected is not None
        )
        return {
            "record_count": record_count,
            "passed_count": passed_count,
            "failed_count": record_count - passed_count,
            "pass_rate": _safe_div(passed_count, record_count),
            "expected_flip_count": len(expected_flip),
            "flip_success_count": flip_success,
            "false_invariance_count": false_invariance,
            "false_invariance_rate": _safe_div(false_invariance, len(expected_flip)),
            "expected_stable_count": len(expected_stable),
            "stable_success_count": stable_success,
            "unexpected_flip_count": unexpected_flip,
            "unexpected_flip_rate": _safe_div(unexpected_flip, len(expected_stable)),
            "original_expected_count": len(original_expected),
            "original_expected_match_count": sum(
                1 for result in original_expected if result.original_matches_expected
            ),
            "original_expected_accuracy": _safe_div(
                sum(1 for result in original_expected if result.original_matches_expected),
                len(original_expected),
            ),
            "counterfactual_expected_count": len(counterfactual_expected),
            "counterfactual_expected_match_count": sum(
                1 for result in counterfactual_expected if result.counterfactual_matches_expected
            ),
            "counterfactual_expected_accuracy": _safe_div(
                sum(1 for result in counterfactual_expected if result.counterfactual_matches_expected),
                len(counterfactual_expected),
            ),
            "by_probe_type": _by_probe_type(results),
        }

    def error_examples(self) -> tuple[dict[str, Any], ...]:
        """Return bounded failed examples for debugging."""
        examples = []
        for result in self.results:
            if result.passed:
                continue
            examples.append(result.to_dict())
            if len(examples) >= self.max_examples:
                break
        return tuple(examples)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report."""
        return {
            "workflow": "counterfactual_verification_audit",
            "summary": self.summary(),
            "results": tuple(result.to_dict() for result in self.results),
            "error_examples": self.error_examples(),
            "metadata": to_jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class CounterfactualProbeGenerator:
    """Dependency-free generator for simple claim counterfactual probes.

    The generator is intentionally conservative. It uses explicit metadata
    variants/replacements when available, then falls back to bounded numeric,
    temporal, and negation perturbations. It does not infer truth; generated
    counterfactuals are audit fixtures that should still be reviewed for
    production-grade evidence sets.
    """

    max_probes_per_claim: int = 3
    probe_types: Sequence[str] = ("metadata", "entity_swap", "quantity", "year", "negation")
    expected_original_status: VerificationStatus | str | None = VerificationStatus.SUPPORTED
    expected_counterfactual_status: VerificationStatus | str | None = VerificationStatus.REFUTED
    expected_flip: bool | str | int = True
    entity_replacements: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        max_probes = int(self.max_probes_per_claim)
        if max_probes < 0:
            raise ValueError("max_probes_per_claim must be non-negative.")
        probe_types = tuple(
            str(probe_type).strip().casefold().replace("-", "_")
            for probe_type in self.probe_types
            if str(probe_type).strip()
        )
        object.__setattr__(self, "max_probes_per_claim", max_probes)
        object.__setattr__(self, "probe_types", probe_types)
        object.__setattr__(
            self,
            "expected_original_status",
            _coerce_optional_status(self.expected_original_status, field_name="expected_original_status"),
        )
        object.__setattr__(
            self,
            "expected_counterfactual_status",
            _coerce_optional_status(
                self.expected_counterfactual_status,
                field_name="expected_counterfactual_status",
            ),
        )
        object.__setattr__(
            self,
            "expected_flip",
            _coerce_bool(self.expected_flip, field_name="expected_flip"),
        )
        object.__setattr__(
            self,
            "entity_replacements",
            {
                str(source): str(target)
                for source, target in self.entity_replacements.items()
                if str(source) and str(target) and str(source) != str(target)
            },
        )

    def generate(self, claims: Sequence[Claim | Mapping[str, Any]]) -> tuple[CounterfactualProbe, ...]:
        """Generate bounded counterfactual probes from claims."""
        probes: list[CounterfactualProbe] = []
        for index, raw_claim in enumerate(claims):
            claim = _coerce_claim(raw_claim, fallback_id=f"claim_{index}")
            variants = _candidate_variants(
                claim,
                probe_types=self.probe_types,
                entity_replacements=self.entity_replacements,
            )
            added = 0
            seen_texts = {claim.text.strip()}
            for variant_type, variant_text, variant_metadata in variants:
                normalized_text = variant_text.strip()
                if not normalized_text or normalized_text in seen_texts:
                    continue
                seen_texts.add(normalized_text)
                probes.append(
                    CounterfactualProbe(
                        original=claim,
                        counterfactual=Claim(
                            text=normalized_text,
                            claim_id=f"{claim.claim_id or f'claim_{index}'}:{variant_type}",
                            span=None,
                            metadata={
                                **dict(variant_metadata),
                                "source_claim_id": claim.claim_id,
                                "generated_by": "CounterfactualProbeGenerator",
                            },
                        ),
                        probe_id=f"{claim.claim_id or f'claim_{index}'}:{variant_type}:{added}",
                        probe_type=variant_type,
                        expected_original_status=self.expected_original_status,
                        expected_counterfactual_status=self.expected_counterfactual_status,
                        expected_flip=self.expected_flip,
                        metadata={
                            "source_claim_id": claim.claim_id,
                            "source": "generated",
                            "generator": "CounterfactualProbeGenerator",
                        },
                    )
                )
                added += 1
                if added >= self.max_probes_per_claim:
                    break
        return tuple(probes)


def generate_counterfactual_probes(
    claims: Sequence[Claim | Mapping[str, Any]],
    *,
    max_probes_per_claim: int = 3,
    probe_types: Sequence[str] = ("metadata", "entity_swap", "quantity", "year", "negation"),
    expected_original_status: VerificationStatus | str | None = VerificationStatus.SUPPORTED,
    expected_counterfactual_status: VerificationStatus | str | None = VerificationStatus.REFUTED,
    expected_flip: bool | str | int = True,
    entity_replacements: Mapping[str, str] | None = None,
) -> tuple[CounterfactualProbe, ...]:
    """Generate simple counterfactual verifier probes from claims."""
    return CounterfactualProbeGenerator(
        max_probes_per_claim=max_probes_per_claim,
        probe_types=probe_types,
        expected_original_status=expected_original_status,
        expected_counterfactual_status=expected_counterfactual_status,
        expected_flip=expected_flip,
        entity_replacements={} if entity_replacements is None else dict(entity_replacements),
    ).generate(claims)


@dataclass(frozen=True)
class CounterfactualVerificationAuditor:
    """Run counterfactual probes through a verifier and summarize robustness."""

    verifier: Verifier

    def audit(
        self,
        probes: Sequence[CounterfactualProbe | Mapping[str, Any]],
        *,
        context: Mapping[str, Any] | None = None,
        max_examples: int = 20,
        metadata: Mapping[str, Any] | None = None,
    ) -> CounterfactualVerificationReport:
        """Verify all probes and return an aggregate report."""
        results = []
        for index, raw_probe in enumerate(probes):
            probe = _coerce_probe(raw_probe)
            original_result = self.verifier.verify(probe.original, context=context)
            counterfactual_result = self.verifier.verify(probe.counterfactual, context=context)
            passed, failure_reason = _evaluate_probe_result(
                probe,
                original_result,
                counterfactual_result,
            )
            if probe.probe_id is None:
                probe = CounterfactualProbe(
                    original=probe.original,
                    counterfactual=probe.counterfactual,
                    probe_id=f"p{index}",
                    probe_type=probe.probe_type,
                    expected_original_status=probe.expected_original_status,
                    expected_counterfactual_status=probe.expected_counterfactual_status,
                    expected_flip=probe.expected_flip,
                    metadata=probe.metadata,
                )
            results.append(CounterfactualProbeResult(
                probe=probe,
                original_result=original_result,
                counterfactual_result=counterfactual_result,
                passed=passed,
                failure_reason=failure_reason,
            ))
        return CounterfactualVerificationReport(
            results=tuple(results),
            max_examples=max_examples,
            metadata={} if metadata is None else dict(metadata),
        )


def audit_counterfactual_verification(
    verifier: Verifier,
    probes: Sequence[CounterfactualProbe | Mapping[str, Any]],
    *,
    context: Mapping[str, Any] | None = None,
    max_examples: int = 20,
    metadata: Mapping[str, Any] | None = None,
) -> CounterfactualVerificationReport:
    """Convenience wrapper for counterfactual verifier robustness audits."""
    return CounterfactualVerificationAuditor(verifier).audit(
        probes,
        context=context,
        max_examples=max_examples,
        metadata=metadata,
    )


def _evaluate_probe_result(
    probe: CounterfactualProbe,
    original_result: VerificationResult,
    counterfactual_result: VerificationResult,
) -> tuple[bool, str | None]:
    original_expected = probe.expected_original_status
    if original_expected is not None and original_result.status != original_expected:
        return False, "original_status_mismatch"
    counterfactual_expected = probe.expected_counterfactual_status
    if counterfactual_expected is not None and counterfactual_result.status != counterfactual_expected:
        return False, "counterfactual_status_mismatch"
    status_changed = original_result.status != counterfactual_result.status
    if probe.expected_flip and not status_changed:
        return False, "false_invariance"
    if not probe.expected_flip and status_changed:
        return False, "unexpected_flip"
    return True, None


def _coerce_probe(value: CounterfactualProbe | Mapping[str, Any]) -> CounterfactualProbe:
    if isinstance(value, CounterfactualProbe):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("counterfactual probes must be CounterfactualProbe or mapping objects.")
    return CounterfactualProbe.from_dict(value)


def _coerce_claim(value: Claim | Mapping[str, Any], *, fallback_id: str) -> Claim:
    if isinstance(value, Claim):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("claim values must be Claim or mapping objects.")
    text = value.get("text", value.get("claim", value.get("statement")))
    if text is None or not str(text).strip():
        raise ValueError("claim mappings must contain text, claim, or statement.")
    span = value.get("span")
    claim_id = value.get("claim_id", value.get("id", fallback_id))
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("claim metadata must be a mapping when provided.")
    return Claim(
        text=str(text),
        claim_id=None if claim_id is None else str(claim_id),
        span=_coerce_span(span),
        metadata=dict(metadata),
    )


_YEAR_RE = re.compile(r"\b(?:18|19|20|21)\d{2}\b")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.-])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?![A-Za-z0-9_.-])")
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|cannot|can't|isn't|aren't|wasn't|weren't)\b",
    re.IGNORECASE,
)
_AUXILIARY_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|can|could|will|would|should|does|do|did)\b",
    re.IGNORECASE,
)
_ORG_HINT_RE = re.compile(
    r"\b(?:corp|corporation|company|inc|llc|ltd|limited|group|systems|technologies|labs|laboratories)\b",
    re.IGNORECASE,
)
_DEFAULT_ENTITY_COUNTERFACTUAL_REPLACEMENTS = {
    "Paris": "Berlin",
    "Berlin": "Paris",
    "London": "Madrid",
    "Madrid": "London",
    "Tokyo": "Seoul",
    "Seoul": "Tokyo",
    "France": "Germany",
    "Germany": "France",
    "Ireland": "Iceland",
    "Iceland": "Ireland",
    "Japan": "South Korea",
    "South Korea": "Japan",
    "United States": "Canada",
    "Canada": "United States",
    "AlphaCorp": "BetaCorp",
    "BetaCorp": "AlphaCorp",
    "Beta Labs": "Gamma Labs",
    "Gamma Labs": "Beta Labs",
}


def _candidate_variants(
    claim: Claim,
    *,
    probe_types: Sequence[str],
    entity_replacements: Mapping[str, str],
) -> tuple[tuple[str, str, Mapping[str, Any]], ...]:
    metadata = dict(claim.metadata) if isinstance(claim.metadata, Mapping) else {}
    variants: list[tuple[str, str, Mapping[str, Any]]] = []
    enabled = set(probe_types)
    if "metadata" in enabled:
        variants.extend(_metadata_variants(claim, metadata))
    if "entity_swap" in enabled:
        replacements = dict(entity_replacements)
        replacements.update(_metadata_replacements(metadata))
        variants.extend(_replacement_variants(claim.text, replacements, probe_type="entity_swap"))
        variants.extend(_entity_candidate_variants(
            claim.text,
            metadata,
            used_sources=tuple(replacements),
        ))
    if "quantity" in enabled:
        quantity = _quantity_variant(claim.text)
        if quantity is not None:
            variants.append(quantity)
    if "year" in enabled:
        year = _year_variant(claim.text)
        if year is not None:
            variants.append(year)
    if "negation" in enabled:
        negation = _negation_variant(claim.text)
        if negation is not None:
            variants.append(negation)
    return tuple(variants)


def _metadata_variants(
    claim: Claim,
    metadata: Mapping[str, Any],
) -> tuple[tuple[str, str, Mapping[str, Any]], ...]:
    variants: list[tuple[str, str, Mapping[str, Any]]] = []
    for key in ("counterfactual_variants", "counterfactuals", "counterfactual_claims"):
        value = metadata.get(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            continue
        for index, item in enumerate(value):
            if isinstance(item, Mapping):
                text = item.get("text", item.get("claim", item.get("statement", item.get("counterfactual_text"))))
                if text is None:
                    continue
                probe_type = str(item.get("probe_type", item.get("type", "metadata"))).strip() or "metadata"
                variants.append((
                    probe_type.casefold().replace("-", "_"),
                    str(text),
                    {"metadata_key": key, "metadata_index": index, **dict(item.get("metadata", {}))},
                ))
            elif item is not None:
                variants.append(("metadata", str(item), {"metadata_key": key, "metadata_index": index}))
    if metadata.get("counterfactual_text") is not None:
        variants.append(("metadata", str(metadata["counterfactual_text"]), {"metadata_key": "counterfactual_text"}))
    if metadata.get("variant_text") is not None:
        variants.append(("metadata", str(metadata["variant_text"]), {"metadata_key": "variant_text"}))
    return tuple(variants)


def _metadata_replacements(metadata: Mapping[str, Any]) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for key in ("counterfactual_replacements", "entity_replacements", "entity_swaps"):
        value = metadata.get(key)
        if isinstance(value, Mapping):
            for source, target in value.items():
                source_text = str(source)
                target_text = str(target)
                if source_text and target_text and source_text != target_text:
                    replacements[source_text] = target_text
    entities = metadata.get("entities")
    if isinstance(entities, Sequence) and not isinstance(entities, (str, bytes, bytearray)):
        for entity in entities:
            if not isinstance(entity, Mapping):
                continue
            source = entity.get("text", entity.get("name", entity.get("value")))
            target = entity.get("counterfactual", entity.get("replacement", entity.get("swap")))
            if source is not None and target is not None and str(source) != str(target):
                replacements[str(source)] = str(target)
    return replacements


def _replacement_variants(
    text: str,
    replacements: Mapping[str, str],
    *,
    probe_type: str,
) -> tuple[tuple[str, str, Mapping[str, Any]], ...]:
    variants = []
    for source, target in replacements.items():
        if source not in text:
            continue
        variants.append((
            probe_type,
            text.replace(source, target, 1),
            {"replacement_source": source, "replacement_target": target},
        ))
    return tuple(variants)


def _entity_candidate_variants(
    text: str,
    metadata: Mapping[str, Any],
    *,
    used_sources: Sequence[str] = (),
) -> tuple[tuple[str, str, Mapping[str, Any]], ...]:
    variants = []
    seen_sources = {str(source) for source in used_sources}
    candidates = _entity_candidates_for_counterfactual(text, metadata)
    for source in candidates:
        if source in seen_sources or source not in text:
            continue
        target = _default_entity_counterfactual_target(source)
        if target is None or target == source or target in text:
            continue
        seen_sources.add(source)
        variants.append((
            "entity_swap",
            text.replace(source, target, 1),
            {
                "replacement_source": source,
                "replacement_target": target,
                "replacement_source_kind": "entity_candidate",
            },
        ))
    return tuple(variants)


def _entity_candidates_for_counterfactual(text: str, metadata: Mapping[str, Any]) -> tuple[str, ...]:
    candidates = []
    explicit_entity_key = False
    for key in ("counterfactual_entity_candidates", "entity_candidates", "named_entities", "entities"):
        if key in metadata:
            explicit_entity_key = True
        for item in _as_sequence(metadata.get(key)):
            if isinstance(item, Mapping):
                value = item.get("text", item.get("name", item.get("value")))
            else:
                value = item
            if value is not None and str(value).strip():
                candidates.append(str(value).strip())
    if not candidates and not explicit_entity_key:
        candidates.extend(claim_entity_candidates(text))
    return tuple(dict.fromkeys(candidates))


def _default_entity_counterfactual_target(source: str) -> str | None:
    source = source.strip()
    if not source:
        return None
    direct = _DEFAULT_ENTITY_COUNTERFACTUAL_REPLACEMENTS.get(source)
    if direct is not None:
        return direct
    for known_source, target in _DEFAULT_ENTITY_COUNTERFACTUAL_REPLACEMENTS.items():
        if source.casefold() == known_source.casefold():
            return _match_case(source, target)
    if _ORG_HINT_RE.search(source):
        if "labs" in source.casefold() or "laboratories" in source.casefold():
            return "Gamma Labs" if source != "Gamma Labs" else "Beta Labs"
        return "BetaCorp" if source != "BetaCorp" else "AlphaCorp"
    if len(source.split()) > 1:
        return "Counterfactual Entity"
    if any(char.isupper() for char in source):
        return "CounterfactualEntity"
    return None


def _match_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source.islower():
        return target.lower()
    return target


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _quantity_variant(text: str) -> tuple[str, str, Mapping[str, Any]] | None:
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(0)
        if _YEAR_RE.fullmatch(raw):
            continue
        replacement = _increment_number_text(raw)
        if replacement == raw:
            continue
        return (
            "quantity",
            f"{text[:match.start()]}{replacement}{text[match.end():]}",
            {"replacement_source": raw, "replacement_target": replacement},
        )
    return None


def _year_variant(text: str) -> tuple[str, str, Mapping[str, Any]] | None:
    match = _YEAR_RE.search(text)
    if match is None:
        return None
    raw = match.group(0)
    replacement = str(int(raw) + 1)
    return (
        "year",
        f"{text[:match.start()]}{replacement}{text[match.end():]}",
        {"replacement_source": raw, "replacement_target": replacement},
    )


def _negation_variant(text: str) -> tuple[str, str, Mapping[str, Any]] | None:
    negation = _NEGATION_RE.search(text)
    if negation is not None:
        variant = f"{text[:negation.start()]}{text[negation.end():]}"
        variant = re.sub(r"\s+", " ", variant).strip()
        if variant:
            return ("negation", variant, {"operation": "remove_negation", "removed": negation.group(0)})
        return None
    auxiliary = _AUXILIARY_RE.search(text)
    if auxiliary is not None:
        return (
            "negation",
            f"{text[:auxiliary.end()]} not{text[auxiliary.end():]}",
            {"operation": "insert_not", "after": auxiliary.group(0)},
        )
    return ("negation", f"Not {text[0].lower()}{text[1:]}" if text else "Not", {"operation": "prefix_not"})


def _increment_number_text(raw: str) -> str:
    if any(char in raw for char in ".eE"):
        value = float(raw)
        replacement = value + 1.0
        return f"{replacement:g}"
    return str(int(raw) + 1)


def _coerce_span(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != 2:
        raise ValueError("claim span must be a two-item sequence when provided.")
    return (int(value[0]), int(value[1]))


def _coerce_optional_status(
    value: VerificationStatus | str | None,
    *,
    field_name: str,
) -> VerificationStatus | None:
    if value is None:
        return None
    if isinstance(value, VerificationStatus):
        return value
    text = str(value).strip().casefold().replace("-", "_")
    if not text:
        return None
    try:
        return VerificationStatus(text)
    except ValueError as exc:
        choices = ", ".join(status.value for status in VerificationStatus)
        raise ValueError(f"{field_name} must be one of: {choices}") from exc


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{field_name} must be a boolean value.")


def _first_present_mapping(data: Mapping[str, Any], keys: Sequence[str]) -> Mapping[str, Any] | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _claim_to_dict(claim: Claim) -> dict[str, Any]:
    return {
        "text": claim.text,
        "claim_id": claim.claim_id,
        "span": claim.span,
        "metadata": to_jsonable(dict(claim.metadata)),
    }


def _verification_result_to_dict(result: VerificationResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "confidence": result.confidence,
        "evidence": tuple(result.evidence),
        "explanation": result.explanation,
        "metadata": to_jsonable(dict(result.metadata)),
    }


def _status_value(status: VerificationStatus | None) -> str | None:
    return None if status is None else status.value


def _by_probe_type(results: Sequence[CounterfactualProbeResult]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[CounterfactualProbeResult]] = {}
    for result in results:
        groups.setdefault(result.probe.probe_type, []).append(result)
    return {
        name: {
            "record_count": len(items),
            "passed_count": sum(1 for item in items if item.passed),
            "failed_count": sum(1 for item in items if not item.passed),
            "pass_rate": _safe_div(sum(1 for item in items if item.passed), len(items)),
            "false_invariance_count": sum(
                1 for item in items if item.probe.expected_flip and not item.status_changed
            ),
            "unexpected_flip_count": sum(
                1 for item in items if not item.probe.expected_flip and item.status_changed
            ),
        }
        for name, items in groups.items()
    }


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
