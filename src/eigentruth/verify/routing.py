"""Default verifier routing presets for product and benchmark loops."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from eigentruth.verify.composite import RoutedVerifier, VerifierRoute
from eigentruth.verify.groundedness import EvidenceDocument, GroundednessVerifier
from eigentruth.verify.planning import (
    DEFAULT_CITATION_FEATURE_FLAGS,
    DEFAULT_CITATION_METADATA_KEYS,
    DEFAULT_TRIPLE_EVIDENCE_FEATURE_FLAGS,
    DEFAULT_TRIPLE_EVIDENCE_METADATA_KEYS,
)
from eigentruth.verify.protocols import VerificationStatus, Verifier
from eigentruth.verify.triples import TripleEvidenceVerifier

_CALCULATION_TEXT_PATTERNS = (r"\d\s*[+*/%-]\s*\d\s*(?:=|equals|is)",)
_CITATION_TEXT_PATTERNS = (
    r"\[[A-Za-z0-9_.:-]+\]",
    r"\b(?:doi:\s*)?10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    r"\barxiv:\s*(?:\d{4}\.\d{4,5}(?:v\d+)?)",
    r"\bhttps?://",
)


def default_verifier_routes(
    *,
    evidence: EvidenceDocument | Mapping[str, Any] | str | Sequence[EvidenceDocument | Mapping[str, Any] | str] = (),
    refutations: Mapping[str, Sequence[str] | str] | None = None,
    qa_verifier: Verifier | None = None,
    state_verifier: Verifier | None = None,
    state: Mapping[str, Any] | None = None,
    transition_verifier: Verifier | None = None,
    world_model: Any | None = None,
    calculator_verifier: Verifier | None = None,
    citation_verifier: Verifier | None = None,
    citation_records: Sequence[Mapping[str, Any]] | None = None,
    triple_evidence_verifier: Verifier | None = None,
    include_calculator: bool = True,
    include_citation: bool = True,
    include_triple_evidence: bool = True,
    include_groundedness: bool = True,
    min_groundedness_overlap: float = 0.65,
    min_triple_slot_coverage: float = 1.0,
    min_world_model_confidence: float = 0.0,
) -> tuple[VerifierRoute, ...]:
    """Build a conservative dependency-free verifier route stack.

    The route order prefers deterministic tools, then strict fact-level triple
    audits for sensitive factual claims, and finally lexical groundedness as the
    fallback evidence baseline.
    """
    evidence_items = _evidence_sequence(evidence)
    routes: list[VerifierRoute] = []

    if include_calculator:
        if calculator_verifier is None:
            from eigentruth.adapters import CalculatorVerifier

            calculator_verifier = CalculatorVerifier()
        routes.append(
            VerifierRoute(
                "calculator",
                calculator_verifier,
                metadata_keys=("calculation", "expression"),
                context_keys=("calculation", "expression"),
                text_patterns=_CALCULATION_TEXT_PATTERNS,
            )
        )

    if include_citation and (citation_verifier is not None or citation_records is not None):
        if citation_verifier is None:
            from eigentruth.verify.citations import CitationVerifier

            citation_verifier = CitationVerifier(records=tuple(citation_records or ()))
        routes.append(
            VerifierRoute(
                "citation",
                citation_verifier,
                feature_flags=DEFAULT_CITATION_FEATURE_FLAGS,
                metadata_keys=DEFAULT_CITATION_METADATA_KEYS,
                text_patterns=_CITATION_TEXT_PATTERNS,
                fallthrough_statuses=(VerificationStatus.NOT_APPLICABLE,),
            )
        )

    if transition_verifier is None and world_model is not None:
        from eigentruth.adapters import StateTransitionVerifier

        transition_verifier = StateTransitionVerifier(
            world_model=world_model,
            state=state or {},
            min_prediction_confidence=min_world_model_confidence,
        )
    if transition_verifier is not None:
        routes.append(
            VerifierRoute(
                "state_transition",
                transition_verifier,
                metadata_keys=("state_transition",),
                context_keys=("state_transition",),
            )
        )

    if state_verifier is None and state is not None:
        from eigentruth.adapters import StructuredStateVerifier

        state_verifier = StructuredStateVerifier(state=state)
    if state_verifier is not None:
        routes.append(
            VerifierRoute(
                "structured_state",
                state_verifier,
                metadata_keys=("state_check", "path", "key", "field"),
                context_keys=("state_check",),
            )
        )

    if qa_verifier is not None:
        routes.append(
            VerifierRoute(
                "structured_qa",
                qa_verifier,
                metadata_keys=("qa_check", "question_answer_check"),
                context_keys=("statement.question", "statement.answer"),
                fallthrough_statuses=(
                    VerificationStatus.NOT_APPLICABLE,
                    VerificationStatus.INSUFFICIENT_EVIDENCE,
                ),
            )
        )

    if include_triple_evidence:
        if triple_evidence_verifier is None:
            triple_evidence_verifier = TripleEvidenceVerifier(
                evidence=evidence_items,
                min_slot_coverage=min_triple_slot_coverage,
            )
        routes.append(
            VerifierRoute(
                "triple_evidence",
                triple_evidence_verifier,
                feature_flags=DEFAULT_TRIPLE_EVIDENCE_FEATURE_FLAGS,
                metadata_keys=DEFAULT_TRIPLE_EVIDENCE_METADATA_KEYS,
                fallthrough_statuses=(VerificationStatus.NOT_APPLICABLE,),
            )
        )

    if include_groundedness:
        routes.append(
            VerifierRoute(
                "groundedness",
                GroundednessVerifier(
                    evidence=evidence_items,
                    refutations={} if refutations is None else refutations,
                    min_overlap=min_groundedness_overlap,
                ),
                fallback=True,
            )
        )

    if not routes:
        raise ValueError("default verifier route stack is empty.")
    return tuple(routes)


def default_routed_verifier(
    *,
    max_attempted_routes: int | None = None,
    **route_kwargs: Any,
) -> RoutedVerifier:
    """Build a `RoutedVerifier` from `default_verifier_routes(...)`."""
    return RoutedVerifier(
        default_verifier_routes(**route_kwargs),
        max_attempted_routes=max_attempted_routes,
    )


def _evidence_sequence(
    evidence: EvidenceDocument | Mapping[str, Any] | str | Sequence[EvidenceDocument | Mapping[str, Any] | str],
) -> tuple[EvidenceDocument | Mapping[str, Any] | str, ...]:
    if isinstance(evidence, (EvidenceDocument, str, Mapping)):
        return (evidence,)
    return tuple(evidence)
