"""Staged verifier gating for cost-aware control loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from eigentruth.control.policy import ControlAction, RiskDecision, RiskLevel
from eigentruth.verify.features import enabled_feature_names, metadata_path_enabled
from eigentruth.verify.protocols import Claim


@dataclass(frozen=True)
class VerificationStageDecision:
    """Decision for whether an expensive verifier stage should run."""

    run_verifier: bool
    reason: str
    verification_scope: str | None = None
    verify_claim_ids: Sequence[str] = ()
    skipped_claim_ids: Sequence[str] = ()
    triggered_claim_ids: Sequence[str] = ()
    triggered_features: Mapping[str, Sequence[str]] = field(default_factory=dict)
    triggered_metadata: Mapping[str, Sequence[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verification_scope is None:
            scope = "all" if self.run_verifier else "none"
        else:
            scope = str(self.verification_scope).strip().lower()
        if scope not in {"all", "triggered", "none"}:
            raise ValueError("verification_scope must be one of: all, triggered, none")
        if self.run_verifier and scope == "none":
            raise ValueError("verification_scope cannot be 'none' when run_verifier is true")
        object.__setattr__(self, "verification_scope", scope)
        object.__setattr__(self, "verify_claim_ids", tuple(str(item) for item in self.verify_claim_ids))
        object.__setattr__(
            self,
            "skipped_claim_ids",
            tuple(str(item) for item in self.skipped_claim_ids),
        )
        object.__setattr__(self, "triggered_claim_ids", tuple(str(item) for item in self.triggered_claim_ids))
        object.__setattr__(
            self,
            "triggered_features",
            {
                str(claim_id): tuple(str(feature) for feature in features)
                for claim_id, features in self.triggered_features.items()
            },
        )
        object.__setattr__(
            self,
            "triggered_metadata",
            {
                str(claim_id): tuple(str(key) for key in keys)
                for claim_id, keys in self.triggered_metadata.items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "run_verifier": self.run_verifier,
            "reason": self.reason,
            "verification_scope": self.verification_scope,
            "verify_claim_ids": tuple(self.verify_claim_ids),
            "skipped_claim_ids": tuple(self.skipped_claim_ids),
            "triggered_claim_ids": tuple(self.triggered_claim_ids),
            "triggered_features": {
                claim_id: tuple(features)
                for claim_id, features in self.triggered_features.items()
            },
            "triggered_metadata": {
                claim_id: tuple(keys)
                for claim_id, keys in self.triggered_metadata.items()
            },
        }


@dataclass(frozen=True)
class StagedVerificationPolicy:
    """Gate expensive claim verification behind cheap diagnostics and claim metadata."""

    verify_risk_levels: Sequence[RiskLevel | str] = (
        RiskLevel.MEDIUM,
        RiskLevel.HIGH,
        RiskLevel.UNKNOWN,
    )
    verify_actions: Sequence[ControlAction | str] = (
        ControlAction.RETRIEVE,
        ControlAction.REWRITE,
        ControlAction.STEER_REGENERATE,
        ControlAction.EXECUTE_TOOL,
        ControlAction.ABSTAIN,
        ControlAction.CLARIFY,
    )
    verify_claim_feature_flags: Sequence[str] = (
        "has_number",
        "has_citation",
        "is_time_sensitive",
    )
    verify_claim_metadata_keys: Sequence[str] = ("requires_verification",)
    verify_triggered_claims_only: bool = False
    fail_closed_on_skip: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verify_risk_levels",
            tuple(_coerce_risk_level(level) for level in self.verify_risk_levels),
        )
        object.__setattr__(
            self,
            "verify_actions",
            tuple(_coerce_control_action(action) for action in self.verify_actions),
        )
        object.__setattr__(
            self,
            "verify_claim_feature_flags",
            tuple(str(flag) for flag in self.verify_claim_feature_flags),
        )
        object.__setattr__(
            self,
            "verify_claim_metadata_keys",
            tuple(str(key) for key in self.verify_claim_metadata_keys),
        )
        object.__setattr__(
            self,
            "verify_triggered_claims_only",
            _coerce_bool(
                self.verify_triggered_claims_only,
                field_name="verify_triggered_claims_only",
            ),
        )
        object.__setattr__(
            self,
            "fail_closed_on_skip",
            _coerce_bool(
                self.fail_closed_on_skip,
                field_name="fail_closed_on_skip",
            ),
        )

    def decide(
        self,
        diagnostic_decision: RiskDecision,
        *,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> VerificationStageDecision:
        """Return whether claim verification should run for this request."""
        del context
        if not claims:
            return VerificationStageDecision(
                run_verifier=False,
                reason="no claims to verify",
                verification_scope="none",
            )
        claim_ids = tuple(_claim_id(claim, index) for index, claim in enumerate(claims))
        if diagnostic_decision.risk_level in self.verify_risk_levels:
            return VerificationStageDecision(
                run_verifier=True,
                reason=f"diagnostic risk level is {diagnostic_decision.risk_level.value}",
                verification_scope="all",
                verify_claim_ids=claim_ids,
            )
        if diagnostic_decision.action in self.verify_actions:
            return VerificationStageDecision(
                run_verifier=True,
                reason=f"diagnostic action is {diagnostic_decision.action.value}",
                verification_scope="all",
                verify_claim_ids=claim_ids,
            )

        feature_hits: dict[str, tuple[str, ...]] = {}
        metadata_hits: dict[str, tuple[str, ...]] = {}
        triggered_claim_ids = []
        for index, claim in enumerate(claims):
            claim_id = _claim_id(claim, index)
            metadata = claim.metadata if isinstance(claim.metadata, Mapping) else {}
            features = metadata.get("features", {})
            if not isinstance(features, Mapping):
                features = {}
            matched_features = enabled_feature_names(features, self.verify_claim_feature_flags)
            matched_metadata = tuple(
                key for key in self.verify_claim_metadata_keys if metadata_path_enabled(metadata, key)
            )
            if matched_features:
                feature_hits[claim_id] = matched_features
            if matched_metadata:
                metadata_hits[claim_id] = matched_metadata
            if matched_features or matched_metadata:
                triggered_claim_ids.append(claim_id)

        if triggered_claim_ids:
            triggered_claim_id_set = set(triggered_claim_ids)
            skipped_claim_ids = tuple(
                claim_id for claim_id in claim_ids if claim_id not in triggered_claim_id_set
            )
            return VerificationStageDecision(
                run_verifier=True,
                reason="claim metadata requires verification",
                verification_scope="triggered" if self.verify_triggered_claims_only else "all",
                verify_claim_ids=(
                    tuple(triggered_claim_ids)
                    if self.verify_triggered_claims_only
                    else claim_ids
                ),
                skipped_claim_ids=(
                    skipped_claim_ids
                    if self.verify_triggered_claims_only
                    else ()
                ),
                triggered_claim_ids=tuple(triggered_claim_ids),
                triggered_features=feature_hits,
                triggered_metadata=metadata_hits,
            )
        return VerificationStageDecision(
            run_verifier=False,
            reason="diagnostics and claim metadata did not require verification",
            verification_scope="none",
            skipped_claim_ids=claim_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "verify_risk_levels": tuple(level.value for level in self.verify_risk_levels),
            "verify_actions": tuple(action.value for action in self.verify_actions),
            "verify_claim_feature_flags": tuple(self.verify_claim_feature_flags),
            "verify_claim_metadata_keys": tuple(self.verify_claim_metadata_keys),
            "verify_triggered_claims_only": self.verify_triggered_claims_only,
            "fail_closed_on_skip": self.fail_closed_on_skip,
        }


def _claim_id(claim: Claim, index: int) -> str:
    return claim.claim_id or f"c{index + 1}"


def _coerce_risk_level(value: RiskLevel | str) -> RiskLevel:
    if isinstance(value, RiskLevel):
        return value
    return RiskLevel(str(value))


def _coerce_control_action(value: ControlAction | str) -> ControlAction:
    if isinstance(value, ControlAction):
        return value
    return ControlAction(str(value))


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")
