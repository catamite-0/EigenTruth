"""Dependency-free calculator verifier for deterministic numeric claims."""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from eigentruth.verify import Claim, VerificationResult, VerificationStatus

_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_CALCULATION_RE = re.compile(
    rf"(?P<expression>[-+*/().%\d\s]+[+*/%-][-+*/().%\d\s]*)\s*(?:=|equals|is)\s*(?P<expected>{_NUMBER_RE})",
    re.IGNORECASE,
)
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


@dataclass(frozen=True)
class CalculationResult:
    """One deterministic calculation result."""

    expression: str
    expected: float
    actual: float
    tolerance: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected", _required_finite_float(self.expected, name="expected"))
        object.__setattr__(self, "actual", _required_finite_float(self.actual, name="actual"))
        object.__setattr__(
            self,
            "tolerance",
            _required_non_negative_finite_float(self.tolerance, name="tolerance"),
        )

    @property
    def matches(self) -> bool:
        """Return whether actual and expected are within tolerance."""
        return math.isclose(self.actual, self.expected, rel_tol=self.tolerance, abs_tol=self.tolerance)

    def to_evidence(self) -> str:
        """Return a compact evidence string."""
        return f"calculator: {self.expression} = {self.actual:g}; expected {self.expected:g}"


@dataclass(frozen=True)
class CalculatorVerifier:
    """Verify arithmetic claims with a safe local calculator.

    The verifier supports either structured metadata/context with
    `expression`/`expected` fields or simple textual claims like `2 + 2 = 4`.
    It is intended as a deterministic tool adapter, not a natural-language math
    parser.
    """

    default_tolerance: float = 1e-9
    max_abs_value: float = 1e12
    max_exponent: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "default_tolerance",
            _required_non_negative_finite_float(
                self.default_tolerance,
                name="default_tolerance",
            ),
        )
        object.__setattr__(
            self,
            "max_abs_value",
            _required_positive_finite_float(self.max_abs_value, name="max_abs_value"),
        )
        object.__setattr__(
            self,
            "max_exponent",
            _required_non_negative_finite_float(self.max_exponent, name="max_exponent"),
        )

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify one arithmetic claim."""
        try:
            parsed = _claim_calculation(claim, context, default_tolerance=self.default_tolerance)
        except ValueError as exc:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                confidence=0.4,
                explanation=str(exc),
                metadata={
                    "verifier": "calculator",
                    "decision_rule": "invalid_calculation_config",
                },
            )
        if parsed is None:
            return VerificationResult(
                status=VerificationStatus.NOT_APPLICABLE,
                confidence=1.0,
                explanation="claim does not contain a supported arithmetic calculation",
                metadata={"verifier": "calculator", "decision_rule": "no_calculation"},
            )

        expression, expected, tolerance = parsed
        try:
            actual = _safe_eval(expression, max_abs_value=self.max_abs_value, max_exponent=self.max_exponent)
        except ArithmeticError as exc:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                confidence=0.4,
                explanation=str(exc),
                metadata={
                    "verifier": "calculator",
                    "decision_rule": "calculation_error",
                    "expression": expression,
                    "expected": expected,
                    "tolerance": tolerance,
                },
            )

        result = CalculationResult(expression=expression, expected=expected, actual=actual, tolerance=tolerance)
        if result.matches:
            return VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=0.99,
                evidence=(result.to_evidence(),),
                explanation="calculator result matches the claimed value",
                metadata={
                    "verifier": "calculator",
                    "decision_rule": "calculation_match",
                    "expression": expression,
                    "expected": expected,
                    "actual": actual,
                    "tolerance": tolerance,
                },
            )
        return VerificationResult(
            status=VerificationStatus.REFUTED,
            confidence=0.99,
            evidence=(result.to_evidence(),),
            explanation="calculator result does not match the claimed value",
            metadata={
                "verifier": "calculator",
                "decision_rule": "calculation_mismatch",
                "expression": expression,
                "expected": expected,
                "actual": actual,
                "tolerance": tolerance,
            },
        )

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationResult, ...]:
        """Verify multiple claims."""
        return tuple(self.verify(claim, context=context) for claim in claims)


def _claim_calculation(
    claim: Claim,
    context: Mapping[str, Any] | None,
    *,
    default_tolerance: float,
) -> tuple[str, float, float] | None:
    for source in _calculation_sources(claim, context):
        parsed = _calculation_from_mapping(source, default_tolerance=default_tolerance)
        if parsed is not None:
            return parsed

    match = _CALCULATION_RE.search(claim.text)
    if match is None:
        return None
    return (
        match.group("expression").strip(),
        _required_finite_float(match.group("expected"), name="expected"),
        default_tolerance,
    )


def _calculation_sources(claim: Claim, context: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], ...]:
    sources: list[Mapping[str, Any]] = []
    if context is not None:
        raw_context = context.get("calculation")
        if isinstance(raw_context, Mapping):
            sources.append(raw_context)
    if isinstance(claim.metadata, Mapping):
        raw_metadata = claim.metadata.get("calculation")
        if isinstance(raw_metadata, Mapping):
            sources.append(raw_metadata)
        sources.append(claim.metadata)
    return tuple(sources)


def _calculation_from_mapping(
    data: Mapping[str, Any],
    *,
    default_tolerance: float,
) -> tuple[str, float, float] | None:
    expression = _optional_text(data.get("expression"))
    raw_expected = data.get("expected", data.get("result", data.get("answer")))
    if expression is None or raw_expected is None:
        return None
    expected = _required_finite_float(raw_expected, name="expected")
    tolerance = _tolerance_or_default(data.get("tolerance"), default_tolerance)
    return expression, expected, tolerance


def _safe_eval(expression: str, *, max_abs_value: float, max_exponent: float) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ArithmeticError(f"invalid arithmetic expression: {expression!r}") from exc
    return _eval_node(tree.body, max_abs_value=max_abs_value, max_exponent=max_exponent)


def _eval_node(node: ast.AST, *, max_abs_value: float, max_exponent: float) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ArithmeticError("calculator only supports numeric constants")
        return _bounded(float(node.value), max_abs_value=max_abs_value)
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand, max_abs_value=max_abs_value, max_exponent=max_exponent)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return _bounded(-value, max_abs_value=max_abs_value)
        raise ArithmeticError("calculator only supports unary plus and minus")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, max_abs_value=max_abs_value, max_exponent=max_exponent)
        right = _eval_node(node.right, max_abs_value=max_abs_value, max_exponent=max_exponent)
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ArithmeticError("unsupported arithmetic operator")
        if isinstance(node.op, ast.Pow) and abs(right) > max_exponent:
            raise ArithmeticError("exponent exceeds calculator bound")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0.0:
            raise ArithmeticError("division by zero")
        return _bounded(_OPERATORS[op_type](left, right), max_abs_value=max_abs_value)
    raise ArithmeticError("calculator only supports numeric arithmetic expressions")


def _bounded(value: float, *, max_abs_value: float) -> float:
    if not math.isfinite(value):
        raise ArithmeticError("calculator result is not finite")
    if abs(value) > max_abs_value:
        raise ArithmeticError("calculator result exceeds configured bound")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number.")
    return parsed


def _required_non_negative_finite_float(value: Any, *, name: str) -> float:
    parsed = _required_finite_float(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed


def _required_positive_finite_float(value: Any, *, name: str) -> float:
    parsed = _required_finite_float(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _tolerance_or_default(value: Any, default: float) -> float:
    if value is None:
        return default
    return _required_non_negative_finite_float(value, name="tolerance")
