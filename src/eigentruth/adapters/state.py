"""Dependency-free structured state and business-rule verifier adapter."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from eigentruth.verify import Claim, VerificationResult, VerificationStatus


class StateSource(Protocol):
    """Interface for loading structured state from an external source."""

    def load_state(self) -> Mapping[str, Any]:
        """Load state as a nested JSON-like mapping."""
        ...


@dataclass(frozen=True)
class StateCheck:
    """One structured assertion over external/domain state."""

    path: str
    operator: str = "eq"
    value: Any = None
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = self.path.strip()
        if not path:
            raise ValueError("state check path must be non-empty.")
        operator = _normalize_operator(self.operator)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "operator", operator)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "StateCheck":
        """Build a state check from a JSON-like mapping."""
        raw_path = data.get("path", data.get("key", data.get("field")))
        if raw_path is None:
            raise ValueError("state check mapping must contain path, key, or field.")
        return cls(
            path=str(raw_path),
            operator=str(data.get("operator", data.get("op", "eq"))),
            value=data.get("value", data.get("expected")),
            source=None if data.get("source") is None else str(data.get("source")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_evidence(self, actual: Any) -> str:
        """Return a compact evidence string."""
        prefix = f"{self.source}: " if self.source else "state: "
        return f"{prefix}{self.path} actual={actual!r} {self.operator} expected={self.value!r}"


@dataclass(frozen=True)
class SQLiteStateQuery:
    """One read-only SQLite query mapped into a structured state path."""

    path: str
    sql: str
    params: Sequence[Any] = ()
    column: str | int | None = None
    default: Any = None
    required: bool = False

    def __post_init__(self) -> None:
        path = self.path.strip()
        sql = self.sql.strip()
        if not path:
            raise ValueError("SQLite state query path must be non-empty.")
        if not sql:
            raise ValueError("SQLite state query SQL must be non-empty.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sql", sql)
        object.__setattr__(self, "params", tuple(self.params))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SQLiteStateQuery":
        """Build a SQLite query from a JSON-like mapping."""
        raw_path = data.get("path", data.get("key", data.get("field")))
        if raw_path is None:
            raise ValueError("SQLite state query mapping must contain path, key, or field.")
        if "sql" not in data:
            raise ValueError("SQLite state query mapping must contain sql.")
        return cls(
            path=str(raw_path),
            sql=str(data["sql"]),
            params=tuple(data.get("params", ())),
            column=data.get("column"),
            default=data.get("default"),
            required=_parse_bool(data.get("required", False), name="required"),
        )


@dataclass(frozen=True)
class SQLiteStateSource:
    """Load structured state from SQLite using explicit read queries.

    Queries are supplied by the caller and each result is assigned to a dotted
    state path. This keeps database integration deterministic, dependency-free,
    and auditable while avoiding a hard dependency on any production database
    driver.
    """

    database_path: str | Path
    queries: Sequence[SQLiteStateQuery | Mapping[str, Any]]

    def __post_init__(self) -> None:
        queries = tuple(
            query if isinstance(query, SQLiteStateQuery) else SQLiteStateQuery.from_mapping(query)
            for query in self.queries
        )
        object.__setattr__(self, "queries", queries)

    def load_state(self) -> Mapping[str, Any]:
        """Execute configured read queries and return nested state."""
        state: dict[str, Any] = {}
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            for query in self.queries:
                value = self._fetch_query_value(connection, query)
                if value is _MISSING:
                    continue
                _set_path(state, query.path, value)
        return state

    def _connect(self) -> sqlite3.Connection:
        path = str(self.database_path)
        if path == ":memory:":
            return sqlite3.connect(path)
        uri = f"file:{Path(path).resolve()}?mode=ro"
        return sqlite3.connect(uri, uri=True)

    @staticmethod
    def _fetch_query_value(connection: sqlite3.Connection, query: SQLiteStateQuery) -> Any:
        try:
            cursor = connection.execute(query.sql, tuple(query.params))
            row = cursor.fetchone()
        except sqlite3.Error as exc:
            raise ValueError(f"SQLite state query failed for {query.path!r}: {exc}") from exc
        if row is None:
            if query.required:
                raise ValueError(f"SQLite state query returned no rows for required path {query.path!r}.")
            return query.default if query.default is not None else _MISSING
        if query.column is None:
            if len(row.keys()) == 1:
                return row[0]
            return {key: row[key] for key in row.keys()}
        if isinstance(query.column, int):
            try:
                return row[query.column]
            except IndexError as exc:
                raise ValueError(
                    f"SQLite state query column index {query.column} is out of range for {query.path!r}."
                ) from exc
        column = str(query.column)
        try:
            return row[column]
        except IndexError as exc:
            raise ValueError(f"SQLite state query column {column!r} is missing for {query.path!r}.") from exc


@dataclass(frozen=True)
class StructuredStateVerifier:
    """Verify structured claims against deterministic external state.

    The verifier consumes a `state_check` mapping from claim metadata or context,
    then evaluates it against adapter state plus optional `context["state"]`.
    This models database, business-rule, policy, or world-state checks without
    adding a database dependency.
    """

    state: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_source(cls, source: StateSource) -> "StructuredStateVerifier":
        """Build a verifier from a structured state source."""
        return cls(source.load_state())

    def verify(self, claim: Claim, context: Mapping[str, Any] | None = None) -> VerificationResult:
        """Verify one claim against structured state."""
        raw_check = _state_check_source(claim, context)
        if raw_check is None:
            return VerificationResult(
                status=VerificationStatus.NOT_APPLICABLE,
                confidence=1.0,
                explanation="claim did not provide a structured state_check",
                metadata={"verifier": "structured_state", "decision_rule": "no_state_check"},
            )
        try:
            check = raw_check if isinstance(raw_check, StateCheck) else StateCheck.from_mapping(raw_check)
        except (TypeError, ValueError) as exc:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                confidence=0.3,
                explanation=str(exc),
                metadata={"verifier": "structured_state", "decision_rule": "invalid_state_check"},
            )

        state = _merged_state(self.state, context)
        found, actual = _get_path(state, check.path)
        metadata = {
            "verifier": "structured_state",
            "path": check.path,
            "operator": check.operator,
            "expected": check.value,
            "source": check.source,
        }
        if not found:
            return VerificationResult(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.25,
                explanation="state path was not found",
                metadata={**metadata, "decision_rule": "state_path_missing"},
            )

        try:
            supported = _evaluate_check(actual, check.operator, check.value)
        except (TypeError, ValueError, ArithmeticError) as exc:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                confidence=0.35,
                explanation=str(exc),
                evidence=(check.to_evidence(actual),),
                metadata={**metadata, "actual": actual, "decision_rule": "evaluation_error"},
            )

        if supported:
            return VerificationResult(
                status=VerificationStatus.SUPPORTED,
                confidence=0.95,
                evidence=(check.to_evidence(actual),),
                explanation="structured state check passed",
                metadata={**metadata, "actual": actual, "decision_rule": "state_check_passed"},
            )
        return VerificationResult(
            status=VerificationStatus.REFUTED,
            confidence=0.9,
            evidence=(check.to_evidence(actual),),
            explanation="structured state check failed",
            metadata={**metadata, "actual": actual, "decision_rule": "state_check_failed"},
        )

    def verify_many(
        self,
        claims: Sequence[Claim],
        context: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationResult, ...]:
        """Verify multiple claims."""
        return tuple(self.verify(claim, context=context) for claim in claims)


def _state_check_source(claim: Claim, context: Mapping[str, Any] | None) -> Mapping[str, Any] | StateCheck | None:
    if context is not None:
        by_claim = context.get("state_checks")
        if isinstance(by_claim, Mapping) and claim.claim_id is not None and claim.claim_id in by_claim:
            candidate = by_claim[claim.claim_id]
            if isinstance(candidate, (Mapping, StateCheck)):
                return candidate
        direct = context.get("state_check")
        if isinstance(direct, (Mapping, StateCheck)):
            return direct
    metadata = claim.metadata if isinstance(claim.metadata, Mapping) else {}
    direct = metadata.get("state_check")
    if isinstance(direct, (Mapping, StateCheck)):
        return direct
    if any(key in metadata for key in ("path", "key", "field")):
        return metadata
    return None


def _merged_state(base_state: Mapping[str, Any], context: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if context is None or not isinstance(context.get("state"), Mapping):
        return base_state
    merged = dict(base_state)
    for key, value in context["state"].items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = {**dict(merged[key]), **dict(value)}
        else:
            merged[key] = value
    return merged


def _get_path(state: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = state
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _set_path(state: dict[str, Any], path: str, value: Any) -> None:
    current = state
    parts = path.split(".")
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            nested: dict[str, Any] = {}
            current[part] = nested
            current = nested
        elif isinstance(existing, dict):
            current = existing
        else:
            raise ValueError(f"state path collision at {part!r} while setting {path!r}.")
    current[parts[-1]] = value


def _normalize_operator(operator: str) -> str:
    normalized = str(operator).strip().lower()
    aliases = {
        "=": "eq",
        "==": "eq",
        "equals": "eq",
        "!=": "ne",
        "<>": "ne",
        ">": "gt",
        ">=": "gte",
        "<": "lt",
        "<=": "lte",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "contains", "between", "exists"}
    if normalized not in allowed:
        raise ValueError(f"unsupported state check operator: {operator!r}")
    return normalized


def _evaluate_check(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "exists":
        return actual is not None
    if operator == "eq":
        return actual == expected
    if operator == "ne":
        return actual != expected
    if operator in {"gt", "gte", "lt", "lte"}:
        actual_number = _finite_float(actual, name="actual")
        expected_number = _finite_float(expected, name="expected")
        if operator == "gt":
            return actual_number > expected_number
        if operator == "gte":
            return actual_number >= expected_number
        if operator == "lt":
            return actual_number < expected_number
        return actual_number <= expected_number
    if operator == "between":
        if not isinstance(expected, Sequence) or isinstance(expected, (str, bytes)) or len(expected) != 2:
            raise ValueError("between operator expects a two-item numeric range.")
        actual_number = _finite_float(actual, name="actual")
        low = _finite_float(expected[0], name="range lower bound")
        high = _finite_float(expected[1], name="range upper bound")
        return low <= actual_number <= high
    if operator in {"in", "not_in"}:
        if isinstance(expected, (str, bytes)) or not isinstance(expected, Sequence):
            raise ValueError("in/not_in operators expect a non-string sequence.")
        present = actual in expected
        return present if operator == "in" else not present
    if operator == "contains":
        if not isinstance(actual, Sequence) and not isinstance(actual, Mapping):
            raise ValueError("contains operator expects an actual sequence or mapping.")
        return expected in actual
    raise ValueError(f"unsupported state check operator: {operator!r}")


def _finite_float(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} value must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ArithmeticError(f"{name} value must be finite.")
    return parsed


def _parse_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean or boolean string.")


_MISSING = object()
