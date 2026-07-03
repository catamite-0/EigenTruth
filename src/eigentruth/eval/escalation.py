"""Post-hoc metrics for uncertainty-escalated verification loops."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from eigentruth.eval.metrics import binomial_confidence_interval
from eigentruth.json_utils import to_jsonable

TRUE_LABEL = 0
FALSE_LABEL = 1
ACCEPT_ACTION = "accept"
RETRIEVE_ACTION = "retrieve"


def uncertainty_escalation_report(
    loop_results: Sequence[Mapping[str, Any] | Any],
    *,
    labels: Sequence[int | bool | str] | None = None,
) -> dict[str, Any]:
    """Summarize uncertainty-escalation impact across verification-loop results.

    ``loop_results`` may contain ``VerificationLoopResult`` objects, their
    ``to_dict()`` payloads, or wrapper mappings with a ``result`` key and
    optional ``label``. Label convention follows benchmark score dumps:
    ``0`` = true/normal, ``1`` = false/anomalous.
    """
    records, embedded_labels = _normalize_loop_records(loop_results)
    resolved_labels = _resolve_labels(embedded_labels, labels, expected_count=len(records))
    initial_actions = tuple(_decision_action(record.get("initial_decision")) for record in records)
    final_actions = tuple(_decision_action(record.get("final_decision")) for record in records)
    escalation_plans = tuple(_optional_mapping(record.get("uncertainty_escalation_plan")) for record in records)
    action_requests = tuple(_sequence_of_mappings(record.get("action_requests")) for record in records)
    action_results = tuple(_sequence_of_mappings(record.get("action_results")) for record in records)
    retrieval_evidence = tuple(_optional_mapping(record.get("retrieval_evidence")) or {} for record in records)

    report = {
        "n_total": len(records),
        "initial_decision": _decision_summary(initial_actions),
        "final_decision": _decision_summary(final_actions),
        "decision_changes": _decision_change_summary(initial_actions, final_actions),
        "verification_status": {
            "initial": _verification_status_summary(record.get("initial_verification_results") for record in records),
            "final": _verification_status_summary(record.get("final_verification_results") for record in records),
        },
        "uncertainty_escalation": _escalation_summary(escalation_plans),
        "action_execution": _action_execution_summary(action_requests, action_results, retrieval_evidence),
        "label_summary": None if resolved_labels is None else _label_summary(resolved_labels),
        "quality": None
        if resolved_labels is None
        else _quality_summary(initial_actions, final_actions, resolved_labels),
    }
    return to_jsonable(report)


def uncertainty_escalation_policy_sweep(
    loop_results: Sequence[Mapping[str, Any] | Any],
    *,
    labels: Sequence[int | bool | str] | None = None,
    min_confidence_values: Sequence[float] = (0.5, 0.6, 0.65, 0.7, 0.75, 0.8),
    uncertain_statuses: Sequence[str] = ("insufficient_evidence", "error"),
    max_final_false_accept_rate: float | None = None,
    min_final_selective_accuracy: float | None = None,
    max_trigger_rate: float | None = None,
    min_trigger_rate: float | None = None,
) -> dict[str, Any]:
    """Sweep second-stage verifier trigger thresholds over recorded loop results.

    The input should come from a run where the stronger verifier outcome is
    available for every record that might be selected. For each candidate
    ``min_confidence`` value, records whose preliminary verification status is
    uncertain or whose confidence falls below the value use the recorded final
    decision; the remaining records keep their initial decision. This gives a
    dependency-free what-if frontier over safety, coverage, and verifier cost.
    """
    records, embedded_labels = _normalize_loop_records(loop_results)
    resolved_labels = _resolve_labels(embedded_labels, labels, expected_count=len(records))
    thresholds = _sweep_thresholds(min_confidence_values)
    uncertain = {str(status).strip().lower() for status in uncertain_statuses if str(status).strip()}
    if not uncertain:
        raise ValueError("uncertain_statuses must contain at least one status.")
    constraints = {
        "max_final_false_accept_rate": _optional_unit_float(
            max_final_false_accept_rate,
            name="max_final_false_accept_rate",
        ),
        "min_final_selective_accuracy": _optional_unit_float(
            min_final_selective_accuracy,
            name="min_final_selective_accuracy",
        ),
        "max_trigger_rate": _optional_unit_float(max_trigger_rate, name="max_trigger_rate"),
        "min_trigger_rate": _optional_unit_float(min_trigger_rate, name="min_trigger_rate"),
    }
    rows = tuple(
        _policy_sweep_row(
            records,
            labels=resolved_labels,
            min_confidence=threshold,
            uncertain_statuses=uncertain,
            constraints=constraints,
        )
        for threshold in thresholds
    )
    recommended = _recommended_policy_sweep_row(rows)
    return to_jsonable({
        "workflow": "uncertainty_escalation_policy_sweep",
        "schema_version": 1,
        "n_total": len(records),
        "config": {
            "min_confidence_values": thresholds,
            "uncertain_statuses": tuple(sorted(uncertain)),
            **constraints,
        },
        "status": "promote" if recommended.get("gate", {}).get("passed") is True else "blocked",
        "recommended": recommended,
        "sweep": rows,
    })


def _normalize_loop_records(
    loop_results: Sequence[Mapping[str, Any] | Any],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Any | None, ...]]:
    records: list[Mapping[str, Any]] = []
    labels: list[Any | None] = []
    for index, item in enumerate(loop_results):
        payload = _to_mapping(item)
        label = payload.get("label")
        if "result" in payload and isinstance(payload.get("result"), Mapping):
            payload = _to_mapping(payload["result"])
        if "loop_result" in payload and isinstance(payload.get("loop_result"), Mapping):
            payload = _to_mapping(payload["loop_result"])
        if "initial_decision" not in payload or "final_decision" not in payload:
            raise ValueError(f"loop_results[{index}] is not a verification-loop result payload.")
        records.append(payload)
        labels.append(label)
    return tuple(records), tuple(labels)


def _resolve_labels(
    embedded_labels: Sequence[Any | None],
    labels: Sequence[int | bool | str] | None,
    *,
    expected_count: int,
) -> tuple[int, ...] | None:
    if labels is not None:
        if len(labels) != expected_count:
            raise ValueError("labels must have the same length as loop_results.")
        return tuple(_label_value(value) for value in labels)
    if not embedded_labels or all(value is None for value in embedded_labels):
        return None
    if any(value is None for value in embedded_labels):
        raise ValueError("embedded labels must be present for every loop result when labels= is omitted.")
    return tuple(_label_value(value) for value in embedded_labels)


def _label_value(value: int | bool | str | Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {TRUE_LABEL, FALSE_LABEL}:
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "true", "normal", "factual", "supported"}:
            return TRUE_LABEL
        if normalized in {"1", "false", "anomalous", "hallucinated", "refuted"}:
            return FALSE_LABEL
    raise ValueError("labels must be binary values in {0, 1}.")


def _decision_summary(actions: Sequence[str]) -> dict[str, Any]:
    accepted = sum(1 for action in actions if action == ACCEPT_ACTION)
    return {
        "action_counts": _counter_dict(actions),
        "acceptance": binomial_confidence_interval(accepted, len(actions)),
        "non_acceptance": binomial_confidence_interval(len(actions) - accepted, len(actions)),
    }


def _decision_change_summary(initial_actions: Sequence[str], final_actions: Sequence[str]) -> dict[str, Any]:
    transitions = Counter(
        f"{initial}->{final}"
        for initial, final in zip(initial_actions, final_actions, strict=True)
        if initial != final
    )
    changed = sum(transitions.values())
    return {
        "changed_records": changed,
        "change_rate": binomial_confidence_interval(changed, len(initial_actions)),
        "transition_counts": dict(sorted(transitions.items())),
        "accept_to_non_accept": sum(
            1
            for initial, final in zip(initial_actions, final_actions, strict=True)
            if initial == ACCEPT_ACTION and final != ACCEPT_ACTION
        ),
        "non_accept_to_accept": sum(
            1
            for initial, final in zip(initial_actions, final_actions, strict=True)
            if initial != ACCEPT_ACTION and final == ACCEPT_ACTION
        ),
    }


def _verification_status_summary(result_sequences: Sequence[Any]) -> dict[str, Any]:
    statuses: list[str] = []
    confidences: list[float] = []
    for raw_results in result_sequences:
        for result in _sequence_of_mappings(raw_results):
            status = result.get("status")
            if status is not None:
                statuses.append(str(status))
            confidence = _finite_float(result.get("confidence"))
            if confidence is not None:
                confidences.append(confidence)
    return {
        "status_counts": _counter_dict(statuses),
        "confidence_observations": len(confidences),
        "mean_confidence": None if not confidences else sum(confidences) / len(confidences),
        "min_confidence": None if not confidences else min(confidences),
    }


def _escalation_summary(plans: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    enabled = sum(1 for plan in plans if plan is not None)
    triggered = sum(1 for plan in plans if _plan_run_verifier(plan))
    verify_claim_total = 0
    skipped_claim_total = 0
    retrieval_query_total = 0
    route_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    selected_claim_counts: list[int] = []
    retrieval_query_counts: list[int] = []
    entity_sensitive_records = 0
    entity_sensitive_claim_total = 0
    entity_candidate_total = 0

    for plan in plans:
        if plan is None:
            continue
        verify_claim_ids = _string_sequence(plan.get("verify_claim_ids"))
        skipped_claim_ids = _string_sequence(plan.get("skipped_claim_ids"))
        retrieval_queries = _sequence_of_mappings(plan.get("retrieval_queries"))
        verify_claim_total += len(verify_claim_ids)
        skipped_claim_total += len(skipped_claim_ids)
        retrieval_query_total += len(retrieval_queries)
        selected_claim_counts.append(len(verify_claim_ids))
        retrieval_query_counts.append(len(retrieval_queries))
        for route_hint in _sequence_of_mappings(plan.get("route_hints")):
            route_counts.update(_string_sequence(route_hint.get("routes")))
        budget = _optional_mapping(plan.get("budget")) or {}
        escalation_budget = _optional_mapping(budget.get("uncertainty_escalation")) or {}
        reasons = _optional_mapping(escalation_budget.get("uncertainty_reasons")) or {}
        for raw_reason_list in reasons.values():
            reason_counts.update(_string_sequence(raw_reason_list))
        entity_candidates = _optional_mapping(escalation_budget.get("entity_candidates")) or {}
        if entity_candidates:
            entity_sensitive_records += 1
            entity_sensitive_claim_total += len(entity_candidates)
            entity_candidate_total += sum(
                len(_string_sequence(raw_candidates))
                for raw_candidates in entity_candidates.values()
            )

    return {
        "enabled_records": enabled,
        "enabled_rate": binomial_confidence_interval(enabled, len(plans)),
        "triggered_records": triggered,
        "trigger_rate": binomial_confidence_interval(triggered, len(plans)),
        "verify_claim_total": verify_claim_total,
        "skipped_claim_total": skipped_claim_total,
        "retrieval_query_total": retrieval_query_total,
        "mean_verify_claim_count": _mean_or_none(selected_claim_counts),
        "mean_retrieval_query_count": _mean_or_none(retrieval_query_counts),
        "route_counts": dict(sorted(route_counts.items())),
        "uncertainty_reason_counts": dict(sorted(reason_counts.items())),
        "entity_sensitive_records": entity_sensitive_records,
        "entity_sensitive_rate": binomial_confidence_interval(entity_sensitive_records, len(plans)),
        "entity_sensitive_claim_total": entity_sensitive_claim_total,
        "entity_candidate_total": entity_candidate_total,
    }


def _action_execution_summary(
    action_requests: Sequence[Sequence[Mapping[str, Any]]],
    action_results: Sequence[Sequence[Mapping[str, Any]]],
    retrieval_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    request_counts = [len(items) for items in action_requests]
    result_counts = [len(items) for items in action_results]
    retrieval_request_records = 0
    retrieval_request_total = 0
    retrieval_success_records = 0
    retrieval_success_total = 0
    retrieval_evidence_records = 0
    retrieval_evidence_total = 0

    for requests, results, evidence in zip(action_requests, action_results, retrieval_evidence, strict=True):
        retrieval_requests = [
            request
            for request in requests
            if _action_value(request.get("action")) == RETRIEVE_ACTION
        ]
        retrieval_results = [result for result in results if _action_value(result.get("action")) == RETRIEVE_ACTION]
        successful_retrieval_results = [
            result
            for result in retrieval_results
            if str(result.get("status", "")).lower() in {"succeeded", "success"}
        ]
        evidence_count = _non_negative_int(evidence.get("total_evidence"))
        if evidence_count is None:
            evidence_items = _sequence_of_mappings(evidence.get("evidence"))
            evidence_count = len(evidence_items)
        retrieval_request_total += len(retrieval_requests)
        retrieval_success_total += len(successful_retrieval_results)
        retrieval_evidence_total += evidence_count
        retrieval_request_records += int(bool(retrieval_requests))
        retrieval_success_records += int(bool(successful_retrieval_results))
        retrieval_evidence_records += int(evidence_count > 0)

    n_total = len(action_requests)
    return {
        "mean_action_request_count": _mean_or_none(request_counts),
        "mean_action_result_count": _mean_or_none(result_counts),
        "retrieval_request_records": retrieval_request_records,
        "retrieval_request_rate": binomial_confidence_interval(retrieval_request_records, n_total),
        "retrieval_request_total": retrieval_request_total,
        "retrieval_success_records": retrieval_success_records,
        "retrieval_success_rate": binomial_confidence_interval(retrieval_success_records, n_total),
        "retrieval_success_total": retrieval_success_total,
        "retrieval_evidence_records": retrieval_evidence_records,
        "retrieval_evidence_rate": binomial_confidence_interval(retrieval_evidence_records, n_total),
        "retrieval_evidence_total": retrieval_evidence_total,
    }


def _label_summary(labels: Sequence[int]) -> dict[str, Any]:
    true_count = sum(1 for label in labels if label == TRUE_LABEL)
    false_count = sum(1 for label in labels if label == FALSE_LABEL)
    return {
        "n_labeled": len(labels),
        "n_true": true_count,
        "n_false": false_count,
        "false_rate": binomial_confidence_interval(false_count, len(labels)),
    }


def _quality_summary(
    initial_actions: Sequence[str],
    final_actions: Sequence[str],
    labels: Sequence[int],
) -> dict[str, Any]:
    initial = _acceptance_quality(initial_actions, labels)
    final = _acceptance_quality(final_actions, labels)
    return {
        "initial": initial,
        "final": final,
        "delta": {
            "accepted_records": final["accepted_records"] - initial["accepted_records"],
            "accepted_true": final["accepted_true"] - initial["accepted_true"],
            "accepted_false": final["accepted_false"] - initial["accepted_false"],
            "coverage": _optional_delta(final["coverage"]["estimate"], initial["coverage"]["estimate"]),
            "selective_accuracy": _optional_delta(
                final["selective_accuracy"]["estimate"],
                initial["selective_accuracy"]["estimate"],
            ),
            "false_accept_rate": _optional_delta(
                final["false_accept_rate"]["estimate"],
                initial["false_accept_rate"]["estimate"],
            ),
            "true_accept_rate": _optional_delta(
                final["true_accept_rate"]["estimate"],
                initial["true_accept_rate"]["estimate"],
            ),
        },
    }


def _acceptance_quality(actions: Sequence[str], labels: Sequence[int]) -> dict[str, Any]:
    accepted = tuple(action == ACCEPT_ACTION for action in actions)
    true_mask = tuple(label == TRUE_LABEL for label in labels)
    false_mask = tuple(label == FALSE_LABEL for label in labels)
    accepted_records = sum(accepted)
    true_count = sum(true_mask)
    false_count = sum(false_mask)
    accepted_true = sum(1 for is_accepted, is_true in zip(accepted, true_mask, strict=True) if is_accepted and is_true)
    accepted_false = sum(
        1
        for is_accepted, is_false in zip(accepted, false_mask, strict=True)
        if is_accepted and is_false
    )
    routed_false = false_count - accepted_false
    return {
        "accepted_records": accepted_records,
        "accepted_true": accepted_true,
        "accepted_false": accepted_false,
        "routed_false": routed_false,
        "coverage": binomial_confidence_interval(accepted_records, len(actions)),
        "selective_accuracy": binomial_confidence_interval(accepted_true, accepted_records),
        "true_accept_rate": binomial_confidence_interval(accepted_true, true_count),
        "false_accept_rate": binomial_confidence_interval(accepted_false, false_count),
        "false_routing_rate": binomial_confidence_interval(routed_false, false_count),
    }


def _policy_sweep_row(
    records: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[int] | None,
    min_confidence: float,
    uncertain_statuses: set[str],
    constraints: Mapping[str, float | None],
) -> dict[str, Any]:
    simulated_records: list[Mapping[str, Any]] = []
    triggered_record_indexes: list[int] = []
    trigger_reason_counts: Counter[str] = Counter()
    for index, record in enumerate(records):
        triggered, reasons = _record_would_trigger_escalation(
            record,
            min_confidence=min_confidence,
            uncertain_statuses=uncertain_statuses,
        )
        trigger_reason_counts.update(reasons)
        if triggered:
            triggered_record_indexes.append(index)
            simulated_records.append(record)
        else:
            simulated_records.append(_record_without_escalation(record))

    report = uncertainty_escalation_report(simulated_records, labels=labels)
    quality = _optional_mapping(report.get("quality"))
    final_quality = _optional_mapping(quality.get("final")) if quality is not None else None
    action_execution = _optional_mapping(report.get("action_execution")) or {}
    escalation = _optional_mapping(report.get("uncertainty_escalation")) or {}
    trigger_rate = _estimate_from_interval(escalation.get("trigger_rate"))
    final_false_accept_rate = _estimate_from_interval(
        None if final_quality is None else final_quality.get("false_accept_rate")
    )
    final_selective_accuracy = _estimate_from_interval(
        None if final_quality is None else final_quality.get("selective_accuracy")
    )
    final_coverage = _estimate_from_interval(
        None if final_quality is None else final_quality.get("coverage")
    )
    retrieval_evidence_rate = _estimate_from_interval(
        action_execution.get("retrieval_evidence_rate")
    )
    gate = _policy_sweep_gate(
        final_false_accept_rate=final_false_accept_rate,
        final_selective_accuracy=final_selective_accuracy,
        trigger_rate=trigger_rate,
        constraints=constraints,
    )
    return {
        "min_confidence": min_confidence,
        "status": "promote" if gate["passed"] else "blocked",
        "gate": gate,
        "triggered_records": len(triggered_record_indexes),
        "triggered_record_indexes": tuple(triggered_record_indexes),
        "trigger_rate": trigger_rate,
        "trigger_reason_counts": dict(sorted(trigger_reason_counts.items())),
        "retrieval_evidence_rate": retrieval_evidence_rate,
        "final_coverage": final_coverage,
        "final_selective_accuracy": final_selective_accuracy,
        "final_false_accept_rate": final_false_accept_rate,
        "final_accepted_records": None if final_quality is None else final_quality.get("accepted_records"),
        "final_accepted_false": None if final_quality is None else final_quality.get("accepted_false"),
        "final_accepted_true": None if final_quality is None else final_quality.get("accepted_true"),
        "report": report,
    }


def _record_would_trigger_escalation(
    record: Mapping[str, Any],
    *,
    min_confidence: float,
    uncertain_statuses: set[str],
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    preliminary_results = _sequence_of_mappings(record.get("initial_verification_results"))
    if not preliminary_results:
        return True, ("missing_preliminary_result",)
    for result in preliminary_results:
        status = str(result.get("status", "")).strip().lower()
        if status in uncertain_statuses:
            reasons.append(f"status:{status}")
        confidence = _finite_float(result.get("confidence"))
        if confidence is None:
            reasons.append("confidence_missing")
        elif confidence < min_confidence:
            reasons.append(f"confidence_below:{min_confidence:g}")
    return bool(reasons), tuple(dict.fromkeys(reasons))


def _record_without_escalation(record: Mapping[str, Any]) -> dict[str, Any]:
    simulated = dict(record)
    simulated["final_decision"] = record.get("initial_decision")
    simulated["final_verification_results"] = record.get("initial_verification_results", ())
    simulated["uncertainty_escalation_plan"] = None
    simulated["action_requests"] = ()
    simulated["action_results"] = ()
    simulated["retrieval_evidence"] = {}
    return simulated


def _policy_sweep_gate(
    *,
    final_false_accept_rate: float | None,
    final_selective_accuracy: float | None,
    trigger_rate: float | None,
    constraints: Mapping[str, float | None],
) -> dict[str, Any]:
    failures: list[str] = []
    max_false_accept = constraints.get("max_final_false_accept_rate")
    min_selective_accuracy = constraints.get("min_final_selective_accuracy")
    max_trigger = constraints.get("max_trigger_rate")
    min_trigger = constraints.get("min_trigger_rate")
    if max_false_accept is not None:
        if final_false_accept_rate is None:
            failures.append("final_false_accept_rate is unavailable")
        elif final_false_accept_rate > max_false_accept:
            failures.append(
                f"final_false_accept_rate {final_false_accept_rate:.6g} "
                f"above {max_false_accept:.6g}"
            )
    if min_selective_accuracy is not None:
        if final_selective_accuracy is None:
            failures.append("final_selective_accuracy is unavailable")
        elif final_selective_accuracy < min_selective_accuracy:
            failures.append(
                f"final_selective_accuracy {final_selective_accuracy:.6g} "
                f"below {min_selective_accuracy:.6g}"
            )
    if max_trigger is not None:
        if trigger_rate is None:
            failures.append("trigger_rate is unavailable")
        elif trigger_rate > max_trigger:
            failures.append(f"trigger_rate {trigger_rate:.6g} above {max_trigger:.6g}")
    if min_trigger is not None:
        if trigger_rate is None:
            failures.append("trigger_rate is unavailable")
        elif trigger_rate < min_trigger:
            failures.append(f"trigger_rate {trigger_rate:.6g} below {min_trigger:.6g}")
    return {"passed": not failures, "blocking_reasons": tuple(failures)}


def _recommended_policy_sweep_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    passing = tuple(
        row
        for row in rows
        if (_optional_mapping(row.get("gate")) or {}).get("passed")
    )
    candidates = passing or tuple(rows)
    recommended = min(candidates, key=_policy_sweep_sort_key)
    payload = {
        key: value
        for key, value in recommended.items()
        if key != "report"
    }
    payload["selected_from_passing_rows"] = bool(passing)
    return payload


def _policy_sweep_sort_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    false_accept = _none_as(row.get("final_false_accept_rate"), 2.0)
    selective_accuracy = _none_as(row.get("final_selective_accuracy"), -1.0)
    coverage = _none_as(row.get("final_coverage"), -1.0)
    trigger_rate = _none_as(row.get("trigger_rate"), 2.0)
    threshold = _none_as(row.get("min_confidence"), 2.0)
    return (false_accept, -selective_accuracy, -coverage, trigger_rate, threshold)


def _sweep_thresholds(values: Sequence[float]) -> tuple[float, ...]:
    thresholds = tuple(sorted({_unit_float(value, name="min_confidence_values") for value in values}))
    if not thresholds:
        raise ValueError("min_confidence_values must contain at least one value.")
    return thresholds


def _unit_float(value: Any, *, name: str) -> float:
    number = _finite_float(value)
    if number is None or not (0.0 <= number <= 1.0):
        raise ValueError(f"{name} must contain values in [0, 1].")
    return number


def _optional_unit_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _unit_float(value, name=name)


def _estimate_from_interval(value: Any) -> float | None:
    mapping = _optional_mapping(value)
    if mapping is None:
        return None
    return _finite_float(mapping.get("estimate"))


def _none_as(value: Any, fallback: float) -> float:
    number = _finite_float(value)
    return fallback if number is None else number


def _to_mapping(item: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(item, Mapping):
        return item
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    raise ValueError("loop result records must be mappings or objects with to_dict().")


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _sequence_of_mappings(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _decision_action(value: Any) -> str:
    mapping = _optional_mapping(value) or {}
    return _action_value(mapping.get("action"))


def _action_value(value: Any) -> str:
    if value is None:
        return "unknown"
    raw_value = getattr(value, "value", value)
    return str(raw_value)


def _plan_run_verifier(plan: Mapping[str, Any] | None) -> bool:
    return bool(plan is not None and plan.get("run_verifier"))


def _counter_dict(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _non_negative_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _mean_or_none(values: Sequence[int]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _optional_delta(final: Any, initial: Any) -> float | None:
    if final is None or initial is None:
        return None
    return float(final) - float(initial)
