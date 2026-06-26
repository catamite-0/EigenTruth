"""Build deterministic state-transition fixtures for verifier-ensemble benchmarks.

The generated fixture checks action consequences instead of static state. Each
record provides a ``state_transition`` rule: apply a local order reservation
action, predict the next state, then verify a structured postcondition over the
predicted inventory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_order_transition_fixture(
    *,
    n_records: int = 12,
    signal: str = "truth_proj",
    rule_based_world_model: bool = False,
    world_model_ensemble: bool = False,
    world_model_ensemble_min_agreement: float = 0.75,
    world_model_ensemble_strategy: str = "label_stress",
) -> dict[str, Any]:
    """Return score, claim, and state payloads for order-reservation transitions."""
    if n_records < 2:
        raise ValueError("n_records must be >= 2.")
    signal = signal.strip()
    if not signal:
        raise ValueError("signal must be non-empty.")
    if not (0.0 < float(world_model_ensemble_min_agreement) <= 1.0):
        raise ValueError("world_model_ensemble_min_agreement must be in (0, 1].")
    world_model_ensemble_strategy = world_model_ensemble_strategy.strip()
    if world_model_ensemble_strategy not in {"label_stress", "policy_replay"}:
        raise ValueError("world_model_ensemble_strategy must be 'label_stress' or 'policy_replay'.")

    state: dict[str, Any] = {"orders": {}, "inventory": {}}
    labels: list[int] = []
    scores: list[float] = []
    statements: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    world_model_rules: list[dict[str, Any]] = []
    ensemble_members = (
        {
            "name": "baseline_a",
            "world_model_rules": [],
            "metadata": {"role": "correct_transition_member"},
        },
        {
            "name": "baseline_b",
            "world_model_rules": [],
            "metadata": {"role": "correct_transition_member"},
        },
        {
            "name": "stress_member",
            "world_model_rules": [],
            "metadata": {"role": "label-correlated_divergence_stress_member"},
        },
    )

    for idx in range(n_records):
        item = _transition_record(idx)
        order_id = item["order_id"]
        sku = item["sku"]
        quantity = item["quantity"]
        available = item["available"]
        expected_remaining = available - quantity
        is_false = idx % 2 == 1
        claimed_remaining = expected_remaining + 1 if is_false else expected_remaining

        state["inventory"][sku] = {"available": available}
        state["orders"][order_id] = {
            "sku": sku,
            "quantity": quantity,
            "status": "pending",
        }

        label = 1 if is_false else 0
        score = _synthetic_score(idx, label)
        claim_id = f"{order_id}_reserve_remaining"
        claim = (
            f"After reserving order {order_id}, inventory for {sku} "
            f"will be {claimed_remaining} units."
        )
        if rule_based_world_model or world_model_ensemble:
            action = {
                "type": "reserve_order",
                "order_id": order_id,
                "sku": sku,
            }
            base_rule = _reservation_rule(
                order_id=order_id,
                sku=sku,
                quantity=quantity,
                action=action,
                member="baseline" if world_model_ensemble else "single",
            )
            if rule_based_world_model and not world_model_ensemble:
                world_model_rules.append(base_rule)
            if world_model_ensemble:
                ensemble_members[0]["world_model_rules"].append(dict(base_rule))
                ensemble_members[1]["world_model_rules"].append(dict(base_rule))
                should_stress = _should_stress_world_model_member(
                    strategy=world_model_ensemble_strategy,
                    quantity=quantity,
                    is_false=is_false,
                )
                stress_quantity = max(0, quantity - 1) if should_stress else quantity
                ensemble_members[2]["world_model_rules"].append(
                    _reservation_rule(
                        order_id=order_id,
                        sku=sku,
                        quantity=stress_quantity,
                        action=action,
                        member="stress",
                        explanation=_stress_member_explanation(
                            strategy=world_model_ensemble_strategy,
                            stressed=should_stress,
                        ),
                    )
                )
        else:
            action = {
                "decrement": {f"inventory.{sku}.available": quantity},
                "set": {f"orders.{order_id}.status": "reserved"},
            }
        transition = {
            "action": action,
            "postcondition": {
                "path": f"inventory.{sku}.available",
                "operator": "eq",
                "value": claimed_remaining,
                "source": "order_reservation_transition",
            },
            "source": "order_transition_fixture",
            "metadata": {
                "order_id": order_id,
                "sku": sku,
                "quantity": quantity,
                "initial_available": available,
                "expected_remaining": expected_remaining,
                "claimed_remaining": claimed_remaining,
            },
        }
        metadata = {
            "domain": "order_reservation",
            "order_id": order_id,
            "sku": sku,
            "quantity": quantity,
            "initial_available": available,
            "expected_remaining": expected_remaining,
            "claimed_remaining": claimed_remaining,
            "claim_is_false": is_false,
        }
        statement = {
            "claim": claim,
            "text": claim,
            "claim_id": claim_id,
            "metadata": {**metadata, "state_transition": transition},
            "state_transition": transition,
        }
        record = {
            "claim": claim,
            "claim_id": claim_id,
            "claim_metadata": {**metadata, "state_transition": transition},
            "metadata": {
                "index": idx,
                "score_label": label,
                "synthetic_score": score,
                **metadata,
            },
        }

        labels.append(label)
        scores.append(score)
        statements.append(statement)
        records.append(record)

    world_model_ensemble_payload = None
    if world_model_ensemble:
        world_model_ensemble_payload = {
            "type": "rule_based",
            "min_agreement": float(world_model_ensemble_min_agreement),
            "members": [
                {
                    "name": str(member["name"]),
                    "world_model_rules": list(member["world_model_rules"]),
                    "metadata": dict(member["metadata"]),
                }
                for member in ensemble_members
            ],
            "metadata": {
                "fixture": "order_reservation_transition",
                "strategy": world_model_ensemble_strategy,
                "divergence_pattern": _ensemble_divergence_pattern(world_model_ensemble_strategy),
            },
        }
    world_model_rule_count = (
        sum(len(member["world_model_rules"]) for member in ensemble_members)
        if world_model_ensemble
        else len(world_model_rules)
    )

    return {
        "scores": {
            "schema_version": 1,
            "config": {
                "model": "synthetic-state-transition",
                "layer": -1,
                "fixture_type": "order_reservation_transition",
                "signal": signal,
                "n_records": n_records,
                "world_model_fixture": (
                    "ensemble"
                    if world_model_ensemble
                    else "rule_based"
                    if rule_based_world_model
                    else "direct_action"
                ),
                "positive_label": "claim_refuted_by_predicted_transition",
            },
            "labels": labels,
            "scores": {signal: scores},
            "statements": statements,
        },
        "claims": {
            "schema_version": 1,
            "fixture_type": "order_transition_state_claims",
            "description": (
                "Synthetic order-reservation claims with explicit state_transition "
                "metadata. True labels match the predicted inventory after the action; "
                "false labels use an off-by-one postcondition."
            ),
            "records": records,
            "summary": {
                "n_records": n_records,
                "n_true": labels.count(0),
                "n_false": labels.count(1),
                "n_world_model_rules": world_model_rule_count,
                "n_world_model_ensemble_members": len(ensemble_members) if world_model_ensemble else 0,
                "world_model_ensemble_strategy": world_model_ensemble_strategy if world_model_ensemble else None,
                "world_model_ensemble_min_agreement": (
                    float(world_model_ensemble_min_agreement) if world_model_ensemble else None
                ),
            },
        },
        "state": {
            "schema_version": 1,
            "fixture_type": "order_reservation_transition_state",
            "state": state,
            **({"world_model_rules": world_model_rules} if rule_based_world_model and not world_model_ensemble else {}),
            **({"world_model_ensemble": world_model_ensemble_payload} if world_model_ensemble else {}),
            "summary": {
                "n_orders": n_records,
                "n_inventory_items": len(state["inventory"]),
                "n_world_model_rules": world_model_rule_count,
                "n_world_model_ensemble_members": len(ensemble_members) if world_model_ensemble else 0,
                "world_model_ensemble_strategy": world_model_ensemble_strategy if world_model_ensemble else None,
                "world_model_ensemble_min_agreement": (
                    float(world_model_ensemble_min_agreement) if world_model_ensemble else None
                ),
            },
        },
    }


def _should_stress_world_model_member(
    *,
    strategy: str,
    quantity: int,
    is_false: bool,
) -> bool:
    if strategy == "label_stress":
        return bool(is_false)
    if strategy == "policy_replay":
        return int(quantity) >= 3
    raise ValueError("unknown world-model ensemble strategy.")


def _stress_member_explanation(*, strategy: str, stressed: bool) -> str:
    if not stressed:
        return "stress member agrees with baseline transition policy"
    if strategy == "label_stress":
        return "stress member under-reserves false-labeled fixture records"
    return "stress member applies conservative high-quantity reservation policy"


def _ensemble_divergence_pattern(strategy: str) -> str:
    if strategy == "label_stress":
        return "stress_member_under_reserves_false_records"
    if strategy == "policy_replay":
        return "stress_member_under_reserves_high_quantity_records"
    raise ValueError("unknown world-model ensemble strategy.")


def _reservation_rule(
    *,
    order_id: str,
    sku: str,
    quantity: int,
    action: dict[str, Any],
    member: str,
    explanation: str = "reserve pending order inventory",
) -> dict[str, Any]:
    name = f"reserve_{order_id}" if member == "single" else f"{member}_reserve_{order_id}"
    return {
        "name": name,
        "action": dict(action),
        "when": (
            {
                "path": f"inventory.{sku}.available",
                "operator": "gte",
                "value": quantity,
                "source": "order_reservation_precondition",
            },
            {
                "path": f"orders.{order_id}.status",
                "operator": "eq",
                "value": "pending",
                "source": "order_reservation_precondition",
            },
        ),
        "decrement": {f"inventory.{sku}.available": quantity},
        "set": {f"orders.{order_id}.status": "reserved"},
        "confidence": 0.95,
        "explanation": explanation,
        "metadata": {
            "order_id": order_id,
            "sku": sku,
            "quantity": quantity,
            "member": member,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fixture builder CLI."""
    payload = build_order_transition_fixture(
        n_records=args.n_records,
        signal=args.signal,
        rule_based_world_model=bool(getattr(args, "rule_based_world_model", False)),
        world_model_ensemble=bool(getattr(args, "world_model_ensemble", False)),
        world_model_ensemble_min_agreement=float(
            getattr(args, "world_model_ensemble_min_agreement", 0.75)
        ),
        world_model_ensemble_strategy=str(
            getattr(args, "world_model_ensemble_strategy", "label_stress")
        ),
    )
    _write_json(Path(args.scores_output), payload["scores"])
    _write_json(Path(args.claims_output), payload["claims"])
    _write_json(Path(args.state_output), payload["state"])
    summary = payload["claims"]["summary"]
    print(
        "Wrote order-transition fixture "
        f"({summary['n_true']} true / {summary['n_false']} false records)"
    )
    return payload


def _transition_record(idx: int) -> dict[str, Any]:
    order_id = f"ord_{idx + 1:04d}"
    sku = f"sku_{idx + 1:04d}"
    quantity = 1 + (idx % 3)
    available = quantity + 5 + (idx % 2)
    return {
        "order_id": order_id,
        "sku": sku,
        "quantity": quantity,
        "available": available,
    }


def _synthetic_score(idx: int, label: int) -> float:
    jitter = 0.015 * (idx % 5)
    if label == 1:
        return round(0.62 + jitter, 6)
    return round(0.22 + jitter, 6)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic state-transition verifier fixtures")
    parser.add_argument("--scores-output", required=True, help="path to write synthetic score dump JSON")
    parser.add_argument("--claims-output", required=True, help="path to write claim fixture JSON")
    parser.add_argument("--state-output", required=True, help="path to write structured state JSON")
    parser.add_argument("--n-records", type=int, default=12, help="number of synthetic order records")
    parser.add_argument("--signal", default="truth_proj", help="score signal name to emit")
    parser.add_argument(
        "--rule-based-world-model",
        action="store_true",
        help="emit typed actions plus world_model_rules for RuleBasedWorldModelAdapter",
    )
    parser.add_argument(
        "--world-model-ensemble",
        action="store_true",
        help="emit a rule-based EnsembleWorldModelAdapter fixture with controlled member disagreement",
    )
    parser.add_argument(
        "--world-model-ensemble-min-agreement",
        type=float,
        default=0.75,
        help="minimum ensemble agreement required before transition verification can decide",
    )
    parser.add_argument(
        "--world-model-ensemble-strategy",
        choices=("label_stress", "policy_replay"),
        default="label_stress",
        help="controlled ensemble disagreement strategy",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
