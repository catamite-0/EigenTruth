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
) -> dict[str, Any]:
    """Return score, claim, and state payloads for order-reservation transitions."""
    if n_records < 2:
        raise ValueError("n_records must be >= 2.")
    signal = signal.strip()
    if not signal:
        raise ValueError("signal must be non-empty.")

    state: dict[str, Any] = {"orders": {}, "inventory": {}}
    labels: list[int] = []
    scores: list[float] = []
    statements: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

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
        transition = {
            "action": {
                "decrement": {f"inventory.{sku}.available": quantity},
                "set": {f"orders.{order_id}.status": "reserved"},
            },
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

    return {
        "scores": {
            "schema_version": 1,
            "config": {
                "model": "synthetic-state-transition",
                "layer": -1,
                "fixture_type": "order_reservation_transition",
                "signal": signal,
                "n_records": n_records,
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
            },
        },
        "state": {
            "schema_version": 1,
            "fixture_type": "order_reservation_transition_state",
            "state": state,
            "summary": {
                "n_orders": n_records,
                "n_inventory_items": len(state["inventory"]),
            },
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fixture builder CLI."""
    payload = build_order_transition_fixture(
        n_records=args.n_records,
        signal=args.signal,
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
    run(parser.parse_args())


if __name__ == "__main__":
    main()
