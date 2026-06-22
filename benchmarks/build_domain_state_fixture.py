"""Build deterministic domain-state fixtures for verifier-ensemble benchmarks.

The generated order-fulfillment fixture is synthetic but not label-derived from
TruthfulQA. It creates explicit business state, claim-level ``state_check``
rules, and a compatible score dump so ``eval_verifier_ensemble.py`` can measure
structured-state route quality and downstream control metrics end to end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_order_fulfillment_fixture(
    *,
    n_records: int = 12,
    signal: str = "truth_proj",
) -> dict[str, Any]:
    """Return score, claim, and state payloads for order-fulfillment checks."""
    if n_records < 2:
        raise ValueError("n_records must be >= 2.")
    signal = signal.strip()
    if not signal:
        raise ValueError("signal must be non-empty.")

    state: dict[str, Any] = {"orders": {}, "inventory": {}, "accounts": {}}
    labels: list[int] = []
    scores: list[float] = []
    statements: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}

    for idx in range(n_records):
        item = _order_record(idx)
        order_id = item["order_id"]
        sku = item["sku"]
        account_id = item["account_id"]
        can_ship = bool(item["can_ship"])
        reason = str(item["reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        state["inventory"][sku] = {"available": item["available"]}
        state["accounts"][account_id] = {"status": item["account_status"]}
        state["orders"][order_id] = {
            "sku": sku,
            "account_id": account_id,
            "quantity": item["quantity"],
            "can_ship": can_ship,
            "reason": reason,
        }

        label = 0 if can_ship else 1
        score = _synthetic_score(idx, label)
        claim_id = f"{order_id}_can_ship"
        claim = f"Order {order_id} can ship now."
        state_check = {
            "path": f"orders.{order_id}.can_ship",
            "operator": "eq",
            "value": True,
            "source": "order_fulfillment_state",
        }
        metadata = {
            "domain": "order_fulfillment",
            "order_id": order_id,
            "sku": sku,
            "account_id": account_id,
            "quantity": item["quantity"],
            "expected_can_ship": can_ship,
            "failure_reason": None if can_ship else reason,
        }
        statement = {
            "claim": claim,
            "text": claim,
            "claim_id": claim_id,
            "metadata": {**metadata, "state_check": state_check},
            "state_check": state_check,
        }
        record = {
            "claim": claim,
            "claim_id": claim_id,
            "claim_metadata": {**metadata, "state_check": state_check},
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
                "model": "synthetic-domain-state",
                "layer": -1,
                "fixture_type": "order_fulfillment_state",
                "signal": signal,
                "n_records": n_records,
                "positive_label": "claim_refuted_by_domain_state",
            },
            "labels": labels,
            "scores": {signal: scores},
            "statements": statements,
        },
        "claims": {
            "schema_version": 1,
            "fixture_type": "order_fulfillment_state_claims",
            "description": (
                "Synthetic order-fulfillment claims with explicit state_check metadata. "
                "True labels are shippable orders; false labels claim shippability when "
                "inventory or account state says the order cannot ship."
            ),
            "records": records,
            "summary": {
                "n_records": n_records,
                "n_true": labels.count(0),
                "n_false": labels.count(1),
                "reason_counts": reason_counts,
            },
        },
        "state": {
            "schema_version": 1,
            "fixture_type": "order_fulfillment_state",
            "state": state,
            "summary": {
                "n_orders": n_records,
                "n_inventory_items": len(state["inventory"]),
                "n_accounts": len(state["accounts"]),
                "reason_counts": reason_counts,
            },
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the fixture builder CLI."""
    payload = build_order_fulfillment_fixture(
        n_records=args.n_records,
        signal=args.signal,
    )
    _write_json(Path(args.scores_output), payload["scores"])
    _write_json(Path(args.claims_output), payload["claims"])
    _write_json(Path(args.state_output), payload["state"])
    summary = payload["claims"]["summary"]
    print(
        "Wrote order-fulfillment fixture "
        f"({summary['n_true']} true / {summary['n_false']} false records)"
    )
    return payload


def _order_record(idx: int) -> dict[str, Any]:
    order_id = f"ord_{idx + 1:04d}"
    sku = f"sku_{idx + 1:04d}"
    account_id = f"acct_{idx + 1:04d}"
    quantity = 2 + (idx % 3)
    scenario = idx % 4
    if scenario in {0, 3}:
        available = quantity + 4
        account_status = "active"
        can_ship = True
        reason = "ok"
    elif scenario == 1:
        available = max(0, quantity - 1)
        account_status = "active"
        can_ship = False
        reason = "insufficient_inventory"
    else:
        available = quantity + 4
        account_status = "suspended"
        can_ship = False
        reason = "account_suspended"
    return {
        "order_id": order_id,
        "sku": sku,
        "account_id": account_id,
        "quantity": quantity,
        "available": available,
        "account_status": account_status,
        "can_ship": can_ship,
        "reason": reason,
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
    parser = argparse.ArgumentParser(description="Build deterministic domain-state verifier fixtures")
    parser.add_argument("--scores-output", required=True, help="path to write synthetic score dump JSON")
    parser.add_argument("--claims-output", required=True, help="path to write claim fixture JSON")
    parser.add_argument("--state-output", required=True, help="path to write structured state JSON")
    parser.add_argument("--n-records", type=int, default=12, help="number of synthetic order records")
    parser.add_argument("--signal", default="truth_proj", help="score signal name to emit")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
