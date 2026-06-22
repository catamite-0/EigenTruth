"""Production-like local tool verification loop demo.

This example does not load a language model or call a network service. It shows
how EigenTruth can sit around a normal product workflow:

1. Check pre-tool business state from a read-only SQLite source.
2. Execute or ingest a local tool result.
3. Map selected tool-output fields into structured verifier state.
4. Verify post-tool claims and emit a route-auditable ``ProductTrace``.

The demo intentionally keeps the "tool" as local JSON so the integration shape
is visible without adding a production tool runtime or new dependencies.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from eigentruth.adapters import (
    SQLiteStateQuery,
    SQLiteStateSource,
    StructuredStateVerifier,
    ToolOutputMapping,
    ToolOutputStateSource,
)
from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control import ProductTrace, RiskController, TraceEvent
from eigentruth.verify import Claim, RoutedVerifier, VerificationResult, VerifierRoute


def demo_artifact() -> CalibrationArtifact:
    """Return a toy artifact whose diagnostic is below threshold by default."""
    return CalibrationArtifact(
        model_id="production-tool-loop-demo",
        target_layer=-1,
        scores=(CalibrationScore("truth_proj", threshold=1.0, direction="higher", conformal_alpha=0.1),),
        eigentruth_version="0.1.0",
        calibration_dataset_metadata={
            "source": "examples/production_tool_loop_demo.py",
            "note": "toy threshold for product-control demonstration only",
        },
    )


def default_diagnostics() -> dict[str, float]:
    """Return low-risk diagnostics so verifier output drives the decision."""
    return {"truth_proj": 0.0}


def default_tool_output() -> dict[str, Any]:
    """Return deterministic local tool output for the reservation step."""
    return {
        "order_id": "ord_1",
        "sku": "sku_123",
        "reserved": 5,
        "remaining": 7,
        "status": "reserved",
        "payment_captured": False,
    }


def build_demo_database(path: Path) -> None:
    """Create a deterministic order/inventory/account SQLite fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.execute("create table inventory (sku text primary key, available integer not null)")
        connection.execute("create table accounts (id text primary key, status text not null)")
        connection.execute(
            "create table orders (id text primary key, sku text not null, account_id text not null, quantity integer)"
        )
        connection.execute("insert into inventory values (?, ?)", ("sku_123", 12))
        connection.execute("insert into accounts values (?, ?)", ("acct_1", "active"))
        connection.execute("insert into orders values (?, ?, ?, ?)", ("ord_1", "sku_123", "acct_1", 5))
        connection.commit()
    finally:
        connection.close()


def database_state_source(database_path: Path) -> SQLiteStateSource:
    """Return a SQLite source for pre-tool state checks."""
    return SQLiteStateSource(
        database_path,
        queries=(
            SQLiteStateQuery(
                path="orders.ord_1.can_reserve",
                sql=_CAN_RESERVE_SQL,
                params=("ord_1",),
                column="can_reserve",
                required=True,
            ),
            SQLiteStateQuery(
                path="orders.ord_1.quantity",
                sql="select quantity from orders where id = ?",
                params=("ord_1",),
                required=True,
            ),
        ),
    )


def tool_output_state_source(tool_output: dict[str, Any]) -> ToolOutputStateSource:
    """Map local reserve-inventory tool output into structured verifier state."""
    return ToolOutputStateSource(
        outputs={"reserve_inventory": tool_output},
        mappings=(
            ToolOutputMapping(
                state_path="reservation.order_id",
                output_path="reserve_inventory.order_id",
                required=True,
            ),
            ToolOutputMapping(
                state_path="reservation.remaining",
                output_path="reserve_inventory.remaining",
                required=True,
            ),
            ToolOutputMapping(
                state_path="reservation.status",
                output_path="reserve_inventory.status",
                required=True,
            ),
            ToolOutputMapping(
                state_path="reservation.payment_captured",
                output_path="reserve_inventory.payment_captured",
                required=True,
            ),
        ),
    )


def pre_tool_claims() -> tuple[Claim, ...]:
    """Return pre-tool claims checked against SQLite state."""
    return (
        Claim(
            "Order ord_1 can be reserved now.",
            claim_id="pre-can-reserve",
            metadata={
                "state_check": {
                    "path": "orders.ord_1.can_reserve",
                    "operator": "eq",
                    "value": 1,
                    "source": "sqlite:orders",
                }
            },
        ),
    )


def post_tool_claims() -> tuple[Claim, ...]:
    """Return post-tool claims checked against local tool output."""
    return (
        Claim(
            "The reservation left seven units of SKU 123 available.",
            claim_id="post-remaining-supported",
            metadata={
                "state_check": {
                    "path": "reservation.remaining",
                    "operator": "eq",
                    "value": 7,
                    "source": "tool:reserve_inventory",
                }
            },
        ),
        Claim(
            "The reservation captured payment.",
            claim_id="post-payment-refuted",
            metadata={
                "state_check": {
                    "path": "reservation.payment_captured",
                    "operator": "eq",
                    "value": True,
                    "source": "tool:reserve_inventory",
                }
            },
        ),
    )


def parse_json_object(value: str | None, *, default: dict[str, Any], name: str) -> dict[str, Any]:
    """Parse a JSON object argument or return a copy of the default."""
    if value is None:
        return dict(default)
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return dict(payload)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the local tool loop and return a JSON-ready trace."""
    if args.database is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            database_path = Path(tmpdir) / "eigentruth_tool_loop.db"
            build_demo_database(database_path)
            return _run_with_database(args, database_path, temporary=True)
    database_path = Path(args.database)
    if args.seed_database:
        build_demo_database(database_path)
    return _run_with_database(args, database_path, temporary=False)


def _run_with_database(args: argparse.Namespace, database_path: Path, *, temporary: bool) -> dict[str, Any]:
    artifact = demo_artifact()
    diagnostics = {
        str(key): float(value)
        for key, value in parse_json_object(
            args.diagnostics,
            default=default_diagnostics(),
            name="--diagnostics",
        ).items()
    }
    tool_output = parse_json_object(args.tool_output, default=default_tool_output(), name="--tool-output")

    database_verifier = RoutedVerifier((
        VerifierRoute(
            "database_state",
            StructuredStateVerifier.from_source(database_state_source(database_path)),
            metadata_keys=("state_check",),
        ),
    ))
    tool_verifier = RoutedVerifier((
        VerifierRoute(
            "tool_output_state",
            StructuredStateVerifier.from_source(tool_output_state_source(tool_output)),
            metadata_keys=("state_check",),
        ),
    ))

    pre_claims = pre_tool_claims()
    post_claims = post_tool_claims()
    pre_results = tuple(database_verifier.verify_many(pre_claims))
    initial_decision = RiskController(artifact).decide(diagnostics, verification_results=pre_results)
    post_results = tuple(tool_verifier.verify_many(post_claims))
    claims = pre_claims + post_claims
    results = pre_results + post_results
    final_decision = RiskController(artifact).decide(diagnostics, verification_results=results)
    route_summary = ProductTrace(verification_results=results).verification_route_summary()

    trace = ProductTrace(
        request_id=args.request_id,
        diagnostics=diagnostics,
        claims=claims,
        verification_results=results,
        risk_decision=final_decision,
        events=(
            TraceEvent(
                "pre_tool_verification",
                {
                    "n_claims": len(pre_claims),
                    "decision": initial_decision.to_dict(),
                    "results": tuple(_result_payload(result) for result in pre_results),
                },
            ),
            TraceEvent(
                "local_tool_output_ingested",
                {
                    "tool": "reserve_inventory",
                    "output_keys": tuple(sorted(tool_output)),
                    "side_effects": False,
                },
            ),
            TraceEvent(
                "post_tool_verification",
                {
                    "n_claims": len(post_claims),
                    "results": tuple(_result_payload(result) for result in post_results),
                },
            ),
            TraceEvent("final_risk_decision", final_decision.to_dict()),
        ),
        metadata={
            "source": "examples/production_tool_loop_demo.py",
            "artifact_model_id": artifact.model_id,
            "database_path": "<temporary>" if temporary else str(database_path),
            "database_seeded": bool(args.seed_database),
            "tool": "reserve_inventory",
            "business_domain": "order_reservation",
            "route_summary": route_summary,
        },
    )
    payload = trace.to_dict()
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _result_payload(result: VerificationResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "confidence": result.confidence,
        "explanation": result.explanation,
        "metadata": dict(result.metadata),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EigenTruth production-like local tool loop demo")
    parser.add_argument("--database", default=None, help="optional SQLite database path")
    parser.add_argument("--no-seed-database", dest="seed_database", action="store_false",
                        help="use an existing database instead of creating the demo fixture")
    parser.set_defaults(seed_database=True)
    parser.add_argument("--diagnostics", default=None,
                        help="diagnostics JSON object; defaults below the toy threshold")
    parser.add_argument("--tool-output", default=None,
                        help="reserve-inventory tool output JSON object; defaults to deterministic output")
    parser.add_argument("--request-id", default="production-tool-loop-demo",
                        help="request id stored in ProductTrace")
    parser.add_argument("--output", default=None, help="optional path to write the trace JSON")
    payload = run(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


_CAN_RESERVE_SQL = """
select
  case
    when inventory.available >= orders.quantity and accounts.status = 'active' then 1
    else 0
  end as can_reserve
from orders
join inventory on inventory.sku = orders.sku
join accounts on accounts.id = orders.account_id
where orders.id = ?
"""


if __name__ == "__main__":
    main()
