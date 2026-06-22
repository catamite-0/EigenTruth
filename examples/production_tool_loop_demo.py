"""Production-like local tool verification loop demo.

This example does not load a language model or call a network service. It shows
how EigenTruth can sit around a normal product workflow:

1. Check pre-tool business state from a read-only SQLite source.
2. Execute a local SQLite-backed reserve-inventory tool.
3. Map selected tool-output fields into structured verifier state.
4. Optionally replay repeated executions from a JSON idempotency ledger.
5. Verify post-tool claims and emit a route-auditable ``ProductTrace``.

The demo intentionally keeps the tool local and deterministic so the integration
shape is visible without adding a production tool runtime or new dependencies.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from dataclasses import dataclass
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
from eigentruth.control import (
    ActionExecutionPolicy,
    ActionExecutionStatus,
    ActionExecutorRegistry,
    ActionRequest,
    ActionResult,
    ControlAction,
    JsonActionExecutionLedger,
    PolicyGuardedActionExecutor,
    ProductTrace,
    RiskController,
    TraceEvent,
)
from eigentruth.verify import Claim, RoutedVerifier, VerificationResult, VerificationStatus, VerifierRoute


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


def default_tool_input() -> dict[str, Any]:
    """Return deterministic local tool input for the reservation step."""
    return {
        "order_id": "ord_1",
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
            "create table orders (id text primary key, sku text not null, account_id text not null, "
            "quantity integer, status text not null)"
        )
        connection.execute("insert into inventory values (?, ?)", ("sku_123", 12))
        connection.execute("insert into accounts values (?, ?)", ("acct_1", "active"))
        connection.execute("insert into orders values (?, ?, ?, ?, ?)", ("ord_1", "sku_123", "acct_1", 5, "pending"))
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


@dataclass(frozen=True)
class SQLiteReserveInventoryExecutor:
    """Local side-effecting reserve-inventory executor for the demo database."""

    database_path: Path

    def execute(self, request: ActionRequest, context: dict[str, Any] | None = None) -> ActionResult:
        """Reserve inventory for one order and return structured tool output."""
        if request.action is not ControlAction.EXECUTE_TOOL:
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.FAILED,
                request_id=request.request_id,
                error="SQLiteReserveInventoryExecutor only supports execute_tool actions.",
                metadata={"executor": type(self).__name__, "side_effects": False},
            )
        tool = request.payload.get("tool")
        if tool != "reserve_inventory":
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.FAILED,
                request_id=request.request_id,
                error=f"unsupported tool: {tool!r}",
                metadata={"executor": type(self).__name__, "side_effects": False},
            )
        input_payload = request.payload.get("input", {})
        if not isinstance(input_payload, dict):
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.FAILED,
                request_id=request.request_id,
                error="execute_tool input must be a JSON object.",
                metadata={"executor": type(self).__name__, "side_effects": False},
            )
        order_id = str(input_payload.get("order_id", "")).strip()
        if not order_id:
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.FAILED,
                request_id=request.request_id,
                error="reserve_inventory requires input.order_id.",
                metadata={"executor": type(self).__name__, "side_effects": False},
            )
        try:
            output = self._reserve(order_id)
        except ValueError as exc:
            return ActionResult(
                action=request.action,
                status=ActionExecutionStatus.FAILED,
                request_id=request.request_id,
                error=str(exc),
                metadata={"executor": type(self).__name__, "side_effects": False},
            )
        return ActionResult(
            action=request.action,
            status=ActionExecutionStatus.SUCCEEDED,
            output=output,
            metadata={
                "executor": type(self).__name__,
                "tool": "reserve_inventory",
                "side_effects": True,
                "context": dict(context or {}),
            },
            request_id=request.request_id,
        )

    def execute_many(
        self,
        requests: tuple[ActionRequest, ...],
        context: dict[str, Any] | None = None,
    ) -> tuple[ActionResult, ...]:
        """Execute multiple reserve requests."""
        return tuple(self.execute(request, context=context) for request in requests)

    def _reserve(self, order_id: str) -> dict[str, Any]:
        path = str(self.database_path)
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("begin immediate")
            row = connection.execute(
                """
                select orders.id, orders.sku, orders.quantity, orders.status, accounts.status as account_status,
                       inventory.available
                from orders
                join accounts on accounts.id = orders.account_id
                join inventory on inventory.sku = orders.sku
                where orders.id = ?
                """,
                (order_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"order not found: {order_id}")
            if row["status"] != "pending":
                raise ValueError(f"order is not pending: {order_id}")
            if row["account_status"] != "active":
                raise ValueError(f"account is not active for order: {order_id}")
            if int(row["available"]) < int(row["quantity"]):
                raise ValueError(f"insufficient inventory for order: {order_id}")
            remaining = int(row["available"]) - int(row["quantity"])
            connection.execute(
                "update inventory set available = ? where sku = ?",
                (remaining, row["sku"]),
            )
            connection.execute(
                "update orders set status = ? where id = ?",
                ("reserved", order_id),
            )
            connection.commit()
            return {
                "order_id": order_id,
                "sku": row["sku"],
                "reserved": int(row["quantity"]),
                "remaining": remaining,
                "status": "reserved",
                "payment_captured": False,
            }
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def tool_output_state_source(action_results: tuple[ActionResult, ...]) -> ToolOutputStateSource:
    """Map local reserve-inventory tool output into structured verifier state."""
    return ToolOutputStateSource(
        action_results=action_results,
        mappings=(
            ToolOutputMapping(
                state_path="reservation.order_id",
                output_path="order_id",
                action=ControlAction.EXECUTE_TOOL,
                request_id="reserve-ord-1",
                required=True,
            ),
            ToolOutputMapping(
                state_path="reservation.remaining",
                output_path="remaining",
                action=ControlAction.EXECUTE_TOOL,
                request_id="reserve-ord-1",
                required=True,
            ),
            ToolOutputMapping(
                state_path="reservation.status",
                output_path="status",
                action=ControlAction.EXECUTE_TOOL,
                request_id="reserve-ord-1",
                required=True,
            ),
            ToolOutputMapping(
                state_path="reservation.payment_captured",
                output_path="payment_captured",
                action=ControlAction.EXECUTE_TOOL,
                request_id="reserve-ord-1",
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
    tool_input = parse_json_object(args.tool_input, default=default_tool_input(), name="--tool-input")

    database_verifier = RoutedVerifier((
        VerifierRoute(
            "database_state",
            StructuredStateVerifier.from_source(database_state_source(database_path)),
            metadata_keys=("state_check",),
        ),
    ))
    pre_claims = pre_tool_claims()
    post_claims = post_tool_claims()
    pre_results = tuple(database_verifier.verify_many(pre_claims))
    initial_decision = RiskController(artifact).decide(diagnostics, verification_results=pre_results)
    order_id = str(tool_input.get("order_id", "")).strip() or "<missing>"
    tool_request = ActionRequest(
        action=ControlAction.EXECUTE_TOOL,
        reason="pre-tool verification supported reservation",
        payload={
            "tool": "reserve_inventory",
            "input": tool_input,
            "instruction": "reserve inventory only after pre-tool state checks pass",
        },
        metadata={
            "idempotency_key": f"reserve_inventory:{args.request_id}:{order_id}",
            "timeout_seconds": 5.0,
        },
        request_id="reserve-ord-1",
    )
    registry = ActionExecutorRegistry().register(
        ControlAction.EXECUTE_TOOL,
        PolicyGuardedActionExecutor(
            SQLiteReserveInventoryExecutor(database_path),
            policy=ActionExecutionPolicy(
                side_effecting=True,
                require_request_id=True,
                require_idempotency_key=True,
                default_timeout_seconds=5.0,
                max_timeout_seconds=30.0,
            ),
            idempotency_ledger=(
                JsonActionExecutionLedger(args.execution_ledger)
                if getattr(args, "execution_ledger", None)
                else None
            ),
        ),
    )
    action_results = (
        registry.execute_many((tool_request,), context={"request_id": args.request_id})
        if initial_decision.action is ControlAction.ACCEPT
        else ()
    )
    if action_results and action_results[0].status is ActionExecutionStatus.SUCCEEDED:
        tool_verifier = RoutedVerifier((
            VerifierRoute(
                "tool_output_state",
                StructuredStateVerifier.from_source(tool_output_state_source(action_results)),
                metadata_keys=("state_check",),
            ),
        ))
        post_results = tuple(tool_verifier.verify_many(post_claims))
    else:
        post_results = tuple(_tool_not_executed_result(claim, action_results=action_results) for claim in post_claims)
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
        actions=(tool_request,),
        action_results=action_results,
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
                "local_tool_executed",
                {
                    "tool": "reserve_inventory",
                    "status": action_results[0].status.value if action_results else "skipped",
                    "output_keys": tuple(sorted(action_results[0].output)) if action_results else (),
                    "side_effects": _action_results_side_effects(action_results),
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
            "execution_ledger_path": (
                None if not getattr(args, "execution_ledger", None) else str(args.execution_ledger)
            ),
            "tool": "reserve_inventory",
            "business_domain": "order_reservation",
            "route_summary": route_summary,
            "action_execution_summary": ProductTrace(action_results=action_results).action_execution_summary(),
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


def _action_results_side_effects(action_results: tuple[ActionResult, ...]) -> bool:
    return any(bool(result.metadata.get("side_effects", False)) for result in action_results)


def _tool_not_executed_result(
    claim: Claim,
    *,
    action_results: tuple[ActionResult, ...],
) -> VerificationResult:
    status = action_results[0].status.value if action_results else "skipped"
    error = action_results[0].error if action_results else "pre-tool verification did not allow execution"
    return VerificationResult(
        status=VerificationStatus.INSUFFICIENT_EVIDENCE,
        confidence=0.2,
        explanation="post-tool verification skipped because tool did not produce successful output",
        metadata={
            "claim_id": claim.claim_id,
            "verifier": "tool_output_state",
            "decision_rule": "tool_output_missing",
            "tool_status": status,
            "tool_error": error,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="EigenTruth production-like local tool loop demo")
    parser.add_argument("--database", default=None, help="optional SQLite database path")
    parser.add_argument("--no-seed-database", dest="seed_database", action="store_false",
                        help="use an existing database instead of creating the demo fixture")
    parser.set_defaults(seed_database=True)
    parser.add_argument("--diagnostics", default=None,
                        help="diagnostics JSON object; defaults below the toy threshold")
    parser.add_argument("--tool-input", default=None,
                        help="reserve-inventory tool input JSON object; defaults to deterministic input")
    parser.add_argument("--request-id", default="production-tool-loop-demo",
                        help="request id stored in ProductTrace")
    parser.add_argument("--execution-ledger", default=None,
                        help="optional JSON idempotency ledger for local tool execution")
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
