"""SQLite-backed structured-state control demo.

This example does not load a language model. It demonstrates a product-facing
closed loop where a calibrated diagnostic is low, but a deterministic SQLite
business-state verifier refutes one claim and drives the final action to
``abstain``.

The database is a small local fixture containing inventory, account, and order
rows. Explicit read-only SQL queries map database values into nested verifier
state, and `StateCheck` metadata on claims defines the business assertions.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from eigentruth.adapters import SQLiteStateQuery, SQLiteStateSource, StructuredStateVerifier
from eigentruth.calibration import CalibrationArtifact, CalibrationScore
from eigentruth.control import ActionExecutorRegistry, RiskController, run_verification_loop
from eigentruth.verify import Claim


def demo_artifact() -> CalibrationArtifact:
    """Return a toy artifact whose diagnostic is below threshold by default."""
    return CalibrationArtifact(
        model_id="sqlite-state-demo",
        target_layer=-1,
        scores=(CalibrationScore("truth_proj", threshold=1.0, direction="higher", conformal_alpha=0.1),),
        eigentruth_version="0.1.0",
        calibration_dataset_metadata={
            "source": "examples/sqlite_state_control_demo.py",
            "note": "toy threshold for product-control demonstration only",
        },
    )


def default_diagnostics() -> dict[str, float]:
    """Return low-risk diagnostics so verifier output drives the decision."""
    return {"truth_proj": 0.0}


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
        connection.executemany(
            "insert into inventory values (?, ?)",
            (
                ("sku_123", 12),
                ("sku_999", 2),
            ),
        )
        connection.executemany(
            "insert into accounts values (?, ?)",
            (
                ("acct_1", "active"),
                ("acct_2", "suspended"),
            ),
        )
        connection.executemany(
            "insert into orders values (?, ?, ?, ?)",
            (
                ("ord_1", "sku_123", "acct_1", 5),
                ("ord_2", "sku_999", "acct_2", 4),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def demo_state_source(database_path: Path) -> SQLiteStateSource:
    """Return a SQLite state source for the demo database."""
    return SQLiteStateSource(
        database_path,
        queries=(
            SQLiteStateQuery(
                path="orders.ord_1.can_ship",
                sql=_CAN_SHIP_SQL,
                params=("ord_1",),
                column="can_ship",
                required=True,
            ),
            SQLiteStateQuery(
                path="orders.ord_2.can_ship",
                sql=_CAN_SHIP_SQL,
                params=("ord_2",),
                column="can_ship",
                required=True,
            ),
            SQLiteStateQuery(
                path="inventory.sku_123.available",
                sql="select available from inventory where sku = ?",
                params=("sku_123",),
                required=True,
            ),
        ),
    )


def demo_claims() -> tuple[Claim, ...]:
    """Return business claims backed by state checks."""
    return (
        Claim(
            "Order ord_1 can ship now.",
            claim_id="order-ord-1",
            metadata={
                "state_check": {
                    "path": "orders.ord_1.can_ship",
                    "operator": "eq",
                    "value": 1,
                    "source": "sqlite:orders",
                }
            },
        ),
        Claim(
            "Order ord_2 can ship now.",
            claim_id="order-ord-2",
            metadata={
                "state_check": {
                    "path": "orders.ord_2.can_ship",
                    "operator": "eq",
                    "value": 1,
                    "source": "sqlite:orders",
                }
            },
        ),
    )


def parse_diagnostics(value: str | None) -> dict[str, float]:
    """Parse diagnostics JSON or return the default low-risk values."""
    if value is None:
        return default_diagnostics()
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--diagnostics must be a JSON object.")
    return {str(key): float(item) for key, item in payload.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the SQLite state control demo and return a JSON-ready trace."""
    if args.database is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            database_path = Path(tmpdir) / "eigentruth_state_demo.db"
            build_demo_database(database_path)
            return _run_with_database(args, database_path, temporary=True)
    database_path = Path(args.database)
    if args.seed_database:
        build_demo_database(database_path)
    return _run_with_database(args, database_path, temporary=False)


def _run_with_database(args: argparse.Namespace, database_path: Path, *, temporary: bool) -> dict[str, Any]:
    artifact = demo_artifact()
    diagnostics = parse_diagnostics(args.diagnostics)
    source = demo_state_source(database_path)
    verifier = StructuredStateVerifier.from_source(source)
    claims = demo_claims()
    loop_result = run_verification_loop(
        request_id=args.request_id,
        diagnostics=diagnostics,
        claims=claims,
        verifier=verifier,
        controller=RiskController(artifact),
        executor_registry=ActionExecutorRegistry(),
        metadata={
            "source": "examples/sqlite_state_control_demo.py",
            "artifact_model_id": artifact.model_id,
            "state_source_type": type(source).__name__,
            "database_path": "<temporary>" if temporary else str(database_path),
            "database_seeded": bool(args.seed_database),
            "business_domain": "order_fulfillment",
        },
    )
    payload = loop_result.trace.to_dict()
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="EigenTruth SQLite structured-state control demo")
    parser.add_argument("--database", default=None, help="optional SQLite database path")
    parser.add_argument("--no-seed-database", dest="seed_database", action="store_false",
                        help="use an existing database instead of creating the demo fixture")
    parser.set_defaults(seed_database=True)
    parser.add_argument("--diagnostics", default=None,
                        help="diagnostics JSON object; defaults below the toy threshold")
    parser.add_argument("--request-id", default="sqlite-state-demo", help="request id stored in ProductTrace")
    parser.add_argument("--output", default=None, help="optional path to write the trace JSON")
    payload = run(parser.parse_args())
    print(json.dumps(payload, indent=2, sort_keys=True))


_CAN_SHIP_SQL = """
select
  case
    when inventory.available >= orders.quantity and accounts.status = 'active' then 1
    else 0
  end as can_ship
from orders
join inventory on inventory.sku = orders.sku
join accounts on accounts.id = orders.account_id
where orders.id = ?
"""


if __name__ == "__main__":
    main()
