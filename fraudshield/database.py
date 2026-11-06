"""SQLite persistence for the single-user FraudShield application."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "fraudshield.db"


def _json_default(value: Any):
    """Convert NumPy/Pandas values into JSON-safe Python values."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the transaction and review tables if they do not exist."""
    with get_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'manual',
                amount REAL NOT NULL,
                fraud_probability REAL NOT NULL,
                decision TEXT NOT NULL
                    CHECK (decision IN ('approve', 'block', 'defer to review')),
                true_label INTEGER NULL CHECK (true_label IN (0, 1)),
                transaction_data TEXT NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'not_required'
                    CHECK (review_status IN ('not_required', 'pending', 'completed')),
                reviewer_decision TEXT NULL
                    CHECK (reviewer_decision IN ('fraud', 'legitimate')),
                review_notes TEXT NULL,
                reviewed_at TEXT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_transactions_created_at
            ON transactions(created_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_transactions_decision
            ON transactions(decision)
            """
        )


def save_transaction(
    *,
    amount: float,
    fraud_probability: float,
    decision: str,
    transaction_data: dict[str, Any],
    source: str = "manual",
    true_label: int | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Save one scored transaction and return its database ID."""
    if decision not in {"approve", "block", "defer to review"}:
        raise ValueError(f"Unsupported decision: {decision}")
    if true_label not in (None, 0, 1):
        raise ValueError("true_label must be 0, 1, or None")

    review_status = "pending" if decision == "defer to review" else "not_required"
    created_at = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(transaction_data, default=_json_default, allow_nan=False)

    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO transactions (
                created_at, source, amount, fraud_probability, decision,
                true_label, transaction_data, review_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                source,
                float(amount),
                float(fraud_probability),
                decision,
                true_label,
                payload,
                review_status,
            ),
        )
        return int(cursor.lastrowid)


def get_transaction_history(
    limit: int = 500,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Return the newest saved transactions first."""
    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, source, amount, fraud_probability, decision,
                   true_label, review_status, reviewer_decision, review_notes,
                   reviewed_at
            FROM transactions
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def get_review_queue(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Return transactions waiting for a human review."""
    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, created_at, amount, fraud_probability, decision,
                   transaction_data, review_status
            FROM transactions
            WHERE review_status = 'pending'
            ORDER BY fraud_probability DESC, id DESC
            """
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def complete_review(
    transaction_id: int,
    reviewer_decision: str,
    review_notes: str = "",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """Record a human decision for a pending deferred transaction."""
    if reviewer_decision not in {"fraud", "legitimate"}:
        raise ValueError("reviewer_decision must be 'fraud' or 'legitimate'")

    reviewed_at = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE transactions
            SET review_status = 'completed',
                reviewer_decision = ?,
                review_notes = ?,
                reviewed_at = ?
            WHERE id = ? AND review_status = 'pending'
            """,
            (reviewer_decision, review_notes, reviewed_at, int(transaction_id)),
        )
        if cursor.rowcount == 0:
            raise ValueError(
                f"Transaction {transaction_id} was not found in the pending review queue."
            )
