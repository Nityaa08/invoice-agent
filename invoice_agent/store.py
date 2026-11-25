"""SQLite storage for runs, saved invoices, the duplicate ledger and an audit log.

This is the accounts-payable side of the system, separate from LangGraph's
checkpoint database (which stores graph state so runs can pause and resume).

- `runs`: one row per uploaded file, keyed by an idempotency key (SHA-256 of
  the file bytes), so the same file is never processed twice.
- `invoices`: the final record for each run (status, invoice JSON, rule
  results, vendor match). Saving is an upsert, so a retried save is harmless.
- `ledger`: (vendor, invoice number) pairs already on file: seeded with paid
  invoices, plus every invoice this system approves. The duplicate rule reads it.
- `vendors`: vendors a reviewer added from the review screen (the vendor
  master in data/vendors.json stays read-only).
- `audit`: an append-only log of what happened to each run.

Pass ":memory:" for a throwaway store (tests, the stateless demo).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from .rules import money, normalize_id

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS invoices (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
    status TEXT NOT NULL,
    vendor_id TEXT,
    invoice_number TEXT,
    total REAL,
    currency TEXT,
    invoice_json TEXT,
    rules_json TEXT NOT NULL,
    vendor_match_json TEXT,
    saved_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger (
    vendor_key TEXT NOT NULL,
    invoice_number TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (vendor_key, invoice_number)
);
CREATE TABLE IF NOT EXISTS po_paid (
    po_key TEXT NOT NULL,
    vendor_key TEXT NOT NULL,
    invoice_number TEXT NOT NULL,
    amount REAL NOT NULL,
    PRIMARY KEY (vendor_key, invoice_number)
);
CREATE TABLE IF NOT EXISTS vendors (
    vendor_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    tax_id TEXT,
    currency TEXT NOT NULL,
    added_by TEXT NOT NULL,
    added_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    at TEXT NOT NULL,
    event TEXT NOT NULL,
    detail_json TEXT NOT NULL
);
"""

APPROVED_STATUSES = {"auto_approved", "approved_by_reviewer", "approved_after_correction",
                     "approved_after_recheck"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class InvoiceStore:
    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # --- idempotency -----------------------------------------------------------

    def register_run(self, run_id: str, idempotency_key: str, filename: str) -> str:
        """Record a new run. If this file was seen before, return the earlier run id instead."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO runs (run_id, idempotency_key, filename, created_at) VALUES (?, ?, ?, ?)",
                (run_id, idempotency_key, filename, _now()),
            )
            row = self._conn.execute(
                "SELECT run_id FROM runs WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
        return row["run_id"]

    def list_runs(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    # --- duplicate ledger ------------------------------------------------------

    def add_to_ledger(self, vendor_key: str, invoice_number: str, source: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO ledger (vendor_key, invoice_number, source) VALUES (?, ?, ?)",
                (vendor_key, normalize_id(invoice_number), source),
            )

    def seed_ledger(self, paid_invoices: list[dict[str, Any]]) -> None:
        for paid in paid_invoices:
            self.add_to_ledger(paid["vendor_id"], paid["invoice_number"], "paid_invoices.json")
            if paid.get("po_number"):  # already paid against a PO: it counts toward the PO's balance
                with self._lock, self._conn:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO po_paid (po_key, vendor_key, invoice_number, amount) VALUES (?, ?, ?, ?)",
                        (normalize_id(paid["po_number"]), paid["vendor_id"], normalize_id(paid["invoice_number"]),
                         paid["total"]))

    def po_spent(self) -> dict[str, Decimal]:
        """Totals paid (paid_invoices.json) or approved here per PO (normalized PO number), for the PO balance check."""
        rows = self._conn.execute(
            "SELECT invoice_json FROM invoices WHERE status IN ({}) AND invoice_json IS NOT NULL".format(
                ",".join("?" * len(APPROVED_STATUSES))), sorted(APPROVED_STATUSES)).fetchall()
        spent: dict[str, Decimal] = {}
        for p in self._conn.execute("SELECT po_key, amount FROM po_paid").fetchall():
            spent[p["po_key"]] = spent.get(p["po_key"], Decimal("0")) + money(p["amount"])
        for r in rows:
            inv = json.loads(r["invoice_json"])
            if inv.get("po_number"):
                key = normalize_id(inv["po_number"])
                spent[key] = spent.get(key, Decimal("0")) + money(inv["total"])
        return spent

    def seen_invoices(self) -> frozenset[tuple[str, str]]:
        rows = self._conn.execute("SELECT vendor_key, invoice_number FROM ledger").fetchall()
        return frozenset((r["vendor_key"], r["invoice_number"]) for r in rows)

    # --- final records -----------------------------------------------------------

    def save_invoice(self, run_id: str, status: str, invoice: dict[str, Any] | None,
                     rules: list[dict[str, Any]], vendor_match: dict[str, Any] | None,
                     duplicate_key: tuple[str, str] | None) -> None:
        """Upsert the final record; approved invoices also join the duplicate ledger."""
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO invoices (run_id, status, vendor_id, invoice_number, total, currency,
                                         invoice_json, rules_json, vendor_match_json, saved_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET
                     status=excluded.status, vendor_id=excluded.vendor_id,
                     invoice_number=excluded.invoice_number, total=excluded.total,
                     currency=excluded.currency, invoice_json=excluded.invoice_json,
                     rules_json=excluded.rules_json, vendor_match_json=excluded.vendor_match_json,
                     saved_at=excluded.saved_at""",
                (
                    run_id, status,
                    (vendor_match or {}).get("vendor_id"),
                    (invoice or {}).get("invoice_number"),
                    (invoice or {}).get("total"),
                    (invoice or {}).get("currency"),
                    json.dumps(invoice) if invoice is not None else None,
                    json.dumps(rules),
                    json.dumps(vendor_match) if vendor_match is not None else None,
                    _now(),
                ),
            )
        if status in APPROVED_STATUSES and duplicate_key:
            self.add_to_ledger(duplicate_key[0], duplicate_key[1], f"run:{run_id}")

    def get_invoice(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM invoices WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        out = dict(row)
        for col in ("invoice_json", "rules_json", "vendor_match_json"):
            out[col.removesuffix("_json")] = json.loads(out.pop(col)) if out[col] else None
        return out

    def count_invoices(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]

    # --- vendors added by reviewers -----------------------------------------------

    def add_vendor(self, vendor: dict[str, Any], added_by: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO vendors (vendor_id, name, tax_id, currency, added_by, added_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (vendor["vendor_id"], vendor["name"], vendor.get("tax_id"), vendor["currency"], added_by, _now()),
            )

    def added_vendor_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM vendors").fetchone()[0]

    def added_vendors(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM vendors ORDER BY added_at, vendor_id").fetchall()
        return [{"vendor_id": r["vendor_id"], "name": r["name"], "tax_id": r["tax_id"], "currency": r["currency"],
                 "source": "added", "added_by": r["added_by"], "added_at": r["added_at"]} for r in rows]

    # --- audit log -------------------------------------------------------------

    def audit(self, run_id: str, event: str, **detail: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO audit (run_id, at, event, detail_json) VALUES (?, ?, ?, ?)",
                (run_id, _now(), event, json.dumps(detail, default=str)),
            )

    def audit_log(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT at, event, detail_json FROM audit WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [{"at": r["at"], "event": r["event"], **json.loads(r["detail_json"])} for r in rows]
