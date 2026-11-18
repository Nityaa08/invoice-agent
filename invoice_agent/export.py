"""Export runs as CSV or JSON: the fields, the decision, the failed rules and the reviewer's action.

The demo page builds the same rows in the browser (EXPORT_COLUMNS in
web/index.html; a test keeps the two lists equal). The full service serves
them from GET /api/runs/export.

Invoice text is untrusted, and a CSV cell that starts with = + - @ (or a tab
or carriage return) runs as a formula when the file is opened in Excel or
Google Sheets. Such text cells get a leading apostrophe; numbers are written
as numbers.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from .store import APPROVED_STATUSES

COLUMNS = (
    "file", "run_id", "status", "decision", "source", "vendor_name", "vendor_id", "vendor_match_score",
    "invoice_number", "invoice_date", "due_date", "currency", "po_number", "subtotal", "discount", "tax", "total",
    "failed_rules", "reviewer_action", "reviewer", "review_note",
)
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def export_row(run: dict[str, Any]) -> dict[str, Any]:
    """One run view (InvoiceService.view) -> one flat row."""
    inv = run.get("invoice") or {}
    vm = run.get("vendor_match") or {}
    decision = run.get("decision") or {}
    status = run.get("status")
    failed = [r["name"] for r in run.get("rules") or [] if not r["passed"]]
    return {
        "file": run.get("filename"),
        "run_id": run.get("id"),
        "status": status,
        "decision": "approved" if status in APPROVED_STATUSES else "rejected" if status == "rejected" else "review",
        "source": run.get("source") or "upload",
        "vendor_name": inv.get("vendor_name"),
        "vendor_id": vm.get("vendor_id") if vm.get("matched") else None,
        "vendor_match_score": vm.get("score"),
        "invoice_number": inv.get("invoice_number"),
        "invoice_date": inv.get("invoice_date"),
        "due_date": inv.get("due_date"),
        "currency": inv.get("currency"),
        "po_number": inv.get("po_number"),
        "subtotal": inv.get("subtotal"),
        "discount": inv.get("discount"),
        "tax": inv.get("tax"),
        "total": inv.get("total"),
        "failed_rules": "; ".join(failed) or ("" if run.get("invoice") else "extraction_failed"),
        "reviewer_action": decision.get("action"),
        "reviewer": decision.get("reviewer"),
        "review_note": decision.get("note"),
    }


def safe_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    text = str(value)
    return "'" + text if text.startswith(_FORMULA_START) else text


def to_csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")
    writer.writerow(COLUMNS)
    for row in rows:
        writer.writerow([safe_cell(row.get(c)) for c in COLUMNS])
    return buf.getvalue()
