"""The invoice flow as a LangGraph `StateGraph`.

    ingest -> extract -> match_vendor -> validate -> auto_approve -> save
                 ^  |                        |
                 +--+ retry once             +-> human_review -> save
                    |                                 |  (interrupt, then resume)
                    +-> human_review                  +-> match_vendor (after a correction,
                                                                         or a recheck once the
                                                                         reviewer added the vendor)

Why a graph: the flow has a retry loop, a branch, and a pause for a human that
may last days. LangGraph gives each of those a name (nodes, conditional edges,
`interrupt()`), and its checkpointer saves state after every step, so a run
survives a restart and resumes exactly where it stopped.

Nodes only move data and call collaborators (extractor, vendor matcher,
rules, store) that are tested on their own. Every node appends its name to
`path`, so the API can show the route a run took.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, Any, Callable, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError
from typing_extensions import TypedDict

from . import ingest as ingest_mod
from .extractors import NOTES_KEY, ExtractionError, Extractor
from .rules import GoodsReceipt, PurchaseOrder, RuleContext, VendorMatch, duplicate_key, run_rules
from .schema import Invoice
from .store import InvoiceStore
from .vendors import VendorMatcher

OCR_CHECK = "ocr_checked_by_person"
OCR_NOTE = ("read by OCR, please check every field against the original image: OCR misreads digits "
            "(1 and 7, 0 and 8, 5 and S) and drops decimal points")
MAX_EXTRACT_ATTEMPTS = 2  # the first try plus one retry with the error fed back
# recheck: run the vendor match and every rule again on the same fields, for
# when something outside the invoice changed (the reviewer added the vendor).
REVIEW_ACTIONS = ("approve", "reject", "correct", "recheck")
RECHECK_ACTIONS = ("correct", "recheck")


class InvoiceState(TypedDict, total=False):
    run_id: str
    filename: str
    file_bytes: bytes | None  # dropped after ingest so checkpoints stay small
    source: str | None  # "ocr": the text came from OCR (the browser's or the server's), so a person checks it
    text: str
    ingest_error: str | None
    ingest_notes: list[str]  # what ingest had to leave out (an over-long document); forces a review
    attempts: int
    extraction_error: str | None  # for people: "could not find invoice date on the invoice"
    model_feedback: str | None  # for the model's retry: the exact schema errors
    invoice: dict[str, Any] | None
    draft: dict[str, Any] | None  # partial extractor output that failed the schema
    extraction_notes: list[str]  # what the extractor had to approximate; forces a review
    vendor_match: dict[str, Any] | None
    rules: list[dict[str, Any]]
    review: dict[str, Any] | None  # the reviewer's last decision
    status: str
    path: Annotated[list[str], operator.add]  # nodes visited, in order


@dataclass
class Deps:
    """Everything the nodes need, injected so tests can swap any piece."""

    extractor: Extractor
    matcher: VendorMatcher
    store: InvoiceStore
    purchase_orders: dict[str, PurchaseOrder] = field(default_factory=dict)
    today: Callable[[], date] = date.today
    goods_receipts: dict[str, tuple[GoodsReceipt, ...]] = field(default_factory=dict)


def _match_from_state(state: InvoiceState) -> VendorMatch:
    m = state.get("vendor_match") or {}
    return VendorMatch(m.get("vendor_id"), m.get("vendor_name"), m.get("score", 0.0), m.get("tax_id"))


FIELD_LABELS = {
    "vendor_name": "vendor name", "vendor_tax_id": "vendor tax id", "invoice_number": "invoice number",
    "invoice_date": "invoice date", "due_date": "due date", "currency": "currency", "po_number": "PO number",
    "line_items": "line items", "subtotal": "subtotal", "tax": "tax", "total": "total",
}


def describe_schema_errors(exc: ValidationError, where: str = "on the invoice") -> str:
    """Pydantic errors in words a reviewer understands: 'could not find invoice date on the invoice'."""
    missing, negative, huge, other = [], [], [], []
    for e in exc.errors():
        loc = [str(p) for p in e["loc"]]
        field = FIELD_LABELS.get(loc[0], ".".join(loc)) if loc else "output"
        if loc[:1] == ["line_items"] and len(loc) > 2:
            field = f"line {int(loc[1]) + 1} {loc[2].replace('_', ' ')}"
        value = e.get("input")
        if e["type"] == "missing" and len(loc) == 1:
            missing.append(field)
        elif loc[:1] == ["line_items"] and e["type"] == "too_short":
            missing.append("line items")
        elif e["type"] in ("greater_than_equal", "greater_than") and isinstance(value, (int, float)) and value < 0:
            negative.append(f"{field} ({value:g})")
        elif e["type"] == "less_than_equal" and isinstance(value, (int, float)):
            huge.append(f"{field} ({value:g})")
        else:
            other.append(f"{'.'.join(loc) or 'output'}: {e['msg']}")
    parts = []
    if missing:
        parts.append(("could not find " if where == "on the invoice" else "missing ") + ", ".join(missing)
                     + (f" {where}" if where else ""))
    if negative:
        parts.append("negative amounts: " + ", ".join(negative[:4]) + (" and more" if len(negative) > 4 else "")
                     + ". A negative invoice is a credit note or a refund, which a person must handle")
    if huge:
        parts.append("amounts too large to be real: " + ", ".join(huge[:4]) + (" and more" if len(huge) > 4 else ""))
    return "; ".join(parts + other)


def review_reasons(state: InvoiceState) -> list[str]:
    """Why a human has to look: the ingest error, the extraction error, or failed checks."""
    if state.get("ingest_error"):
        return [f"could not read the file: {state['ingest_error']}"]
    if state.get("invoice") is None:
        tries = state.get("attempts", 0)
        after = f" after {tries} attempts" if tries > 1 else ""
        return [f"extraction incomplete{after}: {state.get('extraction_error')}. "
                "Use 'Correct fields' to fill in what is missing."]
    return [f"{r['name']}: {r['detail']}" for r in state.get("rules", []) if not r["passed"]]


def rule_context(deps: Deps, match: VendorMatch, document_text: str = "") -> RuleContext:
    """What the rules see: today, the vendor match, what is on file, open POs. Used by the graph and by MCP."""
    return RuleContext(
        today=deps.today(),
        vendor_match=match,
        seen_invoices=deps.store.seen_invoices(),
        purchase_orders=deps.purchase_orders,
        po_spent=deps.store.po_spent(),
        document_text=document_text,
        goods_receipts=deps.goods_receipts,
    )


def build_graph(deps: Deps):
    """Return the uncompiled StateGraph; compile it with a checkpointer."""

    def ingest(state: InvoiceState) -> dict:
        data = state.get("file_bytes") or b""
        try:
            text = ingest_mod.to_text(state["filename"], data)
            notes = []
            if len(text) > ingest_mod.MAX_TEXT_CHARS:
                text, notes = ingest_mod.shorten(text), [ingest_mod.LONG_TEXT_NOTE]
            out = {"text": text, "ingest_error": None, "ingest_notes": notes, "file_bytes": None,
                   "path": ["ingest"]}
            if ingest_mod.sniff_kind(state["filename"], data) == "image":
                out["source"] = "ocr"  # server-side OCR (SERVER_OCR=true): same rule as the browser's
            return out
        except ingest_mod.IngestError as exc:
            return {"text": "", "ingest_error": str(exc), "file_bytes": None, "path": ["ingest"]}

    def extract(state: InvoiceState) -> dict:
        attempt = state.get("attempts", 0) + 1
        raw, notes = None, []
        try:
            raw = deps.extractor.extract(state["text"], feedback=state.get("model_feedback"))
            if isinstance(raw, dict) and NOTES_KEY in raw:
                raw = dict(raw)
                notes = [str(n) for n in raw.pop(NOTES_KEY) or []]
            invoice = Invoice.model_validate(raw)
        except ValidationError as exc:
            # Keep the partial output so a reviewer fixes one field instead of typing all of them.
            draft = raw if isinstance(raw, dict) else state.get("draft")
            exact = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'output'}: {e['msg']}" for e in exc.errors())
            return {"attempts": attempt, "extraction_error": describe_schema_errors(exc), "model_feedback": exact,
                    "draft": draft, "extraction_notes": notes, "path": ["extract"]}
        except ExtractionError as exc:
            return {"attempts": attempt, "extraction_error": str(exc), "model_feedback": str(exc),
                    "path": ["extract"]}
        return {
            "attempts": attempt,
            "extraction_error": None,
            "model_feedback": None,
            "invoice": invoice.model_dump(mode="json"),
            "extraction_notes": notes,
            "path": ["extract"],
        }

    def match_vendor(state: InvoiceState) -> dict:
        m = deps.matcher.match(state["invoice"]["vendor_name"])
        source = deps.matcher.vendors.get(m.vendor_id, {}).get("source", "master") if m.vendor_id else None
        return {
            "vendor_match": {"vendor_id": m.vendor_id, "vendor_name": m.vendor_name,
                             "score": m.score, "matched": m.matched, "tax_id": m.tax_id, "source": source},
            "path": ["match_vendor"],
        }

    def validate(state: InvoiceState) -> dict:
        ctx = rule_context(deps, _match_from_state(state), state.get("text") or "")
        results = [r.__dict__ for r in run_rules(Invoice.model_validate(state["invoice"]), ctx)]
        # Anything the extractor approximated is a failed check, so a human confirms it.
        # A reviewer's correction replaces the extracted fields, so the notes no longer apply.
        corrected = (state.get("review") or {}).get("action") == "correct"
        notes = [] if corrected else list(state.get("ingest_notes") or []) + list(state.get("extraction_notes") or [])
        if notes:
            results.append({"name": "extraction_is_complete", "passed": False, "detail": "; ".join(notes)})
        # OCR text always goes to a person; a re-check does not clear it, a correction does.
        if state.get("source") == "ocr":
            results.append({"name": OCR_CHECK, "passed": corrected,
                            "detail": "a reviewer checked the fields against the image" if corrected else OCR_NOTE})
        return {"rules": results, "path": ["validate"]}

    def auto_approve(state: InvoiceState) -> dict:
        action = (state.get("review") or {}).get("action")
        status = {"correct": "approved_after_correction", "recheck": "approved_after_recheck"}.get(action, "auto_approved")
        return {"status": status, "path": ["auto_approve"]}

    def human_review(state: InvoiceState) -> dict:
        # interrupt() pauses the run here and saves state; the API resumes it with
        # Command(resume=decision). On resume this node runs again from the top,
        # so nothing before interrupt() may have side effects.
        decision = interrupt({
            "reasons": review_reasons(state),
            "invoice": state.get("invoice"),
            "draft": state.get("draft"),
            "vendor_match": state.get("vendor_match"),
            "actions": list(REVIEW_ACTIONS),
        })
        decision = validate_decision(decision, has_invoice=state.get("invoice") is not None)
        update: dict[str, Any] = {"review": decision, "path": ["human_review"]}
        if decision["action"] == "correct":
            update["invoice"] = decision["invoice"]
            update["status"] = "revalidating"
        elif decision["action"] == "recheck":
            update["status"] = "revalidating"
        else:
            update["status"] = "approved_by_reviewer" if decision["action"] == "approve" else "rejected"
        return update

    def save(state: InvoiceState) -> dict:
        invoice = state.get("invoice")
        key = None
        if invoice is not None:
            key = duplicate_key(Invoice.model_validate(invoice), _match_from_state(state))
        deps.store.save_invoice(
            state["run_id"], state["status"], invoice, state.get("rules", []),
            state.get("vendor_match"), key,
        )
        path = state.get("path", []) + ["save"]
        deps.store.audit(state["run_id"], "saved", status=state["status"], path=path)
        return {"path": ["save"]}

    # --- routing (conditional edges) ---------------------------------------------

    def after_ingest(state: InvoiceState) -> Literal["extract", "human_review"]:
        return "human_review" if state.get("ingest_error") else "extract"

    def after_extract(state: InvoiceState) -> Literal["match_vendor", "extract", "human_review"]:
        if state.get("extraction_error") is None:
            return "match_vendor"
        # A deterministic extractor (the demo rules) would return the same thing again.
        if state.get("attempts", 0) < MAX_EXTRACT_ATTEMPTS and not getattr(deps.extractor, "deterministic", False):
            return "extract"
        return "human_review"

    def after_validate(state: InvoiceState) -> Literal["auto_approve", "human_review"]:
        return "auto_approve" if all(r["passed"] for r in state["rules"]) else "human_review"

    def after_review(state: InvoiceState) -> Literal["match_vendor", "save"]:
        return "match_vendor" if state["review"]["action"] in RECHECK_ACTIONS else "save"

    g = StateGraph(InvoiceState)
    g.add_node("ingest", ingest)
    g.add_node("extract", extract)
    g.add_node("match_vendor", match_vendor)
    g.add_node("validate", validate)
    g.add_node("auto_approve", auto_approve)
    g.add_node("human_review", human_review)
    g.add_node("save", save)

    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", after_ingest)
    g.add_conditional_edges("extract", after_extract)
    g.add_edge("match_vendor", "validate")
    g.add_conditional_edges("validate", after_validate)
    g.add_conditional_edges("human_review", after_review)
    g.add_edge("auto_approve", "save")
    g.add_edge("save", END)
    return g


def validate_decision(decision: Any, *, has_invoice: bool) -> dict[str, Any]:
    """Check a reviewer's decision. Raises ValueError with a readable message.

    {"action": "approve" | "reject" | "correct" | "recheck", "invoice": {...} (for correct),
     "reviewer": "name", "note": "why"}
    """
    if not isinstance(decision, dict) or decision.get("action") not in REVIEW_ACTIONS:
        raise ValueError(f"action must be one of {', '.join(REVIEW_ACTIONS)}")
    action = decision["action"]
    out = {"action": action, "reviewer": str(decision.get("reviewer") or "reviewer")[:80],
           "note": str(decision.get("note") or "")[:500]}
    if action in ("approve", "recheck") and not has_invoice:
        raise ValueError(f"nothing to {action}: no invoice was extracted; use 'correct' and enter the fields")
    if action == "correct":
        if not isinstance(decision.get("invoice"), dict):
            raise ValueError("'correct' needs the corrected invoice in 'invoice'")
        try:
            out["invoice"] = Invoice.model_validate(decision["invoice"]).model_dump(mode="json")
        except ValidationError as exc:
            raise ValueError(f"corrected invoice is not valid: {describe_schema_errors(exc, 'in the correction')}") from exc
    return out
