"""The invoice flow as a LangGraph `StateGraph`.

    ingest -> extract -> match_vendor -> validate -> auto_approve -> save
                 ^  |                        |
                 +--+ retry once             +-> human_review -> save
                    |                                 |  (interrupt, then resume)
                    +-> human_review                  +-> match_vendor (after a correction)

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
from .extractors import ExtractionError, Extractor
from .rules import PurchaseOrder, RuleContext, VendorMatch, duplicate_key, run_rules
from .schema import Invoice
from .store import InvoiceStore
from .vendors import VendorMatcher

MAX_EXTRACT_ATTEMPTS = 2  # the first try plus one retry with the error fed back
REVIEW_ACTIONS = ("approve", "reject", "correct")


class InvoiceState(TypedDict, total=False):
    run_id: str
    filename: str
    file_bytes: bytes | None  # dropped after ingest so checkpoints stay small
    text: str
    ingest_error: str | None
    attempts: int
    extraction_error: str | None
    invoice: dict[str, Any] | None
    draft: dict[str, Any] | None  # partial extractor output that failed the schema
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


def _match_from_state(state: InvoiceState) -> VendorMatch:
    m = state.get("vendor_match") or {}
    return VendorMatch(m.get("vendor_id"), m.get("vendor_name"), m.get("score", 0.0))


def review_reasons(state: InvoiceState) -> list[str]:
    """Why a human has to look: the ingest error, the extraction error, or failed rules."""
    if state.get("ingest_error"):
        return [f"could not read the file: {state['ingest_error']}"]
    if state.get("invoice") is None:
        return [f"extraction failed after {state.get('attempts', 0)} attempts: {state.get('extraction_error')}"]
    return [f"{r['name']}: {r['detail']}" for r in state.get("rules", []) if not r["passed"]]


def build_graph(deps: Deps):
    """Return the uncompiled StateGraph; compile it with a checkpointer."""

    def ingest(state: InvoiceState) -> dict:
        try:
            text = ingest_mod.to_text(state["filename"], state.get("file_bytes") or b"")
            return {"text": text, "ingest_error": None, "file_bytes": None, "path": ["ingest"]}
        except ingest_mod.IngestError as exc:
            return {"text": "", "ingest_error": str(exc), "file_bytes": None, "path": ["ingest"]}

    def extract(state: InvoiceState) -> dict:
        attempt = state.get("attempts", 0) + 1
        raw = None
        try:
            raw = deps.extractor.extract(state["text"], feedback=state.get("extraction_error"))
            invoice = Invoice.model_validate(raw)
        except ValidationError as exc:
            errors = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'output'}: {e['msg']}" for e in exc.errors()
            )
            # Keep the partial output so a reviewer fixes one field instead of typing all of them.
            draft = raw if isinstance(raw, dict) else state.get("draft")
            return {"attempts": attempt, "extraction_error": errors, "draft": draft, "path": ["extract"]}
        except ExtractionError as exc:
            return {"attempts": attempt, "extraction_error": str(exc), "path": ["extract"]}
        return {
            "attempts": attempt,
            "extraction_error": None,
            "invoice": invoice.model_dump(mode="json"),
            "path": ["extract"],
        }

    def match_vendor(state: InvoiceState) -> dict:
        m = deps.matcher.match(state["invoice"]["vendor_name"])
        return {
            "vendor_match": {"vendor_id": m.vendor_id, "vendor_name": m.vendor_name,
                             "score": m.score, "matched": m.matched},
            "path": ["match_vendor"],
        }

    def validate(state: InvoiceState) -> dict:
        ctx = RuleContext(
            today=deps.today(),
            vendor_match=_match_from_state(state),
            seen_invoices=deps.store.seen_invoices(),
            purchase_orders=deps.purchase_orders,
        )
        results = run_rules(Invoice.model_validate(state["invoice"]), ctx)
        return {"rules": [r.__dict__ for r in results], "path": ["validate"]}

    def auto_approve(state: InvoiceState) -> dict:
        corrected = (state.get("review") or {}).get("action") == "correct"
        return {"status": "approved_after_correction" if corrected else "auto_approved",
                "path": ["auto_approve"]}

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
        if state.get("attempts", 0) < MAX_EXTRACT_ATTEMPTS:
            return "extract"
        return "human_review"

    def after_validate(state: InvoiceState) -> Literal["auto_approve", "human_review"]:
        return "auto_approve" if all(r["passed"] for r in state["rules"]) else "human_review"

    def after_review(state: InvoiceState) -> Literal["match_vendor", "save"]:
        return "match_vendor" if state["review"]["action"] == "correct" else "save"

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

    {"action": "approve" | "reject" | "correct", "invoice": {...} (for correct),
     "reviewer": "name", "note": "why"}
    """
    if not isinstance(decision, dict) or decision.get("action") not in REVIEW_ACTIONS:
        raise ValueError(f"action must be one of {', '.join(REVIEW_ACTIONS)}")
    action = decision["action"]
    out = {"action": action, "reviewer": str(decision.get("reviewer") or "reviewer")[:80],
           "note": str(decision.get("note") or "")[:500]}
    if action == "approve" and not has_invoice:
        raise ValueError("nothing to approve: no invoice was extracted; use 'correct' and enter the fields")
    if action == "correct":
        if not isinstance(decision.get("invoice"), dict):
            raise ValueError("'correct' needs the corrected invoice in 'invoice'")
        try:
            out["invoice"] = Invoice.model_validate(decision["invoice"]).model_dump(mode="json")
        except ValidationError as exc:
            raise ValueError(f"corrected invoice is not valid: {exc.errors()[0]['loc']}: {exc.errors()[0]['msg']}") from exc
    return out
