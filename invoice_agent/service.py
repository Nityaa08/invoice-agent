"""The application layer: wires the graph to storage and exposes three operations.

    submit(filename, bytes)   -> run the graph until it finishes or pauses for review
    get(run_id)               -> the run as JSON (state, rules, path, pending review)
    review(run_id, decision)  -> resume a paused run with the reviewer's decision

Two ways to build it:
- `InvoiceService.persistent(settings)`: SQLite for invoices and SQLite for
  LangGraph checkpoints. Runs survive restarts; reviews can come days later.
- `InvoiceService.ephemeral(settings)`: everything in memory, for tests and
  the stateless serverless demo (one request = one service).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .config import DATA_DIR, Settings
from .extractors import Extractor, build_extractor
from .graph import Deps, build_graph, validate_decision
from .rules import PurchaseOrder, normalize_id
from .store import InvoiceStore
from .vendors import VendorMatcher

log = logging.getLogger(__name__)


class RunNotFound(LookupError):
    pass


class NotAwaitingReview(RuntimeError):
    pass


def idempotency_key(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_purchase_orders(path: Path = DATA_DIR / "purchase_orders.json") -> dict[str, PurchaseOrder]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {normalize_id(r["po_number"]): PurchaseOrder(**r) for r in rows}


def load_paid_invoices(path: Path = DATA_DIR / "paid_invoices.json") -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


_MATCHER: VendorMatcher | None = None


def default_matcher() -> VendorMatcher:
    """The vendor master is static, so embed it once per process."""
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = VendorMatcher.from_file(DATA_DIR / "vendors.json")
    return _MATCHER


class InvoiceService:
    def __init__(self, deps: Deps, checkpointer: BaseCheckpointSaver):
        self.deps = deps
        self.store = deps.store
        self.graph = build_graph(deps).compile(checkpointer=checkpointer)

    # --- construction ------------------------------------------------------------

    @staticmethod
    def _deps(settings: Settings, store: InvoiceStore, extractor: Extractor | None) -> Deps:
        store.seed_ledger(load_paid_invoices())
        as_of = settings.as_of
        return Deps(
            extractor=extractor or build_extractor(
                settings.extractor, model=settings.llm_model, api_key=settings.llm_api_key,
                base_url=settings.llm_base_url, method=settings.llm_method,
            ),
            matcher=default_matcher(),
            store=store,
            purchase_orders=load_purchase_orders(),
            today=(lambda: as_of) if as_of else date.today,
        )

    @classmethod
    def persistent(cls, settings: Settings, extractor: Extractor | None = None) -> "InvoiceService":
        from langgraph.checkpoint.sqlite import SqliteSaver

        for p in (settings.db_path, settings.checkpoint_path):
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        store = InvoiceStore(settings.db_path)
        saver = SqliteSaver(sqlite3.connect(settings.checkpoint_path, check_same_thread=False))
        return cls(cls._deps(settings, store, extractor), saver)

    @classmethod
    def ephemeral(cls, settings: Settings, extractor: Extractor | None = None) -> "InvoiceService":
        return cls(cls._deps(settings, InvoiceStore(":memory:"), extractor), InMemorySaver())

    # --- operations --------------------------------------------------------------

    @staticmethod
    def _config(run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}}

    def submit(self, filename: str, data: bytes) -> tuple[dict[str, Any], bool]:
        """Process a file. Returns (run view, created). The same bytes twice -> the first run."""
        run_id = uuid.uuid4().hex[:16]
        existing = self.store.register_run(run_id, idempotency_key(data), filename)
        if existing != run_id:
            self.store.audit(existing, "duplicate_submission", filename=filename)
            snap = self.graph.get_state(self._config(existing))
            if snap.next and not snap.interrupts:
                # The first attempt stopped mid-way (crash, restart, timeout):
                # pick up from the last checkpoint instead of starting over.
                self._invoke(existing, None)
            return self.view(existing, duplicate_submission=True), False
        self.store.audit(run_id, "submitted", filename=filename, bytes=len(data))
        self._invoke(run_id, {"run_id": run_id, "filename": filename, "file_bytes": data,
                              "attempts": 0, "path": []})
        return self.view(run_id), True

    def get(self, run_id: str) -> dict[str, Any]:
        return self.view(run_id)

    def review(self, run_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        snap = self.graph.get_state(self._config(run_id))
        if not snap.values:
            raise RunNotFound(run_id)
        if not snap.interrupts:
            raise NotAwaitingReview(f"run {run_id} is not waiting for a review")
        clean = validate_decision(decision, has_invoice=snap.values.get("invoice") is not None)
        self.store.audit(run_id, "reviewed", action=clean["action"], reviewer=clean["reviewer"],
                         note=clean["note"])
        self._invoke(run_id, Command(resume=clean))
        return self.view(run_id)

    def _invoke(self, run_id: str, payload: Any) -> None:
        try:
            self.graph.invoke(payload, self._config(run_id))
        except Exception as exc:  # the checkpoint keeps the progress made so far
            log.exception("run %s stopped", run_id)
            self.store.audit(run_id, "error", error=f"{type(exc).__name__}: {exc}")

    # --- read model --------------------------------------------------------------

    def view(self, run_id: str, duplicate_submission: bool = False) -> dict[str, Any]:
        snap = self.graph.get_state(self._config(run_id))
        if not snap.values:
            raise RunNotFound(run_id)
        v = snap.values
        pending = [i.value for i in snap.interrupts]
        if pending:
            status = "needs_review"
        elif snap.next:
            status = "incomplete"  # stopped by an error; resubmit the same file to resume
        else:
            status = v.get("status", "unknown")
        run = self.store.get_run(run_id) or {}
        return {
            "id": run_id,
            "filename": v.get("filename"),
            "created_at": run.get("created_at"),
            "status": status,
            "duplicate_submission": duplicate_submission,
            "path": v.get("path", []),
            "next": list(snap.next),
            "extractor": self.deps.extractor.name,
            "as_of": self.deps.today().isoformat(),
            "attempts": v.get("attempts", 0),
            "ingest_error": v.get("ingest_error"),
            "extraction_error": v.get("extraction_error"),
            "invoice": v.get("invoice"),
            "vendor_match": v.get("vendor_match"),
            "rules": v.get("rules", []),
            "review_request": pending[0] if pending else None,
            "decision": v.get("review"),
            "text": (v.get("text") or "")[:6000],
            "audit": self.store.audit_log(run_id),
        }
