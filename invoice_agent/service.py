"""The application layer: wires the graph to storage and exposes three operations.

    submit(filename, bytes)   -> run the graph until it finishes or pauses for review
    get(run_id)               -> the run as JSON (state, rules, path, pending review)
    review(run_id, decision)  -> resume a paused run with the reviewer's decision
    add_vendor(vendor)        -> onboard a vendor from the review screen

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
import threading
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
from .ingest import SCANNED_PDF_MESSAGE
from .rules import GoodsReceipt, PurchaseOrder, ReceiptLine, normalize_id
from .store import InvoiceStore
from .vendors import VendorMatcher, check_new_vendor

log = logging.getLogger(__name__)


class RunNotFound(LookupError):
    pass


class NotAwaitingReview(RuntimeError):
    pass


def idempotency_key(data: bytes, source: str | None = None) -> str:
    """The same bytes are the same submission, unless one was read by OCR and the other was not.

    Text read by OCR always goes to a person; keyed on the bytes alone, resending a file's
    text flagged as OCR returned the earlier, unflagged run, so the OCR check never ran.
    """
    digest = hashlib.sha256(data).hexdigest()
    return f"ocr:{digest}" if source == "ocr" else digest


def load_purchase_orders(path: Path = DATA_DIR / "purchase_orders.json") -> dict[str, PurchaseOrder]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {normalize_id(r["po_number"]): PurchaseOrder(**r) for r in rows}


def load_goods_receipts(path: Path = DATA_DIR / "goods_receipts.json") -> dict[str, tuple[GoodsReceipt, ...]]:
    out: dict[str, list[GoodsReceipt]] = {}
    for r in json.loads(path.read_text(encoding="utf-8")):
        grn = GoodsReceipt(r["grn_number"], r["po_number"], r["received_on"],
                           tuple(ReceiptLine(x["description"], float(x["quantity"])) for x in r["lines"]))
        out.setdefault(normalize_id(r["po_number"]), []).append(grn)
    return {k: tuple(v) for k, v in out.items()}


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
    def __init__(self, deps: Deps, checkpointer: BaseCheckpointSaver, shared_vendors: bool = False):
        self.deps = deps
        self.store = deps.store
        self.graph = build_graph(deps).compile(checkpointer=checkpointer)
        # Persistent mode: another process (the MCP server, a second worker) can add vendors
        # to the same database, so the vendor list is re-read when its row count changes.
        self._shared_vendors = shared_vendors
        self._vendor_rows = self.store.added_vendor_count() if shared_vendors else 0
        # One submission at a time per file, so two uploads of the same bytes cannot both
        # see the other's run before its first checkpoint exists.
        self._key_locks: dict[str, threading.Lock] = {}
        self._key_locks_guard = threading.Lock()

    # --- construction ------------------------------------------------------------

    @staticmethod
    def _deps(settings: Settings, store: InvoiceStore, extractor: Extractor | None,
              matcher: VendorMatcher | None = None) -> Deps:
        store.seed_ledger(load_paid_invoices())
        as_of = settings.as_of
        provider = settings.extractor
        if provider == "llm" and not settings.llm_api_key:
            # A missing key on a deployment should not turn every upload into a 500;
            # /api/config already reports the demo extractor in this case.
            log.warning("EXTRACTOR=llm but no LLM_API_KEY or GROQ_API_KEY; using the demo extractor")
            provider = "demo"
        return Deps(
            extractor=extractor or build_extractor(
                provider, model=settings.llm_model, api_key=settings.llm_api_key,
                base_url=settings.llm_base_url, method=settings.llm_method,
                timeout=settings.llm_timeout, max_retries=settings.llm_max_retries,
            ),
            matcher=matcher or default_matcher(),
            store=store,
            purchase_orders=load_purchase_orders(),
            goods_receipts=load_goods_receipts(),
            today=(lambda: as_of) if as_of else date.today,
        )

    @classmethod
    def persistent(cls, settings: Settings, extractor: Extractor | None = None) -> "InvoiceService":
        from langgraph.checkpoint.sqlite import SqliteSaver

        for p in (settings.db_path, settings.checkpoint_path):
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        store = InvoiceStore(settings.db_path)
        saver = SqliteSaver(sqlite3.connect(settings.checkpoint_path, check_same_thread=False))
        matcher = default_matcher().with_vendors(store.added_vendors())
        return cls(cls._deps(settings, store, extractor, matcher), saver, shared_vendors=True)

    @classmethod
    def ephemeral(cls, settings: Settings, extractor: Extractor | None = None,
                  vendors: list[dict[str, Any]] | None = None) -> "InvoiceService":
        """In memory. `vendors`: added vendors, already checked (the demo gets them from the browser)."""
        matcher = default_matcher().with_vendors(vendors or [])
        return cls(cls._deps(settings, InvoiceStore(":memory:"), extractor, matcher), InMemorySaver())

    # --- operations --------------------------------------------------------------

    @staticmethod
    def _config(run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}}

    def submit(self, filename: str, data: bytes, source: str | None = None) -> tuple[dict[str, Any], bool]:
        """Process a file. Returns (run view, created). The same bytes twice -> the first run.

        source="ocr": the bytes are text a browser read from a photo; the run always goes to review.
        """
        key = idempotency_key(data, source)
        with self._key_lock(key):
            self._refresh_vendors()
            run_id = uuid.uuid4().hex[:16]
            existing = self.store.register_run(run_id, key, filename)
            if existing != run_id:
                self.store.audit(existing, "duplicate_submission", filename=filename)
                snap = self.graph.get_state(self._config(existing))
                if not snap.values:
                    # Registered, but the graph never wrote a checkpoint (a crash right after
                    # registering). Before 2026-11-25 this file then failed with a 500 forever.
                    self._start(existing, filename, data, source)
                elif snap.next and not snap.interrupts:
                    # The first attempt stopped mid-way (crash, restart, timeout):
                    # pick up from the last checkpoint instead of starting over.
                    self._invoke(existing, None)
                return self.view(existing, duplicate_submission=True), False
            self.store.audit(run_id, "submitted", filename=filename, bytes=len(data))
            self._start(run_id, filename, data, source)
            return self.view(run_id), True

    def _start(self, run_id: str, filename: str, data: bytes, source: str | None) -> None:
        self._invoke(run_id, {"run_id": run_id, "filename": filename, "file_bytes": data,
                              "source": "ocr" if source == "ocr" else None, "attempts": 0, "path": [],
                              "trace": []})

    def _key_lock(self, key: str) -> threading.Lock:
        with self._key_locks_guard:
            if len(self._key_locks) > 1000:  # forget idle locks; a held one is re-created at worst
                self._key_locks = {k: v for k, v in self._key_locks.items() if v.locked()}
            return self._key_locks.setdefault(key, threading.Lock())

    def _refresh_vendors(self) -> None:
        if not self._shared_vendors:
            return
        rows = self.store.added_vendor_count()
        if rows != self._vendor_rows:
            self.deps.matcher = default_matcher().with_vendors(self.store.added_vendors())
            self._vendor_rows = rows

    def get(self, run_id: str) -> dict[str, Any]:
        return self.view(run_id)

    def review(self, run_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        snap = self.graph.get_state(self._config(run_id))
        if not snap.values:
            raise RunNotFound(run_id)
        if not snap.interrupts:
            raise NotAwaitingReview(f"run {run_id} is not waiting for a review")
        clean = validate_decision(decision, has_invoice=snap.values.get("invoice") is not None)
        self._refresh_vendors()
        self.store.audit(run_id, "reviewed", action=clean["action"], reviewer=clean["reviewer"],
                         note=clean["note"])
        self._invoke(run_id, Command(resume=clean))
        return self.view(run_id)

    def trace(self, run_id: str) -> dict[str, Any]:
        """The run's timeline (node, start, duration, outcome) plus its audit log."""
        v = self.view(run_id)
        return {"run_id": run_id, "status": v["status"], "path": v["path"], "trace": v["trace"],
                "total_ms": round(sum(e["ms"] for e in v["trace"]), 2), "audit": v["audit"]}

    def list_runs(self, limit: int = 500) -> list[dict[str, Any]]:
        """The most recent runs, newest first, each as a full view."""
        out = []
        for r in self.store.list_runs(limit):
            try:
                out.append(self.view(r["run_id"]))
            except RunNotFound:  # registered, but the graph never started (a crash before the first step)
                continue
        return out

    def list_vendors(self) -> list[dict[str, Any]]:
        self._refresh_vendors()
        return self.deps.matcher.all()

    def add_vendor(self, data: Any, added_by: str = "reviewer") -> dict[str, Any]:
        """Check and save a vendor a reviewer added. Raises VendorRejected with a readable reason."""
        self._refresh_vendors()
        vendor = check_new_vendor(data, self.deps.matcher)
        if vendor["vendor_id"] not in self.deps.matcher.vendors:
            added_by = str(added_by or "reviewer")[:80]
            self.store.add_vendor(vendor, added_by)
            self.deps.matcher = self.deps.matcher.with_vendors([vendor])  # the graph reads it on every run
            if self._shared_vendors:
                self._vendor_rows = self.store.added_vendor_count()
            self.store.audit(f"vendor:{vendor['vendor_id']}", "vendor_added", vendor_id=vendor["vendor_id"],
                             by=added_by)
        return vendor

    def _invoke(self, run_id: str, payload: Any) -> None:
        try:
            self.graph.invoke(payload, self._config(run_id))
        except Exception as exc:  # the checkpoint keeps the progress made so far
            # The audit log is returned by the API, so it gets the kind of error only; the
            # message (which can name a path or quote the document) stays in the server log.
            log.exception("run %s stopped", run_id)
            self.store.audit(run_id, "error", error=type(exc).__name__)

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
            "source": v.get("source"),
            "created_at": run.get("created_at"),
            "status": status,
            "duplicate_submission": duplicate_submission,
            "path": v.get("path", []),
            "trace": v.get("trace", []),
            "next": list(snap.next),
            "extractor": self.deps.extractor.name,
            "as_of": self.deps.today().isoformat(),
            "attempts": v.get("attempts", 0),
            "ingest_error": v.get("ingest_error"),
            # A PDF with no text layer: the demo page reads it by OCR in the browser and sends the text again.
            "scanned_pdf": v.get("ingest_error") == SCANNED_PDF_MESSAGE,
            "extraction_error": v.get("extraction_error"),
            "invoice": v.get("invoice"),
            "vendor_match": v.get("vendor_match"),
            "rules": v.get("rules", []),
            "review_request": pending[0] if pending else None,
            "decision": v.get("review"),
            "text": (v.get("text") or "")[:6000],
            "audit": self.store.audit_log(run_id),
        }
