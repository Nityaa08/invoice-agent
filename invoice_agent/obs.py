"""Observability: a timeline of every run, structured JSON logs, optional LangSmith.

- **Timeline.** Each graph node is wrapped by `traced()`, which appends one entry
  to the run's `trace`: the node, when it started, how long it took and a short
  outcome (the rules that failed, the vendor id and score, the review action).
  It is part of the graph state, so it is checkpointed with the run and comes
  back with GET /invoices/{id} and GET /api/runs/{id}/trace.
- **Logs.** One JSON line per node (`event: node`) and per request outcome.
  Nothing from the document goes into a log: no file names, vendor names,
  amounts or text; ids, timings, counts and rule names only.
- **LangSmith (optional).** Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY and
  LangGraph sends its own traces there. Off by default, because a trace
  carries the invoice text and fields to a third party.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable

log = logging.getLogger("invoice_agent.trace")

# Keys that may appear in a node's outcome. Anything else is dropped before it is
# stored or logged, so a careless outcome() cannot leak document content.
SAFE_KEYS = {"chars", "error_kind", "ok", "attempt", "extractor", "vendor_id", "score", "matched", "source",
             "passed", "failed", "status", "action", "notes"}


def _safe(outcome: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in outcome.items() if k in SAFE_KEYS}


def traced(name: str, fn: Callable[[dict], dict], outcome: Callable[[dict, dict], dict[str, Any]]):
    """Wrap a node: time it, add a trace entry to its update, log one JSON line.

    An interrupt() inside `fn` raises; it propagates untouched (the pause is not
    a finished step; the resumed call records the entry).
    """

    def node(state: dict) -> dict:
        started = time.perf_counter()
        at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        update = fn(state)
        ms = round((time.perf_counter() - started) * 1000, 2)
        entry = {"node": name, "at": at, "ms": ms, **_safe(outcome(state, update))}
        log.info("node", extra={"fields": {"event": "node", "run_id": state.get("run_id"), **entry}})
        return {**update, "trace": [entry]}

    node.__name__ = name
    return node


def tracing_backend() -> str:
    on = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "")).strip().lower() in {"1", "true", "yes"}
    has_key = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
    return "langsmith" if on and has_key else "off"


class JsonFormatter(logging.Formatter):
    """One JSON object per line: time, level, logger, message and any `extra={"fields": {...}}`."""

    def format(self, record: logging.LogRecord) -> str:
        out = {"time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
               "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            out.update(fields)
        if record.exc_info:
            out["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None  # no message, no stack
        return json.dumps(out, default=str)


def setup_logging() -> None:
    """JSON logs by default (LOG_FORMAT=text for plain lines while developing)."""
    handler = logging.StreamHandler()
    if os.getenv("LOG_FORMAT", "json").lower() == "text":
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    else:
        handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
