"""Audit trail and observability: the per-node timeline, the trace endpoint, JSON logs without PII."""

import json
import logging

import pytest

from invoice_agent.config import load_settings
from invoice_agent.obs import JsonFormatter, _safe, tracing_backend
from invoice_agent.service import InvoiceService

from .conftest import ROOT
from .test_api import make_client


@pytest.fixture
def svc(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("AS_OF_DATE", "2026-11-13")
    return InvoiceService.ephemeral(load_settings())


def test_every_node_gets_a_timed_entry(svc, sample_bytes):
    run, _ = svc.submit("01.pdf", sample_bytes("01_northwind_ok.pdf"))
    trace = run["trace"]
    assert [e["node"] for e in trace] == run["path"]
    assert all(e["ms"] >= 0 and e["at"].endswith("+00:00") for e in trace)
    by = {e["node"]: e for e in trace}
    assert by["ingest"]["chars"] > 100 and by["extract"]["ok"] and by["extract"]["attempt"] == 1
    assert by["match_vendor"] == by["match_vendor"] | {"vendor_id": "V001", "matched": True}
    assert by["validate"]["failed"] == [] and by["validate"]["passed"] == len(run["rules"])
    assert by["save"]["status"] == "auto_approved"


def test_a_paused_and_resumed_run_records_the_review(svc, sample_bytes):
    run, _ = svc.submit("04.pdf", sample_bytes("04_saffron_wrong_total.pdf"))
    assert run["trace"][-1]["node"] == "validate"  # the pause itself is not a finished step
    assert run["trace"][-1]["failed"] == ["subtotal_plus_tax_equals_total"]
    done = svc.review(run["id"], {"action": "reject", "reviewer": "asha"})
    review = [e for e in done["trace"] if e["node"] == "human_review"]
    assert len(review) == 1 and review[0]["action"] == "reject" and review[0]["status"] == "rejected"
    t = svc.trace(run["id"])
    assert t["total_ms"] == pytest.approx(sum(e["ms"] for e in t["trace"]), abs=0.05)
    assert [a["event"] for a in t["audit"]] == ["submitted", "reviewed", "saved"]


def test_the_trace_holds_no_document_content(svc, sample_bytes):
    run, _ = svc.submit("secret-name.txt", sample_bytes("06_unknown_vendor.txt"))
    dumped = json.dumps(run["trace"])
    for leak in ("Quillmark", "QS-311", "7788", "33AAQFQ9876P1Z4", "secret-name", "Whiteboard"):
        assert leak not in dumped
    assert _safe({"vendor_name": "x", "text": "y", "status": "ok"}) == {"status": "ok"}


def test_json_logs_carry_fields_but_no_names_or_contents(monkeypatch, tmp_path, caplog):
    client = make_client(monkeypatch, tmp_path, demo=True)
    with caplog.at_level(logging.INFO):
        data = (ROOT / "samples" / "06_unknown_vendor.txt").read_bytes()
        client.post("/api/demo/run", files={"file": ("private-name.txt", data, "text/plain")})
    fmt = JsonFormatter()
    lines = [json.loads(fmt.format(r)) for r in caplog.records if r.name.startswith("invoice_agent")]
    events = [line.get("event") for line in lines]
    assert "node" in events and "demo_run" in events
    demo = next(line for line in lines if line.get("event") == "demo_run")
    assert demo["status"] == "needs_review" and demo["failed_rules"] == ["vendor_is_known"] and demo["ext"] == "txt"
    text = "\n".join(json.dumps(line) for line in lines)
    for leak in ("private-name", "Quillmark", "QS-311", "7,788", "7788"):
        assert leak not in text


def test_json_formatter_never_writes_a_stack_trace():
    try:
        raise ValueError("secret detail /home/user/path")
    except ValueError:
        import sys

        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "boom", None, sys.exc_info())
    out = json.loads(JsonFormatter().format(record))
    assert out["exc_type"] == "ValueError" and "secret" not in json.dumps(out)


def test_trace_endpoint_in_the_full_service(monkeypatch, tmp_path):
    full = make_client(monkeypatch, tmp_path, demo=False)
    run = full.post("/invoices", data={"sample": "06_unknown_vendor.txt"}).json()
    t = full.get(f"/api/runs/{run['id']}/trace")
    assert t.status_code == 200 and t.headers["cache-control"] == "no-store"
    body = t.json()
    assert [e["node"] for e in body["trace"]] == body["path"] and body["status"] == "needs_review"
    assert full.get("/api/runs/nope/trace").status_code == 404
    demo = make_client(monkeypatch, tmp_path / "d", demo=True)
    assert demo.get(f"/api/runs/{run['id']}/trace").status_code == 404  # the demo stores nothing


@pytest.mark.parametrize("env, want", [
    ({}, "off"),
    ({"LANGSMITH_TRACING": "true"}, "off"),  # no key: nothing is sent
    ({"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "k"}, "langsmith"),
    ({"LANGCHAIN_TRACING_V2": "true", "LANGCHAIN_API_KEY": "k"}, "langsmith"),
])
def test_langsmith_is_opt_in(monkeypatch, env, want):
    for k in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert tracing_backend() == want
