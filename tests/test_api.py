"""HTTP API: the full service (SQLite) and the stateless demo."""

import ast
import importlib
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from invoice_agent.api import register_routes
from invoice_agent.config import load_settings
from invoice_agent.ingest import MAX_UPLOAD_BYTES

from .conftest import ROOT


def make_client(monkeypatch, tmp_path, demo: bool) -> TestClient:
    monkeypatch.setenv("DEMO_MODE", "true" if demo else "false")
    monkeypatch.setenv("AS_OF_DATE", "2026-11-13")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "invoices.db"))
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "checkpoints.db"))
    monkeypatch.delenv("EXTRACTOR", raising=False)
    app = FastAPI()
    register_routes(app, load_settings())
    return TestClient(app)


@pytest.fixture
def full(monkeypatch, tmp_path):
    return make_client(monkeypatch, tmp_path, demo=False)


@pytest.fixture
def demo(monkeypatch, tmp_path):
    return make_client(monkeypatch, tmp_path, demo=True)


def test_page_config_and_samples(full):
    assert full.get("/").status_code == 200
    assert "invoice-agent" in full.get("/").text
    cfg = full.get("/api/config").json()
    assert cfg.pop("rules")[:2] == ["line_items_sum_to_subtotal", "subtotal_plus_tax_equals_total"]
    assert cfg == {
        "demo_mode": False, "extractor": "demo", "llm_model": None, "as_of": "2026-11-13",
        "as_of_uploads": "2026-11-13", "max_upload_mb": 4,
    }
    samples = full.get("/api/samples").json()
    assert len(samples) == 11 and samples[0]["file"] == "01_northwind_ok.pdf"
    assert full.get("/api/samples/01_northwind_ok.pdf").content.startswith(b"%PDF")
    assert full.get("/api/samples/..%2Fmain.py").status_code == 404


def test_upload_get_and_idempotency(full):
    pdf = (ROOT / "samples" / "01_northwind_ok.pdf").read_bytes()
    r = full.post("/invoices", files={"file": ("inv.pdf", pdf, "application/pdf")})
    assert r.status_code == 201
    run = r.json()
    assert run["status"] == "auto_approved"
    assert run["invoice"]["invoice_number"] == "NW-2026-0142"

    assert full.get(f"/invoices/{run['id']}").json()["status"] == "auto_approved"
    again = full.post("/invoices", files={"file": ("again.pdf", pdf, "application/pdf")})
    assert again.status_code == 200 and again.json()["id"] == run["id"]
    assert again.json()["duplicate_submission"] is True


def test_review_resumes_the_graph(full):
    run = full.post("/invoices", data={"sample": "06_unknown_vendor.txt"}).json()
    assert run["status"] == "needs_review"
    assert run["review_request"]["actions"] == ["approve", "reject", "correct"]

    r = full.post(f"/invoices/{run['id']}/review", json={"action": "approve", "reviewer": "asha",
                                                           "note": "new supplier, onboarding"})
    assert r.status_code == 200
    assert r.json()["status"] == "approved_by_reviewer"
    assert r.json()["path"][-2:] == ["human_review", "save"]
    # Reviewing again: the run is no longer waiting.
    assert full.post(f"/invoices/{run['id']}/review", json={"action": "approve"}).status_code == 409


def test_review_with_correction(full):
    run = full.post("/invoices", data={"sample": "04_saffron_wrong_total.pdf"}).json()
    fixed = dict(run["invoice"], total=44100.0)
    r = full.post(f"/invoices/{run['id']}/review", json={"action": "correct", "invoice": fixed})
    assert r.json()["status"] == "approved_after_correction"


@pytest.mark.parametrize(
    "call, status",
    [
        (lambda c: c.get("/invoices/nope"), 404),
        (lambda c: c.post("/invoices/nope/review", json={"action": "approve"}), 404),
        (lambda c: c.post("/invoices"), 422),
        (lambda c: c.post("/invoices", data={"sample": "secret.txt"}), 404),
        (lambda c: c.post("/invoices", files={"file": ("big.txt", b"x" * (MAX_UPLOAD_BYTES + 1))}), 413),
        (lambda c: c.post("/api/demo/run", data={"sample": "01_northwind_ok.pdf"}), 404),  # demo off
    ],
)
def test_full_mode_errors(full, call, status):
    assert call(full).status_code == status


def test_bad_review_input_is_422(full):
    run = full.post("/invoices", data={"sample": "07_missing_invoice_number.txt"}).json()
    assert full.post(f"/invoices/{run['id']}/review", json={"action": "approve"}).status_code == 422
    assert full.post(f"/invoices/{run['id']}/review", json={"action": "delete"}).status_code == 422
    assert full.post(f"/invoices/{run['id']}/review", json={"nope": 1}).status_code == 422


def test_demo_runs_are_stateless_and_replay_decisions(demo):
    assert demo.get("/api/config").json()["demo_mode"] is True
    assert demo.post("/invoices", data={"sample": "01_northwind_ok.pdf"}).status_code in (404, 405)

    first = demo.post("/api/demo/run", data={"sample": "09_saffron_po_exceeded.txt"}).json()
    assert first["status"] == "needs_review"
    decided = demo.post("/api/demo/run", data={
        "sample": "09_saffron_po_exceeded.txt",
        "decisions": json.dumps([{"action": "reject", "note": "PO exhausted"}]),
    }).json()
    assert decided["status"] == "rejected"
    assert decided["path"][-2:] == ["human_review", "save"]

    # Nothing persists: the same file again is a new, fresh run.
    again = demo.post("/api/demo/run", data={"sample": "09_saffron_po_exceeded.txt"}).json()
    assert again["status"] == "needs_review" and again["id"] != first["id"]


def test_demo_upload_and_bad_decisions(demo):
    text = (ROOT / "samples" / "02_bluepeak_ok.txt").read_bytes()
    r = demo.post("/api/demo/run", files={"file": ("mine.txt", text, "text/plain")})
    assert r.json()["status"] == "auto_approved"
    assert demo.post("/api/demo/run", data={"sample": "06_unknown_vendor.txt", "decisions": "nope"}).status_code == 422
    too_many = json.dumps([{"action": "approve"}] * 6)
    assert demo.post("/api/demo/run", data={"sample": "06_unknown_vendor.txt", "decisions": too_many}).status_code == 422
    bad = json.dumps([{"action": "approve"}])
    assert demo.post("/api/demo/run", data={"sample": "07_missing_invoice_number.txt", "decisions": bad}).status_code == 422


def test_demo_rate_limit(demo, monkeypatch):
    import invoice_agent.api as api

    # Rebuild with a tiny limit so the test is quick.
    monkeypatch.setattr(api, "DEMO_RUNS_PER_HOUR", 2)
    app = FastAPI()
    api.register_routes(app, load_settings())
    client = TestClient(app)
    codes = [client.post("/api/demo/run", data={"sample": "01_northwind_ok.pdf"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_vercel_defaults_to_demo_mode(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("VERCEL", "1")
    assert load_settings().demo_mode is True
    monkeypatch.setenv("DEMO_MODE", "false")  # an explicit setting still wins
    assert load_settings().demo_mode is False


def test_main_defines_a_top_level_fastapi_app_for_vercel(monkeypatch, tmp_path):
    # Vercel's FastAPI preset needs `app = FastAPI(...)` written in the entry file itself.
    tree = ast.parse((ROOT / "main.py").read_text())
    assigns = [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "app" for t in n.targets)]
    assert assigns and isinstance(assigns[0].value, ast.Call) and assigns[0].value.func.id == "FastAPI"

    monkeypatch.setenv("DEMO_MODE", "true")
    import main

    importlib.reload(main)
    assert isinstance(main.app, FastAPI)
    paths = {r.path for r in main.app.routes}
    assert "/api/demo/run" in paths and "/invoices" not in paths
