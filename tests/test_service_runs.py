"""Submissions that race, crash before their first checkpoint, or differ only by source."""

import threading

import pytest

from invoice_agent.config import load_settings
from invoice_agent.service import InvoiceService, idempotency_key


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("AS_OF_DATE", "2026-11-13")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "invoices.db"))
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "checkpoints.db"))
    return load_settings()


def test_a_run_registered_but_never_started_is_started_on_resubmit(settings, sample_bytes):
    # A crash between registering a run and its first checkpoint left a row with no
    # graph state. Before 2026-11-25 every later upload of that file was a 500.
    data = sample_bytes("02_bluepeak_ok.txt")
    svc = InvoiceService.persistent(settings)
    svc.store.register_run("orphan0000000001", idempotency_key(data), "02_bluepeak_ok.txt")
    run, created = svc.submit("02_bluepeak_ok.txt", data)
    assert not created and run["id"] == "orphan0000000001"
    assert run["status"] == "auto_approved"


def test_the_same_file_uploaded_at_once_gives_one_run(settings, sample_bytes):
    data = sample_bytes("02_bluepeak_ok.txt")
    svc = InvoiceService.persistent(settings)
    results, errors = [], []

    def go():
        try:
            results.append(svc.submit("02_bluepeak_ok.txt", data))
        except Exception as exc:  # pragma: no cover - the failure this test guards against
            errors.append(repr(exc))

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len({run["id"] for run, _ in results}) == 1
    assert sorted(created for _, created in results) == [False, False, False, True]
    assert {run["status"] for run, _ in results} == {"auto_approved"}


def test_resending_the_same_text_as_ocr_is_a_new_run_that_goes_to_review(settings, sample_bytes):
    data = sample_bytes("02_bluepeak_ok.txt")
    svc = InvoiceService.persistent(settings)
    first, _ = svc.submit("02_bluepeak_ok.txt", data)
    assert first["status"] == "auto_approved"
    ocr, created = svc.submit("photo.png.txt", data, source="ocr")
    assert created and ocr["id"] != first["id"]
    assert ocr["source"] == "ocr" and ocr["status"] == "needs_review"


def test_the_audit_log_names_the_error_kind_not_its_text(settings, monkeypatch):
    # The audit log is returned by GET /invoices/{id} and /trace.
    svc = InvoiceService.persistent(settings)

    def boom(*a, **k):
        raise RuntimeError("/srv/app/secret/path.db is locked")

    monkeypatch.setattr(svc.graph, "invoke", boom)
    svc.store.register_run("r1", "k1", "f")
    svc._invoke("r1", {})
    errors = [e for e in svc.store.audit_log("r1") if e["event"] == "error"]
    assert errors and errors[0]["error"] == "RuntimeError"
    assert "/srv" not in str(errors)
