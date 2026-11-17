"""The LangGraph flow end to end: every path through the graph."""

from dataclasses import replace
from datetime import date

import pytest

from invoice_agent.config import load_settings
from invoice_agent.extractors import ExtractionError
from invoice_agent.graph import MAX_EXTRACT_ATTEMPTS
from invoice_agent.service import InvoiceService, NotAwaitingReview, RunNotFound

AS_OF = date(2026, 11, 13)

GOOD = {
    "vendor_name": "Bluepeak Cloud Services Inc", "vendor_tax_id": None, "invoice_number": "BP-20001",
    "invoice_date": "2026-11-01", "due_date": "2026-12-01", "currency": "USD", "po_number": None,
    "line_items": [{"description": "Compute", "quantity": 2, "unit_price": 50, "amount": 100}],
    "subtotal": 100, "tax": 0, "total": 100,
}


class ScriptedExtractor:
    """Returns the scripted outputs in order and records the feedback it got."""

    name = "scripted"

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.feedback = []

    def extract(self, text, feedback=None):
        self.feedback.append(feedback)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("AS_OF_DATE", AS_OF.isoformat())
    monkeypatch.setenv("DB_PATH", str(tmp_path / "invoices.db"))
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "checkpoints.db"))
    return load_settings()


def service(settings, extractor=None):
    return InvoiceService.ephemeral(settings, extractor=extractor)


def test_happy_path_auto_approves_and_saves(settings, sample_bytes):
    svc = service(settings)
    run, created = svc.submit("01_northwind_ok.pdf", sample_bytes("01_northwind_ok.pdf"))
    assert created
    assert run["status"] == "auto_approved"
    assert run["path"] == ["ingest", "extract", "match_vendor", "validate", "auto_approve", "save"]
    assert all(r["passed"] for r in run["rules"])
    assert run["vendor_match"]["vendor_id"] == "V001"
    saved = svc.store.get_invoice(run["id"])
    assert saved["status"] == "auto_approved" and saved["total"] == 23954.0
    assert [e["event"] for e in run["audit"]] == ["submitted", "saved"]


def test_failed_rule_interrupts_for_review_then_approve_resumes(settings, sample_bytes):
    svc = service(settings)
    run, _ = svc.submit("04.pdf", sample_bytes("04_saffron_wrong_total.pdf"))
    assert run["status"] == "needs_review"
    assert run["path"] == ["ingest", "extract", "match_vendor", "validate"]
    assert run["next"] == ["human_review"]
    reasons = run["review_request"]["reasons"]
    assert len(reasons) == 1 and reasons[0].startswith("subtotal_plus_tax_equals_total")
    assert svc.store.get_invoice(run["id"]) is None  # nothing saved while waiting

    done = svc.review(run["id"], {"action": "approve", "reviewer": "asha", "note": "vendor confirmed by phone"})
    assert done["status"] == "approved_by_reviewer"
    assert done["path"][-2:] == ["human_review", "save"]
    assert done["decision"]["reviewer"] == "asha"
    assert svc.store.get_invoice(run["id"])["status"] == "approved_by_reviewer"


def test_reject_saves_as_rejected_and_stays_out_of_the_ledger(settings, sample_bytes):
    svc = service(settings)
    run, _ = svc.submit("06.txt", sample_bytes("06_unknown_vendor.txt"))
    before = svc.store.seen_invoices()
    done = svc.review(run["id"], {"action": "reject", "note": "not our supplier"})
    assert done["status"] == "rejected"
    assert svc.store.seen_invoices() == before


def test_correction_goes_back_through_vendor_match_and_validation(settings, sample_bytes):
    svc = service(settings)
    run, _ = svc.submit("04.pdf", sample_bytes("04_saffron_wrong_total.pdf"))
    fixed = dict(run["invoice"], total=44100.0)
    done = svc.review(run["id"], {"action": "correct", "invoice": fixed})
    assert done["status"] == "approved_after_correction"
    assert done["path"][4:] == ["human_review", "match_vendor", "validate", "auto_approve", "save"]
    assert svc.store.get_invoice(run["id"])["total"] == 44100.0


def test_a_bad_correction_pauses_again(settings, sample_bytes):
    svc = service(settings)
    run, _ = svc.submit("04.pdf", sample_bytes("04_saffron_wrong_total.pdf"))
    still_wrong = dict(run["invoice"], total=1.0)
    again = svc.review(run["id"], {"action": "correct", "invoice": still_wrong})
    assert again["status"] == "needs_review"
    assert again["path"].count("human_review") == 1 and again["path"].count("validate") == 2


def test_retry_once_on_schema_failure_with_error_fed_back(settings, sample_bytes):
    bad = dict(GOOD)
    del bad["invoice_number"]
    extractor = ScriptedExtractor(bad, GOOD)
    run, _ = service(settings, extractor).submit("x.txt", b"some invoice text")
    assert run["status"] == "auto_approved"
    assert run["path"][:3] == ["ingest", "extract", "extract"]
    assert run["attempts"] == 2
    assert extractor.feedback[0] is None
    assert "invoice_number: Field required" in extractor.feedback[1]


def test_a_model_gets_max_attempts_then_a_human_sees_the_fields_in_words(settings):
    bad = dict(GOOD)
    del bad["invoice_date"]
    extractor = ScriptedExtractor(*[bad] * MAX_EXTRACT_ATTEMPTS)
    run, _ = service(settings, extractor).submit("x.txt", b"some invoice text")
    assert run["path"] == ["ingest"] + ["extract"] * MAX_EXTRACT_ATTEMPTS
    assert "invoice_date: Field required" in extractor.feedback[1]  # the model gets the exact error
    reason = run["review_request"]["reasons"][0]  # the reviewer gets words
    assert f"after {MAX_EXTRACT_ATTEMPTS} attempts: could not find invoice date on the invoice" in reason


def test_retry_on_non_json_model_output(settings):
    extractor = ScriptedExtractor(ExtractionError("model returned output that is not JSON"), GOOD)
    run, _ = service(settings, extractor).submit("x.txt", b"some invoice text")
    assert run["status"] == "auto_approved"
    assert "not JSON" in extractor.feedback[1]


def test_gives_up_after_max_attempts_and_asks_a_human(settings, sample_bytes):
    svc = service(settings)  # demo extractor cannot find the missing number
    run, _ = svc.submit("07.txt", sample_bytes("07_missing_invoice_number.txt"))
    assert run["status"] == "needs_review"
    # The demo extractor is deterministic, so the graph does not retry it.
    assert run["path"] == ["ingest", "extract"]
    assert run["invoice"] is None
    assert "could not find invoice number on the invoice" in run["review_request"]["reasons"][0]
    draft = run["review_request"]["draft"]  # partial output, so the reviewer only adds what is missing
    assert draft["vendor_name"] == "Orbit Facility Management LLP" and "invoice_number" not in draft

    with pytest.raises(ValueError, match="nothing to approve"):
        svc.review(run["id"], {"action": "approve"})

    typed_in = draft | {"invoice_number": "OFM/26/402"}
    done = svc.review(run["id"], {"action": "correct", "invoice": typed_in, "reviewer": "asha"})
    assert done["status"] == "approved_after_correction"
    assert done["vendor_match"]["vendor_id"] == "V006"


def test_unreadable_file_goes_straight_to_review(settings):
    run, _ = service(settings).submit("scan.png", b"\x89PNG\r\n\x1a\n....")
    assert run["status"] == "needs_review"
    assert run["path"] == ["ingest"]
    assert "could not read the file" in run["review_request"]["reasons"][0]


def test_duplicate_detected_against_paid_ledger_and_own_approvals(settings, sample_bytes):
    svc = service(settings)
    dup, _ = svc.submit("05.txt", sample_bytes("05_bluepeak_duplicate.txt"))
    assert dup["status"] == "needs_review"
    assert [r["name"] for r in dup["rules"] if not r["passed"]] == ["not_a_duplicate"]

    first, _ = svc.submit("01.pdf", sample_bytes("01_northwind_ok.pdf"))
    assert first["status"] == "auto_approved"
    # Same invoice, different file (text copy instead of the PDF): different bytes, same invoice.
    again, created = svc.submit("01-copy.txt", first["text"].encode())
    assert created
    # Caught twice over: same vendor + number, and PO-7781 (30,000) cannot pay 23,954 a second time.
    assert [r["name"] for r in again["rules"] if not r["passed"]] == ["not_a_duplicate", "matches_purchase_order"]
    po = next(r for r in again["rules"] if r["name"] == "matches_purchase_order")
    assert "already approved" in po["detail"]


def test_same_file_twice_is_processed_once(settings, sample_bytes):
    svc = service(settings)
    data = sample_bytes("02_bluepeak_ok.txt")
    first, created1 = svc.submit("a.txt", data)
    second, created2 = svc.submit("renamed.txt", data)
    assert (created1, created2) == (True, False)
    assert second["id"] == first["id"] and second["duplicate_submission"]
    assert svc.store.count_invoices() == 1


def test_review_errors(settings, sample_bytes):
    svc = service(settings)
    with pytest.raises(RunNotFound):
        svc.review("nope", {"action": "approve"})
    run, _ = svc.submit("01.pdf", sample_bytes("01_northwind_ok.pdf"))
    with pytest.raises(NotAwaitingReview):
        svc.review(run["id"], {"action": "approve"})
    pending, _ = svc.submit("06.txt", sample_bytes("06_unknown_vendor.txt"))
    for bad in ({"action": "delete"}, {"action": "correct"}, {"action": "correct", "invoice": {"total": 1}}):
        with pytest.raises(ValueError):
            svc.review(pending["id"], bad)
    assert svc.get(pending["id"])["status"] == "needs_review"  # bad input did not move the run


def test_review_survives_a_restart(settings, sample_bytes):
    """SQLite checkpointer: pause in one process, resume in a 'new' one."""
    svc1 = InvoiceService.persistent(settings)
    run, _ = svc1.submit("09.txt", sample_bytes("09_saffron_po_exceeded.txt"))
    assert run["status"] == "needs_review"
    assert [r["name"] for r in run["rules"] if not r["passed"]] == ["matches_purchase_order"]
    del svc1

    svc2 = InvoiceService.persistent(settings)  # fresh graph, same database files
    assert svc2.get(run["id"])["status"] == "needs_review"
    done = svc2.review(run["id"], {"action": "approve", "note": "PO being amended"})
    assert done["status"] == "approved_by_reviewer"
    assert svc2.store.get_invoice(run["id"])["status"] == "approved_by_reviewer"


def test_a_crashed_run_resumes_from_its_checkpoint(settings):
    """The extractor blows up (not a schema error); resubmitting picks up from the checkpoint."""
    extractor = ScriptedExtractor(RuntimeError("provider outage"), GOOD)
    svc = InvoiceService.persistent(settings, extractor=extractor)
    run, _ = svc.submit("x.txt", b"invoice text")
    assert run["status"] == "incomplete"
    assert run["path"] == ["ingest"] and run["next"] == ["extract"]
    assert run["audit"][-1]["event"] == "error"

    again, created = svc.submit("x.txt", b"invoice text")
    assert not created and again["id"] == run["id"]
    assert again["status"] == "auto_approved"
    assert again["path"] == ["ingest", "extract", "match_vendor", "validate", "auto_approve", "save"]


def test_dates_use_the_configured_as_of_date(settings, sample_bytes):
    later = replace(settings, as_of=date(2028, 1, 1))
    run, _ = service(later).submit("01.pdf", sample_bytes("01_northwind_ok.pdf"))
    assert run["as_of"] == "2028-01-01"
    assert [r["name"] for r in run["rules"] if not r["passed"]] == ["dates_are_sane"]
