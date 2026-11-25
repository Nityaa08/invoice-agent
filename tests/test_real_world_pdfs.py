"""Uploads that look like real invoices, through the same endpoint the demo page uses.

The PDFs in tests/fixtures/pdfs/ are made by scripts/make_test_pdfs.py and
are deliberately unlike the bundled samples: bordered tables, two-column
headers, labels above their values, GST with CGST/SGST/IGST, lakh grouping,
US and EU number and date formats, three pages, a prose invoice, a scan.
Labels are in tests/fixtures/pdfs/labels.json.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from invoice_agent.api import register_routes
from invoice_agent.config import load_settings
from invoice_agent.extractors import (
    DemoExtractor, LLMExtractor, detect_decimal_comma, find_date, parse_amount,
)
from invoice_agent.ingest import MAX_UPLOAD_BYTES, SCANNED_PDF_MESSAGE, IngestError, to_text

from .conftest import ROOT
from .test_llm_extractor import fake_openai

FIXTURES = ROOT / "tests" / "fixtures" / "pdfs"
LABELS = json.loads((FIXTURES / "labels.json").read_text(encoding="utf-8"))
READABLE = [i for i in LABELS["invoices"] if not i.get("extraction_fails")]


def pdf(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.fixture
def demo(monkeypatch, tmp_path):
    """The public demo's configuration (Vercel sets VERCEL=1 -> DEMO_MODE)."""
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("AS_OF_DATE", LABELS["as_of"])
    monkeypatch.delenv("EXTRACTOR", raising=False)
    app = FastAPI()
    register_routes(app, load_settings())
    return TestClient(app)


def upload(client, name: str, data: bytes, content_type: str = "application/pdf"):
    return client.post("/api/demo/run", files={"file": (name, data, content_type)})


# --- end to end through the demo endpoint -------------------------------------


@pytest.mark.parametrize("label", READABLE, ids=lambda x: x["file"])
def test_real_world_pdf_is_read_and_routed(demo, label):
    r = upload(demo, label["file"], pdf(label["file"]))
    assert r.status_code == 200
    run = r.json()
    assert run["ingest_error"] is None and run["extraction_error"] is None
    got = dict(run["invoice"], line_count=len(run["invoice"]["line_items"]))
    for field, want in label["expected"].items():
        assert got[field] == want, field
    failed = sorted(x["name"] for x in run["rules"] if not x["passed"])
    assert failed == label["failed_rules"]
    assert run["status"] == ("auto_approved" if label["decision"] == "auto_approve" else "needs_review")


def test_indian_gst_invoice_reads_lakh_grouping_and_both_gst_halves(demo):
    run = upload(demo, "gst.pdf", pdf("gst_tax_invoice.pdf")).json()
    inv = run["invoice"]
    assert inv["line_items"][0] == {"description": "Steel almirah, 2 door", "quantity": 12.0,
                                    "unit_price": 8500.0, "discount": 0.0, "amount": 102000.0}
    assert inv["tax"] == 10449.0 * 2  # CGST 9% + SGST 9%
    passed = {x["name"] for x in run["rules"] if x["passed"]}
    assert {"line_items_sum_to_subtotal", "subtotal_plus_tax_equals_total", "dates_are_sane"} <= passed


def test_us_invoice_reads_month_first_dates(demo):
    inv = upload(demo, "us.pdf", pdf("us_invoice.pdf")).json()["invoice"]
    assert (inv["invoice_date"], inv["due_date"]) == ("2026-11-03", "2026-12-03")


def test_multi_page_invoice_keeps_every_line_and_the_last_page_totals(demo):
    run = upload(demo, "multi.pdf", pdf("multipage_invoice.pdf")).json()
    assert len(run["invoice"]["line_items"]) == 90
    assert run["invoice"]["total"] == 6305.23
    assert next(x for x in run["rules"] if x["name"] == "line_items_sum_to_subtotal")["passed"]


def test_unreadable_table_falls_back_to_one_total_line_and_asks_a_human(demo):
    run = upload(demo, "services.pdf", pdf("services_invoice.pdf")).json()
    assert run["status"] == "needs_review"
    assert run["invoice"]["line_items"] == [{"description": "Invoice total (line items not itemised)",
                                             "quantity": 1.0, "unit_price": 85000.0, "discount": 0.0, "amount": 85000.0}]
    reasons = " ".join(run["review_request"]["reasons"])
    assert "extraction_is_complete: could not read the line-item table" in reasons


def test_reviewer_correction_clears_the_extraction_note(demo):
    data = pdf("services_invoice.pdf")
    first = upload(demo, "s.pdf", data).json()
    fixed = dict(first["invoice"], line_items=[
        {"description": "Festive packaging design", "quantity": 1, "unit_price": 85000, "amount": 85000}])
    decisions = json.dumps([{"action": "correct", "invoice": fixed}])
    run = demo.post("/api/demo/run", files={"file": ("s.pdf", data, "application/pdf")},
                    data={"decisions": decisions}).json()
    assert "extraction_is_complete" not in {x["name"] for x in run["rules"]}
    assert [x["name"] for x in run["rules"] if not x["passed"]] == ["vendor_is_known"]


def test_scanned_pdf_gets_a_clear_message_not_a_fake_extraction(demo):
    r = upload(demo, "scan.pdf", pdf("scanned_invoice.pdf"))
    assert r.status_code == 200
    run = r.json()
    assert run["ingest_error"] == SCANNED_PDF_MESSAGE
    assert run["invoice"] is None and run["path"] == ["ingest"]
    assert run["status"] == "needs_review"
    assert "scanned image" in run["review_request"]["reasons"][0]


@pytest.mark.parametrize(
    "name, data, message",
    [
        ("truncated.pdf", pdf("truncated.pdf"), "could not be read as a PDF"),
        ("invoice.pdf", b"<html><body>404 Not Found</body></html>", "is named .pdf but is not a PDF"),
        ("invoice.docx", b"PK\x03\x04" + b"\x00" * 200, "Word or Excel"),
        ("photo.heic", bytes(range(256)) * 4, "unsupported file type"),
    ],
)
def test_corrupt_or_not_a_pdf_is_explained_never_a_500(demo, name, data, message):
    r = upload(demo, name, data, "application/octet-stream")
    assert r.status_code == 200
    run = r.json()
    assert message in run["ingest_error"]
    assert run["status"] == "needs_review"


def test_oversize_upload_is_413_with_the_limit(demo):
    big = b"%PDF-1.4\n" + b"0" * MAX_UPLOAD_BYTES
    r = upload(demo, "big.pdf", big)
    assert r.status_code == 413
    assert "larger than 4 MB" in r.json()["detail"]


def test_empty_upload_is_422(demo):
    r = upload(demo, "empty.pdf", b"")
    assert r.status_code == 422


def test_demo_checks_upload_dates_against_today(monkeypatch):
    """Without AS_OF_DATE the fixed demo date applies to samples only, so a
    real invoice dated today is not "in the future"."""
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.delenv("AS_OF_DATE", raising=False)
    app = FastAPI()
    register_routes(app, load_settings())
    client = TestClient(app)
    from datetime import date

    assert upload(client, "us.pdf", pdf("us_invoice.pdf")).json()["as_of"] == date.today().isoformat()
    sample = client.post("/api/demo/run", data={"sample": "01_northwind_ok.pdf"}).json()
    assert sample["as_of"] == "2026-11-13"
    assert client.get("/api/config").json()["as_of_uploads"] == "today"


def test_llm_mode_reads_uploads_too(monkeypatch, tmp_path):
    """EXTRACTOR=llm: the uploaded PDF's text reaches the model and its answer runs the graph."""
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("AS_OF_DATE", LABELS["as_of"])
    monkeypatch.setenv("EXTRACTOR", "llm")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    answer = {
        "vendor_name": "Sri Lakshmi Traders Pvt. Ltd.", "vendor_tax_id": "33AAKCS1234M1Z9",
        "invoice_number": "SLT/2026-27/0457", "invoice_date": "2026-11-14", "due_date": None,
        "currency": "INR", "po_number": None,
        "line_items": [{"description": "Steel almirah", "quantity": 12, "unit_price": 8500, "amount": 102000},
                       {"description": "Office chair", "quantity": 6, "unit_price": 2350, "amount": 14100}],
        "subtotal": 116100, "tax": 20898, "total": 136998,
    }
    requests = []
    import invoice_agent.service as service_mod

    monkeypatch.setattr(service_mod, "build_extractor",
                        lambda *a, **k: LLMExtractor("m", "k", chat_model=fake_openai([json.dumps(answer)], requests)))
    app = FastAPI()
    register_routes(app, load_settings())
    client = TestClient(app)
    assert client.get("/api/config").json()["extractor"] == "llm"
    run = upload(client, "gst.pdf", pdf("gst_tax_invoice.pdf")).json()
    assert "SLT/2026-27/0457" in requests[0]["messages"][-1]["content"]  # the PDF text was sent
    assert run["invoice"]["total"] == 136998.0
    assert [x["name"] for x in run["rules"] if not x["passed"]] == ["vendor_is_known"]


def test_llm_without_a_key_falls_back_to_demo_instead_of_failing(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("EXTRACTOR", "llm")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    app = FastAPI()
    register_routes(app, load_settings())
    r = upload(TestClient(app), "us.pdf", pdf("us_invoice.pdf"))
    assert r.status_code == 200 and r.json()["extractor"] == "demo-rules"


# --- the pieces ------------------------------------------------------------------


def test_scanned_pdf_ingest_message():
    with pytest.raises(IngestError, match="scanned image"):
        to_text("scan.pdf", pdf("scanned_invoice.pdf"))


@pytest.mark.parametrize(
    "raw, decimal_comma, expected",
    [
        ("₹ 1,36,998.00", False, 136998.0),
        ("Rs.2,24,200.00", False, 224200.0),
        ("1,20,000", False, 120000.0),
        ("€ 1.500,00", True, 1500.0),
        ("1.250", True, 1250.0),
        ("1.250", False, 1.25),
        ("$7,247.30", False, 7247.3),
        ("IGST18 (18%)", False, None),
    ],
)
def test_parse_amount_with_document_locale(raw, decimal_comma, expected):
    assert parse_amount(raw, decimal_comma) == expected


def test_detect_decimal_comma():
    assert detect_decimal_comma("Amount  € 1.500,00\nVAT  € 315,00")
    assert not detect_decimal_comma("Total  ₹ 1,02,000.00\nCGST  9,180.00")


@pytest.mark.parametrize(
    "raw, mdy, expected",
    [
        ("14-Nov-2026", False, "2026-11-14"),
        ("12 Nov 2026", False, "2026-11-12"),
        ("2nd November, 2026", False, "2026-11-02"),
        ("November 2, 2026", False, "2026-11-02"),
        ("2026/11/10", False, "2026-11-10"),
        ("03.11.2026", False, "2026-11-03"),
        ("11/03/2026", True, "2026-11-03"),
        ("11/03/2026", False, "2026-03-11"),
        ("25/11/2026", True, "2026-11-25"),  # a part over 12 settles it
        ("14-11-26", False, "2026-11-14"),
        ("Net 30", False, None),
    ],
)
def test_find_date_formats(raw, mdy, expected):
    assert find_date(raw, prefer_mdy=mdy) == expected


@pytest.mark.parametrize(
    "text, field, expected",
    [
        ("ACME Ltd\nInvoice No.: 77-A\nGrand Total  100.00", "invoice_number", "77-A"),
        ("ACME Ltd\nInv # 5521\nTotal 100.00", "invoice_number", "5521"),
        ("ACME Ltd\nBill No  : B/9\nTotal 100.00", "invoice_number", "B/9"),
        ("ACME Ltd\nInvoice No.      Dated\nX-1              1-Jan-2026\nTotal 1", "invoice_number", "X-1"),
        ("ACME Ltd\nInvoice No.      Dated\nX-1              1-Jan-2026\nTotal 1", "invoice_date", "2026-01-01"),
        ("ACME Ltd  USD\nSubtotal  90.00\nTax  10.00\nAmount Due  100.00", "total", 100.0),
        ("ACME Ltd  USD\nSubtotal  90.00\nTax  10.00\nBalance Due  100.00", "total", 100.0),
        ("ACME Ltd\nGrand Total  ₹ 1,23,456.00", "total", 123456.0),
        ("ACME Ltd\nGrand Total  ₹ 1,23,456.00", "currency", "INR"),
        ("ACME Ltd\nTotal   INR 500", "currency", "INR"),
    ],
)
def test_label_synonyms(text, field, expected):
    assert DemoExtractor().extract(text).get(field) == expected


def test_missing_fields_are_left_out_not_guessed():
    out = DemoExtractor().extract("Some Company Pvt Ltd\nThank you for your business")
    for field in ("invoice_number", "invoice_date", "total", "subtotal", "currency"):
        assert field not in out
    assert out["line_items"] == []


def test_missing_fields_reach_the_reviewer_in_words(demo):
    text = b"Some Company Pvt Ltd\nInvoice No: 12\nThank you for your business\n"
    run = upload(demo, "note.txt", text, "text/plain").json()
    assert run["status"] == "needs_review" and run["invoice"] is None
    reason = run["review_request"]["reasons"][0]
    assert "could not find invoice date, currency, line items, subtotal, total on the invoice" in reason
    assert run["review_request"]["draft"]["invoice_number"] == "12"


def test_fixtures_stay_small():
    for f in FIXTURES.glob("*.pdf"):
        assert f.stat().st_size < 60 * 1024, f.name


# --- round-off lines (Tally prints one on most invoices) --------------------------


def test_round_off_invoice_from_a_known_vendor_is_auto_approved(demo):
    # Before 2026-11-23 the round-off line was ignored, so this correct invoice
    # failed subtotal + tax = total by 0.40 and always went to a person.
    run = upload(demo, "gst_round_off.pdf", pdf("gst_round_off.pdf")).json()
    assert run["invoice"]["rounding"] == -0.4
    assert run["status"] == "auto_approved"
    detail = next(r["detail"] for r in run["rules"] if r["name"] == "subtotal_plus_tax_equals_total")
    assert "- round-off 0.40 = 17322.00" in detail


@pytest.mark.parametrize("line, expected", [
    ("Round Off                     (-)0.40", -0.4),
    ("Rounded Off                     0.35", 0.35),
    ("Rounding Off                   -0.25", -0.25),
    ("Less: Round Off                 0.30", -0.3),
    ("Round off                      (0.45)", -0.45),
])
def test_round_off_line_is_read_with_its_sign(line, expected):
    text = "\n".join([
        "TAX INVOICE", "Northwind Office Supplies Pvt Ltd", "Invoice No.: NW-1  Dated: 12-Nov-2026",
        "Description                 Quantity     Rate        Amount",
        "Copier paper                      10   100.00      1,000.00",
        "Taxable Value                                     1,000.00",
        "IGST @ 18%                                          180.00",
        line,
        "Total                                    ₹ " + f"{1180 + expected:,.2f}",
    ])
    out = DemoExtractor().extract(text)
    assert out["rounding"] == expected
    assert out["total"] == round(1180 + expected, 2)


# --- a total that wrapped in a narrow amount column ----------------------------


def test_wrapped_total_is_read_not_the_quantity_total(demo):
    # Before 2026-11-24 "Total   15 Nos   ₹" with "4,071.00" wrapped below it was
    # read as a total of 15.00 (the quantity column), so a correct invoice went
    # to review showing the wrong amount.
    run = upload(demo, "gst_wrapped_total.pdf", pdf("gst_wrapped_total.pdf")).json()
    inv = run["invoice"]
    assert (inv["subtotal"], inv["tax"], inv["total"]) == (3450.0, 621.0, 4071.0)
    assert run["status"] == "auto_approved"


@pytest.mark.parametrize("total_lines, expected", [
    (["Total                 15 Nos             ₹ 1,180.00"], 1180.0),     # quantity on the line, amount too
    (["Total                 15 Nos             ₹", "                                    1,180.00"], 1180.0),
    (["Total                 24 Pcs", "                                    1,180.00"], 1180.0),
    (["Grand Total           3.5 Hrs", "                                  ₹ 1,180.00"], 1180.0),
])
def test_total_line_skips_the_quantity_and_finds_a_wrapped_amount(total_lines, expected):
    text = "\n".join([
        "TAX INVOICE", "Northwind Office Supplies Pvt Ltd", "Invoice No.: NW-1  Dated: 12-Nov-2026",
        "Description                 Quantity     Rate        Amount",
        "Copier paper                      10   100.00      1,000.00",
        "Taxable Value                                     1,000.00",
        "IGST @ 18%                                          180.00",
        *total_lines,
        "Amount Chargeable (in words): INR One Thousand One Hundred Eighty Only",
    ])
    assert DemoExtractor().extract(text)["total"] == expected


def test_a_quantity_total_alone_is_never_the_invoice_total():
    # No amount anywhere near the Total line: better no total (and a person
    # looks) than the item count shown as the amount.
    text = "\n".join([
        "TAX INVOICE", "Northwind Office Supplies Pvt Ltd", "Invoice No.: NW-1  Dated: 12-Nov-2026",
        "Description                 Quantity     Rate        Amount",
        "Copier paper                      10   100.00      1,000.00",
        "Total                 15 Nos",
        "Amount Chargeable (in words): INR One Thousand Only",
    ])
    assert DemoExtractor().extract(text).get("total") != 15.0


# --- line-item descriptions that wrap in their column ---------------------------


def test_wrapped_descriptions_are_joined():
    # Before 2026-11-24 only the line holding the numbers was kept:
    # "Whiteboard markers, pack" lost "of 4".
    text = "\n".join([
        "TAX INVOICE", "Northwind Office Supplies Pvt Ltd", "Invoice No.: NW-1  Dated: 12-Nov-2026",
        "Sl   Item &                      Quantity     Rate        Amount",
        "     Description",
        "1    Whiteboard markers, pack          10   100.00      1,000.00",
        "     of 4",
        "2    Copier paper                      2    50.00        100.00",
        "Page 1 of 1",
        "Taxable Value                                     1,100.00",
        "IGST @ 18%                                          198.00",
        "Total                                    ₹ 1,298.00",
    ])
    items = DemoExtractor().extract(text)["line_items"]
    assert [i["description"] for i in items] == ["Whiteboard markers, pack of 4", "Copier paper"]


def test_vertically_centred_description_keeps_the_words_above_its_numbers():
    # Zoho-style rows centre the text: "Cloud" sits above the numbers line and
    # "November" below it. Before, the first item read "hosting -".
    text = to_text("blind2_gst_line_tax.pdf", pdf("blind2_gst_line_tax.pdf"))
    items = DemoExtractor().extract(text)["line_items"]
    assert [i["description"] for i in items] == ["Cloud hosting - November", "Managed backups"]


# --- per-line discounts (a "Disc. %" column) ---------------------------------------


def test_line_discount_invoice_from_a_known_vendor_is_auto_approved(demo):
    # Before 2026-11-25 the discounted line (2 x 4,500.00 less 10% = 8,100.00) was
    # dropped because quantity x rate is not the amount, so this correct invoice
    # went to review: "line items add up to 2650.00 but the subtotal says 10750.00".
    run = upload(demo, "gst_line_discount.pdf", pdf("gst_line_discount.pdf")).json()
    items = run["invoice"]["line_items"]
    assert [(i["quantity"], i["unit_price"], i["discount"], i["amount"]) for i in items] == [
        (2.0, 4500.0, 900.0, 8100.0), (10.0, 265.0, 0.0, 2650.0)]
    assert run["status"] == "auto_approved"
    detail = next(r["detail"] for r in run["rules"] if r["name"] == "line_items_sum_to_subtotal")
    assert "after line discounts of 900.00" in detail


def _discount_table(header: str, row: str) -> str:
    return "\n".join([
        "TAX INVOICE", "Northwind Office Supplies Pvt Ltd", "Invoice No.: NW-1  Dated: 12-Nov-2026",
        header,
        row,
        "Copier paper              10 Nos      100.00                   1,000.00",
        "Taxable Value                                                  9,100.00",
        "Total                                                        ₹ 9,100.00",
    ])


@pytest.mark.parametrize("header, row, discount", [
    # a percentage column
    ("Description               Qty         Rate       Disc. %      Amount",
     "Office chair               2 Nos    4,500.00       10%       8,100.00", 900.0),
    ("Description               Qty         Rate       Disc %       Amount",
     "Office chair               2 Nos    4,500.00      10 %       8,100.00", 900.0),
    # an amount column
    ("Description               Qty         Rate       Discount     Amount",
     "Office chair               2         4,500.00      900.00     8,100.00", 900.0),
])
def test_a_discount_column_is_read(header, row, discount):
    items = DemoExtractor().extract(_discount_table(header, row))["line_items"]
    assert len(items) == 2
    assert (items[0]["quantity"], items[0]["unit_price"], items[0]["discount"], items[0]["amount"]) == (
        2.0, 4500.0, discount, 8100.0)


@pytest.mark.parametrize("header, row", [
    # no discount column in the header: a % cell is a tax rate or noise, not a discount
    ("Description               Qty         Rate       GST %        Amount",
     "Office chair               2 Nos    4,500.00       10%       8,100.00"),
    # the printed discount does not explain the amount (10% of 9,000 is 900, not 1,000)
    ("Description               Qty         Rate       Disc. %      Amount",
     "Office chair               2 Nos    4,500.00       10%       8,000.00"),
])
def test_a_discount_is_only_read_when_printed_and_it_adds_up(header, row):
    items = DemoExtractor().extract(_discount_table(header, row))["line_items"]
    assert [i["description"] for i in items] == ["Copier paper"]
