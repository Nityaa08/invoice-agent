"""Ingest (file -> text) and the demo extractor (text -> fields)."""

import pytest

from invoice_agent.extractors import DemoExtractor, build_extractor, parse_amount, parse_date
from invoice_agent.ingest import MAX_UPLOAD_BYTES, IngestError, sniff_kind, to_text
from invoice_agent.schema import Invoice


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1,450.00", 1450.0),
        ("$1,861.90", 1861.9),
        ("3.200,00", 3200.0),
        ("722,00", 722.0),
        ("₹ 23,954", 23954.0),
        ("1.234.567", 1234567.0),
        ("10", 10.0),
        ("n/a", None),
    ],
)
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2026-10-02", "2026-10-02"),
        ("14.10.2026", "2026-10-14"),
        ("20/10/2026", "2026-10-20"),
        ("Oct 5, 2026", "2026-10-05"),
        ("5 October 2026", "2026-10-05"),
        ("next Tuesday", None),
    ],
)
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


def test_pdf_ingest_keeps_table_columns(sample_bytes):
    text = to_text("01_northwind_ok.pdf", sample_bytes("01_northwind_ok.pdf"))
    assert "Invoice No: NW-2026-0142" in text
    assert "\n\n\n" not in text  # blank padding collapsed
    row = next(line for line in text.splitlines() if line.startswith("A4 copier paper"))
    assert "  " in row.split("reams", 1)[1]  # columns still separated by runs of spaces


def test_demo_extractor_reads_a_pdf(sample_bytes):
    data = DemoExtractor().extract(to_text("x.pdf", sample_bytes("03_kestrel_ok.pdf")))
    invoice = Invoice.model_validate(data)
    assert invoice.vendor_name == "Kestrel Logistics GmbH"
    assert invoice.vendor_tax_id == "DE811234567"
    assert invoice.currency == "EUR"
    assert invoice.po_number == "PO-5520"
    assert (invoice.subtotal, invoice.tax, invoice.total) == (3800.0, 722.0, 4522.0)
    assert [i.amount for i in invoice.line_items] == [3200.0, 600.0]


def test_demo_extractor_ignores_header_ids_when_reading_tax(sample_bytes):
    # "VAT Reg No: GB123456789" sits in the header; the tax is the VAT line in the totals.
    data = DemoExtractor().extract(to_text("x.txt", sample_bytes("10_tidewater_bad_dates.txt")))
    assert data["tax"] == 290.0
    assert data["vendor_tax_id"] == "GB123456789"


def test_demo_extractor_leaves_missing_fields_out(sample_bytes):
    data = DemoExtractor().extract(to_text("x.txt", sample_bytes("07_missing_invoice_number.txt")))
    assert "invoice_number" not in data  # so schema validation fails, as it should


@pytest.mark.parametrize(
    "name, data, kind",
    [
        ("a.pdf", b"%PDF-1.7 ...", "pdf"),
        ("a.bin", b"\x89PNG\r\n", "image"),
        ("a.jpg", b"\xff\xd8\xff\xe0", "image"),
        ("a.txt", b"hello", "text"),
        ("a.bin", b"\xff\xfe\x00\x81", "unknown"),
    ],
)
def test_sniff_kind_uses_bytes_first(name, data, kind):
    assert sniff_kind(name, data) == kind


@pytest.mark.parametrize(
    "name, data, message",
    [
        ("a.txt", b"", "empty"),
        ("a.txt", b"x" * (MAX_UPLOAD_BYTES + 1), "larger"),
        ("a.bin", b"\xff\xfe\x00\x81", "unsupported"),
        ("a.pdf", b"%PDF-1.4 not really a pdf", "could not be read as a PDF"),
        ("a.txt", b"   \n  ", "no text"),
    ],
)
def test_ingest_errors_are_readable(name, data, message):
    with pytest.raises(IngestError, match=message):
        to_text(name, data)


def test_build_extractor():
    assert build_extractor("demo").name == "demo-rules"
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        build_extractor("llm", model="m")
    with pytest.raises(ValueError, match="unknown"):
        build_extractor("magic")
