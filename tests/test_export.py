"""Batch export: rows, CSV formula injection, and the full service's export endpoint."""

import csv
import io
import json
import re

import pytest

from invoice_agent.export import COLUMNS, export_row, safe_cell, to_csv

from .conftest import ROOT
from .test_api import make_client


@pytest.fixture
def full(monkeypatch, tmp_path):
    return make_client(monkeypatch, tmp_path, demo=False)


def test_the_page_and_the_server_export_the_same_columns():
    html = (ROOT / "invoice_agent" / "web" / "index.html").read_text()
    block = re.search(r"const EXPORT_COLUMNS = \[(.*?)\];", html, re.S).group(1)
    assert tuple(re.findall(r'"([a-z_]+)"', block)) == COLUMNS


@pytest.mark.parametrize("value, cell", [
    ("=HYPERLINK(\"http://evil\",\"x\")", "'=HYPERLINK(\"http://evil\",\"x\")"),
    ("+1-555", "'+1-555"), ("-2+3", "'-2+3"), ("@SUM(A1)", "'@SUM(A1)"), ("\tx", "'\tx"),
    ("Northwind", "Northwind"), (None, ""), (-12.5, -12.5), (0, 0),
])
def test_csv_cells_cannot_become_formulas(value, cell):
    assert safe_cell(value) == cell


def test_full_service_lists_and_exports_runs(full):
    for sample in ("01_northwind_ok.pdf", "04_saffron_wrong_total.pdf", "06_unknown_vendor.txt"):
        full.post("/invoices", data={"sample": sample})
    wrong = next(r for r in full.get("/api/runs").json() if r["filename"] == "04_saffron_wrong_total.pdf")
    assert wrong["status"] == "needs_review" and wrong["failed_rules"] == ["subtotal_plus_tax_equals_total"]
    full.post(f"/invoices/{wrong['id']}/review", json={"action": "reject", "reviewer": "asha", "note": "=cmd|' /C calc'!A0"})

    r = full.get("/api/runs/export")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert r.headers["cache-control"] == "no-store" and "attachment" in r.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert tuple(rows[0]) == COLUMNS and len(rows) == 3
    by_file = {row["file"]: row for row in rows}
    assert by_file["01_northwind_ok.pdf"]["decision"] == "approved"
    assert by_file["01_northwind_ok.pdf"]["vendor_id"] == "V001" and by_file["01_northwind_ok.pdf"]["total"] == "23954.0"
    assert by_file["04_saffron_wrong_total.pdf"]["decision"] == "rejected"
    assert by_file["04_saffron_wrong_total.pdf"]["reviewer_action"] == "reject"
    assert by_file["04_saffron_wrong_total.pdf"]["review_note"].startswith("'=")  # neutralised
    assert by_file["06_unknown_vendor.txt"]["decision"] == "review"
    assert by_file["06_unknown_vendor.txt"]["failed_rules"] == "vendor_is_known"
    assert by_file["06_unknown_vendor.txt"]["vendor_id"] == ""  # not matched: no id

    as_json = full.get("/api/runs/export?format=json").json()
    assert len(as_json) == 3 and set(as_json[0]) == set(COLUMNS)
    assert full.get("/api/runs/export?format=xml").status_code == 422


def test_export_of_an_attack_file_keeps_html_as_text():
    text = (ROOT / "tests" / "fixtures" / "redteam" / "xss_in_fields.txt").read_bytes()
    from invoice_agent.config import load_settings
    from invoice_agent.service import InvoiceService

    svc = InvoiceService.ephemeral(load_settings())
    run, _ = svc.submit("xss.txt", text)
    out = to_csv([export_row(run)])
    assert "<" in out  # kept as data; escaping is the viewer's job (the page escapes, a CSV is not HTML)
    assert json.loads(json.dumps(export_row(run)))["file"] == "xss.txt"


def test_extraction_failure_is_explicit_in_the_row():
    row = export_row({"id": "x", "filename": "f.txt", "status": "needs_review", "invoice": None, "rules": []})
    assert row["failed_rules"] == "extraction_failed" and row["decision"] == "review"
