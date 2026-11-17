"""Generate the QA invoices in tests/fixtures/qa/ (written 2026-11-17).

A third set, written for the end-to-end QA pass and run once against the
extractor before any fix (see tests/fixtures/qa/labels.json for what that
first run found). Unlike the other fixture sets, most of these come from
vendors in the vendor master, so a correct read should be *auto-approved*;
that tests the happy path on layouts the extractor was not written for.

    GST tax invoice, CGST + SGST     (Northwind, PO-7781)
    GST bill, IGST, lakh grouping    (Harbor Legal)
    US invoice, MM/DD/YYYY           (Bluepeak)
    UK VAT invoice, £                (Tidewater, PO-3300)
    three-page invoice, 36 lines     (Kestrel, PO-5520)
    invoice with a discount line     (Saffron)
    amounts as "4 800.00 SGD"        (Lumen Analytics)
    plain-text invoice               (Orbit)
    PNG and JPEG photos of an invoice (no OCR in the demo: clear message, review)

Every company, tax id and number is made up or comes from the made-up vendor master.

    pip install reportlab playwright pillow && playwright install chromium
    python scripts/make_qa_invoices.py
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "qa"
sys.path.insert(0, str(Path(__file__).resolve().parent))

CSS = """<style>body{font-family:'DejaVu Sans',Arial,sans-serif;font-size:11px;color:#111}
table{border-collapse:collapse;width:100%} td,th{border:1px solid #bbb;padding:4px 6px;text-align:left}
th{background:#f0f0f0} .r{text-align:right} .no td{border:none;padding:1px 4px}
.pb{page-break-after:always}</style>"""


def _html_pdf(html: str, fmt: str = "A4") -> bytes:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        exe = os.getenv("CHROMIUM_PATH")  # a system Chromium, if `playwright install` was not run
        browser = pw.chromium.launch(**({"executable_path": exe} if exe else {}))
        page = browser.new_page()
        page.set_content(html)
        data = page.pdf(format=fmt, print_background=True, margin={"top": "12mm", "bottom": "12mm",
                                                                   "left": "12mm", "right": "12mm"})
        browser.close()
    return data


def gst_cgst_sgst() -> bytes:
    return _html_pdf(CSS + """
<h3 style="text-align:center;margin:0">TAX INVOICE</h3>
<table class="no"><tr><td style="width:60%"><b>Northwind Office Supplies Pvt Ltd</b><br>
14 Residency Road, Bengaluru 560025<br>GSTIN: 29ABCDE1234F1Z5 &nbsp; State: Karnataka (29)</td>
<td>Invoice No: NW-2026-0187<br>Invoice Date: 10/11/2026<br>Due Date: 10/12/2026<br>PO No: PO-7781</td></tr></table>
<p>Bill to: Example Retail Co, GSTIN 29AAECE1111A1Z1</p>
<table><tr><th>#</th><th>Description</th><th>HSN</th><th class="r">Qty</th><th class="r">Rate</th><th class="r">Amount</th></tr>
<tr><td>1</td><td>A4 copier paper, 500 sheets</td><td>4802</td><td class="r">20</td><td class="r">245.00</td><td class="r">4,900.00</td></tr>
<tr><td>2</td><td>Laser toner cartridge</td><td>8443</td><td class="r">4</td><td class="r">3,150.00</td><td class="r">12,600.00</td></tr></table>
<table class="no" style="width:45%;margin-left:55%;margin-top:6px">
<tr><td>Taxable Value</td><td class="r">17,500.00</td></tr>
<tr><td>CGST @ 9%</td><td class="r">1,575.00</td></tr>
<tr><td>SGST @ 9%</td><td class="r">1,575.00</td></tr>
<tr><td><b>Grand Total</b></td><td class="r"><b>₹ 20,650.00</b></td></tr></table>
<p>Amount in words: Rupees Twenty Thousand Six Hundred Fifty Only</p>
""")


def gst_igst() -> bytes:
    return _html_pdf(CSS + """
<table class="no"><tr><td><h2 style="margin:0">Harbor Legal Associates LLP</h2>
Nariman Point, Mumbai 400021 · GSTIN 27AAGFH4321M1Z3</td><td class="r"><h3>TAX INVOICE</h3></td></tr></table>
<table class="no" style="width:auto;margin:8px 0">
<tr><td>Invoice No.</td><td>HLA/26-27/088</td></tr><tr><td>Date</td><td>05-Nov-2026</td></tr>
<tr><td>Due Date</td><td>05-Dec-2026</td></tr><tr><td>Place of supply</td><td>Karnataka (29)</td></tr></table>
<table><tr><th>Particulars</th><th>SAC</th><th class="r">Qty</th><th class="r">Rate</th><th class="r">Amount (Rs.)</th></tr>
<tr><td>Contract review and advisory, October 2026</td><td>998212</td><td class="r">1</td><td class="r">1,20,000.00</td><td class="r">1,20,000.00</td></tr></table>
<table class="no" style="width:45%;margin-left:55%;margin-top:6px">
<tr><td>Sub Total</td><td class="r">1,20,000.00</td></tr>
<tr><td>IGST 18%</td><td class="r">21,600.00</td></tr>
<tr><td><b>Total</b></td><td class="r"><b>Rs. 1,41,600.00</b></td></tr></table>
""")


def us_invoice() -> bytes:
    return _html_pdf(CSS + """
<table class="no"><tr><td><h2 style="margin:0">Bluepeak Cloud Services, Inc.</h2>
200 Harbor Street, Suite 4, Portland, OR 97204</td><td class="r"><h1 style="margin:0">INVOICE</h1></td></tr></table>
<table class="no" style="width:auto;margin:8px 0">
<tr><td>Invoice #</td><td>BP-10501</td></tr><tr><td>Invoice Date</td><td>11/02/2026</td></tr>
<tr><td>Due Date</td><td>12/02/2026</td></tr></table>
<table><tr><th>Description</th><th class="r">Qty</th><th class="r">Unit Price</th><th class="r">Amount</th></tr>
<tr><td>Managed compute, November</td><td class="r">1</td><td class="r">$1,250.00</td><td class="r">$1,250.00</td></tr>
<tr><td>Object storage (TB)</td><td class="r">3</td><td class="r">$85.00</td><td class="r">$255.00</td></tr>
<tr><td>Support hours</td><td class="r">2.5</td><td class="r">$120.00</td><td class="r">$300.00</td></tr></table>
<table class="no" style="width:40%;margin-left:60%;margin-top:6px">
<tr><td>Subtotal</td><td class="r">$1,805.00</td></tr>
<tr><td>Sales Tax (8.25%)</td><td class="r">$148.91</td></tr>
<tr><td><b>Total Due</b></td><td class="r"><b>$1,953.91</b></td></tr></table>
""", fmt="Letter")


def uk_vat() -> bytes:
    return _html_pdf(CSS + """
<table class="no"><tr><td><h2 style="margin:0">Tidewater Print &amp; Signage Ltd</h2>
3 Quay Street, Bristol BS1 4DB<br>VAT Reg No: GB 123 4567 89</td><td class="r"><h3>VAT INVOICE</h3></td></tr></table>
<table class="no" style="width:auto;margin:8px 0">
<tr><td>Invoice No</td><td>TPS-8890</td></tr><tr><td>Invoice Date</td><td>06/11/2026</td></tr>
<tr><td>Payment due</td><td>06/12/2026</td></tr><tr><td>Purchase Order</td><td>PO-3300</td></tr></table>
<table><tr><th>Description</th><th class="r">Qty</th><th class="r">Unit £</th><th class="r">Net £</th></tr>
<tr><td>Window graphics, printed and fitted</td><td class="r">4</td><td class="r">300.00</td><td class="r">1,200.00</td></tr>
<tr><td>Design proofs</td><td class="r">1</td><td class="r">300.00</td><td class="r">300.00</td></tr></table>
<table class="no" style="width:40%;margin-left:60%;margin-top:6px">
<tr><td>Net Total</td><td class="r">£1,500.00</td></tr>
<tr><td>VAT @ 20%</td><td class="r">£300.00</td></tr>
<tr><td><b>Total</b></td><td class="r"><b>£1,800.00</b></td></tr></table>
""")


def multipage() -> bytes:
    rows, subtotal = [], 0.0
    for i in range(1, 37):
        qty, unit = (i % 4) + 1, 25.0
        amount = qty * unit
        subtotal += amount
        rows.append(f"<tr><td>{i}</td><td>Pallet transfer, lane {i:02d}</td><td class='r'>{qty}</td>"
                    f"<td class='r'>{unit:,.2f}</td><td class='r'>{amount:,.2f}</td></tr>")
    head = ("<table><tr><th>No</th><th>Description</th><th class='r'>Qty</th>"
            "<th class='r'>Unit Price</th><th class='r'>Amount</th></tr>")
    pages = [rows[:14], rows[14:28], rows[28:]]
    body = ""
    for n, chunk in enumerate(pages, start=1):
        body += head + "".join(chunk) + "</table>"
        body += f"<p class='r'>Page {n} of 3</p>"
        if n < 3:
            body += "<p>Continued on next page</p><div class='pb'></div>"
    vat = round(subtotal * 0.19, 2)
    assert subtotal == 2250.0 and vat == 427.5
    return _html_pdf(CSS + f"""
<h2 style="margin:0">Kestrel Logistics GmbH</h2><p>Hafenstrasse 12, 20457 Hamburg · VAT ID DE811234567</p>
<table class="no" style="width:auto;margin:8px 0">
<tr><td>Invoice Number</td><td>KL-2026-5602</td></tr><tr><td>Invoice Date</td><td>2026-11-09</td></tr>
<tr><td>Due Date</td><td>2026-12-09</td></tr><tr><td>PO Number</td><td>PO-5520</td></tr>
<tr><td>Currency</td><td>EUR</td></tr></table>
{body}
<table class="no" style="width:40%;margin-left:60%;margin-top:6px">
<tr><td>Subtotal</td><td class="r">{subtotal:,.2f}</td></tr>
<tr><td>VAT 19%</td><td class="r">{vat:,.2f}</td></tr>
<tr><td><b>Total</b></td><td class="r"><b>{subtotal + vat:,.2f}</b></td></tr></table>
""")


def discount() -> bytes:
    return _html_pdf(CSS + """
<h2 style="margin:0">Saffron Catering Co</h2><p>Andheri East, Mumbai · GSTIN 27AAACS1234K1Z2</p>
<table class="no" style="width:auto;margin:8px 0">
<tr><td>Invoice No</td><td>SC/2026/0988</td></tr><tr><td>Invoice Date</td><td>12/11/2026</td></tr>
<tr><td>Due Date</td><td>27/11/2026</td></tr></table>
<table><tr><th>Description</th><th class="r">Qty</th><th class="r">Rate</th><th class="r">Amount</th></tr>
<tr><td>Team lunch buffet</td><td class="r">60</td><td class="r">400.00</td><td class="r">24,000.00</td></tr>
<tr><td>Evening snacks</td><td class="r">60</td><td class="r">100.00</td><td class="r">6,000.00</td></tr></table>
<table class="no" style="width:45%;margin-left:55%;margin-top:6px">
<tr><td>Subtotal</td><td class="r">30,000.00</td></tr>
<tr><td>Less: Discount (10%)</td><td class="r">-3,000.00</td></tr>
<tr><td>Taxable Value</td><td class="r">27,000.00</td></tr>
<tr><td>GST 5%</td><td class="r">1,350.00</td></tr>
<tr><td><b>Total</b></td><td class="r"><b>₹28,350.00</b></td></tr></table>
""")


def currency_layout() -> bytes:
    return _html_pdf(CSS + """
<h2 style="margin:0">Lumen Analytics Pte Ltd</h2><p>1 Fusionopolis Way, Singapore 138632</p>
<table class="no" style="width:auto;margin:8px 0">
<tr><td>Invoice No</td><td>LA-2026-118</td></tr><tr><td>Issue Date</td><td>3 November 2026</td></tr>
<tr><td>Due Date</td><td>3 December 2026</td></tr></table>
<table><tr><th>Description</th><th class="r">Qty</th><th class="r">Unit Price</th><th class="r">Amount</th></tr>
<tr><td>Dashboard build, fixed fee</td><td class="r">1</td><td class="r">4 800.00 SGD</td><td class="r">4 800.00 SGD</td></tr>
<tr><td>Data pipeline hours</td><td class="r">12</td><td class="r">150.00 SGD</td><td class="r">1 800.00 SGD</td></tr></table>
<table class="no" style="width:45%;margin-left:55%;margin-top:6px">
<tr><td>Subtotal</td><td class="r">6 600.00 SGD</td></tr>
<tr><td>GST 9%</td><td class="r">594.00 SGD</td></tr>
<tr><td><b>Total</b></td><td class="r"><b>7 194.00 SGD</b></td></tr></table>
""")


TEXT_INVOICE = """Orbit Facility Management LLP
7th Floor, Prestige Tower, Bengaluru 560001
GSTIN: 29AAFFO5678L1Z9

TAX INVOICE
Invoice No: OFM/26/344
Invoice Date: 01-11-2026
Due Date: 30-11-2026

Description                               Qty      Rate          Amount
Housekeeping, November                      1   60,000.00     60,000.00
Pest control visit                          2    2,500.00      5,000.00

                                              Sub Total:      65,000.00
                                              CGST 9%:         5,850.00
                                              SGST 9%:         5,850.00
                                              Grand Total:    76,700.00
"""


def _image(fmt: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (700, 520), (250, 250, 248))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    y = 30
    for line in ["INVOICE", "Bluepeak Cloud Services, Inc.", "Invoice #: BP-10533", "Date: Nov 6, 2026",
                 "", "Managed compute       1   $1,250.00", "Subtotal                 $1,250.00",
                 "Sales Tax                  $103.13", "Total Due                $1,353.13"]:
        d.text((40, y), line, fill=(20, 20, 20), font=font)
        y += 44
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


FILES = {
    "qa_gst_cgst_sgst.pdf": gst_cgst_sgst,
    "qa_gst_igst.pdf": gst_igst,
    "qa_us_invoice.pdf": us_invoice,
    "qa_uk_vat.pdf": uk_vat,
    "qa_multipage.pdf": multipage,
    "qa_discount.pdf": discount,
    "qa_currency_after_amount.pdf": currency_layout,
    "qa_text_invoice.txt": lambda: TEXT_INVOICE.encode(),
    "qa_photo.png": lambda: _image("PNG"),
    "qa_photo.jpg": lambda: _image("JPEG"),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in FILES.items():
        data = make()
        (OUT / name).write_bytes(data)
        print(f"{name:32} {len(data) / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
