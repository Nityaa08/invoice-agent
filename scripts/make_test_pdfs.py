"""Generate the "real-world" test PDFs in tests/fixtures/pdfs/.

The bundled samples/ were written alongside the demo extractor, so they only
prove it reads its own layouts. These files imitate what people actually
upload: a GST tax invoice from an accounting package (bordered table, two
columns, CGST/SGST, lakh grouping, rupee sign), an HTML-to-PDF IGST bill, a
US invoice, an EU invoice, a three-page invoice, a services invoice whose line
items cannot be read as a table, and a scanned (image-only) PDF.

Every company, tax id and number is made up.

    pip install reportlab playwright pillow && playwright install chromium
    python scripts/make_test_pdfs.py
"""

from __future__ import annotations

import io
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures" / "pdfs"

# A Unicode font so the rupee and euro signs end up in the text layer, as they
# do in invoices from Tally, Zoho or a browser.
_DEJAVU = Path("/usr/share/fonts/truetype/dejavu")
if (_DEJAVU / "DejaVuSans.ttf").exists() and (_DEJAVU / "DejaVuSans-Bold.ttf").exists():
    pdfmetrics.registerFont(TTFont("DejaVu", str(_DEJAVU / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(_DEJAVU / "DejaVuSans-Bold.ttf")))
    FONT, FONT_BOLD = "DejaVu", "DejaVu-Bold"
else:  # the rupee sign will be missing from the text layer
    FONT, FONT_BOLD = "Helvetica", "Helvetica-Bold"


def _styles():
    ss = getSampleStyleSheet()
    for s in ss.byName.values():
        s.fontName = FONT
    return ss


GRID = TableStyle([
    ("FONT", (0, 0), (-1, -1), FONT, 8),
    ("FONT", (0, 0), (-1, 0), FONT_BOLD, 8),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
    ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
])


def gst_tax_invoice() -> bytes:
    """Indian GST tax invoice: two-column header, HSN table, CGST + SGST, lakh grouping."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=12 * mm)
    ss = _styles()
    p = lambda t, s="Normal": Paragraph(t, ss[s])  # noqa: E731
    seller = p("<b>Sri Lakshmi Traders Pvt. Ltd.</b><br/>No. 22, Gandhi Bazaar, Tiruchirappalli 620008<br/>"
               "GSTIN/UIN: 33AAKCS1234M1Z9<br/>State Name: Tamil Nadu, Code: 33")
    # Tally-style boxes: the label sits above its value, not beside it.
    meta = Table([
        ["Invoice No.", "Dated"],
        ["SLT/2026-27/0457", "14-Nov-2026"],
        ["Buyer's Order No.", "Mode/Terms of Payment"],
        ["PO-8812", "30 days"],
    ], colWidths=[42 * mm, 43 * mm], style=TableStyle([
        ("FONT", (0, 0), (-1, -1), FONT, 8), ("BOX", (0, 0), (0, 1), 0.5, colors.grey),
        ("BOX", (1, 0), (1, 1), 0.5, colors.grey), ("BOX", (0, 2), (0, 3), 0.5, colors.grey),
        ("BOX", (1, 2), (1, 3), 0.5, colors.grey)]))
    head = Table([[seller, meta]], colWidths=[95 * mm, 87 * mm])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.5, colors.grey)]))
    buyer = p("<b>Buyer (Bill to)</b><br/>Example Retail Co<br/>5 Lake View Road, Chennai 600034<br/>"
              "GSTIN/UIN: 33AABCE9876K1Z2")
    rows = [
        ["Sl No.", "Description of Goods", "HSN/SAC", "Quantity", "Rate", "per", "Amount"],
        ["1", "Steel almirah, 2 door", "9403", "12 Nos", "8,500.00", "Nos", "1,02,000.00"],
        ["2", "Office chair, mesh back", "9401", "6 Nos", "2,350.00", "Nos", "14,100.00"],
        ["", "", "", "", "", "Taxable Value", "1,16,100.00"],
        ["", "CGST @ 9%", "", "", "", "", "10,449.00"],
        ["", "SGST @ 9%", "", "", "", "", "10,449.00"],
        ["", "Grand Total", "", "18 Nos", "", "", "₹ 1,36,998.00"],
    ]
    table = Table(rows, colWidths=[12 * mm, 58 * mm, 20 * mm, 20 * mm, 22 * mm, 22 * mm, 28 * mm], style=GRID)
    doc.build([
        p("<b>TAX INVOICE</b>", "Title"), head, Spacer(1, 4 * mm), buyer, Spacer(1, 4 * mm), table,
        Spacer(1, 3 * mm),
        p("Amount Chargeable (in words): INR One Lakh Thirty Six Thousand Nine Hundred Ninety Eight Only"),
        p("Declaration: We declare that this invoice shows the actual price of the goods described."),
        p("for Sri Lakshmi Traders Pvt. Ltd. &nbsp; Authorised Signatory"),
    ])
    return buf.getvalue()


def _html_pdf(html: str, fmt: str = "A4") -> bytes:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.set_content(html)
        data = page.pdf(format=fmt, print_background=True, margin={"top": "12mm", "bottom": "12mm",
                                                                   "left": "12mm", "right": "12mm"})
        browser.close()
    return data


CSS = """<style>body{font-family:'DejaVu Sans',Arial,sans-serif;font-size:11px;color:#111}
table{border-collapse:collapse;width:100%} td,th{border:1px solid #bbb;padding:4px 6px;text-align:left}
th{background:#f0f0f0} .r{text-align:right} .cols{display:flex;justify-content:space-between}
.no td{border:none;padding:1px 4px}</style>"""


def gst_igst_html() -> bytes:
    """Interstate GST bill from a web app: 'Bill No', IGST, Rs., dates like 12 Nov 2026."""
    return _html_pdf(CSS + """
<h2 style="text-align:center">Tax Invoice</h2>
<div class="cols">
 <div><b>Kaveri Digital Services LLP</b><br>3rd Floor, Indiranagar, Bengaluru 560038<br>
 GSTIN: 29AAQFK4321L1ZP<br>PAN: AAQFK4321L</div>
 <table class="no" style="width:auto">
  <tr><td>Bill No</td><td>: KDS/1124</td></tr>
  <tr><td>Bill Date</td><td>: 12 Nov 2026</td></tr>
  <tr><td>Due Date</td><td>: 27 Nov 2026</td></tr>
  <tr><td>Place of Supply</td><td>: Tamil Nadu (33)</td></tr>
 </table>
</div>
<p><b>Bill To:</b> Example Retail Co, Chennai &middot; GSTIN 33AABCE9876K1Z2</p>
<table>
 <tr><th>#</th><th>Item &amp; Description</th><th>SAC</th><th class="r">Qty</th><th class="r">Rate (Rs.)</th><th class="r">Amount (Rs.)</th></tr>
 <tr><td>1</td><td>Website maintenance, Nov 2026</td><td>998314</td><td class="r">1</td><td class="r">45,000.00</td><td class="r">45,000.00</td></tr>
 <tr><td>2</td><td>Cloud hosting (annual)</td><td>998315</td><td class="r">1</td><td class="r">1,20,000.00</td><td class="r">1,20,000.00</td></tr>
 <tr><td>3</td><td>SEO audit</td><td>998361</td><td class="r">2</td><td class="r">12,500.00</td><td class="r">25,000.00</td></tr>
</table>
<table class="no" style="width:45%;margin-left:55%;margin-top:8px">
 <tr><td>Sub Total</td><td class="r">1,90,000.00</td></tr>
 <tr><td>IGST18 (18%)</td><td class="r">34,200.00</td></tr>
 <tr><td><b>Total</b></td><td class="r"><b>Rs.2,24,200.00</b></td></tr>
 <tr><td>Balance Due</td><td class="r"><b>Rs.2,24,200.00</b></td></tr>
</table>
<p>Total In Words: Indian Rupee Two Lakh Twenty-Four Thousand Two Hundred Only</p>
""")


def us_invoice() -> bytes:
    """US invoice: 'Inv #', MM/DD/YYYY, sales tax, 'Balance Due'."""
    return _html_pdf(CSS + """
<div class="cols">
 <div><h1 style="margin:0">Granite Peak Consulting LLC</h1>1450 Market St, Suite 300<br>Denver, CO 80202<br>EIN 84-1234567</div>
 <div style="text-align:right"><h1 style="margin:0;color:#555">INVOICE</h1>
  Inv #: GPC-2026-0311<br>Invoice Date: 11/03/2026<br>Terms: Net 30<br>Due Date: 12/03/2026<br>P.O. #: PO-7790</div>
</div>
<p><b>Bill To</b><br>Example Retail Co<br>5 Lake View Road</p>
<table>
 <tr><th>Item</th><th>Description</th><th class="r">Hours</th><th class="r">Rate</th><th class="r">Amount</th></tr>
 <tr><td>CONSULT</td><td>Inventory process review</td><td class="r">32</td><td class="r">$175.00</td><td class="r">$5,600.00</td></tr>
 <tr><td>TRAIN</td><td>Staff training, on site</td><td class="r">8</td><td class="r">$150.00</td><td class="r">$1,200.00</td></tr>
 <tr><td>TRAVEL</td><td>Travel expenses</td><td class="r">1</td><td class="r">$412.50</td><td class="r">$412.50</td></tr>
</table>
<table class="no" style="width:40%;margin-left:60%;margin-top:8px">
 <tr><td>Subtotal</td><td class="r">$7,212.50</td></tr>
 <tr><td>Sales Tax (2.9%)</td><td class="r">$34.80</td></tr>
 <tr><td>Total</td><td class="r">$7,247.30</td></tr>
 <tr><td>Payments/Credits</td><td class="r">$0.00</td></tr>
 <tr><td><b>Balance Due</b></td><td class="r"><b>$7,247.30</b></td></tr>
</table>
<p>Sales tax applies to training only (CO 2.9%). Thank you for your business!</p>
""", fmt="Letter")


def eu_invoice() -> bytes:
    """Dutch/EU invoice: 'Invoice number', 03.11.2026, VAT 21 %, euro sign, 1.234,56."""
    return _html_pdf(CSS + """
<div class="cols">
 <div><b>Van der Berg Logistiek B.V.</b><br>Keizersgracht 120<br>1015 CW Amsterdam<br>
 VAT ID: NL812345678B01<br>KvK 12345678</div>
 <div style="text-align:right"><h2 style="margin:0">Invoice</h2></div>
</div>
<table class="no" style="width:auto;margin:10px 0">
 <tr><td>Invoice number</td><td>2026-00873</td></tr>
 <tr><td>Invoice date</td><td>03.11.2026</td></tr>
 <tr><td>Due date</td><td>03.12.2026</td></tr>
 <tr><td>Customer</td><td>Example Retail Co</td></tr>
</table>
<table>
 <tr><th>Description</th><th class="r">Qty</th><th class="r">Unit price</th><th class="r">VAT</th><th class="r">Amount</th></tr>
 <tr><td>Pallet storage, October</td><td class="r">40</td><td class="r">€ 18,50</td><td class="r">21%</td><td class="r">€ 740,00</td></tr>
 <tr><td>Outbound handling</td><td class="r">1.250</td><td class="r">€ 1,20</td><td class="r">21%</td><td class="r">€ 1.500,00</td></tr>
 <tr><td>Customs documentation</td><td class="r">3</td><td class="r">€ 95,00</td><td class="r">21%</td><td class="r">€ 285,00</td></tr>
</table>
<table class="no" style="width:40%;margin-left:60%;margin-top:8px">
 <tr><td>Total excl. VAT</td><td class="r">€ 2.525,00</td></tr>
 <tr><td>VAT 21 %</td><td class="r">€ 530,25</td></tr>
 <tr><td><b>Amount due</b></td><td class="r"><b>€ 3.055,25</b></td></tr>
</table>
<p>Please pay within 30 days to IBAN NL91 ABNA 0417 1643 00 quoting the invoice number.</p>
""")


def multipage_invoice() -> bytes:
    """Three pages of line items with a repeated header; totals on the last page."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, topMargin=14 * mm, bottomMargin=14 * mm)
    ss = _styles()
    rows = [["#", "Description", "Qty", "Unit Price", "Line Total"]]
    subtotal = 0.0
    for i in range(1, 91):
        qty, price = (i % 5) + 1, 12.0 + (i % 7) * 3.25
        subtotal += qty * price
        rows.append([str(i), f"Replacement part SKU-{4000 + i}", str(qty), f"{price:,.2f}", f"{qty * price:,.2f}"])
    tax = round(subtotal * 0.0725, 2)
    table = Table(rows, colWidths=[12 * mm, 80 * mm, 15 * mm, 28 * mm, 30 * mm], repeatRows=1, style=GRID)

    def footer(c: canvas.Canvas, d) -> None:
        c.setFont(FONT, 8)
        c.drawRightString(LETTER[0] - 15 * mm, 8 * mm, f"Page {d.page}")

    p = lambda t: Paragraph(t, ss["Normal"])  # noqa: E731
    doc.build([
        p("<b>Harbor Industrial Supply Co.</b><br/>88 Dockside Ave, Oakland, CA 94607"),
        p("<b>INVOICE</b> &nbsp; Invoice No.: HIS-55021 &nbsp; Date: November 2, 2026 &nbsp; Due Date: December 2, 2026"),
        p("Currency: USD"),
        Spacer(1, 4 * mm), table, Spacer(1, 4 * mm),
        Table([["Subtotal", f"{subtotal:,.2f}"], ["Sales Tax 7.25%", f"{tax:,.2f}"],
               ["Amount Due (USD)", f"{subtotal + tax:,.2f}"]],
              colWidths=[40 * mm, 30 * mm], hAlign="RIGHT",
              style=TableStyle([("FONT", (0, 0), (-1, -1), FONT, 9), ("ALIGN", (1, 0), (1, -1), "RIGHT")])),
    ], onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


def services_invoice() -> bytes:
    """A freelancer's invoice: prose instead of a table, so only the total is readable."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont(FONT_BOLD, 16)
    c.drawString(20 * mm, 270 * mm, "Meera Iyer Design Studio")
    c.setFont(FONT, 10)
    y = 262
    for line in [
        "Freelance brand and packaging design · Chennai",
        "",
        "Inv # MI-0093",
        "Date of issue: 2026/11/10",
        "",
        "For the design of a new festive packaging range for Example Retail Co, including",
        "three rounds of revisions and print-ready files, as agreed in our email of 2 October.",
        "",
        "Fee for the above work: ₹85,000",
        "",
        "Total Amount Payable: ₹ 85,000.00",
        "",
        "Not registered under GST (turnover below the threshold).",
    ]:
        c.drawString(20 * mm, y * mm, line)
        y -= 7
    c.save()
    return buf.getvalue()


def scanned_invoice() -> bytes:
    """An image-only PDF, like a phone scan: no text layer at all."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("L", (620, 820), 245)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    y = 40
    for line in ["TAX INVOICE", "Annapurna Caterers", "Invoice No: AC-2211", "Date: 08/11/2026",
                 "", "Lunch buffet x 40      12,000.00", "CGST 2.5%                300.00",
                 "SGST 2.5%                300.00", "Grand Total          12,600.00"]:
        d.text((50, y), line, fill=20, font=font)
        y += 32
    png = io.BytesIO()
    img.save(png, format="PNG", optimize=True)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    from reportlab.lib.utils import ImageReader

    c.drawImage(ImageReader(io.BytesIO(png.getvalue())), 0, 0, width=A4[0], height=A4[1])
    c.save()
    return buf.getvalue()


# --- blind set ------------------------------------------------------------------
# Written after the demo extractor was frozen and run once without changes, so
# the eval on these says something about invoices the rules were not tuned on.


def blind_uk_invoice() -> bytes:
    return _html_pdf(CSS + """
<table class="no"><tr><td style="width:60%"><h2 style="margin:0">Thistle &amp; Pine Joinery Ltd</h2>
Unit 4, Riverside Works, Leeds LS10 1AB<br>VAT Registration No. GB 284 1937 62</td>
<td style="text-align:right"><h3 style="margin:0">SALES INVOICE</h3></td></tr></table>
<table class="no" style="width:auto;margin:10px 0">
<tr><td>Invoice Ref</td><td>TPJ-4471</td></tr>
<tr><td>Tax Point / Date</td><td>5th November 2026</td></tr>
<tr><td>Payment Terms</td><td>14 days</td></tr>
<tr><td>Your Order</td><td>PO-3300</td></tr></table>
<table><tr><th>Qty</th><th>Item</th><th class="r">Unit £</th><th class="r">Net £</th></tr>
<tr><td>6</td><td>Oak shelving unit, bespoke</td><td class="r">320.00</td><td class="r">1,920.00</td></tr>
<tr><td>2</td><td>Fitting (half day)</td><td class="r">95.00</td><td class="r">190.00</td></tr></table>
<table class="no" style="width:40%;margin-left:60%;margin-top:8px">
<tr><td>Net Total</td><td class="r">£2,110.00</td></tr>
<tr><td>VAT @ 20%</td><td class="r">£422.00</td></tr>
<tr><td><b>Invoice Total</b></td><td class="r"><b>£2,532.00</b></td></tr></table>
""")


def blind_gst_with_hsn_summary() -> bytes:
    return _html_pdf(CSS + """
<h3 style="text-align:center;margin:0">TAX INVOICE</h3><p style="text-align:center;margin:0">(Original for Recipient)</p>
<p><b>Aravind Agro Foods</b><br>Plot 17, SIPCOT, Hosur 635126<br>GSTIN : 33BXKPA7788Q1ZC</p>
<p>Invoice Date : 05/11/2026 &nbsp;&nbsp; Invoice No : AAF/26-27/219 &nbsp;&nbsp; Due Date : 20/11/2026</p>
<table><tr><th>S.No</th><th>Product Description</th><th>HSN</th><th class="r">Qty</th><th class="r">Rate</th><th class="r">Taxable Amt</th></tr>
<tr><td>1</td><td>Cold pressed groundnut oil 1L</td><td>1508</td><td class="r">120</td><td class="r">210.00</td><td class="r">25,200.00</td></tr>
<tr><td>2</td><td>Ragi flour 1kg</td><td>1102</td><td class="r">200</td><td class="r">68.00</td><td class="r">13,600.00</td></tr></table>
<table class="no" style="width:45%;margin-left:55%;margin-top:6px">
<tr><td>Total Taxable Value</td><td class="r">38,800.00</td></tr>
<tr><td>CGST</td><td class="r">970.00</td></tr><tr><td>SGST</td><td class="r">970.00</td></tr>
<tr><td><b>Total Invoice Value</b></td><td class="r"><b>40,740.00</b></td></tr></table>
<p>HSN-wise tax summary</p>
<table><tr><th>HSN</th><th class="r">Taxable Value</th><th class="r">CGST Rate</th><th class="r">CGST Amt</th><th class="r">SGST Rate</th><th class="r">SGST Amt</th></tr>
<tr><td>1508</td><td class="r">25,200.00</td><td class="r">2.5%</td><td class="r">630.00</td><td class="r">2.5%</td><td class="r">630.00</td></tr>
<tr><td>1102</td><td class="r">13,600.00</td><td class="r">2.5%</td><td class="r">340.00</td><td class="r">2.5%</td><td class="r">340.00</td></tr>
<tr><td>Total</td><td class="r">38,800.00</td><td></td><td class="r">970.00</td><td></td><td class="r">970.00</td></tr></table>
<p>Rupees Forty Thousand Seven Hundred Forty Only</p>
""")


def blind_canada_invoice() -> bytes:
    return _html_pdf(CSS + """
<div class="cols"><div><b>Northern Lights Print Shop Inc.</b><br>88 Queen St W, Toronto ON M5H 2M9<br>GST/HST No. 81234 5678 RT0001</div>
<div style="text-align:right">INVOICE<br>Invoice: NLP-90017<br>Date: 2026-11-09<br>Payment due: 2026-12-09</div></div>
<table style="margin-top:10px"><tr><th>Description</th><th class="r">Quantity</th><th class="r">Price</th><th class="r">Total</th></tr>
<tr><td>Business cards, 500 pack</td><td class="r">4</td><td class="r">CA$45.00</td><td class="r">CA$180.00</td></tr>
<tr><td>Vinyl banner 3x6 ft</td><td class="r">1</td><td class="r">CA$129.99</td><td class="r">CA$129.99</td></tr></table>
<table class="no" style="width:40%;margin-left:60%;margin-top:8px">
<tr><td>Subtotal</td><td class="r">CA$309.99</td></tr><tr><td>HST 13%</td><td class="r">CA$40.30</td></tr>
<tr><td><b>Amount Owing (CAD)</b></td><td class="r"><b>CA$350.29</b></td></tr></table>
""")


BLIND = {
    "blind_uk_invoice.pdf": blind_uk_invoice,
    "blind_gst_hsn_summary.pdf": blind_gst_with_hsn_summary,
    "blind_canada_invoice.pdf": blind_canada_invoice,
}

FILES = {
    "gst_tax_invoice.pdf": gst_tax_invoice,
    "gst_igst_bill.pdf": gst_igst_html,
    "us_invoice.pdf": us_invoice,
    "eu_invoice.pdf": eu_invoice,
    "multipage_invoice.pdf": multipage_invoice,
    "services_invoice.pdf": services_invoice,
    "scanned_invoice.pdf": scanned_invoice,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in FILES.items():
        data = make()
        (OUT / name).write_bytes(data)
        print(f"{name:28} {len(data) / 1024:7.1f} KB")
    for name, make in BLIND.items():
        data = make()
        (OUT / name).write_bytes(data)
        print(f"{name:28} {len(data) / 1024:7.1f} KB")
    # A file that says it is a PDF but is cut off half way, as from a failed download.
    good = (OUT / "us_invoice.pdf").read_bytes()
    (OUT / "truncated.pdf").write_bytes(good[: len(good) // 3])
    print(f"{'truncated.pdf':28} {len(good) // 3 / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
