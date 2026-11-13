"""Generate the synthetic sample invoices in samples/.

Every company, tax id and number here is made up. Some invoices are broken on
purpose (see the `why` notes) so the demo and the eval exercise each path
through the graph. The expected answers live in evals/labels.json.

    pip install reportlab   # in requirements-dev.txt
    python scripts/make_samples.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples"

# Each sample: header lines, a table (description, qty, unit price, amount),
# footer lines. Rendered as .txt (column-aligned) or as a .pdf.
SAMPLES = [
    {
        "file": "01_northwind_ok.pdf",
        "why": "Clean INR invoice with GST and a matching PO: should auto-approve.",
        "header": [
            "TAX INVOICE",
            "Northwind Office Supplies Pvt Ltd",
            "14 Residency Road, Bengaluru 560025",
            "GSTIN: 29ABCDE1234F1Z5",
            "",
            "Invoice No: NW-2026-0142",
            "Invoice Date: 2026-10-02",
            "Due Date: 2026-11-01",
            "PO Number: PO-7781",
            "Currency: INR",
            "",
            "Bill To: Example Retail Co, 5 Lake View Road, Chennai",
        ],
        "columns": ("Description", "Qty", "Unit Price", "Amount"),
        "rows": [
            ("A4 copier paper, box of 5 reams", "10", "1,450.00", "14,500.00"),
            ("Laser toner cartridge, black", "2", "2,900.00", "5,800.00"),
        ],
        "footer": ["Subtotal: 20,300.00", "GST 18%: 3,654.00", "Total: 23,954.00"],
    },
    {
        "file": "02_bluepeak_ok.txt",
        "why": "US-style USD invoice, month-name dates, $ amounts, no PO: should auto-approve.",
        "header": [
            "INVOICE",
            "Bluepeak Cloud Services, Inc.",
            "200 Harbor Street, Suite 4, Portland, OR",
            "",
            "Invoice #: BP-10422",
            "Date: Oct 5, 2026",
            "Due: Nov 4, 2026",
            "",
            "Bill To: Example Retail Co",
        ],
        "columns": ("Description", "Qty", "Unit Price", "Amount"),
        "rows": [
            ("Managed compute, October", "1", "$1,250.00", "$1,250.00"),
            ("Object storage (TB)", "2", "$85.00", "$170.00"),
            ("Priority support", "1", "$300.00", "$300.00"),
        ],
        "footer": ["Subtotal: $1,720.00", "Sales Tax (8.25%): $141.90", "Total Due: $1,861.90"],
    },
    {
        "file": "03_kestrel_ok.pdf",
        "why": "German EUR invoice: dd.mm.yyyy dates, 1.234,56 number format, VAT id, PO: should auto-approve.",
        "header": [
            "Rechnung / Invoice",
            "Kestrel Logistics GmbH",
            "Hafenstrasse 12, 20457 Hamburg",
            "VAT ID: DE811234567",
            "",
            "Invoice No: KL-2026-5531",
            "Invoice Date: 14.10.2026",
            "Due Date: 13.11.2026",
            "Purchase Order: PO-5520",
            "Currency: EUR",
        ],
        "columns": ("Description", "Qty", "Unit Price", "Amount"),
        "rows": [
            ("Sea freight, Hamburg to Chennai", "1", "3.200,00", "3.200,00"),
            ("Port handling", "4", "150,00", "600,00"),
        ],
        "footer": ["Subtotal: 3.800,00", "VAT 19%: 722,00", "Total: 4.522,00"],
    },
    {
        "file": "04_saffron_wrong_total.pdf",
        "why": "Total is 1,000 too high (subtotal + tax = 44,100, printed 45,100): should go to review.",
        "header": [
            "TAX INVOICE",
            "Saffron Catering Co",
            "GSTIN: 27AAACS1234K1Z2",
            "",
            "Invoice No: SC/2026/0907",
            "Invoice Date: 2026-10-21",
            "Due Date: 2026-11-20",
            "Currency: INR",
        ],
        "columns": ("Description", "Qty", "Unit Price", "Amount"),
        "rows": [("Lunch buffet, quarterly offsite (per plate)", "120", "350.00", "42,000.00")],
        "footer": ["Subtotal: 42,000.00", "GST 5%: 2,100.00", "Total: 45,100.00"],
    },
    {
        "file": "05_bluepeak_duplicate.txt",
        "why": "BP-10388 was already paid (data/paid_invoices.json): should go to review as a duplicate.",
        "header": [
            "INVOICE",
            "Bluepeak Cloud Services, Inc.",
            "",
            "Invoice #: BP-10388",
            "Date: Sep 6, 2026",
            "Due: Oct 6, 2026",
        ],
        "columns": ("Description", "Qty", "Unit Price", "Amount"),
        "rows": [
            ("Managed compute, September", "1", "$1,250.00", "$1,250.00"),
            ("Object storage (TB)", "2", "$85.00", "$170.00"),
            ("Priority support", "1", "$300.00", "$300.00"),
        ],
        "footer": ["Subtotal: $1,720.00", "Sales Tax (8.25%): $141.90", "Total Due: $1,861.90"],
    },
    {
        "file": "06_unknown_vendor.txt",
        "why": "Quillmark Stationers is not in the vendor master: should go to review.",
        "header": [
            "TAX INVOICE",
            "Quillmark Stationers",
            "GSTIN: 33AAQFQ9876P1Z4",
            "",
            "Invoice No: QS-311",
            "Invoice Date: 2026-11-02",
            "Due Date: 2026-11-17",
            "Currency: INR",
        ],
        "columns": ("Description", "Qty", "Unit Price", "Amount"),
        "rows": [
            ("Whiteboard markers, pack of 12", "15", "240.00", "3,600.00"),
            ("Spiral notebooks", "50", "60.00", "3,000.00"),
        ],
        "footer": ["Subtotal: 6,600.00", "GST 18%: 1,188.00", "Total: 7,788.00"],
    },
    {
        "file": "07_missing_invoice_number.txt",
        "why": "No invoice number anywhere: extraction fails schema validation twice, then a human fills it in.",
        "header": [
            "TAX INVOICE",
            "Orbit Facility Management LLP",
            "GSTIN: 29AAFFO5678L1Z9",
            "",
            "Invoice Date: 2026-11-01",
            "Due Date: 2026-11-30",
            "Currency: INR",
        ],
        "columns": ("Description", "Qty", "Unit Price", "Amount"),
        "rows": [("Housekeeping services, October", "1", "75,000.00", "75,000.00")],
        "footer": ["Subtotal: 75,000.00", "GST 18%: 13,500.00", "Total: 88,500.00"],
    },
    {
        "file": "08_northwind_name_variant.pdf",
        "why": "Vendor printed as 'NORTHWIND OFFICE SUPPLY CO.': the vector match still finds V001; should auto-approve.",
        "header": [
            "TAX INVOICE",
            "NORTHWIND OFFICE SUPPLY CO.",
            "GSTIN: 29ABCDE1234F1Z5",
            "",
            "Invoice No: NW-2026-0151",
            "Invoice Date: 2026-10-28",
            "Due Date: 2026-11-27",
            "Currency: INR",
        ],
        "columns": ("Description", "Qty", "Unit Price", "Amount"),
        "rows": [
            ("Ergonomic office chair", "3", "6,200.00", "18,600.00"),
            ("Desk lamp, LED", "3", "1,100.00", "3,300.00"),
        ],
        "footer": ["Subtotal: 21,900.00", "GST 18%: 3,942.00", "Total: 25,842.00"],
    },
    {
        "file": "09_saffron_po_exceeded.txt",
        "why": "Invoice total 47,250 is above PO-9012's 40,000: should go to review.",
        "header": [
            "TAX INVOICE",
            "Saffron Catering Co",
            "GSTIN: 27AAACS1234K1Z2",
            "",
            "Invoice No: SC/2026/0951",
            "Invoice Date: 2026-11-04",
            "Due Date: 2026-12-04",
            "PO Number: PO-9012",
            "Currency: INR",
        ],
        "columns": ("Description", "Qty", "Unit Price", "Amount"),
        "rows": [("Diwali celebration dinner (per plate)", "150", "300.00", "45,000.00")],
        "footer": ["Subtotal: 45,000.00", "GST 5%: 2,250.00", "Total: 47,250.00"],
    },
    {
        "file": "10_tidewater_bad_dates.txt",
        "why": "Due date is before the invoice date: should go to review.",
        "header": [
            "INVOICE",
            "Tidewater Print & Signage Ltd",
            "VAT Reg No: GB123456789",
            "",
            "Invoice No: TPS-8812",
            "Invoice Date: 20/10/2026",
            "Due Date: 10/10/2026",
            "PO Number: PO-3300",
            "Currency: GBP",
        ],
        "columns": ("Description", "Qty", "Unit Price", "Amount"),
        "rows": [("Vinyl banners, 3m x 1m", "5", "290.00", "1,450.00")],
        "footer": ["Subtotal: 1,450.00", "VAT 20%: 290.00", "Total: 1,740.00"],
    },
]

WIDTHS = (46, 6, 14, 14)


def as_text(sample: dict) -> str:
    def row(cells):
        first, *rest = cells
        return first.ljust(WIDTHS[0]) + "".join(c.rjust(w) for c, w in zip(rest, WIDTHS[1:]))

    lines = list(sample["header"]) + ["", row(sample["columns"]), "-" * sum(WIDTHS)]
    lines += [row(r) for r in sample["rows"]]
    lines += ["-" * sum(WIDTHS)] + [f.rjust(sum(WIDTHS)) for f in sample["footer"]]
    lines += ["", "Thank you for your business."]
    return "\n".join(lines) + "\n"


def write_pdf(sample: dict, path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4, invariant=1)  # invariant: same bytes every run
    c.setTitle(f"Synthetic invoice {sample['file']}")
    width, height = A4
    y = height - 60
    for i, line in enumerate(sample["header"]):
        c.setFont("Helvetica-Bold", 16 if i == 0 else 12) if i <= 1 else c.setFont("Helvetica", 10)
        c.drawString(50, y, line)
        y -= 22 if i <= 1 else 15
    y -= 15
    x_desc, x_qty, x_unit, x_amt = 50, 330, 420, width - 50
    c.setFont("Helvetica-Bold", 10)
    cols = sample["columns"]
    c.drawString(x_desc, y, cols[0])
    c.drawRightString(x_qty, y, cols[1])
    c.drawRightString(x_unit, y, cols[2])
    c.drawRightString(x_amt, y, cols[3])
    y -= 6
    c.line(50, y, width - 50, y)
    y -= 15
    c.setFont("Helvetica", 10)
    for desc, qty, unit, amt in sample["rows"]:
        c.drawString(x_desc, y, desc)
        c.drawRightString(x_qty, y, qty)
        c.drawRightString(x_unit, y, unit)
        c.drawRightString(x_amt, y, amt)
        y -= 15
    c.line(50, y + 6, width - 50, y + 6)
    y -= 10
    for line in sample["footer"]:
        c.setFont("Helvetica-Bold" if line.lower().startswith("total") else "Helvetica", 10)
        c.drawRightString(x_amt, y, line)
        y -= 15
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 50, "Synthetic sample invoice for the invoice-agent demo. Not a real company.")
    c.save()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for sample in SAMPLES:
        path = OUT / sample["file"]
        if path.suffix == ".pdf":
            write_pdf(sample, path)
        else:
            path.write_text(as_text(sample), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}  ({sample['why']})")


if __name__ == "__main__":
    main()
