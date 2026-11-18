"""Three-way match: purchase order + goods receipt + invoice, and the split-PO attack sample."""

from datetime import date

import pytest

from invoice_agent.config import load_settings
from invoice_agent.rules import (
    GoodsReceipt,
    PurchaseOrder,
    ReceiptLine,
    RuleContext,
    VendorMatch,
    matches_goods_receipt,
    run_rules,
)
from invoice_agent.schema import Invoice
from invoice_agent.service import InvoiceService, load_goods_receipts, load_purchase_orders

TODAY = date(2026, 11, 13)
KNOWN = VendorMatch("V001", "Northwind Office Supplies Pvt Ltd", 1.0, "29ABCDE1234F1Z5")
GOODS = PurchaseOrder("PO-1", "V001", "INR", 100000.0, requires_receipt=True)
SERVICES = PurchaseOrder("PO-2", "V001", "INR", 100000.0)
GRN = GoodsReceipt("GRN-1", "PO-1", "2026-11-01", (ReceiptLine("Ergonomic office chair", 8),
                                                    ReceiptLine("Monitor arm, single", 10)))
GRN2 = GoodsReceipt("GRN-2", "PO-1", "2026-11-05", (ReceiptLine("Ergonomic office chair", 2),))


def invoice(lines, po="PO-1"):
    items = [{"description": d, "quantity": q, "unit_price": 100, "amount": q * 100} for d, q in lines]
    sub = sum(i["amount"] for i in items)
    return Invoice.model_validate({
        "vendor_name": "Northwind Office Supplies Pvt Ltd", "invoice_number": "N-1", "invoice_date": "2026-11-09",
        "currency": "INR", "po_number": po, "line_items": items, "subtotal": sub, "tax": 0, "total": sub})


def ctx(receipts=(GRN,), pos=(GOODS, SERVICES)):
    return RuleContext(today=TODAY, vendor_match=KNOWN, purchase_orders={p.po_number: p for p in pos},
                       goods_receipts={"PO-1": tuple(receipts)} if receipts else {})


@pytest.mark.parametrize("lines, receipts, po, passed, words", [
    ([("Ergonomic office chair", 8)], (GRN,), "PO-1", True, "within the quantities received (GRN-1)"),
    ([("Office chair, ergonomic", 5), ("Monitor arm single", 10)], (GRN,), "PO-1", True, "2 lines"),  # wording differs
    ([("Ergonomic office chair", 10)], (GRN,), "PO-1", False, "billed 10 x 'Ergonomic office chair' but only 8 received"),
    ([("Ergonomic office chair", 10)], (GRN, GRN2), "PO-1", True, "GRN-1, GRN-2"),  # two partial deliveries add up
    # The same item on two lines cannot add up to more than arrived.
    ([("Ergonomic office chair", 5), ("Ergonomic office chair (black)", 5)], (GRN,), "PO-1", False, "billed 10"),
    ([("Standing desk", 1)], (GRN,), "PO-1", False, "'Standing desk' is not on any goods receipt"),
    ([("Ergonomic office chair", 1)], (), "PO-1", False, "no goods receipt has been recorded"),
    ([("Consulting", 5)], (), "PO-2", True, "for services"),
    ([("Anything", 1)], (GRN,), None, True, "no PO on the invoice"),
    ([("Anything", 1)], (GRN,), "PO-404", True, "PO not found"),
])
def test_matches_goods_receipt(lines, receipts, po, passed, words):
    r = matches_goods_receipt(invoice(lines, po), ctx(receipts))
    assert r.passed is passed and words in r.detail, r.detail


def test_rule_runs_with_the_others():
    names = [r.name for r in run_rules(invoice([("Ergonomic office chair", 8)]), ctx())]
    assert names.index("matches_goods_receipt") == names.index("matches_purchase_order") + 1


def test_sample_data_is_consistent():
    pos, grns = load_purchase_orders(), load_goods_receipts()
    assert set(grns) <= set(pos)  # every receipt belongs to a known PO
    assert all(pos[k].requires_receipt for k in grns)
    assert grns["PO-7781"][0].grn_number == "GRN-4410"


@pytest.fixture
def svc(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("AS_OF_DATE", TODAY.isoformat())
    return InvoiceService.ephemeral(load_settings())


def test_short_delivery_sample_stops_and_a_corrected_quantity_passes(svc, sample_bytes):
    run, _ = svc.submit("12.txt", sample_bytes("12_northwind_short_delivery.txt"))
    failed = [r for r in run["rules"] if not r["passed"]]
    assert [r["name"] for r in failed] == ["matches_goods_receipt"]
    assert "only 8 received" in failed[0]["detail"]
    # The vendor sends a corrected invoice for the 8 chairs that arrived: the reviewer corrects and it passes.
    fixed = dict(run["invoice"], line_items=[dict(run["invoice"]["line_items"][0], quantity=8, amount=52000)],
                 subtotal=52000, tax=9360, total=61360)
    done = svc.review(run["id"], {"action": "correct", "invoice": fixed, "reviewer": "asha"})
    assert done["status"] == "approved_after_correction"


def test_clean_po_invoice_matches_its_receipt(svc, sample_bytes):
    run, _ = svc.submit("01.pdf", sample_bytes("01_northwind_ok.pdf"))
    grn = next(r for r in run["rules"] if r["name"] == "matches_goods_receipt")
    assert run["status"] == "auto_approved" and grn["passed"] and "GRN-4410" in grn["detail"]


def test_split_po_sample_is_caught_by_the_paid_ledger(svc, sample_bytes):
    """PO-6610 is for SGD 10,000 and 6,000 was already paid against it: a 5,450 'second half' does not fit."""
    run, _ = svc.submit("13.txt", sample_bytes("13_lumen_split_po.txt"))
    po = next(r for r in run["rules"] if r["name"] == "matches_purchase_order")
    assert run["status"] == "needs_review" and not po["passed"]
    assert "6000.00 already paid or approved = 4000.00" in po["detail"]
