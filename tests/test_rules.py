"""Table-driven tests for the validation rules. Each row: a change to a good
invoice (or context), and whether the rule should pass."""

from datetime import date

import pytest
from pydantic import ValidationError

from invoice_agent import rules
from invoice_agent.rules import PurchaseOrder, RuleContext, VendorMatch
from invoice_agent.schema import Invoice

TODAY = date(2026, 11, 13)
KNOWN = VendorMatch("V001", "Northwind Office Supplies Pvt Ltd", 0.97)
UNKNOWN = VendorMatch("V004", "Kestrel Logistics GmbH", 0.31)
PO = PurchaseOrder("PO-7781", "V001", "INR", 30000.0)


def good_invoice(**changes) -> Invoice:
    data = {
        "vendor_name": "Northwind Office Supplies Pvt Ltd",
        "vendor_tax_id": "29ABCDE1234F1Z5",
        "invoice_number": "NW-2026-0142",
        "invoice_date": "2026-10-02",
        "due_date": "2026-11-01",
        "currency": "INR",
        "po_number": "PO-7781",
        "line_items": [
            {"description": "Copier paper", "quantity": 10, "unit_price": 1450.0, "amount": 14500.0},
            {"description": "Toner", "quantity": 2, "unit_price": 2900.0, "amount": 5800.0},
        ],
        "subtotal": 20300.0,
        "tax": 3654.0,
        "total": 23954.0,
    }
    data.update(changes)
    return Invoice.model_validate(data)


def ctx(**changes) -> RuleContext:
    base = {"today": TODAY, "vendor_match": KNOWN, "purchase_orders": {"PO-7781": PO}}
    base.update(changes)
    return RuleContext(**base)


def test_good_invoice_passes_every_rule():
    results = rules.run_rules(good_invoice(), ctx())
    assert [r.name for r in results] == [r.__name__ for r in rules.RULES]
    assert all(r.passed for r in results), [r for r in results if not r.passed]


@pytest.mark.parametrize(
    "rule, invoice_changes, ctx_changes, should_pass",
    [
        # line items -> subtotal
        ("line_items_sum_to_subtotal", {}, {}, True),
        ("line_items_sum_to_subtotal", {"subtotal": 20300.01}, {}, True),  # within tolerance
        ("line_items_sum_to_subtotal", {"subtotal": 20000.0}, {}, False),
        (
            "line_items_sum_to_subtotal",
            {"line_items": [{"description": "Paper", "quantity": 10, "unit_price": 1450.0, "amount": 20300.0}]},
            {},
            False,  # qty x price != amount
        ),
        # subtotal + tax -> total
        ("subtotal_plus_tax_equals_total", {}, {}, True),
        ("subtotal_plus_tax_equals_total", {"total": 23954.02}, {}, True),
        ("subtotal_plus_tax_equals_total", {"total": 24954.0}, {}, False),
        ("subtotal_plus_tax_equals_total", {"tax": 0.0}, {}, False),
        # dates
        ("dates_are_sane", {}, {}, True),
        ("dates_are_sane", {"due_date": None}, {}, True),
        ("dates_are_sane", {"invoice_date": "2026-12-01", "due_date": None}, {}, False),  # future
        ("dates_are_sane", {"invoice_date": "2025-01-01", "due_date": None}, {}, False),  # too old
        ("dates_are_sane", {"due_date": "2026-09-30"}, {}, False),  # due before issue
        ("dates_are_sane", {"due_date": "2027-03-01"}, {}, False),  # 150-day terms
        # currency
        ("currency_is_iso4217", {}, {}, True),
        ("currency_is_iso4217", {"currency": "eur"}, {}, True),
        ("currency_is_iso4217", {"currency": "RSX"}, {}, False),
        ("currency_is_iso4217", {"currency": "BTC"}, {}, False),
        # vendor
        ("vendor_is_known", {}, {}, True),
        ("vendor_is_known", {}, {"vendor_match": UNKNOWN}, False),
        ("vendor_is_known", {}, {"vendor_match": VendorMatch(None, None, 0.0)}, False),
        # duplicates (case and spaces in the number do not hide one)
        ("not_a_duplicate", {}, {}, True),
        ("not_a_duplicate", {}, {"seen_invoices": frozenset({("V001", "NW-2026-0142")})}, False),
        ("not_a_duplicate", {"invoice_number": "nw-2026- 0142"}, {"seen_invoices": frozenset({("V001", "NW-2026-0142")})}, False),
        ("not_a_duplicate", {}, {"seen_invoices": frozenset({("V002", "NW-2026-0142")})}, True),
        # purchase order
        ("matches_purchase_order", {}, {}, True),
        ("matches_purchase_order", {"po_number": None}, {}, True),  # PO is optional
        ("matches_purchase_order", {"po_number": "po-7781"}, {}, True),
        ("matches_purchase_order", {"po_number": "PO-0000"}, {}, False),  # unknown PO
        ("matches_purchase_order", {}, {"purchase_orders": {"PO-7781": PurchaseOrder("PO-7781", "V002", "INR", 30000.0)}}, False),
        ("matches_purchase_order", {}, {"purchase_orders": {"PO-7781": PurchaseOrder("PO-7781", "V001", "USD", 30000.0)}}, False),
        ("matches_purchase_order", {}, {"purchase_orders": {"PO-7781": PurchaseOrder("PO-7781", "V001", "INR", 20000.0)}}, False),
    ],
)
def test_rule_table(rule, invoice_changes, ctx_changes, should_pass):
    result = getattr(rules, rule)(good_invoice(**invoice_changes), ctx(**ctx_changes))
    assert result.name == rule
    assert result.passed is should_pass, result.detail
    assert result.detail  # every result explains itself


def test_unknown_vendor_duplicate_key_uses_name():
    invoice = good_invoice(vendor_name="  Some   NEW Vendor ")
    assert rules.duplicate_key(invoice, UNKNOWN) == ("name:some new vendor", "NW-2026-0142")


def test_money_is_compared_in_cents_not_floats():
    # 0.1 + 0.2 != 0.3 in floats; the rule must not care.
    invoice = good_invoice(
        line_items=[
            {"description": "a", "quantity": 1, "unit_price": 0.1, "amount": 0.1},
            {"description": "b", "quantity": 1, "unit_price": 0.2, "amount": 0.2},
        ],
        subtotal=0.3,
        tax=0.0,
        total=0.3,
        po_number=None,
    )
    assert rules.line_items_sum_to_subtotal(invoice, ctx()).passed
    assert rules.subtotal_plus_tax_equals_total(invoice, ctx()).passed


@pytest.mark.parametrize(
    "changes",
    [
        {"currency": "RUPEES"},
        {"line_items": []},
        {"invoice_number": ""},
        {"invoice_date": "02/10/2026"},
        {"total": -5},
        {"surprise_field": 1},
    ],
)
def test_schema_rejects_malformed_output(changes):
    with pytest.raises(ValidationError):
        good_invoice(**changes)


def test_schema_requires_core_fields():
    data = good_invoice().model_dump(mode="json")
    del data["invoice_number"]
    with pytest.raises(ValidationError, match="invoice_number"):
        Invoice.model_validate(data)
