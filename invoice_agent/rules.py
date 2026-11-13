"""Validation rules: plain, pure Python that decides whether an invoice is fine.

The model never decides. Each rule takes the extracted `Invoice` plus a
`RuleContext` (today's date, invoices already on file, open purchase orders,
the vendor match) and returns a `RuleResult`. No I/O, no clock, no globals,
so every rule is table-tested in tests/test_rules.py.

Money is compared in Decimal cents to avoid float surprises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

from .schema import Invoice

# Rounding allowed when comparing printed amounts (one cent / one paisa per
# comparison, plus a little for per-line rounding on long invoices).
MONEY_TOLERANCE = Decimal("0.02")
# Invoices older than this are suspicious (stale or a re-sent old bill).
MAX_INVOICE_AGE_DAYS = 365
# Payment terms longer than this are unusual enough for a human to look.
MAX_PAYMENT_TERMS_DAYS = 120
# Cosine similarity needed to treat a vendor as known.
VENDOR_MATCH_THRESHOLD = 0.75

# Active ISO 4217 currency codes (list maintained by SIX; checked 2026-11).
ISO_4217 = frozenset(
    """
    AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB BOV
    BRL BSD BTN BWP BYN BZD CAD CDF CHE CHF CHW CLF CLP CNY COP COU CRC CUP CVE CZK
    DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GNF GTQ GYD HKD HNL
    HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR KMF KPW KRW KWD KYD KZT
    LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR MVR MWK MXN MXV MYR
    MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN PYG QAR RON RSD RUB RWF
    SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP STN SVC SYP SZL THB TJS TMT TND TOP
    TRY TTD TWD TZS UAH UGX USD USN UYI UYU UYW UZS VED VES VND VUV WST XAF XAG XAU
    XCD XCG XDR XOF XPD XPF XPT XSU XUA YER ZAR ZMW ZWG
    """.split()
)


@dataclass(frozen=True)
class RuleResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class PurchaseOrder:
    po_number: str
    vendor_id: str
    currency: str
    amount: float


@dataclass(frozen=True)
class VendorMatch:
    vendor_id: str | None
    vendor_name: str | None
    score: float

    @property
    def matched(self) -> bool:
        return self.vendor_id is not None and self.score >= VENDOR_MATCH_THRESHOLD


@dataclass(frozen=True)
class RuleContext:
    today: date
    vendor_match: VendorMatch
    # (vendor key, invoice number) pairs already on file. See duplicate_key().
    seen_invoices: frozenset[tuple[str, str]] = frozenset()
    purchase_orders: dict[str, PurchaseOrder] = field(default_factory=dict)


def money(value: float) -> Decimal:
    """Float from the model -> Decimal rounded to cents."""
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def normalize_id(value: str) -> str:
    """Invoice and PO numbers: case and spacing should not hide a duplicate."""
    return "".join(value.upper().split())


def duplicate_key(invoice: Invoice, match: VendorMatch) -> tuple[str, str]:
    """Who sent it + their invoice number. Unknown vendors fall back to the name."""
    vendor = match.vendor_id if match.matched else "name:" + " ".join(invoice.vendor_name.lower().split())
    return (vendor, normalize_id(invoice.invoice_number))


# --- the rules ---------------------------------------------------------------


def line_items_sum_to_subtotal(invoice: Invoice, ctx: RuleContext) -> RuleResult:
    lines = sum((money(item.amount) for item in invoice.line_items), Decimal("0"))
    bad_lines = [
        item.description
        for item in invoice.line_items
        if abs(money(item.quantity * item.unit_price) - money(item.amount)) > MONEY_TOLERANCE
    ]
    if bad_lines:
        return RuleResult(
            "line_items_sum_to_subtotal",
            False,
            f"quantity x unit price does not match the amount on: {', '.join(bad_lines)}",
        )
    diff = lines - money(invoice.subtotal)
    if abs(diff) > MONEY_TOLERANCE:
        return RuleResult(
            "line_items_sum_to_subtotal",
            False,
            f"line items add up to {lines} but the subtotal says {money(invoice.subtotal)} (off by {diff})",
        )
    n = len(invoice.line_items)
    return RuleResult("line_items_sum_to_subtotal", True, f"{n} line{'s' if n > 1 else ''} add{'' if n > 1 else 's'} up to {lines}")


def subtotal_plus_tax_equals_total(invoice: Invoice, ctx: RuleContext) -> RuleResult:
    expected = money(invoice.subtotal) + money(invoice.tax)
    diff = money(invoice.total) - expected
    if abs(diff) > MONEY_TOLERANCE:
        return RuleResult(
            "subtotal_plus_tax_equals_total",
            False,
            f"subtotal {money(invoice.subtotal)} + tax {money(invoice.tax)} = {expected}, "
            f"but the total says {money(invoice.total)} (off by {diff})",
        )
    return RuleResult("subtotal_plus_tax_equals_total", True, f"{expected} matches the total")


def dates_are_sane(invoice: Invoice, ctx: RuleContext) -> RuleResult:
    issued, due = invoice.invoice_date, invoice.due_date
    problems = []
    if issued > ctx.today:
        problems.append(f"invoice date {issued} is in the future")
    if issued < ctx.today - timedelta(days=MAX_INVOICE_AGE_DAYS):
        problems.append(f"invoice date {issued} is more than {MAX_INVOICE_AGE_DAYS} days old")
    if due is not None:
        if due < issued:
            problems.append(f"due date {due} is before the invoice date {issued}")
        elif (due - issued).days > MAX_PAYMENT_TERMS_DAYS:
            problems.append(f"payment terms of {(due - issued).days} days exceed {MAX_PAYMENT_TERMS_DAYS}")
    if problems:
        return RuleResult("dates_are_sane", False, "; ".join(problems))
    terms = f", due in {(due - issued).days} days" if due else ", no due date"
    return RuleResult("dates_are_sane", True, f"issued {issued}{terms}")


def currency_is_iso4217(invoice: Invoice, ctx: RuleContext) -> RuleResult:
    if invoice.currency in ISO_4217:
        return RuleResult("currency_is_iso4217", True, f"{invoice.currency} is a valid ISO 4217 code")
    return RuleResult("currency_is_iso4217", False, f"{invoice.currency} is not an ISO 4217 currency code")


def vendor_is_known(invoice: Invoice, ctx: RuleContext) -> RuleResult:
    m = ctx.vendor_match
    if m.matched:
        return RuleResult(
            "vendor_is_known", True, f"matched {m.vendor_name} ({m.vendor_id}) with similarity {m.score:.2f}"
        )
    closest = f"; closest was {m.vendor_name} at {m.score:.2f}" if m.vendor_name else ""
    return RuleResult(
        "vendor_is_known",
        False,
        f"'{invoice.vendor_name}' is not in the vendor master (needs {VENDOR_MATCH_THRESHOLD:.2f}{closest})",
    )


def not_a_duplicate(invoice: Invoice, ctx: RuleContext) -> RuleResult:
    key = duplicate_key(invoice, ctx.vendor_match)
    if key in ctx.seen_invoices:
        return RuleResult(
            "not_a_duplicate",
            False,
            f"invoice {invoice.invoice_number} from this vendor is already on file",
        )
    return RuleResult("not_a_duplicate", True, "no earlier invoice with this vendor and number")


def matches_purchase_order(invoice: Invoice, ctx: RuleContext) -> RuleResult:
    if not invoice.po_number:
        return RuleResult("matches_purchase_order", True, "no PO on the invoice (not required)")
    po = ctx.purchase_orders.get(normalize_id(invoice.po_number))
    if po is None:
        return RuleResult("matches_purchase_order", False, f"PO {invoice.po_number} does not exist")
    problems = []
    if ctx.vendor_match.matched and po.vendor_id != ctx.vendor_match.vendor_id:
        problems.append(f"PO {po.po_number} belongs to a different vendor")
    if po.currency != invoice.currency:
        problems.append(f"PO is in {po.currency}, invoice is in {invoice.currency}")
    if money(invoice.total) > money(po.amount) + MONEY_TOLERANCE:
        problems.append(f"invoice total {money(invoice.total)} exceeds the PO amount {money(po.amount)}")
    if problems:
        return RuleResult("matches_purchase_order", False, "; ".join(problems))
    return RuleResult(
        "matches_purchase_order", True, f"within PO {po.po_number} ({po.currency} {money(po.amount)})"
    )


RULES = (
    line_items_sum_to_subtotal,
    subtotal_plus_tax_equals_total,
    dates_are_sane,
    currency_is_iso4217,
    vendor_is_known,
    not_a_duplicate,
    matches_purchase_order,
)


def run_rules(invoice: Invoice, ctx: RuleContext) -> list[RuleResult]:
    """Run every rule (no short-circuit, so a reviewer sees all problems at once)."""
    return [rule(invoice, ctx) for rule in RULES]
