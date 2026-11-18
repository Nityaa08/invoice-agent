"""Validation rules: plain, pure Python that decides whether an invoice is fine.

The model never decides. Each rule takes the extracted `Invoice` plus a
`RuleContext` (today's date, invoices already on file, open purchase orders,
the vendor match) and returns a `RuleResult`. No I/O, no clock, no globals,
so every rule is table-tested in tests/test_rules.py.

Money is compared in Decimal cents to avoid float surprises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal, localcontext

from .schema import Invoice
from .textguard import fold, instructions, lookalikes

# Rounding allowed when comparing printed amounts (one cent / one paisa per
# comparison, plus a little for per-line rounding on long invoices).
MONEY_TOLERANCE = Decimal("0.02")
# Invoices older than this are suspicious (stale or a re-sent old bill).
MAX_INVOICE_AGE_DAYS = 365
# Payment terms longer than this are unusual enough for a human to look.
MAX_PAYMENT_TERMS_DAYS = 120
# Cosine similarity needed to treat a vendor as known.
VENDOR_MATCH_THRESHOLD = 0.75
# Above this an amount is a typo or an attack, not an invoice (in any currency;
# 10 billion covers large JPY or IDR invoices). A person looks.
MAX_INVOICE_TOTAL = Decimal("10000000000")
# GST tops out at 28% (+ cess), VAT at 27% (Hungary). Above this, tax was tampered with.
MAX_TAX_RATE_PERCENT = Decimal("30")

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
    # Goods: an invoice may only bill what a goods receipt (GRN) says arrived.
    # Services (consulting, catering) have no receipt to match.
    requires_receipt: bool = False


@dataclass(frozen=True)
class ReceiptLine:
    description: str
    quantity: float


@dataclass(frozen=True)
class GoodsReceipt:
    """A goods received note: what the warehouse counted in against a PO."""

    grn_number: str
    po_number: str
    received_on: str
    lines: tuple[ReceiptLine, ...]


@dataclass(frozen=True)
class VendorMatch:
    vendor_id: str | None
    vendor_name: str | None
    score: float
    tax_id: str | None = None  # the vendor master's GSTIN / VAT number, if it has one

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
    # Totals already approved against each PO (normalized PO number -> amount),
    # so a PO cannot be overspent by splitting the bill across invoices.
    po_spent: dict[str, Decimal] = field(default_factory=dict)
    # The text the extractor read, for the checks that look at the whole document.
    document_text: str = ""
    # Goods receipts per PO (normalized PO number), for the three-way match.
    goods_receipts: dict[str, tuple[GoodsReceipt, ...]] = field(default_factory=dict)


def money(value: float) -> Decimal:
    """Float from the model -> Decimal rounded to cents (enough precision for any schema-valid amount)."""
    with localcontext() as c:
        c.prec = 40
        return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def normalize_id(value: str) -> str:
    """Invoice and PO numbers: case, spacing and lookalike characters should not hide a duplicate."""
    return "".join(fold(value).upper().split())


def duplicate_key(invoice: Invoice, match: VendorMatch) -> tuple[str, str]:
    """Who sent it + their invoice number. Unknown vendors fall back to the name."""
    vendor = match.vendor_id if match.matched else "name:" + " ".join(fold(invoice.vendor_name).lower().split())
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
    discount = money(invoice.discount)
    expected = money(invoice.subtotal) - discount + money(invoice.tax)
    less = f" - discount {discount}" if discount else ""
    diff = money(invoice.total) - expected
    if abs(diff) > MONEY_TOLERANCE:
        return RuleResult(
            "subtotal_plus_tax_equals_total",
            False,
            f"subtotal {money(invoice.subtotal)}{less} + tax {money(invoice.tax)} = {expected}, "
            f"but the total says {money(invoice.total)} (off by {diff})",
        )
    return RuleResult("subtotal_plus_tax_equals_total", True,
                      f"subtotal {money(invoice.subtotal)}{less} + tax {money(invoice.tax)} = {expected}, matches the total")


def _taxable(invoice: Invoice) -> Decimal:
    return money(invoice.subtotal) - money(invoice.discount)


def tax_rate_is_consistent(invoice: Invoice, ctx: RuleContext) -> RuleResult:
    """The tax must be the printed rate of the taxable amount, and no rate may be implausibly high."""
    name = "tax_rate_is_consistent"
    base, tax = _taxable(invoice), money(invoice.tax)
    if invoice.tax_rate is not None:
        rate = Decimal(str(invoice.tax_rate)).normalize()
        rate = Decimal(f"{rate:f}")  # 18.0 -> 18, and never 2E+1
        if rate > MAX_TAX_RATE_PERCENT:
            return RuleResult(name, False, f"a tax rate of {rate}% is above {MAX_TAX_RATE_PERCENT}%, higher than any GST or VAT rate")
        expected = (base * rate / 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        # Each tax line is rounded on its own (CGST + SGST): allow a few cents, or 0.5% of the tax.
        allowed = max(Decimal("0.05"), expected * Decimal("0.005"))
        if abs(tax - expected) <= allowed:
            return RuleResult(name, True, f"{rate}% of {base} = {expected}, matches the tax")
        taxed = _taxed_lines(invoice, rate, tax)
        if taxed:
            return RuleResult(name, True, f"{rate}% of {taxed[1]} (taxed lines: {', '.join(taxed[0])}) matches the tax")
        return RuleResult(name, False, f"the invoice prints a tax rate of {rate}%, which is {expected} on "
                                       f"{base}, but the tax charged is {tax}")
    if base > 0 and tax * 100 / base > MAX_TAX_RATE_PERCENT:
        pct = (tax * 100 / base).quantize(Decimal("0.1"))
        return RuleResult(name, False, f"tax {tax} is {pct}% of {base}; above {MAX_TAX_RATE_PERCENT}% is higher than any GST or VAT rate")
    return RuleResult(name, True, "no rate printed; tax is within the normal range" if tax else "no tax charged")


def _taxed_lines(invoice: Invoice, rate: Decimal, tax: Decimal) -> tuple[list[str], Decimal] | None:
    """US sales tax often applies to some lines only. Find lines whose sum at `rate` gives `tax`."""
    if money(invoice.discount) or not 1 < len(invoice.line_items) <= 12:
        return None
    from itertools import combinations

    items = invoice.line_items
    for size in range(1, len(items)):
        for combo in combinations(items, size):
            part = sum((money(i.amount) for i in combo), Decimal("0"))
            if abs((part * rate / 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) - tax) <= Decimal("0.01"):
                return [i.description for i in combo], part
    return None


def amounts_are_plausible(invoice: Invoice, ctx: RuleContext) -> RuleResult:
    """Zero, absurdly large, or discount-larger-than-the-bill: a person should look."""
    total, problems = money(invoice.total), []
    if total <= 0:
        problems.append("the total is zero; nothing to pay, so this is probably a mistake or a credit note")
    if total > MAX_INVOICE_TOTAL or money(invoice.subtotal) > MAX_INVOICE_TOTAL:
        problems.append(f"amounts above {MAX_INVOICE_TOTAL:,} are not accepted without a person checking them")
    if money(invoice.discount) > money(invoice.subtotal):
        problems.append(f"the discount {money(invoice.discount)} is larger than the subtotal {money(invoice.subtotal)}")
    if problems:
        return RuleResult("amounts_are_plausible", False, "; ".join(problems))
    return RuleResult("amounts_are_plausible", True, f"total {total} {invoice.currency} is within the normal range")


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


_GSTIN = re.compile(r"^(?:0[1-9]|[1-3]\d)[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_EU_PREFIXES = {"AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT",
                "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK", "XI", "CHE", "NO"}


def tax_id_problem(tax_id: str) -> str | None:
    """Why `tax_id` is not a well-formed GSTIN / UK or EU VAT number / EIN / Canadian BN, or None."""
    t = re.sub(r"[\s.]", "", tax_id).upper()
    if re.match(r"^\d{2}[A-Z]", t):
        return None if _GSTIN.match(t) else (
            f"{tax_id} looks like a GSTIN but is not in the 15-character GSTIN format (e.g. 29ABCDE1234F1Z5)")
    if t.startswith("GB"):
        return None if re.fullmatch(r"GB(?:\d{9}|\d{12}|GD\d{3}|HA\d{3})", t) else (
            f"{tax_id} is not a valid UK VAT number (GB + 9 or 12 digits)")
    prefix = next((p for p in sorted(_EU_PREFIXES, key=len, reverse=True) if t.startswith(p)), None)
    if prefix:
        return None if re.fullmatch(rf"{prefix}[0-9A-Z+*]{{2,12}}", t) and re.search(r"\d{2}", t) else (
            f"{tax_id} is not a valid {prefix} VAT number")
    if re.fullmatch(r"\d{2}-\d{7}", t) or re.fullmatch(r"\d{9}(?:[A-Z]{2}\d{4})?", t):
        return None  # US EIN, Canadian business number
    if re.fullmatch(r"[A-Z0-9\-/]{6,20}", t) and re.search(r"\d{4}", t):
        return None  # another country's format; not checked further
    return f"{tax_id} does not look like a tax registration number"


def vendor_tax_id_is_valid(invoice: Invoice, ctx: RuleContext) -> RuleResult:
    """A printed tax id must be well formed, and must be the known vendor's own id."""
    name, printed, m = "vendor_tax_id_is_valid", invoice.vendor_tax_id, ctx.vendor_match
    if not printed:
        if m.matched and m.tax_id:
            return RuleResult(name, False, f"no tax id on the invoice, but {m.vendor_name} is registered as {m.tax_id}")
        return RuleResult(name, True, "no tax id printed (not required for this vendor)")
    problem = tax_id_problem(printed)
    if problem:
        return RuleResult(name, False, problem)
    if m.matched and m.tax_id and normalize_id(printed) != normalize_id(m.tax_id):
        return RuleResult(name, False, f"the invoice says tax id {printed}, but {m.vendor_name} is registered as "
                                       f"{m.tax_id}; the name may be borrowed by someone else")
    return RuleResult(name, True, f"{printed} is well formed" + (" and matches the vendor master" if m.matched and m.tax_id else ""))


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
    spent = ctx.po_spent.get(normalize_id(po.po_number), Decimal("0"))
    left = money(po.amount) - spent
    if money(invoice.total) > money(po.amount) + MONEY_TOLERANCE:
        problems.append(f"invoice total {money(invoice.total)} exceeds the PO amount {money(po.amount)}")
    elif money(invoice.total) > left + MONEY_TOLERANCE:
        problems.append(f"invoice total {money(invoice.total)} exceeds what is left on the PO: {money(po.amount)} "
                        f"minus {spent} already paid or approved = {left}")
    if problems:
        return RuleResult("matches_purchase_order", False, "; ".join(problems))
    used = f", {spent} already paid or approved" if spent else ""
    return RuleResult(
        "matches_purchase_order", True, f"within PO {po.po_number} ({po.currency} {money(po.amount)}{used})"
    )


_STOP = {"of", "and", "the", "a", "an", "for", "with", "per", "x", "pack", "box"}
LINE_MATCH_THRESHOLD = 0.5


def _words(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z0-9]+", fold(text).lower()) if w not in _STOP)


def _similarity(a: str, b: str) -> float:
    """Word overlap (Jaccard) between two line descriptions; 1.0 for the same words in any order."""
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def matches_goods_receipt(invoice: Invoice, ctx: RuleContext) -> RuleResult:
    """Three-way match: PO + goods receipt + invoice. Bill only what was received.

    Receipts against the PO are added up per item; each invoice line is paired
    with the received item whose description shares the most words, and its
    quantity may not exceed what was received.
    """
    name = "matches_goods_receipt"
    if not invoice.po_number:
        return RuleResult(name, True, "no PO on the invoice, so no goods receipt to match")
    key = normalize_id(invoice.po_number)
    po = ctx.purchase_orders.get(key)
    if po is None:
        return RuleResult(name, True, "PO not found, so no receipt to match (matches_purchase_order reports it)")
    receipts = ctx.goods_receipts.get(key, ())
    if not receipts:
        if po.requires_receipt:
            return RuleResult(name, False, f"PO {po.po_number} is for goods, but no goods receipt has been recorded "
                                           "against it yet: nothing to bill for")
        return RuleResult(name, True, f"PO {po.po_number} is for services; no goods receipt needed")
    received: dict[str, float] = {}
    for grn in receipts:
        for line in grn.lines:
            received[line.description] = received.get(line.description, 0.0) + line.quantity
    billed: dict[str, float] = {}
    problems = []
    for item in invoice.line_items:
        best, score = max(((d, _similarity(item.description, d)) for d in received), key=lambda t: t[1])
        if score < LINE_MATCH_THRESHOLD:
            problems.append(f"'{item.description}' is not on any goods receipt for {po.po_number}")
            continue
        billed[best] = billed.get(best, 0.0) + item.quantity
    for desc, qty in billed.items():
        if qty > received[desc] + 1e-9:
            problems.append(f"billed {qty:g} x '{desc}' but only {received[desc]:g} received")
    grns = ", ".join(g.grn_number for g in receipts)
    if problems:
        return RuleResult(name, False, "; ".join(problems) + f" (goods receipts: {grns})")
    n = len(invoice.line_items)
    return RuleResult(name, True, f"{n} line{'s' if n > 1 else ''} within the quantities received ({grns})")


def no_lookalike_characters(invoice: Invoice, ctx: RuleContext) -> RuleResult:
    """Vendor name and ids must not hide Cyrillic/Greek lookalikes, foreign digits or invisible characters."""
    found = []
    for label, value in (("vendor name", invoice.vendor_name), ("invoice number", invoice.invoice_number),
                         ("PO number", invoice.po_number), ("tax id", invoice.vendor_tax_id)):
        hits = lookalikes(value or "")
        if hits:
            found.append(f"{label} contains " + ", ".join(hits[:3]))
    if found:
        return RuleResult("no_lookalike_characters", False,
                          "; ".join(found) + ". It looks like plain text but is not; this is how lookalike vendors "
                          "and re-sent duplicates get past a match")
    return RuleResult("no_lookalike_characters", True, "names and numbers use plain characters")


def no_instructions_in_document(invoice: Invoice, ctx: RuleContext) -> RuleResult:
    """Text that talks to an AI or a reviewer has no place on an invoice."""
    fields = " ".join(str(v) for v in (invoice.vendor_name, invoice.invoice_number, invoice.po_number) if v)
    fields += " " + " ".join(li.description for li in invoice.line_items)
    hits = instructions(ctx.document_text + "\n" + fields)
    if hits:
        quoted = "; ".join(f"'{h}'" for h in hits)
        return RuleResult("no_instructions_in_document", False,
                          f"the document contains text addressed to an AI or a reviewer ({quoted}). It was ignored: "
                          "only these rules decide. A person should check where this invoice came from")
    return RuleResult("no_instructions_in_document", True, "no instructions to an AI or a reviewer in the text")


RULES = (
    line_items_sum_to_subtotal,
    subtotal_plus_tax_equals_total,
    tax_rate_is_consistent,
    amounts_are_plausible,
    dates_are_sane,
    currency_is_iso4217,
    vendor_is_known,
    vendor_tax_id_is_valid,
    not_a_duplicate,
    matches_purchase_order,
    matches_goods_receipt,
    no_lookalike_characters,
    no_instructions_in_document,
)


def run_rules(invoice: Invoice, ctx: RuleContext) -> list[RuleResult]:
    """Run every rule (no short-circuit, so a reviewer sees all problems at once)."""
    return [rule(invoice, ctx) for rule in RULES]
