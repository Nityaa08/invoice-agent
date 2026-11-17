"""The invoice schema the model must fill in.

This is the contract between "the model reads" and "code decides". The LLM
(or the demo extractor) must return JSON that validates against `Invoice`;
anything else is rejected and retried. The checks here are about *shape*
(types, required fields, formats). Whether the numbers add up is checked
later by the business rules in `rules.py`, so a well-formed but wrong invoice
still reaches validation and gets a clear reason.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator

# A hard ceiling on any number, far above any real invoice, so arithmetic on
# it stays exact (the plausibility rule flags much smaller amounts for review).
MAX_NUMBER = 1e15


class LineItem(BaseModel):
    """One line on the invoice."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    description: str = Field(min_length=1, max_length=500, description="What was bought, as written on the invoice")
    quantity: float = Field(gt=0, le=MAX_NUMBER, description="Number of units")
    unit_price: float = Field(ge=0, le=MAX_NUMBER, description="Price per unit, before tax")
    amount: float = Field(ge=0, le=MAX_NUMBER, description="Line total before tax, as printed")


class Invoice(BaseModel):
    """Structured invoice data, ready for accounts payable.

    Amounts are numbers in the invoice currency, without symbols or thousands
    separators. Dates are ISO 8601 (YYYY-MM-DD).
    """

    # allow_inf_nan=False: JSON parsers accept NaN and Infinity, and either one
    # would slip past `ge=0` or break the Decimal arithmetic in the rules.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)

    vendor_name: str = Field(min_length=2, max_length=200, description="Legal name of the seller, as printed")
    vendor_tax_id: str | None = Field(
        default=None, max_length=40, description="Seller's GSTIN or VAT number if printed, else null"
    )
    invoice_number: str = Field(min_length=1, max_length=64, description="The seller's invoice number")
    invoice_date: date = Field(description="Date the invoice was issued, YYYY-MM-DD")
    due_date: date | None = Field(default=None, description="Payment due date, YYYY-MM-DD, or null")
    currency: str = Field(description="ISO 4217 code, for example INR, USD, EUR")
    po_number: str | None = Field(default=None, max_length=64, description="Buyer's purchase order number, or null")
    line_items: list[LineItem] = Field(min_length=1, max_length=500, description="Every line item on the invoice")
    subtotal: float = Field(ge=0, le=MAX_NUMBER, description="Sum of line amounts before tax and before any invoice-level discount")
    discount: float = Field(
        default=0, ge=0, le=MAX_NUMBER,
        description="Invoice-level discount taken off the subtotal, as a positive number; 0 if there is none",
    )
    tax_rate: float | None = Field(
        default=None, ge=0, le=100,
        description="Total tax rate in percent as printed (CGST 9% + SGST 9% = 18), or null if not printed",
    )
    tax: float = Field(ge=0, le=MAX_NUMBER, description="Total tax (GST, VAT or sales tax)")
    total: float = Field(ge=0, le=MAX_NUMBER, description="Amount payable: subtotal - discount + tax")

    @field_validator("currency")
    @classmethod
    def _currency_shape(cls, value: str) -> str:
        # Shape only: three letters, upper-cased. Whether the code really
        # exists in ISO 4217 is a business rule (rules.currency_is_iso4217),
        # so an unknown code shows up as a failed check a reviewer can see.
        value = value.strip().upper()
        if len(value) != 3 or not value.isalpha():
            raise ValueError("currency must be a three-letter code such as INR or USD")
        return value

    @field_validator("invoice_number", "po_number")
    @classmethod
    def _strip_ids(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value
