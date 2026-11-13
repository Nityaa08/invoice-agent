"""Extractors: invoice text in, a dict that should validate as `Invoice` out.

Two implementations behind one small interface:

- `DemoExtractor` (default): regular expressions and layout rules written for
  the kind of invoices in samples/. Free, offline, deterministic, so the tests,
  the eval and the public demo never need an API key. It is a stand-in for the
  model, not a general invoice reader.
- `LLMExtractor`: any OpenAI-compatible chat model (Groq's free tier by
  default) through LangChain's `with_structured_output(Invoice)`, so the model
  is constrained by the same Pydantic schema.

Neither one decides anything. The graph validates what they return against
the schema, retries once with the error fed back, and hands the result to the
rules.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Protocol

from pydantic import ValidationError

from .schema import Invoice


class ExtractionError(RuntimeError):
    """The extractor could not produce output at all (network, refusal, ...)."""


class Extractor(Protocol):
    name: str

    def extract(self, text: str, feedback: str | None = None) -> dict[str, Any]:
        """Return the extracted fields. `feedback` is the validation error from
        the previous attempt, if any, so a model can correct itself."""
        ...


# --- demo extractor ----------------------------------------------------------

_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "Rs.": "INR"}
_DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y")
_HEADER_LINE = re.compile(r"^\s*(tax\s+)?invoice\s*$|^\s*rechnung\b", re.I)
_ID = r"([A-Z0-9][A-Z0-9\-/]*)"


def parse_amount(raw: str) -> float | None:
    """'1,450.00' / '$1,450.00' / '1.450,00' / '₹ 1,450' -> 1450.0."""
    s = re.sub(r"[^\d.,\-]", "", raw)
    if not re.search(r"\d", s):
        return None
    if "," in s and "." in s:
        # whichever separator comes last is the decimal point
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".") if re.search(r",\d{2}$", s) else s.replace(",", "")
    elif s.count(".") > 1:
        s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_date(raw: str) -> str | None:
    raw = raw.strip().rstrip(".")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _field(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text, re.I | re.M)
    return m.group(1).strip() if m else None


def _last_amount(line: str) -> float | None:
    numbers = re.findall(r"[-$€£₹]?\s?\d[\d.,]*", line)
    return parse_amount(numbers[-1]) if numbers else None


class DemoExtractor:
    """Rules + regex for the sample invoices. See the module docstring."""

    name = "demo-rules"

    def extract(self, text: str, feedback: str | None = None) -> dict[str, Any]:
        lines = [ln.rstrip() for ln in text.splitlines()]
        out: dict[str, Any] = {}

        vendor = next((ln.strip() for ln in lines if ln.strip() and not _HEADER_LINE.match(ln)), None)
        if vendor:
            out["vendor_name"] = vendor
        out["vendor_tax_id"] = _field(text, r"(?:gstin|vat[ \t]*(?:id|reg\.?[ \t]*no\.?|no\.?|number)|tax[ \t]*id)[ \t]*[:#]?[ \t]*([A-Z0-9]{8,15})\b")

        number = _field(text, r"(?:invoice[ \t]*(?:no\.?|number|#)|bill[ \t]*no\.?)[ \t]*[:#][ \t]*" + _ID)
        if number:
            out["invoice_number"] = number

        issued = _field(text, r"^[ \t]*(?:invoice[ \t]*date|date[ \t]*of[ \t]*issue|date)[ \t]*:[ \t]*(.+?)[ \t]*$")
        if issued and parse_date(issued):
            out["invoice_date"] = parse_date(issued)
        due = _field(text, r"^[ \t]*(?:due[ \t]*date|payment[ \t]*due|due)[ \t]*:[ \t]*(.+?)[ \t]*$")
        out["due_date"] = parse_date(due) if due else None

        out["po_number"] = _field(text, r"(?:p\.?o\.?[ \t]*(?:number|no\.?|#)|purchase[ \t]*order)[ \t]*[:#][ \t]*" + _ID)

        currency = _field(text, r"^[ \t]*currency[ \t]*:[ \t]*([A-Za-z]{3})\b")
        if not currency:
            currency = next((code for sym, code in _SYMBOLS.items() if sym in text), None)
        if currency:
            out["currency"] = currency.upper()

        out["line_items"] = self._line_items(lines)

        # Totals live below the line-item table; reading only there keeps
        # header lines such as "VAT Reg No: GB123456789" out of the tax.
        start = next((i for i, ln in enumerate(lines) if ln.strip().lower().startswith("description")), 0)
        for line in lines[start:]:
            low = line.strip().lower()
            if re.match(r"^sub\s*-?total\b", low):
                out["subtotal"] = _last_amount(line)
            elif re.match(r"^(grand\s+)?total(\s+due)?\b|^amount\s+due\b", low):
                out["total"] = _last_amount(line)
            elif re.match(r"^(gst|igst|cgst|sgst|vat|tax|sales\s+tax)\b", low):
                out["tax"] = round(out.get("tax", 0.0) + (_last_amount(line) or 0.0), 2)
        out.setdefault("tax", 0.0)
        return out

    @staticmethod
    def _line_items(lines: list[str]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        in_table = False
        for line in lines:
            low = line.strip().lower()
            if low.startswith("description"):
                in_table = True
                continue
            if not in_table or not low or set(low) <= {"-", "="}:
                continue
            if re.match(r"^sub\s*-?total\b", low):
                break
            cells = [c for c in re.split(r"\s{2,}|\t|\|", line.strip()) if c.strip()]
            if len(cells) < 4:
                continue
            desc, qty, unit, amount = cells[0], cells[-3], cells[-2], cells[-1]
            q = parse_amount(re.match(r"[\d.,]+", qty.strip()).group(0)) if re.match(r"[\d.,]+", qty.strip()) else None
            u, a = parse_amount(unit), parse_amount(amount)
            if q is None or u is None or a is None:
                continue
            items.append({"description": desc.strip(), "quantity": q, "unit_price": u, "amount": a})
        return items


# --- real LLM extractor --------------------------------------------------------

SYSTEM_PROMPT = """You extract data from invoices for an accounts payable team.
Return only what is printed on the invoice. Do not calculate, correct or guess:
if the total printed is wrong, return the printed total. Use null for optional
fields that are not on the invoice. Dates are YYYY-MM-DD. Amounts are plain
numbers without currency symbols or thousands separators. Currency is the
ISO 4217 code (for example INR for rupees, EUR for euros)."""


class LLMExtractor:
    """Any OpenAI-compatible chat model, constrained by the `Invoice` schema.

    `with_structured_output(Invoice, include_raw=True)` asks the provider for
    JSON matching the schema. Output that breaks the schema surfaces as a
    Pydantic `ValidationError` (or as raw JSON the graph then validates); the
    graph records the message and the retry passes it back as `feedback`.
    """

    def __init__(self, model: str, api_key: str, base_url: str | None = None,
                 method: str = "json_schema", timeout: float = 30.0, chat_model: Any = None):
        self.name = f"llm:{model}"
        if chat_model is None:
            from langchain_openai import ChatOpenAI

            chat_model = ChatOpenAI(
                model=model, api_key=api_key, base_url=base_url,
                temperature=0, timeout=timeout, max_retries=2,
            )
        self._structured = chat_model.with_structured_output(Invoice, method=method, include_raw=True)

    def extract(self, text: str, feedback: str | None = None) -> dict[str, Any]:
        messages = [("system", SYSTEM_PROMPT), ("human", f"Invoice text:\n\n{text}")]
        if feedback:
            messages.append((
                "human",
                "Your previous answer failed schema validation with this error. "
                f"Fix it and answer again:\n{feedback}",
            ))
        try:
            result = self._structured.invoke(messages)
        except ValidationError:
            # Valid JSON that breaks the schema (json_schema mode parses inside
            # the OpenAI client). The graph reports it and retries.
            raise
        except Exception as exc:  # network, auth, rate limit: not a schema problem
            raise ExtractionError(f"model call failed: {type(exc).__name__}: {exc}") from exc
        if result.get("parsed") is not None:
            return result["parsed"].model_dump(mode="json")
        # Hand back whatever the model produced so the schema error is precise.
        raw = result.get("raw")
        content = getattr(raw, "content", "") if raw is not None else ""
        try:
            return json.loads(content) if content else {}
        except (TypeError, json.JSONDecodeError):
            raise ExtractionError(f"model returned output that is not JSON: {result.get('parsing_error')}")


def build_extractor(provider: str, *, model: str = "", api_key: str = "", base_url: str = "",
                    method: str = "json_schema") -> Extractor:
    if provider == "demo":
        return DemoExtractor()
    if provider == "llm":
        if not api_key:
            raise ValueError("EXTRACTOR=llm needs LLM_API_KEY")
        return LLMExtractor(model=model, api_key=api_key, base_url=base_url or None, method=method)
    raise ValueError(f"unknown extractor {provider!r}; use 'demo' or 'llm'")
