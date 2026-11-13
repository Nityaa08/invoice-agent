"""Turn an uploaded file into text the extractor can read.

- PDF: the text layer, via pypdf in layout mode (keeps table columns apart).
- Plain text: decoded as UTF-8.
- Images (PNG/JPEG): OCR only if the optional `pytesseract` + Tesseract are
  installed. Otherwise a clear error, and the graph sends the invoice to a
  human, who can type the fields in.

Scanned PDFs with no text layer get the same treatment as images.
"""

from __future__ import annotations

import io
import re

MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB is plenty for an invoice; keeps the demo cheap
MAX_PDF_PAGES = 10

_PDF_MAGIC = b"%PDF"
_PNG_MAGIC = b"\x89PNG"
_JPEG_MAGIC = b"\xff\xd8\xff"


class IngestError(ValueError):
    """The file cannot be turned into text. The message is shown to the reviewer."""


def sniff_kind(filename: str, data: bytes) -> str:
    """Decide the file type from its bytes first, the extension second."""
    if data.startswith(_PDF_MAGIC):
        return "pdf"
    if data.startswith(_PNG_MAGIC) or data.startswith(_JPEG_MAGIC):
        return "image"
    name = filename.lower()
    if name.endswith((".txt", ".text", ".md")):
        return "text"
    try:
        data.decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return "unknown"


def pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = reader.pages[:MAX_PDF_PAGES]
        return "\n".join(page.extract_text(extraction_mode="layout") or "" for page in pages)
    except (PdfReadError, ValueError, KeyError) as exc:
        raise IngestError(f"could not read the PDF: {exc}") from exc


def image_to_text(data: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise IngestError(
            "this is an image and OCR is not installed (pip install pytesseract pillow, "
            "plus the Tesseract binary); a reviewer can enter the fields by hand"
        ) from exc
    return pytesseract.image_to_string(Image.open(io.BytesIO(data)))


def to_text(filename: str, data: bytes) -> str:
    if not data:
        raise IngestError("the file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestError(f"the file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    kind = sniff_kind(filename, data)
    if kind == "pdf":
        text = pdf_to_text(data)
    elif kind == "image":
        text = image_to_text(data)
    elif kind == "text":
        text = data.decode("utf-8", errors="replace")
    else:
        raise IngestError("unsupported file type; upload a PDF, PNG, JPEG or text file")
    if not text.strip():
        raise IngestError("no text found (a scanned PDF needs OCR); a reviewer can enter the fields by hand")
    # Layout-mode PDF text pads pages with blank lines; keep the state small.
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
