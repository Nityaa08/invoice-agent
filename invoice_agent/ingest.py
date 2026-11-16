"""Turn an uploaded file into text the extractor can read.

- PDF: the text layer, via pypdf in layout mode (keeps table columns apart).
- Plain text: decoded as UTF-8.
- Images (PNG/JPEG): OCR only if the optional `pytesseract` + Tesseract are
  installed. Otherwise a clear error, and the graph sends the invoice to a
  human, who can type the fields in.

Scanned PDFs with no text layer are detected and reported plainly; the demo
does not pretend to read them. Every failure is an `IngestError` with a
message written for the person who uploaded the file.
"""

from __future__ import annotations

import io
import re

# Vercel caps a serverless request body at 4.5 MB; 4 MB leaves room for the
# multipart envelope. A text-layer invoice is usually well under 1 MB.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_UPLOAD_MB = MAX_UPLOAD_BYTES // (1024 * 1024)
MAX_PDF_PAGES = 10  # the first 9 pages plus the last one (where the totals are)
MIN_TEXT_CHARS = 30  # fewer letters and digits than this: treat the PDF as a scan

SCANNED_PDF_MESSAGE = (
    "This PDF is a scanned image. The demo reads PDFs with a text layer; OCR is listed as a next step. "
    "A reviewer can type the fields in by hand."
)

_PDF_MAGIC = b"%PDF"
_PNG_MAGIC = b"\x89PNG"
_JPEG_MAGIC = b"\xff\xd8\xff"
_ZIP_MAGIC = b"PK\x03\x04"  # .docx / .xlsx


class IngestError(ValueError):
    """The file cannot be turned into text. The message is shown to the reviewer."""


def sniff_kind(filename: str, data: bytes) -> str:
    """Decide the file type from its bytes first, the extension second."""
    if _PDF_MAGIC in data[:1024]:  # the spec allows junk before the header
        return "pdf"
    if data.startswith(_PNG_MAGIC) or data.startswith(_JPEG_MAGIC):
        return "image"
    name = filename.lower()
    if data.startswith(_ZIP_MAGIC) or name.endswith(".pdf"):
        return "unknown"  # an Office file, or something renamed to .pdf
    if name.endswith((".txt", ".text", ".md")):
        return "text"
    try:
        data.decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return "unknown"


def _page_text(page) -> str:
    try:
        return page.extract_text(extraction_mode="layout") or ""
    except Exception:  # layout mode trips on some fonts; plain mode is more forgiving
        try:
            return page.extract_text() or ""
        except Exception:
            return ""


def pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import FileNotDecryptedError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise FileNotDecryptedError("password required")
        pages = list(reader.pages)
    except FileNotDecryptedError as exc:
        raise IngestError("this PDF is password-protected; remove the password and upload it again") from exc
    except Exception as exc:  # pypdf raises many types on damaged files; none may reach the user as a 500
        raise IngestError(
            f"this file could not be read as a PDF; it may be damaged or incomplete ({type(exc).__name__})"
        ) from exc
    if not pages:
        raise IngestError("this PDF has no pages")
    if len(pages) > MAX_PDF_PAGES:
        pages = pages[: MAX_PDF_PAGES - 1] + pages[-1:]
    text = "\n".join(_page_text(p) for p in pages)
    if len(re.findall(r"[A-Za-z0-9]", text)) < MIN_TEXT_CHARS:
        raise IngestError(SCANNED_PDF_MESSAGE)
    return text


def image_to_text(data: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise IngestError(
            "this is an image and OCR is not installed (pip install pytesseract pillow, "
            "plus the Tesseract binary); a reviewer can enter the fields by hand"
        ) from exc
    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(data)))
    except Exception as exc:  # unreadable image, missing tesseract binary, ...
        raise IngestError(f"OCR failed ({type(exc).__name__}); a reviewer can enter the fields by hand") from exc


def to_text(filename: str, data: bytes) -> str:
    if not data:
        raise IngestError("the file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestError(f"the file is larger than {MAX_UPLOAD_MB} MB")
    kind = sniff_kind(filename, data)
    if kind == "pdf":
        text = pdf_to_text(data)
    elif kind == "image":
        text = image_to_text(data)
    elif kind == "text":
        text = data.decode("utf-8", errors="replace")
    elif filename.lower().endswith(".pdf"):
        raise IngestError("this file is named .pdf but is not a PDF (it does not start with %PDF)")
    elif data.startswith(_ZIP_MAGIC):
        raise IngestError("this looks like a Word or Excel file; save it as PDF and upload that")
    else:
        raise IngestError("unsupported file type; upload a PDF, PNG, JPEG or text file")
    if not text.strip():
        raise IngestError("no text found in the file; a reviewer can enter the fields by hand")
    # Layout-mode PDF text pads pages with blank lines; keep the state small.
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
