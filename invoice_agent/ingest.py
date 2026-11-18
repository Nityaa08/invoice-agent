"""Turn an uploaded file into text the extractor can read.

- PDF: the text layer, via pypdf in layout mode (keeps table columns apart).
- Plain text: decoded as UTF-8.
- Images (PNG/JPEG): the server does not run OCR (it would add a 30 MB
  binary to a free serverless function). The demo page reads photos in the
  visitor's browser with Tesseract.js and sends the recognised text, flagged
  as OCR, so it always goes to a person. For a local service, server-side OCR
  can be switched on with SERVER_OCR=true plus the optional `pytesseract` and
  the Tesseract binary. Otherwise an image gets a clear message and the graph
  sends it to a human, who can type the fields in.

Scanned PDFs with no text layer are detected and reported plainly; the demo
does not pretend to read them. Every failure is an `IngestError` with a
message written for the person who uploaded the file.
"""

from __future__ import annotations

import io
import logging
import os
import re

log = logging.getLogger(__name__)

# Vercel caps a serverless request body at 4.5 MB; 4 MB leaves room for the
# multipart envelope. A text-layer invoice is usually well under 1 MB.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_UPLOAD_MB = MAX_UPLOAD_BYTES // (1024 * 1024)
MAX_PDF_PAGES = 10  # the first 9 pages plus the last one (where the totals are)
MIN_TEXT_CHARS = 30  # fewer letters and digits than this: treat the PDF as a scan
# A page whose drawing instructions are bigger than this is not an invoice page;
# text extraction time grows with it (about 2 s per MB), so it is not read.
MAX_PAGE_CONTENT_BYTES = 1024 * 1024
# A long invoice is a few thousand characters. Beyond this the text is cut to its
# start and end (where the header and the totals are) and a person reviews it.
MAX_TEXT_CHARS = 60_000
LONG_TEXT_NOTE = (f"the document has more than {MAX_TEXT_CHARS:,} characters of text; only its start and end "
                  "were read, so a person should check it")

SCANNED_PDF_MESSAGE = (
    "This PDF is a scanned image. The server reads PDFs with a text layer and does not run OCR. "
    "Upload a photo or screenshot of the page instead (the page reads images in your browser), "
    "or a reviewer can type the fields in by hand."
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


IMAGE_MESSAGE = (
    "This is a photo or image ({kind}). The server does not run OCR on images; the demo page reads photos "
    "in your browser and sends the text. Upload it through the page, export the invoice to PDF, "
    "or use 'Correct fields' to type the fields in."
)


def server_ocr_enabled() -> bool:
    return os.getenv("SERVER_OCR", "").strip().lower() in {"1", "true", "yes", "on"}


def shorten(text: str) -> str:
    """Keep the start and the end of an over-long text (header and totals)."""
    head, tail = MAX_TEXT_CHARS * 2 // 3, MAX_TEXT_CHARS // 3
    return text[:head] + "\n[... text omitted ...]\n" + text[-tail:]


def _content_size(page) -> int:
    try:
        contents = page.get_contents()
        return len(contents.get_data()) if contents is not None else 0
    except Exception:
        return 0  # unreadable contents: extract_text reports the problem


def _page_text(page) -> str | None:
    """The page's text, or None if it could not be read at all."""
    if _content_size(page) > MAX_PAGE_CONTENT_BYTES:
        raise IngestError("a page of this PDF holds more than 1 MB of drawing instructions, far more than an "
                          "invoice page; it was not read")
    try:
        return page.extract_text(extraction_mode="layout") or ""
    except Exception:  # layout mode trips on some fonts; plain mode is more forgiving
        try:
            return page.extract_text() or ""
        except Exception as exc:
            log.info("page text failed: %s", type(exc).__name__)
            return None


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
        log.info("pdf open failed: %s", type(exc).__name__)  # the type helps debugging; the user gets plain words
        raise IngestError("this file could not be read as a PDF; it may be damaged or incomplete") from exc
    if not pages:
        raise IngestError("this PDF has no pages")
    if len(pages) > MAX_PDF_PAGES:
        pages = pages[: MAX_PDF_PAGES - 1] + pages[-1:]
    texts = [_page_text(p) for p in pages]
    if all(t is None for t in texts):
        raise IngestError("the text of this PDF could not be read; it may be damaged or use an unusual format")
    text = "\n".join(t or "" for t in texts)
    if len(re.findall(r"[A-Za-z0-9]", text)) < MIN_TEXT_CHARS:
        raise IngestError(SCANNED_PDF_MESSAGE)
    return text


def image_to_text(data: bytes) -> str:
    kind = "PNG" if data.startswith(_PNG_MAGIC) else "JPEG"
    if not server_ocr_enabled():
        raise IngestError(IMAGE_MESSAGE.format(kind=kind))
    try:  # optional: pip install pytesseract pillow, plus the Tesseract binary
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise IngestError(IMAGE_MESSAGE.format(kind=kind)) from exc
    try:
        Image.MAX_IMAGE_PIXELS = 40_000_000  # a decompression bomb raises instead of filling memory
        return pytesseract.image_to_string(Image.open(io.BytesIO(data)))
    except Exception as exc:  # unreadable image, missing tesseract binary, ...
        log.info("ocr failed: %s", type(exc).__name__)
        raise IngestError("OCR could not read this image; use 'Correct fields' to type the fields in") from exc


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
