"""HTTP API and the demo page.

Full service (default, `uvicorn main:app`):
    POST /invoices                 upload a file (or pick a sample) -> runs the graph
    GET  /invoices/{id}            the run: extracted JSON, rule results, path, pending review
    POST /invoices/{id}/review     approve / reject / correct / recheck -> resumes the paused graph

Stateless demo (`DEMO_MODE=true`, used on Vercel):
    POST /api/demo/run             one request = one complete run in memory.
                                   Review decisions are sent back with the
                                   same file and replayed, because a serverless
                                   function keeps nothing between requests.

Both: GET / (page), GET /api/config, GET /api/samples, GET /api/samples/{file}, GET /api/health,
GET /api/vendors (the vendor master, plus added vendors in the full service) and
POST /api/vendors (add a vendor from the review screen: saved in SQLite by the
full service; in the demo only checked, and the page keeps it in the browser).

The GET routes import nothing heavy: LangGraph, LangChain and the extractor
load on the first POST, so a cold start that only serves the page is fast,
and the page, config and samples carry CDN cache headers so Vercel's edge can
answer them without starting the function at all.

Hardening (see docs/RED-TEAM.md): security headers and a CSP on every
response, a same-origin check on POSTs, a per-client rate limit on the demo,
size and depth limits on every input, and no stack traces in error responses.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
import unicodedata
from collections import deque
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .config import SAMPLES_DIR, Settings
from .ingest import MAX_UPLOAD_BYTES, MAX_UPLOAD_MB

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_DEMO_DECISIONS = 5
MAX_DECISIONS_CHARS = 64 * 1024  # five corrected invoices fit easily in this
DEMO_RUNS_PER_HOUR = 60  # per client IP and per function instance; DEMO_RUNS_PER_HOUR overrides it
VENDOR_ADDS_PER_HOUR = 30  # POST /api/vendors, per client IP and instance
MAX_VENDORS_CHARS = 16 * 1024  # the demo's list of added vendors, sent with every run

# Edge caching. Vercel's CDN honours s-maxage and purges on every deploy, so these
# can be long; browsers revalidate quickly. The health check is never cached.
CACHE_PAGE = "public, max-age=0, s-maxage=86400, stale-while-revalidate=604800"
CACHE_STATIC = "public, max-age=300, s-maxage=86400, stale-while-revalidate=604800"
NO_STORE = "no-store"

BASE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}
# JSON and files: nothing in them may run or be framed.
API_CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
# FastAPI's /docs and /redoc load their UI from jsdelivr with an inline bootstrap script.
DOCS_CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; img-src 'self' data: https://fastapi.tiangolo.com "
            "https://cdn.redoc.ly; worker-src blob:; frame-ancestors 'none'; base-uri 'none'")


# Browser OCR: Tesseract.js, loaded by the page only when someone picks a photo.
# Pinned versions, and the CSP allows exactly these package folders on jsDelivr:
# the library (main script, checked with SRI by the page, and its web worker),
# the WebAssembly core, and the English language data. 'wasm-unsafe-eval' lets
# the page compile WebAssembly; it does not allow eval() of JavaScript.
OCR_SCRIPT_SOURCES = (
    "https://cdn.jsdelivr.net/npm/tesseract.js@7.0.0/",
    "https://cdn.jsdelivr.net/npm/tesseract.js-core@7.0.0/",
)
OCR_DATA_SOURCE = "https://cdn.jsdelivr.net/npm/@tesseract.js-data/eng@1.0.0/"


@lru_cache(maxsize=1)
def page_csp() -> str:
    """The demo page's CSP: its one inline script is allowed by hash, plus the pinned OCR library."""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    hashes = " ".join(
        "'sha256-" + base64.b64encode(hashlib.sha256(m.encode()).digest()).decode() + "'"
        for m in re.findall(r"<script>(.*?)</script>", html, re.S))
    ocr = " ".join(OCR_SCRIPT_SOURCES)
    return ("default-src 'none'; "
            f"script-src {hashes} {ocr} 'wasm-unsafe-eval'; worker-src blob:; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; img-src 'self' data: blob:; "
            f"connect-src 'self' {ocr} {OCR_DATA_SOURCE}; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


def _safe_filename(name: str) -> str:
    """Display name for an upload: no folders, no control characters, at most 120 characters.

    The file is never written to disk under this name; this keeps logs, the
    audit trail and the page tidy. The page escapes it anyway.
    """
    name = re.split(r"[\\/]", name)[-1]
    name = "".join(ch for ch in unicodedata.normalize("NFC", name) if unicodedata.category(ch)[0] != "C")
    name = name.strip(" .") or "upload"
    if len(name) > 120:
        stem, dot, ext = name.rpartition(".")
        name = (stem[:110] + "…." + ext[:8]) if dot and len(ext) <= 8 else name[:120]
    return name


class ReviewDecision(BaseModel):
    action: str = Field(description="approve, reject, correct or recheck")
    invoice: dict[str, Any] | None = Field(default=None, description="the corrected invoice, for 'correct'")
    reviewer: str | None = None
    note: str | None = None


class NewVendor(BaseModel):
    name: str = Field(max_length=200, description="the vendor's legal name, as printed")
    tax_id: str | None = Field(default=None, max_length=40, description="GSTIN / VAT number, if it has one")
    currency: str = Field(max_length=8, description="ISO 4217 code the vendor bills in")
    reviewer: str | None = Field(default=None, max_length=80)


def load_manifest() -> list[dict[str, Any]]:
    return json.loads((SAMPLES_DIR / "manifest.json").read_text(encoding="utf-8"))["samples"]


def _sample_path(name: str) -> Path:
    allowed = {s["file"] for s in load_manifest()}
    if name not in allowed:  # also blocks path tricks such as ../
        raise HTTPException(404, f"no sample called {name!r}")
    return SAMPLES_DIR / name


def _source(value: str | None) -> str | None:
    """'ocr' marks text the page read from a photo with Tesseract.js. Anything else is refused."""
    if value in (None, "", "upload"):
        return None
    if value == "ocr":
        return "ocr"
    raise HTTPException(422, "source must be 'ocr' or left out")


async def _read_input(file: UploadFile | None, sample: str | None) -> tuple[str, bytes]:
    if file is not None and file.filename:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"the file is larger than {MAX_UPLOAD_MB} MB, the upload limit of this demo")
        if not data:
            raise HTTPException(422, "the file is empty")
        return _safe_filename(file.filename), data
    if sample:
        path = _sample_path(sample)
        return path.name, path.read_bytes()
    raise HTTPException(422, "send a file or the name of a sample")


class _RateLimiter:
    """Per-client sliding window, in the memory of one function instance.

    Serverless instances do not share memory, so this is a politeness limit
    per instance, not a global quota (that needs a shared store such as Redis,
    or Vercel's firewall rate-limit rules; see docs/RED-TEAM.md). Old clients
    are pruned so a flood of spoofed or rotating IPs cannot grow it without bound.
    """

    MAX_CLIENTS = 10_000

    def __init__(self, limit: int, window_s: float = 3600):
        self.limit, self.window = limit, window_s
        self.hits: dict[str, deque] = {}
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            if key not in self.hits and len(self.hits) >= self.MAX_CLIENTS:
                self._prune(now)
                if len(self.hits) >= self.MAX_CLIENTS:  # still full: drop the oldest client
                    self.hits.pop(next(iter(self.hits)))
            q = self.hits.setdefault(key, deque())
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True

    def _prune(self, now: float) -> None:
        for k in [k for k, q in self.hits.items() if not q or now - q[-1] > self.window]:
            del self.hits[k]


def client_ip(request: Request) -> str:
    """Vercel sets x-real-ip and overwrites x-forwarded-for, so a client cannot choose its own key."""
    ip = request.headers.get("x-real-ip") or request.headers.get("x-forwarded-for", "").split(",")[0]
    return ip.strip() or (request.client.host if request.client else "?")


def _same_origin(request: Request) -> bool:
    """A browser POST from another site carries its Origin; refuse it (CSRF).

    Requests without an Origin header (curl, scripts, the test client) pass:
    they are not a cross-site browser request, and the API has no cookies or
    sessions to ride on anyway.
    """
    origin = request.headers.get("origin")
    if not origin:
        return True
    if origin == "null":
        return False
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    return urlsplit(origin).netloc.lower() == host.lower()


def install_hardening(app: FastAPI) -> None:
    """Security headers on every response, a same-origin check on POSTs, no stack traces."""

    @app.middleware("http")
    async def harden(request: Request, call_next):
        if request.method == "POST" and not _same_origin(request):
            response: Response = JSONResponse({"detail": "cross-site requests are not accepted"}, status_code=403)
        else:
            try:
                response = await call_next(request)
            except Exception:  # never show a stack trace or a server path
                log.exception("unhandled error on %s %s", request.method, request.url.path)
                response = JSONResponse({"detail": "the server hit an unexpected error; nothing was saved"},
                                        status_code=500)
        for k, v in BASE_HEADERS.items():
            response.headers.setdefault(k, v)
        path = request.url.path
        if path == "/":
            csp = page_csp()
        elif path in ("/docs", "/redoc") or path.startswith("/docs/"):
            csp = DOCS_CSP
        else:
            csp = API_CSP
        response.headers.setdefault("Content-Security-Policy", csp)
        if request.method == "POST" or path.startswith("/invoices"):
            response.headers.setdefault("Cache-Control", NO_STORE)
        return response


def _json(body: Any, cache: str) -> JSONResponse:
    return JSONResponse(body, headers={"Cache-Control": cache})


def register_routes(app: FastAPI, settings: Settings) -> None:
    install_hardening(app)
    from .rules import RULES  # light: pydantic and the standard library only

    # GET and HEAD on every read route: uptime checks and link previews use HEAD.
    @app.head("/", include_in_schema=False)
    @app.get("/", include_in_schema=False)
    def page():
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": CACHE_PAGE})

    @app.head("/api/health", include_in_schema=False)
    @app.get("/api/health")
    def health():
        return _json({"status": "ok"}, NO_STORE)

    @app.head("/api/config", include_in_schema=False)
    @app.get("/api/config")
    def config():
        return _json({
            "demo_mode": settings.demo_mode,
            "extractor": "llm" if settings.llm_configured else "demo",
            "llm_model": settings.llm_model if settings.llm_configured else None,
            "as_of": settings.as_of.isoformat() if settings.as_of else None,
            # In the demo the fixed date applies to the samples only; uploads use today.
            "as_of_uploads": (settings.as_of.isoformat() if settings.as_of_from_env and settings.as_of
                              else None if not settings.demo_mode else "today"),
            "max_upload_mb": MAX_UPLOAD_MB,
            "rules": [r.__name__ for r in RULES],
        }, CACHE_STATIC)

    @app.head("/api/samples", include_in_schema=False)
    @app.get("/api/samples")
    def samples():
        return _json([{"file": s["file"], "title": s["title"], "decision": s["decision"]} for s in load_manifest()],
                     CACHE_STATIC)

    @app.head("/api/samples/{name}", include_in_schema=False)
    @app.get("/api/samples/{name}")
    def sample_file(name: str):
        return FileResponse(_sample_path(name), headers={"Cache-Control": CACHE_STATIC})

    vendor_limiter = _RateLimiter(int(os.getenv("VENDOR_ADDS_PER_HOUR") or VENDOR_ADDS_PER_HOUR))
    service = None if settings.demo_mode else _register_full(app, settings)
    if settings.demo_mode:
        _register_demo(app, settings)

    @app.head("/api/vendors", include_in_schema=False)
    @app.get("/api/vendors")
    def vendors():
        if service is not None:
            return _json(service.list_vendors(), NO_STORE)
        from .config import DATA_DIR  # the demo: the read-only master, no numpy needed

        rows = json.loads((DATA_DIR / "vendors.json").read_text(encoding="utf-8"))
        return _json([dict(v, source="master") for v in rows], CACHE_STATIC)

    @app.post("/api/vendors")
    def add_vendor(request: Request, vendor: NewVendor):
        if not vendor_limiter.allow(client_ip(request)):
            raise HTTPException(429, "vendor limit reached; try again in an hour", headers={"Retry-After": "3600"})
        from .vendors import VendorRejected

        data = vendor.model_dump()
        try:
            if service is not None:
                added = service.add_vendor(data, added_by=data.get("reviewer") or "reviewer")
                return JSONResponse(dict(added, stored="server"), status_code=201)
            from .service import default_matcher
            from .vendors import check_new_vendor

            # The demo stores nothing: the page keeps the checked vendor in the browser
            # and sends it back with each run, where it is checked again.
            return JSONResponse(dict(check_new_vendor(data, default_matcher()), stored="browser"), status_code=200)
        except VendorRejected as exc:
            raise HTTPException(422, str(exc))


def _register_full(app: FastAPI, settings: Settings):
    from .service import InvoiceService, NotAwaitingReview, RunNotFound

    service = InvoiceService.persistent(settings)
    app.state.service = service

    @app.post("/invoices", status_code=201)
    async def submit(file: UploadFile | None = File(None), sample: str | None = Form(None),
                     source: str | None = Form(None)):
        origin = _source(source)
        filename, data = await _read_input(file, sample)
        run, created = service.submit(filename, data, source=origin)
        # Same file again: 200 with the original run instead of a second one.
        return run if created else JSONResponse(run, status_code=200)

    @app.get("/invoices/{run_id}")
    def get_run(run_id: str):
        try:
            return service.get(run_id)
        except RunNotFound:
            raise HTTPException(404, f"no run {run_id}")

    @app.post("/invoices/{run_id}/review")
    def review(run_id: str, decision: ReviewDecision):
        try:
            return service.review(run_id, decision.model_dump())
        except RunNotFound:
            raise HTTPException(404, f"no run {run_id}")
        except NotAwaitingReview as exc:
            raise HTTPException(409, str(exc))
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    return service


def _parse_vendors(raw: str) -> list[dict[str, Any]]:
    """The demo's added vendors, sent back by the browser: parsed, bounded and checked again."""
    if raw in ("", "[]"):
        return []
    if len(raw) > MAX_VENDORS_CHARS:
        raise HTTPException(413, "the list of added vendors is too long")
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise HTTPException(422, "vendors must be a JSON list") from exc
    from .service import default_matcher
    from .vendors import VendorRejected, check_vendor_list

    try:
        return check_vendor_list(items, default_matcher())
    except VendorRejected as exc:
        raise HTTPException(422, str(exc))


def _parse_decisions(raw: str) -> list[dict[str, Any]]:
    if len(raw) > MAX_DECISIONS_CHARS:
        raise HTTPException(413, "the review decisions are too long")
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise HTTPException(422, "decisions must be a JSON list of review decisions") from exc
    if not isinstance(items, list):
        raise HTTPException(422, "decisions must be a JSON list of review decisions")
    if len(items) > MAX_DEMO_DECISIONS:
        raise HTTPException(422, f"at most {MAX_DEMO_DECISIONS} review decisions")
    try:
        return [ReviewDecision.model_validate(d).model_dump() for d in items]
    except ValueError as exc:
        raise HTTPException(422, "each decision needs an 'action' (approve, reject or correct)") from exc


def _register_demo(app: FastAPI, settings: Settings) -> None:
    limiter = _RateLimiter(int(os.getenv("DEMO_RUNS_PER_HOUR") or DEMO_RUNS_PER_HOUR))

    @app.post("/api/demo/run")
    async def demo_run(request: Request, file: UploadFile | None = File(None),
                       sample: str | None = Form(None), decisions: str = Form("[]"),
                       vendors: str = Form("[]"), source: str | None = Form(None)):
        if not limiter.allow(client_ip(request)):
            raise HTTPException(429, "demo limit reached; try again in an hour",
                                headers={"Retry-After": "3600"})
        steps = _parse_decisions(decisions)
        origin = _source(source)
        filename, data = await _read_input(file, sample)
        added = _parse_vendors(vendors)
        uploaded = not (sample and (file is None or not file.filename))
        log.info("demo run: %s, %d bytes, %s", ("ocr text" if origin else "upload") if uploaded else "sample", len(data),
                 filename.rsplit(".", 1)[-1].lower() if "." in filename else "no extension")

        # A fresh in-memory service per request: nothing is shared between visitors.
        # The fixed demo date keeps the samples fresh; a real upload is checked against today.
        run_settings = replace(settings, as_of=None) if uploaded and not settings.as_of_from_env else settings
        from .service import InvoiceService  # first POST pays for LangGraph; GETs never do

        service = InvoiceService.ephemeral(run_settings, vendors=added)
        run, _ = service.submit(filename, data, source=origin)
        if run.get("ingest_error") or run.get("extraction_error"):
            log.info("demo run %s needs review: %s", run["id"], run.get("ingest_error") or run.get("extraction_error"))
        for step in steps:
            if run["status"] != "needs_review":
                break
            try:
                run = service.review(run["id"], step)
            except ValueError as exc:
                raise HTTPException(422, str(exc))
        return run
