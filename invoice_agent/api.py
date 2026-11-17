"""HTTP API and the demo page.

Full service (default, `uvicorn main:app`):
    POST /invoices                 upload a file (or pick a sample) -> runs the graph
    GET  /invoices/{id}            the run: extracted JSON, rule results, path, pending review
    POST /invoices/{id}/review     approve / reject / correct -> resumes the paused graph

Stateless demo (`DEMO_MODE=true`, used on Vercel):
    POST /api/demo/run             one request = one complete run in memory.
                                   Review decisions are sent back with the
                                   same file and replayed, because a serverless
                                   function keeps nothing between requests.

Both: GET / (page), GET /api/config, GET /api/samples, GET /api/samples/{file}, GET /api/health.

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


@lru_cache(maxsize=1)
def page_csp() -> str:
    """The demo page's CSP: its one inline script is allowed by hash, nothing else may run."""
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    hashes = " ".join(
        "'sha256-" + base64.b64encode(hashlib.sha256(m.encode()).digest()).decode() + "'"
        for m in re.findall(r"<script>(.*?)</script>", html, re.S))
    return ("default-src 'none'; "
            f"script-src {hashes}; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
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
    action: str = Field(description="approve, reject or correct")
    invoice: dict[str, Any] | None = Field(default=None, description="the corrected invoice, for 'correct'")
    reviewer: str | None = None
    note: str | None = None


def load_manifest() -> list[dict[str, Any]]:
    return json.loads((SAMPLES_DIR / "manifest.json").read_text(encoding="utf-8"))["samples"]


def _sample_path(name: str) -> Path:
    allowed = {s["file"] for s in load_manifest()}
    if name not in allowed:  # also blocks path tricks such as ../
        raise HTTPException(404, f"no sample called {name!r}")
    return SAMPLES_DIR / name


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

    if settings.demo_mode:
        _register_demo(app, settings)
    else:
        _register_full(app, settings)


def _register_full(app: FastAPI, settings: Settings) -> None:
    from .service import InvoiceService, NotAwaitingReview, RunNotFound

    service = InvoiceService.persistent(settings)
    app.state.service = service

    @app.post("/invoices", status_code=201)
    async def submit(file: UploadFile | None = File(None), sample: str | None = Form(None)):
        filename, data = await _read_input(file, sample)
        run, created = service.submit(filename, data)
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
                       sample: str | None = Form(None), decisions: str = Form("[]")):
        if not limiter.allow(client_ip(request)):
            raise HTTPException(429, "demo limit reached; try again in an hour",
                                headers={"Retry-After": "3600"})
        steps = _parse_decisions(decisions)
        filename, data = await _read_input(file, sample)
        uploaded = not (sample and (file is None or not file.filename))
        log.info("demo run: %s, %d bytes, %s", "upload" if uploaded else "sample", len(data),
                 filename.rsplit(".", 1)[-1].lower() if "." in filename else "no extension")

        # A fresh in-memory service per request: nothing is shared between visitors.
        # The fixed demo date keeps the samples fresh; a real upload is checked against today.
        run_settings = replace(settings, as_of=None) if uploaded and not settings.as_of_from_env else settings
        from .service import InvoiceService  # first POST pays for LangGraph; GETs never do

        service = InvoiceService.ephemeral(run_settings)
        run, _ = service.submit(filename, data)
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
