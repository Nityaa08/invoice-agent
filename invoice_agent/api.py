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
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import SAMPLES_DIR, Settings
from .ingest import MAX_UPLOAD_BYTES, MAX_UPLOAD_MB
from .service import InvoiceService, NotAwaitingReview, RunNotFound

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_DEMO_DECISIONS = 5
DEMO_RUNS_PER_HOUR = 60


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
        return file.filename, data
    if sample:
        path = _sample_path(sample)
        return path.name, path.read_bytes()
    raise HTTPException(422, "send a file or the name of a sample")


class _RateLimiter:
    """Per-client sliding window, per instance. Enough to keep a free demo polite."""

    def __init__(self, limit: int, window_s: float = 3600):
        self.limit, self.window = limit, window_s
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


def register_routes(app: FastAPI, settings: Settings) -> None:
    @app.get("/", include_in_schema=False)
    def page():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/config")
    def config():
        return {
            "demo_mode": settings.demo_mode,
            "extractor": "llm" if settings.llm_configured else "demo",
            "llm_model": settings.llm_model if settings.llm_configured else None,
            "as_of": settings.as_of.isoformat() if settings.as_of else None,
            # In the demo the fixed date applies to the samples only; uploads use today.
            "as_of_uploads": (settings.as_of.isoformat() if settings.as_of_from_env and settings.as_of
                              else None if not settings.demo_mode else "today"),
            "max_upload_mb": MAX_UPLOAD_MB,
        }

    @app.get("/api/samples")
    def samples():
        return [{"file": s["file"], "title": s["title"], "decision": s["decision"]} for s in load_manifest()]

    @app.get("/api/samples/{name}")
    def sample_file(name: str):
        return FileResponse(_sample_path(name))

    if settings.demo_mode:
        _register_demo(app, settings)
    else:
        _register_full(app, settings)


def _register_full(app: FastAPI, settings: Settings) -> None:
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


def _register_demo(app: FastAPI, settings: Settings) -> None:
    limiter = _RateLimiter(DEMO_RUNS_PER_HOUR)

    @app.post("/api/demo/run")
    async def demo_run(request: Request, file: UploadFile | None = File(None),
                       sample: str | None = Form(None), decisions: str = Form("[]")):
        client = request.headers.get("x-forwarded-for", request.client.host if request.client else "?")
        if not limiter.allow(client.split(",")[0].strip()):
            raise HTTPException(429, "demo limit reached; try again in an hour")
        try:
            steps = [ReviewDecision.model_validate(d).model_dump() for d in json.loads(decisions)]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise HTTPException(422, f"decisions must be a JSON list of review decisions: {exc}")
        if len(steps) > MAX_DEMO_DECISIONS:
            raise HTTPException(422, f"at most {MAX_DEMO_DECISIONS} review decisions")
        filename, data = await _read_input(file, sample)
        uploaded = not (sample and (file is None or not file.filename))
        log.info("demo run: %s, %d bytes, %s", "upload" if uploaded else "sample", len(data),
                 filename.rsplit(".", 1)[-1].lower() if "." in filename else "no extension")

        # A fresh in-memory service per request: nothing is shared between visitors.
        # The fixed demo date keeps the samples fresh; a real upload is checked against today.
        run_settings = replace(settings, as_of=None) if uploaded and not settings.as_of_from_env else settings
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
