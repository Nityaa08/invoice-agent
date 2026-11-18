"""Settings from environment variables (see .env.example). No secrets in code."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLES_DIR = ROOT / "samples"

# The sample invoices are dated October/November 2026. The public demo checks
# the samples' dates "as of" this day so they do not go stale; uploaded files
# are checked against the real date (an invoice dated today must not look
# "in the future"). The full service uses the real date unless AS_OF_DATE is set.
DEMO_AS_OF = "2026-11-13"


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


GROQ_BASE_URL = "https://api.groq.com/openai/v1"


@dataclass(frozen=True)
class Settings:
    demo_mode: bool
    extractor: str  # "demo" or "llm"
    llm_model: str
    llm_api_key: str = field(repr=False)  # never printed, logged or sent to the page
    llm_base_url: str
    llm_method: str
    db_path: str
    checkpoint_path: str
    as_of: date | None
    as_of_from_env: bool = False  # AS_OF_DATE was set: use it for uploads too
    llm_timeout: float = 30.0  # seconds per model call
    llm_max_retries: int = 2  # HTTP retries inside one call (429 / 5xx), with backoff

    @property
    def llm_configured(self) -> bool:
        return self.extractor == "llm" and bool(self.llm_api_key)

    @property
    def llm_provider(self) -> str:
        host = urlsplit(self.llm_base_url).hostname or ""
        for name in ("groq", "openai", "together", "fireworks", "openrouter", "deepseek", "mistral"):
            if name in host:
                return name
        return "openai-compatible"


def load_settings() -> Settings:
    # On Vercel (which sets VERCEL=1) default to the stateless demo: its
    # functions keep no files between requests, so SQLite would not persist.
    demo = _flag("DEMO_MODE", default=bool(os.getenv("VERCEL")))
    explicit = os.getenv("AS_OF_DATE", "").strip()
    as_of = explicit or (DEMO_AS_OF if demo else "")
    var = Path(os.getenv("DATA_DIR", ROOT / "var"))
    # A free Groq key is the one-step switch: GROQ_API_KEY alone turns the real
    # model on with Groq's endpoint. LLM_API_KEY + LLM_BASE_URL work for any
    # OpenAI-compatible provider, and EXTRACTOR=demo forces the free extractor.
    key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("GROQ_API_KEY", "").strip()
    default_extractor = "llm" if key else "demo"
    # Vercel stops a request at 30 s (maxDuration). Two extraction attempts must
    # fit: 12 s each and no HTTP retry inside a call on the demo; the graph's own
    # retry (with the schema error fed back) is the second chance.
    timeout = float(os.getenv("LLM_TIMEOUT_S") or (12 if demo else 30))
    retries = int(os.getenv("LLM_MAX_RETRIES") or (0 if demo else 2))
    return Settings(
        demo_mode=demo,
        extractor=(os.getenv("EXTRACTOR") or default_extractor).strip().lower(),
        llm_model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        llm_api_key=key,
        llm_base_url=os.getenv("LLM_BASE_URL", GROQ_BASE_URL),
        llm_method=os.getenv("LLM_STRUCTURED_METHOD", "json_schema"),
        llm_timeout=max(1.0, min(timeout, 60.0)),
        llm_max_retries=max(0, min(retries, 5)),
        db_path=os.getenv("DB_PATH", str(var / "invoices.db")),
        checkpoint_path=os.getenv("CHECKPOINT_PATH", str(var / "checkpoints.db")),
        as_of=date.fromisoformat(as_of) if as_of else None,
        as_of_from_env=bool(explicit),
    )
