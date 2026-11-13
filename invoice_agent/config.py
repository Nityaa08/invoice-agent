"""Settings from environment variables (see .env.example). No secrets in code."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLES_DIR = ROOT / "samples"

# The sample invoices are dated October/November 2026. The public demo checks
# dates "as of" this day so the samples do not go stale; the full service
# uses the real date unless AS_OF_DATE is set.
DEMO_AS_OF = "2026-11-13"


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    demo_mode: bool
    extractor: str  # "demo" or "llm"
    llm_model: str
    llm_api_key: str
    llm_base_url: str
    llm_method: str
    db_path: str
    checkpoint_path: str
    as_of: date | None

    @property
    def llm_configured(self) -> bool:
        return self.extractor == "llm" and bool(self.llm_api_key)


def load_settings() -> Settings:
    # On Vercel (which sets VERCEL=1) default to the stateless demo: its
    # functions keep no files between requests, so SQLite would not persist.
    demo = _flag("DEMO_MODE", default=bool(os.getenv("VERCEL")))
    as_of = os.getenv("AS_OF_DATE") or (DEMO_AS_OF if demo else "")
    var = Path(os.getenv("DATA_DIR", ROOT / "var"))
    return Settings(
        demo_mode=demo,
        extractor=os.getenv("EXTRACTOR", "demo").strip().lower(),
        llm_model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        llm_method=os.getenv("LLM_STRUCTURED_METHOD", "json_schema"),
        db_path=os.getenv("DB_PATH", str(var / "invoices.db")),
        checkpoint_path=os.getenv("CHECKPOINT_PATH", str(var / "checkpoints.db")),
        as_of=date.fromisoformat(as_of) if as_of else None,
    )
