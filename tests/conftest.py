from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


@pytest.fixture
def sample_bytes():
    def load(name: str) -> bytes:
        return (SAMPLES / name).read_bytes()

    return load


@pytest.fixture(autouse=True)
def _no_live_model(monkeypatch):
    """A GROQ_API_KEY or LLM_API_KEY in the developer's shell must not make tests call a real model."""
    for key in ("LLM_API_KEY", "GROQ_API_KEY", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(key, raising=False)
