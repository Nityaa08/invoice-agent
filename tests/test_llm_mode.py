"""LLM mode, ready to switch on: Groq key handling, timeouts that fit Vercel, the cache, safe errors, compare."""

import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from evals.run_eval import compare, evaluate
from invoice_agent import extractors
from invoice_agent.api import register_routes
from invoice_agent.config import GROQ_BASE_URL, load_settings
from invoice_agent.extractors import ExtractionError, LLMExtractor, build_extractor

from .test_llm_extractor import GOOD, fake_openai


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in ("EXTRACTOR", "LLM_BASE_URL", "LLM_MODEL", "LLM_TIMEOUT_S", "LLM_MAX_RETRIES", "DEMO_MODE", "VERCEL"):
        monkeypatch.delenv(k, raising=False)


def test_no_key_means_the_free_demo_extractor():
    s = load_settings()
    assert s.extractor == "demo" and not s.llm_configured


def test_a_groq_key_alone_switches_the_model_on(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    s = load_settings()
    assert s.extractor == "llm" and s.llm_configured and s.llm_base_url == GROQ_BASE_URL
    assert s.llm_provider == "groq" and s.llm_model == "openai/gpt-oss-20b"
    assert "gsk_test" not in repr(s)  # the key is never printed
    monkeypatch.setenv("EXTRACTOR", "demo")  # the off switch
    assert load_settings().extractor == "demo"


def test_llm_api_key_wins_and_any_openai_compatible_endpoint_works(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("LLM_API_KEY", "sk_other")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    s = load_settings()
    assert s.llm_api_key == "sk_other" and s.llm_provider == "openai"


def test_timeouts_fit_inside_vercels_30_seconds(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    s = load_settings()
    # Two extraction attempts (the graph's retry), each one call with no HTTP retry.
    assert 2 * s.llm_timeout * (1 + s.llm_max_retries) <= 25
    monkeypatch.delenv("VERCEL")
    full = load_settings()
    assert full.llm_timeout == 30 and full.llm_max_retries == 2

    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def with_structured_output(self, *a, **k):
            return None

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", Recorder)
    ex = build_extractor("llm", model="m", api_key="k", base_url="http://x/v1", timeout=12, max_retries=0)
    assert seen | {"api_key": None} == {"model": "m", "api_key": None, "base_url": "http://x/v1", "temperature": 0,
                                        "timeout": 12, "max_retries": 0}
    assert ex._cache_key  # the extractor the service builds caches


def test_config_reports_the_live_model_but_never_the_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret_value")
    monkeypatch.setenv("DEMO_MODE", "true")
    app = FastAPI()
    register_routes(app, load_settings())
    cfg = TestClient(app).get("/api/config")
    body = cfg.json()
    assert body["extractor"] == "llm" and body["llm_provider"] == "groq" and body["llm_model"] == "openai/gpt-oss-20b"
    assert "gsk_secret" not in cfg.text


def test_identical_text_is_extracted_once_when_cached(monkeypatch):
    monkeypatch.setattr(extractors, "_CACHE", type(extractors._CACHE)())
    requests = []
    chat = fake_openai([json.dumps(GOOD)], requests)
    ex = LLMExtractor("m", "k", chat_model=chat, cache=True)
    first = ex.extract("INVOICE cached")
    first["total"] = -1  # callers cannot change what is cached
    second = LLMExtractor("m", "k", chat_model=chat, cache=True).extract("INVOICE cached")  # a new request
    assert len(requests) == 1 and second["total"] == 10.0


@pytest.mark.parametrize("status, words", [
    (401, "API key was refused"), (429, "rate limit"), (500, "HTTP 500"),
])
def test_model_errors_never_quote_the_provider(status, words):
    def handler(request):
        return httpx.Response(status, json={"error": {"message": "Incorrect API key provided: gsk_ab***xyz"}})
    from langchain_openai import ChatOpenAI

    chat = ChatOpenAI(model="m", api_key="k", base_url="http://llm.test/v1", max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ExtractionError) as exc:
        LLMExtractor("m", "k", chat_model=chat).extract("INVOICE")
    assert words in str(exc.value) and "gsk_" not in str(exc.value)


def test_compare_prints_both_extractors_side_by_side():
    calls = []

    def fake(which, extractor=None):
        calls.append((which, extractor))
        r = evaluate(which)  # demo both times: no network in tests
        return dict(r, extractor="llm:test" if extractor == "llm" else r["extractor"])
    table = compare(["samples"], run=fake)
    assert calls == [("samples", "demo"), ("samples", "llm")]
    assert "`demo-rules`" in table and "`llm:test`" in table and table.count("| samples |") == 2


def test_compare_without_a_key_skips_cleanly(capsys, monkeypatch):
    import sys

    from evals import run_eval

    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--compare"])
    assert run_eval.main() == 0
    assert "Skipped" in capsys.readouterr().out
