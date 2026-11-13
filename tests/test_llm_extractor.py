"""The real-LLM extractor, with the HTTP call faked (no network, no key).

This goes through LangChain's ChatOpenAI and `with_structured_output(Invoice)`
for real, so it checks that the Pydantic schema is sent to the provider and
that its answer is parsed back."""

import json

import httpx
import pytest
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from invoice_agent.extractors import ExtractionError, LLMExtractor

GOOD = {
    "vendor_name": "Bluepeak Cloud Services Inc", "vendor_tax_id": None, "invoice_number": "BP-1",
    "invoice_date": "2026-10-05", "due_date": None, "currency": "USD", "po_number": None,
    "line_items": [{"description": "Compute", "quantity": 1, "unit_price": 10, "amount": 10}],
    "subtotal": 10, "tax": 0, "total": 10,
}


def fake_openai(replies, requests):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        reply = replies.pop(0)
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": {"message": "boom"}})
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": reply}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    return ChatOpenAI(model="m", api_key="test-key", base_url="http://llm.test/v1", max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_schema_is_sent_and_answer_parsed():
    requests = []
    extractor = LLMExtractor("m", "test-key", chat_model=fake_openai([json.dumps(GOOD)], requests))
    out = extractor.extract("INVOICE ...")
    assert out["invoice_number"] == "BP-1" and out["total"] == 10.0
    fmt = requests[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert "line_items" in fmt["json_schema"]["schema"]["properties"]


def test_schema_violation_surfaces_as_validation_error():
    # Valid JSON, wrong shape: the graph catches this, records it and retries.
    bad = dict(GOOD)
    del bad["invoice_number"]
    extractor = LLMExtractor("m", "k", chat_model=fake_openai([json.dumps(bad)], []))
    with pytest.raises(ValidationError, match="invoice_number"):
        extractor.extract("INVOICE ...")


def test_feedback_is_sent_back_to_the_model():
    requests = []
    extractor = LLMExtractor("m", "k", chat_model=fake_openai([json.dumps(GOOD)], requests))
    extractor.extract("INVOICE ...", feedback="invoice_number: Field required")
    last = requests[0]["messages"][-1]["content"]
    assert "invoice_number: Field required" in last


def test_invalid_json_is_a_retryable_schema_error():
    extractor = LLMExtractor("m", "k", chat_model=fake_openai(["this is not json"], []))
    with pytest.raises((ValidationError, ExtractionError)):
        extractor.extract("INVOICE ...")


def test_http_errors_raise_extraction_error():
    extractor = LLMExtractor("m", "k", chat_model=fake_openai([500], []))
    with pytest.raises(ExtractionError, match="model call failed"):
        extractor.extract("INVOICE ...")
