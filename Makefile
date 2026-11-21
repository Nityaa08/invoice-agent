# Shortcuts for the commands in the README. Nothing here is needed on Vercel.
PY ?= python

.PHONY: install test run demo mcp mcp-http mcp-demo eval eval-all compare

install:          ## dev dependencies (tests, MCP server, PDF generators)
	$(PY) -m pip install -r requirements-dev.txt

test:             ## the whole suite, no network or key needed
	$(PY) -m pytest

run:              ## full service: SQLite in ./var, reviews survive restarts
	$(PY) -m uvicorn main:app --reload

demo:             ## the stateless demo, exactly as on Vercel
	VERCEL=1 $(PY) -m uvicorn main:app --port 8765

mcp:              ## MCP server over stdio (what Claude Desktop runs)
	$(PY) -m invoice_agent.mcp_server

mcp-http:         ## MCP server over streamable HTTP on 127.0.0.1:8770
	$(PY) -m invoice_agent.mcp_server --http

mcp-demo:         ## start the MCP server over stdio and call every tool
	$(PY) scripts/mcp_client_demo.py

eval:             ## the samples
	$(PY) evals/run_eval.py --markdown

eval-all:         ## every labelled set
	for s in samples realworld blind blind2 blind3 qa redteam; do $(PY) evals/run_eval.py --set $$s --markdown; echo; done

compare:          ## demo extractor vs the model on every set (needs GROQ_API_KEY or LLM_API_KEY)
	$(PY) evals/run_eval.py --compare --set all
