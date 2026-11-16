"""Entry point: `uvicorn main:app --reload`.

Vercel's FastAPI preset looks for a top-level `app = FastAPI(...)` in main.py
(or app.py / index.py / server.py), so the app is created right here rather
than imported from a factory. DEMO_MODE only decides which routes it gets.
"""

import logging

from fastapi import FastAPI

from invoice_agent.api import register_routes
from invoice_agent.config import load_settings

# INFO lines (what kind of file each demo run got, why it went to review) show
# up in the Vercel runtime logs; no file contents or names are logged.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
settings = load_settings()

app = FastAPI(
    title="invoice-agent",
    description=(
        "A LangGraph agent that turns invoices into validated, structured data. "
        "The model reads; tested code decides."
    ),
    version="0.1.0",
)

register_routes(app, settings)
