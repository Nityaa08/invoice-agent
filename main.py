"""Entry point: `uvicorn main:app --reload`.

Vercel's FastAPI preset looks for a top-level `app = FastAPI(...)` in main.py
(or app.py / index.py / server.py), so the app is created right here rather
than imported from a factory. DEMO_MODE only decides which routes it gets.
"""

from fastapi import FastAPI

from invoice_agent.api import register_routes
from invoice_agent.config import load_settings

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
