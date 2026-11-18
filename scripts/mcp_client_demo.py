"""Talk to the invoice-agent MCP server the way Claude Desktop does: start it over stdio and call its tools.

    python scripts/mcp_client_demo.py

Uses a throwaway database in a temporary folder, so it does not touch ./var.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import anyio
from mcp import Client
from mcp.client.stdio import StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent


def show(title: str, result) -> None:
    print(f"\n>>> {title}")
    if result.is_error:
        print("    error:", result.content[0].text)
        return
    data = result.structured_content or {}
    keep = {k: data[k] for k in ("status", "needs_person", "reasons", "decision", "count", "vendor") if k in data}
    if "failed_rules" in data:
        keep["failed_rules"] = [r["name"] for r in data["failed_rules"]]
    print("   ", json.dumps(keep, indent=2, ensure_ascii=False).replace("\n", "\n    "))


async def main() -> None:
    tmp = tempfile.mkdtemp(prefix="invoice-mcp-")
    env = dict(os.environ, DB_PATH=f"{tmp}/invoices.db", CHECKPOINT_PATH=f"{tmp}/checkpoints.db",
               AS_OF_DATE="2026-11-13", PYTHONPATH=str(ROOT), MCP_LOG_LEVEL="WARNING")
    params = StdioServerParameters(command=sys.executable, args=["-m", "invoice_agent.mcp_server"], env=env, cwd=str(ROOT))
    async with Client(params) as client:
        tools = await client.list_tools()
        print("tools:", ", ".join(t.name for t in tools.tools))
        show("process_invoice(sample='01_northwind_ok.pdf')",
             await client.call_tool("process_invoice", {"sample": "01_northwind_ok.pdf"}))
        show("process_invoice(sample='04_saffron_wrong_total.pdf')",
             await client.call_tool("process_invoice", {"sample": "04_saffron_wrong_total.pdf"}))
        text = (ROOT / "samples" / "06_unknown_vendor.txt").read_text()
        show("process_invoice(text=<unknown vendor>)", await client.call_tool("process_invoice", {"text": text}))
        show("add_vendor('NORTHWIND OFFICE SUPPLY', 'INR')  # shadowing a known vendor",
             await client.call_tool("add_vendor", {"name": "NORTHWIND OFFICE SUPPLY", "currency": "INR"}))
        show("add_vendor('Quillmark Stationers', 'INR', '33AAQFQ9876P1Z4')",
             await client.call_tool("add_vendor", {"name": "Quillmark Stationers", "currency": "INR",
                                                   "tax_id": "33AAQFQ9876P1Z4"}))
        extracted = await client.call_tool("extract_invoice", {"text": text})
        show("validate_invoice(<the same invoice, vendor now known>)",
             await client.call_tool("validate_invoice", {"invoice": extracted.structured_content["invoice"]}))


if __name__ == "__main__":
    anyio.run(main)
