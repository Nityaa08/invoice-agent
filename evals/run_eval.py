"""Evaluate the agent on the labelled sample invoices.

Reports two things:
- field-level accuracy: each extracted field vs the hand label in
  samples/manifest.json (strings compared after trimming; amounts to the cent)
- decision accuracy: did the graph auto-approve or stop for review as the
  label says, and did exactly the expected rules fail?

Demo mode (default, free, offline):
    python evals/run_eval.py

With a real model (any OpenAI-compatible API; Groq has a free tier):
    EXTRACTOR=llm LLM_API_KEY=... LLM_MODEL=openai/gpt-oss-20b python evals/run_eval.py
    # other providers: also set LLM_BASE_URL, e.g. https://api.openai.com/v1

Two sets:
    --set samples    the eleven synthetic samples in samples/ (default)
    --set realworld  the real-world-style PDFs in tests/fixtures/pdfs/
                     (GST, US, EU, multi-page, prose, scanned; labels.json)
    --set blind      three more PDFs written after the extractor was frozen
                     (blind_labels.json); the honest number
    --set qa         ten files from the 2026-11-17 QA pass (tests/fixtures/qa/),
                     mostly known vendors, so a correct read is auto-approved
    --set redteam    nine attack files (tests/fixtures/redteam/); each must go
                     to review, failed by the rule that defends against it

Add --markdown to print a table for the README, --out FILE to save JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from invoice_agent.config import DEMO_AS_OF, SAMPLES_DIR, load_settings  # noqa: E402
from invoice_agent.service import InvoiceService  # noqa: E402

FIELDS = ["vendor_name", "vendor_tax_id", "invoice_number", "invoice_date", "due_date", "currency",
          "po_number", "line_count", "subtotal", "tax", "total"]
MONEY = {"subtotal", "tax", "total"}


def same(field: str, got, want) -> bool:
    if field in MONEY:
        return got is not None and want is not None and round(float(got), 2) == round(float(want), 2)
    if isinstance(got, str):
        got = got.strip()
    return got == want


SETS = {
    # name: (folder, labels file, key holding the list)
    "samples": (SAMPLES_DIR, SAMPLES_DIR / "manifest.json", "samples"),
    "realworld": (ROOT / "tests" / "fixtures" / "pdfs", ROOT / "tests" / "fixtures" / "pdfs" / "labels.json", "invoices"),
    "blind": (ROOT / "tests" / "fixtures" / "pdfs", ROOT / "tests" / "fixtures" / "pdfs" / "blind_labels.json", "invoices"),
    "qa": (ROOT / "tests" / "fixtures" / "qa", ROOT / "tests" / "fixtures" / "qa" / "labels.json", "invoices"),
    "redteam": (ROOT / "tests" / "fixtures" / "redteam", ROOT / "tests" / "fixtures" / "redteam" / "labels.json", "invoices"),
}


def evaluate(which: str = "samples") -> dict:
    folder, labels_path, key = SETS[which]
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    # The invoices are dated Oct/Nov 2026; check them as of a fixed day so they do not go stale.
    # AS_OF_DATE in the environment still wins. Nothing is written back to os.environ,
    # so evaluating one set does not change the date used for the next.
    as_of = os.environ.get("AS_OF_DATE") or labels.get("as_of", DEMO_AS_OF)
    settings = replace(load_settings(), as_of=date.fromisoformat(as_of), as_of_from_env=True)
    manifest = labels[key]
    rows, field_hits, field_total = [], {f: 0 for f in FIELDS}, 0

    for sample in manifest:
        service = InvoiceService.ephemeral(settings)  # fresh ledger per invoice: no cross-talk
        started = time.perf_counter()
        run, _ = service.submit(sample["file"], (folder / sample["file"]).read_bytes())
        seconds = time.perf_counter() - started

        decision = "auto_approve" if run["status"] == "auto_approved" else "review"
        failed = sorted(r["name"] for r in run["rules"] if not r["passed"])
        extraction_failed = run["invoice"] is None
        row = {
            "file": sample["file"],
            "expected_decision": sample["decision"],
            "decision": decision,
            "decision_ok": decision == sample["decision"],
            "expected_failed_rules": sorted(sample["failed_rules"]),
            "failed_rules": failed,
            "rules_ok": failed == sorted(sample["failed_rules"])
                        and extraction_failed == bool(sample.get("extraction_fails")),
            "extraction_failed": extraction_failed,
            "attempts": run["attempts"],
            "seconds": round(seconds, 3),
            "field_errors": {},
        }
        # Field accuracy only where an invoice was produced and one was expected.
        if not extraction_failed and not sample.get("extraction_fails") and sample.get("expected"):
            got = dict(run["invoice"], line_count=len(run["invoice"]["line_items"]))
            field_total += 1
            for f in FIELDS:
                if same(f, got.get(f), sample["expected"][f]):
                    field_hits[f] += 1
                else:
                    row["field_errors"][f] = {"got": got.get(f), "want": sample["expected"][f]}
        rows.append(row)

    n = len(rows)
    all_fields = sum(field_hits.values())
    return {
        "set": which,
        "extractor": InvoiceService.ephemeral(settings).deps.extractor.name,
        "as_of": settings.as_of.isoformat() if settings.as_of else None,
        "invoices": n,
        "field_accuracy": round(all_fields / (field_total * len(FIELDS)), 4) if field_total else None,
        "field_accuracy_by_field": {f: round(h / field_total, 4) for f, h in field_hits.items()} if field_total else {},
        "invoices_scored_for_fields": field_total,
        "decision_accuracy": round(sum(r["decision_ok"] for r in rows) / n, 4),
        "failed_rules_exact_match": round(sum(r["rules_ok"] for r in rows) / n, 4),
        "sent_to_review": sum(r["decision"] == "review" for r in rows),
        "mean_seconds_per_invoice": round(sum(r["seconds"] for r in rows) / n, 3),
        "rows": rows,
    }


def markdown(report: dict) -> str:
    lines = [
        f"Set `{report['set']}`, extractor `{report['extractor']}`, {report['invoices']} invoices, dates as of {report['as_of']}.",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Field accuracy ({len(FIELDS)} fields x {report['invoices_scored_for_fields']} invoices) | "
        + (f"{report['field_accuracy']:.1%} |" if report["field_accuracy"] is not None else "not scored |"),
        f"| Decision accuracy (auto-approve vs review) | {report['decision_accuracy']:.1%} |",
        f"| Exactly the expected rules failed | {report['failed_rules_exact_match']:.1%} |",
        f"| Sent to review | {report['sent_to_review']} of {report['invoices']} |",
        f"| Mean time per invoice | {report['mean_seconds_per_invoice'] * 1000:.0f} ms |",
        "",
        "| Invoice | Expected | Got | Failed rules | Field errors |",
        "|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        failed = ", ".join(r["failed_rules"]) or ("extraction failed" if r["extraction_failed"] else "none")
        errors = ", ".join(r["field_errors"]) or "none"
        mark = "" if r["decision_ok"] and r["rules_ok"] else " (wrong)"
        lines.append(f"| `{r['file']}` | {r['expected_decision']} | {r['decision']}{mark} | {failed} | {errors} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", choices=sorted(SETS), default="samples", help="which labelled set to run")
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    parser.add_argument("--out", help="write the full JSON report here")
    args = parser.parse_args()
    report = evaluate(args.set)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown:
        print(markdown(report))
    else:
        summary = {k: v for k, v in report.items() if k != "rows"}
        print(json.dumps(summary, indent=2))
        for r in report["rows"]:
            if not (r["decision_ok"] and r["rules_ok"]) or r["field_errors"]:
                print("MISS", json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
