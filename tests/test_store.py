"""SQLite store: idempotency, upsert, ledger and audit log."""

from invoice_agent.store import InvoiceStore


def test_register_run_is_idempotent_on_the_file_hash():
    store = InvoiceStore()
    assert store.register_run("r1", "hash-a", "a.pdf") == "r1"
    assert store.register_run("r2", "hash-a", "copy.pdf") == "r1"  # same bytes -> first run
    assert store.register_run("r3", "hash-b", "b.pdf") == "r3"


def test_save_is_an_upsert_and_only_approvals_join_the_ledger():
    store = InvoiceStore()
    store.register_run("r1", "h", "a.pdf")
    invoice = {"invoice_number": "INV-1", "total": 10.0, "currency": "USD"}
    store.save_invoice("r1", "rejected", invoice, [], {"vendor_id": "V002"}, ("V002", "INV-1"))
    assert store.seen_invoices() == frozenset()
    store.save_invoice("r1", "approved_by_reviewer", invoice, [], {"vendor_id": "V002"}, ("V002", "INV-1"))
    assert store.count_invoices() == 1
    assert store.get_invoice("r1")["status"] == "approved_by_reviewer"
    assert ("V002", "INV-1") in store.seen_invoices()


def test_seed_ledger_normalizes_numbers():
    store = InvoiceStore()
    store.seed_ledger([{"vendor_id": "V001", "invoice_number": "nw 0098"}])
    assert store.seen_invoices() == frozenset({("V001", "NW0098")})


def test_audit_log_is_ordered_per_run(tmp_path):
    path = str(tmp_path / "a.db")
    store = InvoiceStore(path)
    store.audit("r1", "submitted", filename="a.pdf")
    store.audit("r2", "submitted", filename="b.pdf")
    store.audit("r1", "saved", status="auto_approved")
    store.close()
    reopened = InvoiceStore(path)  # survives a restart
    assert [e["event"] for e in reopened.audit_log("r1")] == ["submitted", "saved"]
    assert reopened.audit_log("r1")[0]["filename"] == "a.pdf"
