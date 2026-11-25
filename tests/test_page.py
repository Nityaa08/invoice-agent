"""The demo page: structure that the features rely on (the page itself is exercised in a browser
by the Playwright demo script; these checks keep the wiring from silently breaking)."""

import re

from .conftest import ROOT

HTML = (ROOT / "invoice_agent" / "web" / "index.html").read_text()
SCRIPT = re.search(r"<script>(.*?)</script>", HTML, re.S).group(1)


def test_one_inline_script_and_no_inline_handlers():
    assert len(re.findall(r"<script", HTML)) == 1  # the CSP allows exactly one, by hash
    assert not re.search(r"\son[a-z]+=", HTML)  # no onclick= etc.: they would need 'unsafe-inline'


def test_how_it_works_diagram_is_static_svg():
    how = re.search(r'<details class="how".*?</details>', HTML, re.S).group(0)
    assert "<svg" in how and "<script" not in how
    for node in ("ingest", "extract", "match_vendor", "validate", "auto_approve", "human_review", "save"):
        assert f">{node}</text>" in how


def test_attack_samples_have_their_own_group():
    assert 'id="attacks"' in HTML and 'x.group === "attack"' in SCRIPT


def test_shortcuts_are_listed_and_ignored_while_typing():
    for key in ("j", "k", "a", "r", "c", "v", "u", "b", "?"):
        assert f"<kbd>{key}</kbd>" in HTML
    assert 'e.target.closest("input, textarea, select, [contenteditable]")' in SCRIPT
    assert "e.ctrlKey || e.metaKey || e.altKey" in SCRIPT  # browser shortcuts are left alone


def test_batch_upload_and_export_controls():
    assert re.search(r'<input type="file" id="file" multiple', HTML)
    for id_ in ("runall", "export-csv", "export-json", "queue"):
        assert f'id="{id_}"' in HTML


def test_storage_access_is_guarded():
    for line in SCRIPT.splitlines():
        if "localStorage." in line:
            assert "try" in line, line  # a private window can throw; the page must keep working


def test_queue_totals_keep_two_decimals():
    # "Run all samples" showed "USD 1,861.9": a bare toLocaleString() drops trailing zeros.
    line = next(ln for ln in SCRIPT.splitlines() if "inv.vendor_name} · ${inv.currency}" in ln)
    assert "minimumFractionDigits: 2" in line
