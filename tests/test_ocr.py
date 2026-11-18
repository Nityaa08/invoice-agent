"""OCR: photos are read in the browser; the server never runs OCR unless SERVER_OCR is set,
and text that came from OCR always goes to a person."""

import re
import sys
import types

import pytest

from invoice_agent import api, ingest
from invoice_agent.graph import OCR_CHECK

from .conftest import ROOT
from .test_api import make_client

PHOTO = ROOT / "tests" / "fixtures" / "qa" / "qa_photo.png"
TEXT = (ROOT / "samples" / "02_bluepeak_ok.txt").read_bytes()  # what the browser's OCR would send


@pytest.fixture
def demo(monkeypatch, tmp_path):
    monkeypatch.delenv("SERVER_OCR", raising=False)
    return make_client(monkeypatch, tmp_path, demo=True)


@pytest.fixture
def full(monkeypatch, tmp_path):
    monkeypatch.delenv("SERVER_OCR", raising=False)
    return make_client(monkeypatch, tmp_path, demo=False)


def failed(run):
    return [r["name"] for r in run["rules"] if not r["passed"]]


def test_the_server_does_not_ocr_images_by_default(monkeypatch):
    monkeypatch.delenv("SERVER_OCR", raising=False)
    # Even where pytesseract happens to be installed, the honest default is no server OCR.
    with pytest.raises(ingest.IngestError, match=r"server does not run OCR.*reads photos in your browser"):
        ingest.to_text("photo.png", PHOTO.read_bytes())


def test_ocr_text_from_the_browser_always_goes_to_review(demo):
    run = demo.post("/api/demo/run", files={"file": ("photo.ocr.txt", TEXT, "text/plain")},
                    data={"source": "ocr"}).json()
    assert run["source"] == "ocr" and run["status"] == "needs_review"
    assert failed(run) == [OCR_CHECK]  # everything else passed: only the OCR check stops it
    assert "read by OCR, please check" in run["review_request"]["reasons"][0]
    # The same text without the flag is an ordinary text upload and is approved.
    plain = demo.post("/api/demo/run", files={"file": ("a.txt", TEXT, "text/plain")}).json()
    assert plain["status"] == "auto_approved" and plain["source"] is None


@pytest.mark.parametrize("decision, status, ocr_passes", [
    ({"action": "approve", "note": "checked against the photo"}, "approved_by_reviewer", False),
    ({"action": "recheck"}, "needs_review", False),  # a re-check is not a person checking the fields
    ({"action": "reject"}, "rejected", False),
])
def test_reviewer_decisions_on_ocr_text(demo, decision, status, ocr_passes):
    import json

    run = demo.post("/api/demo/run", files={"file": ("p.ocr.txt", TEXT, "text/plain")},
                    data={"source": "ocr", "decisions": json.dumps([decision])}).json()
    assert run["status"] == status
    assert (OCR_CHECK not in failed(run)) == ocr_passes


def test_a_correction_clears_the_ocr_check(demo):
    import json

    first = demo.post("/api/demo/run", files={"file": ("p.ocr.txt", TEXT, "text/plain")}, data={"source": "ocr"}).json()
    fix = {"action": "correct", "invoice": first["invoice"], "reviewer": "asha"}
    done = demo.post("/api/demo/run", files={"file": ("p.ocr.txt", TEXT, "text/plain")},
                     data={"source": "ocr", "decisions": json.dumps([fix])}).json()
    assert done["status"] == "approved_after_correction"
    check = next(r for r in done["rules"] if r["name"] == OCR_CHECK)
    assert check["passed"] and "checked the fields against the image" in check["detail"]


def test_full_service_takes_the_flag_too(full):
    run = full.post("/invoices", files={"file": ("p.ocr.txt", TEXT, "text/plain")}, data={"source": "ocr"}).json()
    assert run["status"] == "needs_review" and failed(run) == [OCR_CHECK]


@pytest.mark.parametrize("value", ["OCR", "server", "<script>", "ocr "])
def test_unknown_source_values_are_refused(demo, value):
    r = demo.post("/api/demo/run", data={"sample": "02_bluepeak_ok.txt", "source": value})
    assert r.status_code == 422 and "source" in r.json()["detail"]


def test_server_ocr_when_switched_on_is_flagged_as_ocr(monkeypatch, demo):
    """SERVER_OCR=true (local use) runs pytesseract; its text gets the same OCR check."""
    monkeypatch.setenv("SERVER_OCR", "true")
    fake_tess = types.SimpleNamespace(image_to_string=lambda img: TEXT.decode())
    fake_pil = types.ModuleType("PIL")
    fake_pil.Image = types.SimpleNamespace(open=lambda f: object(), MAX_IMAGE_PIXELS=None)
    monkeypatch.setitem(sys.modules, "pytesseract", fake_tess)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    run = demo.post("/api/demo/run", files={"file": ("photo.png", PHOTO.read_bytes(), "image/png")}).json()
    assert run["source"] == "ocr" and failed(run) == [OCR_CHECK]


def test_page_loads_the_pinned_ocr_library_with_sri_and_the_csp_allows_only_that():
    html = (ROOT / "invoice_agent" / "web" / "index.html").read_text()
    script = re.search(r'script: "([^"]+)"', html).group(1)
    integrity = re.search(r'integrity: "([^"]+)"', html).group(1)
    assert script == "https://cdn.jsdelivr.net/npm/tesseract.js@7.0.0/dist/tesseract.min.js"
    assert integrity.startswith("sha384-") and len(integrity) == 71
    for key in ("workerPath", "corePath", "langPath"):
        url = re.search(rf'{key}: "([^"]+)"', html).group(1)
        allowed = api.OCR_SCRIPT_SOURCES + (api.OCR_DATA_SOURCE,)
        assert any((url + "/").startswith(src) for src in allowed), url
        assert "@" in url.split("/npm/")[1]  # every path carries a pinned version
    # The library is loaded lazily, only for photos: no <script src> in the page itself.
    assert "<script src" not in html


def test_scanned_pdf_message_points_at_the_photo_path():
    assert "photo or screenshot" in ingest.SCANNED_PDF_MESSAGE and "does not run OCR" in ingest.SCANNED_PDF_MESSAGE
