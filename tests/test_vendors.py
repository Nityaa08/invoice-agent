"""The vendor vector store: embedder, store and matcher."""

import numpy as np
import pytest

from invoice_agent.rules import VENDOR_MATCH_THRESHOLD
from invoice_agent.vendors import HashingEmbedder, NumpyVectorStore, VendorMatcher, normalize_vendor_name

from .conftest import ROOT


@pytest.fixture(scope="module")
def matcher():
    return VendorMatcher.from_file(ROOT / "data" / "vendors.json")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Northwind Office Supplies Pvt. Ltd.", "northwind office supplies"),
        ("Tidewater Print & Signage Ltd", "tidewater print and signage"),
        ("The Company Ltd", "the company ltd"),  # all legal words: keep them rather than return ""
    ],
)
def test_normalize_vendor_name(raw, expected):
    assert normalize_vendor_name(raw) == expected


def test_embedder_is_deterministic_and_unit_length():
    emb = HashingEmbedder(dim=256)
    a, b = emb.embed(["Kestrel Logistics GmbH", "Kestrel Logistics GmbH"])
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)


@pytest.mark.parametrize(
    "printed, vendor_id",
    [
        ("Northwind Office Supplies Pvt Ltd", "V001"),  # exact
        ("NORTHWIND OFFICE SUPPLY CO.", "V001"),  # case, plural, legal form
        ("Bluepeak Cloud Services, Inc.", "V002"),  # punctuation
        ("Kestrel Logistik GmbH", "V003"),  # one-letter typo
        ("Tidewater Print and Signage Limited", "V005"),  # & vs and
    ],
)
def test_matches_known_vendors_above_threshold(matcher, printed, vendor_id):
    match = matcher.match(printed)
    assert match.vendor_id == vendor_id
    assert match.score >= VENDOR_MATCH_THRESHOLD
    assert match.matched


@pytest.mark.parametrize("printed", ["Quillmark Stationers", "Acme Corp", "Northwind Stationery"])
def test_unknown_or_ambiguous_vendors_stay_below_threshold(matcher, printed):
    assert not matcher.match(printed).matched


def test_candidates_are_ranked(matcher):
    hits = matcher.candidates("Saffron Catering", k=3)
    assert hits[0].id == "V004"
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_store_contract():
    store = NumpyVectorStore(dim=2)
    assert store.query(np.array([1.0, 0.0])) == []
    store.add(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]]), [{"n": 1}, {"n": 2}])
    hit = store.query(np.array([0.0, 1.0]), k=1)[0]
    assert (hit.id, hit.score, hit.metadata) == ("b", 1.0, {"n": 2})
    with pytest.raises(ValueError):
        store.add(["c"], np.zeros((2, 2)), [{}])


def test_empty_vendor_master():
    assert VendorMatcher([]).match("Anyone").vendor_id is None
