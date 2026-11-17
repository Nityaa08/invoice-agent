"""Vendor matching with a small vector store.

The extracted vendor name is rarely an exact copy of the vendor master entry
("NORTHWIND OFFICE SUPPLY CO." vs "Northwind Office Supplies Pvt Ltd"). Exact
lookup misses these; an LLM judging "same company?" is slow and not
repeatable. Here both names become vectors and the closest vendor wins, with a
similarity score the rules can threshold.

Pieces, each swappable:
- `HashingEmbedder`: character 3-grams + words, hashed into a fixed-size
  vector. No model download, no API key, deterministic, about 1 ms. It
  captures spelling similarity, not meaning, which is what vendor names need.
- `VectorStore`: the two-method interface (`add`, `query`). `NumpyVectorStore`
  is an in-memory cosine-similarity store. A Chroma, pgvector or FAISS
  adapter would implement the same two methods.
- `VendorMatcher`: loads the vendor master into the store and answers
  "who is this?" with a `VendorMatch` (id, name, score).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .rules import VendorMatch
from .textguard import fold

# Words that say what kind of company it is, not which company.
_LEGAL_WORDS = {
    "pvt", "private", "ltd", "limited", "llp", "llc", "inc", "incorporated", "corp",
    "corporation", "co", "company", "gmbh", "ag", "pte", "plc", "sa", "bv", "the",
}


def normalize_vendor_name(name: str) -> str:
    """Fold lookalike letters, lower-case, '&' -> 'and', drop punctuation and legal-form words.

    Folding means "Orbit Fаcility" (Cyrillic а) still matches Orbit; the
    no_lookalike_characters rule then reports the odd letter to a person.
    """
    text = fold(name).lower().replace("&", " and ")
    words = re.findall(r"[a-z0-9]+", text)
    kept = [w for w in words if w not in _LEGAL_WORDS]
    return " ".join(kept or words)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Character n-gram hashing embedder (the "hashing trick").

    Each normalized name is split into padded character 3-grams plus whole
    words; each feature is hashed (stable blake2b, not Python's salted
    `hash`) into one of `dim` buckets. Vectors are L2-normalized, so a dot
    product is cosine similarity.
    """

    def __init__(self, dim: int = 1024, ngram: int = 3):
        self.dim = dim
        self.ngram = ngram

    def _features(self, text: str) -> list[str]:
        norm = normalize_vendor_name(text)
        feats = [f"w:{w}" for w in norm.split()]
        padded = f" {norm} "
        feats += [padded[i : i + self.ngram] for i in range(len(padded) - self.ngram + 1)]
        return feats

    def _bucket(self, feature: str) -> int:
        digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                out[row, self._bucket(feat)] += 1.0
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


@dataclass(frozen=True)
class Hit:
    id: str
    score: float
    metadata: dict[str, Any]


class VectorStore(Protocol):
    def add(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict[str, Any]]) -> None: ...

    def query(self, vector: np.ndarray, k: int = 3) -> list[Hit]: ...


class NumpyVectorStore:
    """In-memory store: one matrix of unit vectors, cosine similarity by dot product."""

    def __init__(self, dim: int):
        self._ids: list[str] = []
        self._meta: list[dict[str, Any]] = []
        self._matrix = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict[str, Any]]) -> None:
        if not (len(ids) == len(vectors) == len(metadatas)):
            raise ValueError("ids, vectors and metadatas must have the same length")
        self._ids.extend(ids)
        self._meta.extend(metadatas)
        self._matrix = np.vstack([self._matrix, vectors.astype(np.float32)])

    def query(self, vector: np.ndarray, k: int = 3) -> list[Hit]:
        if not self._ids:
            return []
        scores = self._matrix @ vector.astype(np.float32)
        top = np.argsort(-scores)[:k]
        return [Hit(self._ids[i], float(scores[i]), self._meta[i]) for i in top]


class VendorMatcher:
    """Find the closest vendor in the vendor master for an extracted name."""

    def __init__(self, vendors: list[dict[str, Any]], embedder: Embedder | None = None,
                 store: VectorStore | None = None):
        self.embedder = embedder or HashingEmbedder()
        self.store = store or NumpyVectorStore(self.embedder.dim)
        self.vendors = {v["vendor_id"]: v for v in vendors}
        if vendors:
            self.store.add(
                [v["vendor_id"] for v in vendors],
                self.embedder.embed([v["name"] for v in vendors]),
                [dict(v) for v in vendors],
            )

    @classmethod
    def from_file(cls, path: str | Path) -> "VendorMatcher":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def candidates(self, name: str, k: int = 3) -> list[Hit]:
        return self.store.query(self.embedder.embed([name])[0], k)

    def match(self, name: str) -> VendorMatch:
        hits = self.candidates(name, k=1)
        if not hits:
            return VendorMatch(None, None, 0.0)
        best = hits[0]
        return VendorMatch(best.id, best.metadata["name"], round(best.score, 4), best.metadata.get("tax_id"))
