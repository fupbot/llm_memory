"""Vector (semantic) search over an OKF bundle: the retrieval half of the agent's RAG
memory. See requirements.md's Search section.

The OKF bundle stays the single source of truth. The vector index is a derived,
disposable cache: `index()` brings it up to date with the bundle incrementally (by
content hash), and deleting the index directory loses nothing that a re-index can't
rebuild. So memory stays portable, git-diffable markdown, and semantic search sits on
top of it.

Dependencies are optional and imported lazily, so `llm_memory_core` itself keeps
needing only PyYAML:
- `ChromaVectorStore` / `MiniLMEmbedder` need `chromadb` (pip). MiniLM runs locally
  via ONNX; it downloads about 80 MB on first use and needs no API key.
- `InMemoryVectorStore` / `HashingEmbedder` are pure Python. They're for tests and
  offline use; the hashing embedder matches words, not meaning.
"""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .. import frontmatter
from ..bundle import Bundle
from .base import SearchBackend, SearchResult


# --------------------------------------------------------------------------- embedders


class Embedder(Protocol):
    name: str

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic feature-hashing embedder over word unigrams and bigrams. No model
    download and no network, so tests are reproducible. It only matches overlapping
    words (lexical), so use MiniLMEmbedder for real semantic retrieval."""

    name = "hashing-v1"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return words + [f"{a} {b}" for a, b in zip(words, words[1:])]

    def __call__(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for feat in self._features(text):
                digest = hashlib.md5(feat.encode()).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                vec[idx] += 1.0 if digest[4] & 1 else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 sentence embeddings (384-d) via chromadb's bundled ONNX
    runtime. Runs locally on CPU."""

    name = "all-MiniLM-L6-v2"

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]


# ------------------------------------------------------------------------ vector stores


@dataclass
class VectorHit:
    id: str
    document: str
    metadata: dict
    similarity: float


class VectorStore(ABC):
    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...

    @abstractmethod
    def query(
        self, embedding: list[float], k: int, where: dict | None = None
    ) -> list[VectorHit]: ...

    @abstractmethod
    def all_metadata(self) -> dict[str, dict]:
        """{id: metadata} for every stored chunk, used for incremental re-indexing."""


class InMemoryVectorStore(VectorStore):
    """Brute-force cosine similarity in pure Python. Fine for the few thousand chunks
    one robot's memory produces; not meant for more than that."""

    def __init__(self):
        self._rows: dict[str, tuple[list[float], str, dict]] = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, e, d, m in zip(ids, embeddings, documents, metadatas):
            self._rows[i] = (e, d, m)

    def delete(self, ids):
        for i in ids:
            self._rows.pop(i, None)

    def query(self, embedding, k, where=None):
        if where and "$and" in where:  # Chroma-style conjunction -> flat equality map
            where = {key: val for cond in where["$and"] for key, val in cond.items()}
        hits = []
        for id_, (emb, doc, meta) in self._rows.items():
            if where and any(meta.get(key) != val for key, val in where.items()):
                continue
            sim = sum(a * b for a, b in zip(embedding, emb))
            hits.append(VectorHit(id_, doc, meta, sim))
        hits.sort(key=lambda h: h.similarity, reverse=True)
        return hits[:k]

    def all_metadata(self):
        return {i: m for i, (_, _, m) in self._rows.items()}


class ChromaVectorStore(VectorStore):
    """Chroma collection with cosine distance. `persist_dir=None` gives an ephemeral
    in-process client. Otherwise the index survives restarts in that directory, which
    should sit outside the bundle (it's a cache, not memory)."""

    def __init__(self, persist_dir: str | Path | None = None, collection: str = "okf_bundle"):
        import chromadb

        if persist_dir is None:
            self._client = chromadb.EphemeralClient()
        else:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        # embedding_function=None: embeddings are always computed by our Embedder and
        # passed in explicitly, so the store never silently re-embeds with a
        # different model than the one used at query time.
        self._col = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def upsert(self, ids, embeddings, documents, metadatas):
        if ids:
            self._col.upsert(
                ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
            )

    def delete(self, ids):
        if ids:
            self._col.delete(ids=ids)

    def query(self, embedding, k, where=None):
        n = self._col.count()
        if n == 0:
            return []
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=min(k, n),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            VectorHit(i, d, m, 1.0 - dist)
            for i, d, m, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def all_metadata(self):
        res = self._col.get(include=["metadatas"])
        return dict(zip(res["ids"], res["metadatas"]))


# ------------------------------------------------------------------------- the backend


@dataclass
class IndexStats:
    indexed_docs: int = 0
    unchanged_docs: int = 0
    removed_docs: int = 0
    chunks_written: int = 0


@dataclass
class _Chunk:
    text: str
    line_number: int


def _chunk_body(body: str, max_chars: int, first_line: int) -> list[_Chunk]:
    """Split a markdown body on blank lines into chunks of at most ~max_chars,
    remembering each chunk's starting line number in the original file."""
    chunks: list[_Chunk] = []
    current: list[str] = []
    current_start = first_line
    size = 0
    line_no = first_line
    for para in re.split(r"(\n\s*\n)", body):
        if not para.strip():
            line_no += para.count("\n")
            continue
        if current and size + len(para) > max_chars:
            chunks.append(_Chunk("\n\n".join(current), current_start))
            current, size = [], 0
        if not current:
            current_start = line_no
        current.append(para.strip())
        size += len(para)
        line_no += para.count("\n")
    if current:
        chunks.append(_Chunk("\n\n".join(current), current_start))
    return chunks


class VectorSearchBackend(SearchBackend):
    """Semantic search over a bundle's concept documents.

    Each chunk is embedded together with its concept's title, type and description,
    so a short body like "door was closed" still retrieves with its context.
    """

    def __init__(
        self,
        bundle: Bundle,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        *,
        max_chunk_chars: int = 1200,
    ):
        self.bundle = bundle
        self.store = store or InMemoryVectorStore()
        self.embedder = embedder or HashingEmbedder()
        self.max_chunk_chars = max_chunk_chars

    def _doc_chunks(self, rel_path: Path, text: str) -> tuple[dict, list[_Chunk]]:
        data, body = frontmatter.parse(text)
        # Body starts after the closing '---' plus the blank line serialize() adds.
        header_lines = text[: len(text) - len(body)].count("\n") + 1
        chunks = _chunk_body(body, self.max_chunk_chars, header_lines)
        if not chunks:
            chunks = [_Chunk("", header_lines)]
        prefix_parts = [f"[{data.get('type', 'Concept')}] {data.get('title') or rel_path.stem}"]
        if data.get("description"):
            prefix_parts.append(str(data["description"]))
        prefix = "\n".join(prefix_parts)
        return data, [_Chunk(f"{prefix}\n{c.text}".strip(), c.line_number) for c in chunks]

    def index(self) -> IndexStats:
        """Bring the vector index up to date with the bundle. Only documents whose
        content hash changed are re-embedded; chunks of deleted documents are removed."""
        stats = IndexStats()
        existing = self.store.all_metadata()
        by_path: dict[str, list[str]] = {}
        hash_by_path: dict[str, str] = {}
        for id_, meta in existing.items():
            by_path.setdefault(meta["path"], []).append(id_)
            hash_by_path[meta["path"]] = meta.get("sha", "")

        seen: set[str] = set()
        ids, docs, metas = [], [], []
        for rel_path in self.bundle.walk():
            key = rel_path.as_posix()
            seen.add(key)
            text = (self.bundle.root / rel_path).read_text(encoding="utf-8")
            sha = hashlib.sha1(text.encode()).hexdigest()
            if hash_by_path.get(key) == sha:
                stats.unchanged_docs += 1
                continue
            try:
                data, chunks = self._doc_chunks(rel_path, text)
            except frontmatter.FrontmatterError:
                continue
            self.store.delete(by_path.get(key, []))
            stats.indexed_docs += 1
            for n, chunk in enumerate(chunks):
                meta = {
                    "path": key,
                    "chunk": n,
                    "line": chunk.line_number,
                    "sha": sha,
                    "type": str(data.get("type", "")),
                    "title": str(data.get("title") or ""),
                    "resource": str(data.get("resource") or ""),
                    "tags": "," + ",".join(map(str, data.get("tags") or [])) + ",",
                }
                ids.append(f"{key}#{n}")
                docs.append(chunk.text)
                metas.append(meta)

        if docs:
            self.store.upsert(ids, self.embedder(docs), docs, metas)
            stats.chunks_written = len(docs)

        for path, chunk_ids in by_path.items():
            if path not in seen:
                self.store.delete(chunk_ids)
                stats.removed_docs += 1
        return stats

    def search(
        self,
        query: str,
        *,
        type_filter: str | None = None,
        tags: list[str] | None = None,
        k: int = 8,
        resource: str | None = None,
    ) -> list[SearchResult]:
        where: dict = {}
        if type_filter is not None:
            where["type"] = type_filter
        if resource is not None:
            where["resource"] = resource
        if len(where) > 1:
            where = {"$and": [{key: val} for key, val in where.items()]}
        # Chroma can't filter on substrings of metadata, so tag filtering happens
        # after retrieval, over-fetching to compensate.
        fetch = k * 4 if tags else k
        hits = self.store.query(self.embedder([query])[0], fetch, where or None)
        if tags:
            hits = [h for h in hits if any(f",{t}," in h.metadata.get("tags", "") for t in tags)]
        return [
            SearchResult(
                Path(h.metadata["path"]),
                int(h.metadata.get("line", 0)),
                h.document,
                score=round(h.similarity, 4),
            )
            for h in hits[:k]
        ]
