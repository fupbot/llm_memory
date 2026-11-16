from pathlib import Path

import pytest

from llm_memory_core.bundle import Bundle
from llm_memory_core.concepts.types import (
    EnvironmentObservation,
    FailureNote,
    TaskAttempt,
)
from llm_memory_core.search.vector_backend import (
    HashingEmbedder,
    InMemoryVectorStore,
    VectorSearchBackend,
    _chunk_body,
)


def _bundle(tmp_path):
    bundle = Bundle(tmp_path)
    bundle.write_concept(
        "f1.md",
        FailureNote(title="Blocked door", resource="sim://house",
                    body="Navigation to the lab failed: lab_door_west is blocked.\n"),
    )
    bundle.write_concept(
        "o1.md",
        EnvironmentObservation(title="Garage", resource="sim://house",
                               body="Saw a toolbox and a ladder in the garage.\n"),
    )
    bundle.write_concept(
        "t1.md", TaskAttempt(title="Kitchen trip", resource="sim://other",
                             tags=["nav"], body="Drove to the kitchen.\n"),
    )
    return bundle


def test_search_ranks_relevant_doc_first(tmp_path):
    backend = VectorSearchBackend(_bundle(tmp_path))
    backend.index()
    results = backend.search("where is the toolbox", k=3)
    assert results[0].path == Path("o1.md")
    assert results[0].score is not None
    assert results[0].score >= results[-1].score


def test_filters_by_type_resource_and_tags(tmp_path):
    backend = VectorSearchBackend(_bundle(tmp_path))
    backend.index()
    assert {r.path for r in backend.search("door", type_filter="Failure/Recovery Note")} == {
        Path("f1.md")
    }
    assert Path("t1.md") not in {r.path for r in backend.search("kitchen", resource="sim://house")}
    assert [r.path for r in backend.search("kitchen", tags=["nav"])] == [Path("t1.md")]


def test_index_is_incremental(tmp_path):
    bundle = _bundle(tmp_path)
    backend = VectorSearchBackend(bundle)
    first = backend.index()
    assert first.indexed_docs == 3

    second = backend.index()
    assert second.indexed_docs == 0 and second.unchanged_docs == 3

    bundle.write_concept("o1.md", EnvironmentObservation(body="The garage is empty now.\n"))
    (tmp_path / "t1.md").unlink()
    third = backend.index()
    assert third.indexed_docs == 1 and third.removed_docs == 1
    assert Path("t1.md") not in {r.path for r in backend.search("kitchen")}


def test_hashing_embedder_is_deterministic_and_normalized():
    emb = HashingEmbedder(dim=64)
    a, b = emb(["red mug in kitchen"]), emb(["red mug in kitchen"])
    assert a == b
    assert sum(v * v for v in a[0]) == pytest.approx(1.0)


def test_chunking_tracks_line_numbers():
    body = "para one\n\npara two\n\npara three\n"
    chunks = _chunk_body(body, max_chars=10, first_line=5)
    assert [c.text for c in chunks] == ["para one", "para two", "para three"]
    assert [c.line_number for c in chunks] == [5, 7, 9]


def test_chroma_store_roundtrip(tmp_path):
    pytest.importorskip("chromadb")
    from llm_memory_core.search.vector_backend import ChromaVectorStore

    backend = VectorSearchBackend(
        _bundle(tmp_path / "bundle"),
        store=ChromaVectorStore(tmp_path / "index", collection="test"),
    )
    backend.index()
    assert backend.search("toolbox", k=1)[0].path == Path("o1.md")
    # Re-opening the persisted index sees the same chunks, so nothing is re-embedded.
    reopened = VectorSearchBackend(
        backend.bundle, store=ChromaVectorStore(tmp_path / "index", collection="test")
    )
    assert reopened.index().unchanged_docs == 3


def test_in_memory_store_handles_and_filter():
    store = InMemoryVectorStore()
    store.upsert(["a", "b"], [[1.0, 0.0], [1.0, 0.0]], ["A", "B"],
                 [{"type": "x", "resource": "r"}, {"type": "x", "resource": "s"}])
    hits = store.query([1.0, 0.0], 5, {"$and": [{"type": "x"}, {"resource": "s"}]})
    assert [h.id for h in hits] == ["b"]
