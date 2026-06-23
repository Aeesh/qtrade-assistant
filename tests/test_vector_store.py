import pytest
from pathlib import Path

from src.ingestion.document_loader import load_chunks_from_directory
from src.retrieval.vector_store import QTradeVectorStore

DOCS_DIR = Path(__file__).parent.parent / "data" / "help-docs"


@pytest.fixture(scope="module")
def indexed_store(tmp_path_factory):
    """
    Build and index the store once for all tests in this module.
    Uses a temp ChromaDB directory so tests are fully isolated.
    """
    tmp_dir = str(tmp_path_factory.mktemp("chroma"))
    store = QTradeVectorStore(persist_dir=tmp_dir)
    chunks = load_chunks_from_directory(DOCS_DIR)
    store.index(chunks)
    return store


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def test_store_reports_correct_chunk_count(indexed_store):
    chunks = load_chunks_from_directory(DOCS_DIR)
    assert indexed_store.num_chunks == len(chunks)


def test_indexing_empty_list_raises(tmp_path):
    store = QTradeVectorStore(persist_dir=str(tmp_path / "chroma"))
    with pytest.raises(ValueError, match="empty"):
        store.index([])


def test_empty_collection_returns_empty_list(tmp_path):
    """Before indexing, retrieve should return [] not raise."""
    store = QTradeVectorStore(persist_dir=str(tmp_path / "chroma"))
    results = store.retrieve("any question")
    assert results == []


def test_upsert_is_idempotent(tmp_path):
    """Calling index() twice with the same chunks must not duplicate entries."""
    store = QTradeVectorStore(persist_dir=str(tmp_path / "chroma"))
    chunks = load_chunks_from_directory(DOCS_DIR)
    store.index(chunks)
    store.index(chunks)   # second call — upsert, not insert
    assert store.num_chunks == len(chunks)


# ---------------------------------------------------------------------------
# Retrieval — semantic relevance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected_source_file", [
    ("how do I return an opened item",       "returns_and_refunds"),
    ("reset the SmartHub device",            "smarthub_setup_and_troubleshooting"),
    ("when will my order arrive",            "shipping"),
    ("does warranty cover accidental damage","warranty"),
])
def test_top_result_is_from_expected_doc(indexed_store, query, expected_source_file):
    results = indexed_store.retrieve(query, top_k=5)
    assert results, f"Expected at least one result for: {query!r}"
    assert results[0].chunk.source_file == expected_source_file, (
        f"Top result for {query!r} was {results[0].chunk.source_file!r}, "
        f"expected {expected_source_file!r} (score={results[0].score:.3f})"
    )


def test_scores_are_descending(indexed_store):
    results = indexed_store.retrieve("shipping and delivery", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_scores_are_in_valid_range(indexed_store):
    results = indexed_store.retrieve("refund policy", top_k=5)
    for r in results:
        assert -1.0 <= r.score <= 1.0


def test_reset_clears_collection(tmp_path):
    store = QTradeVectorStore(persist_dir=str(tmp_path / "chroma"))
    chunks = load_chunks_from_directory(DOCS_DIR)
    store.index(chunks)
    assert store.num_chunks > 0
    store.reset()
    assert store.num_chunks == 0