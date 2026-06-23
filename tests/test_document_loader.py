import pytest
from pathlib import Path

from src.ingestion.document_loader import (
    DocumentChunk,
    _chunk_by_sentences,
    load_chunks_from_directory,
)

DOCS_DIR = Path(__file__).parent.parent / "data" / "help-docs"


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------

def test_short_text_produces_single_chunk():
    text = "This is a short sentence."
    chunks = _chunk_by_sentences(text, size=300, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_produces_multiple_chunks():
    # 10 sentences of ~40 chars each → ~400 chars total; size=150 → ≥2 chunks
    text = " ".join(["This is sentence number %d." % i for i in range(10)])
    chunks = _chunk_by_sentences(text, size=150, overlap=30)
    assert len(chunks) >= 2


def test_chunks_are_non_empty():
    text = "First sentence. Second sentence. Third sentence."
    chunks = _chunk_by_sentences(text, size=50, overlap=10)
    assert all(c.strip() for c in chunks)


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def test_load_chunks_returns_list_of_document_chunks():
    chunks = load_chunks_from_directory(DOCS_DIR)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(c, DocumentChunk) for c in chunks)


def test_all_chunks_have_required_fields():
    chunks = load_chunks_from_directory(DOCS_DIR)
    for chunk in chunks:
        assert chunk.chunk_id, "chunk_id must be non-empty"
        assert chunk.source_doc, "source_doc must be non-empty"
        assert chunk.source_file, "source_file must be non-empty"
        assert chunk.text.strip(), "text must be non-empty"


def test_chunk_ids_are_unique():
    chunks = load_chunks_from_directory(DOCS_DIR)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be globally unique"


def test_expected_docs_are_loaded():
    chunks = load_chunks_from_directory(DOCS_DIR)
    source_files = {c.source_file for c in chunks}
    expected = {"returns_and_refunds", "shipping", "smarthub_setup_and_troubleshooting", "warranty"}
    assert expected <= source_files


def test_missing_directory_raises():
    with pytest.raises(FileNotFoundError):
        load_chunks_from_directory("/tmp/this_dir_does_not_exist_qtrade")