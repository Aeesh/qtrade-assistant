import pytest

from src.generation.answer_generator import GeneratedAnswer, generate_answer
from src.llm import LLMProvider

from src.ingestion.document_loader import DocumentChunk
from src.retrieval.vector_store import RetrievedChunk


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------

class _MockProvider(LLMProvider):
    """Returns a fixed string as if the LLM generated it."""

    def __init__(self, response: str) -> None:
        self._response = response

    def complete(self, system: str, user: str) -> str:
        return self._response


def _make_chunk(text: str, source_doc: str = "Returns & Refunds") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id="test::0",
            source_doc=source_doc,
            source_file="returns_and_refunds",
            text=text,
            char_start=0,
        ),
        score=0.85,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_generate_returns_grounded_answer():
    provider = _MockProvider("You can return it with a 15% fee. [Source: Returns & Refunds]")
    chunks = [_make_chunk("Opened items incur a 15% restocking fee.")]
    result = generate_answer("Can I return an opened item?", chunks, provider)

    assert isinstance(result, GeneratedAnswer)
    assert result.is_grounded is True
    assert "15%" in result.text
    assert "Returns & Refunds" in result.cited_docs


def test_generate_marks_escalation_response_as_ungrounded():
    provider = _MockProvider(
        "I don't have enough information in our help docs to answer that. [Escalate]"
    )
    chunks = [_make_chunk("Some unrelated text.")]
    result = generate_answer("Do you do bulk discounts?", chunks, provider)

    assert result.is_grounded is False


def test_empty_chunks_skips_llm_call():
    """With no retrieved chunks, the LLM must NOT be called."""

    class _FailProvider(LLMProvider):
        def complete(self, system, user):
            raise AssertionError("LLM was called with empty retrieved chunks")

    result = generate_answer("anything", [], _FailProvider())
    assert result.is_grounded is False
    assert "[Escalate]" in result.text


def test_cited_docs_includes_all_source_docs():
    provider = _MockProvider("Answer from multiple docs. [Source: Shipping]")
    chunks = [
        _make_chunk("Shipping takes 3–5 days.", source_doc="Shipping"),
        _make_chunk("Returns are 30 days.", source_doc="Returns & Refunds"),
    ]
    result = generate_answer("Tell me about shipping and returns.", chunks, provider)

    assert "Shipping" in result.cited_docs
    assert "Returns & Refunds" in result.cited_docs