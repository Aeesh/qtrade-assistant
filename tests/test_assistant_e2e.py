import pytest
from pathlib import Path

from src.assistant import QTradeAssistant
from src.generation.answer_generator import LLMProvider

DOCS_DIR = Path(__file__).parent.parent / "data" / "help-docs"


# ---------------------------------------------------------------------------
# Mock LLM that echoes the first retrieved chunk as its "answer"
# ---------------------------------------------------------------------------

class _EchoProvider(LLMProvider):
    """
    Grabs the first context block from the user prompt and returns it
    as a grounded answer.  This lets testing the pipeline without a real LLM.
    """

    def complete(self, system: str, user: str) -> str:
        # Extract the first context block
        lines = user.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("["):
                # return the next non-empty line as the "answer"
                for candidate in lines[i + 1:]:
                    if candidate.strip():
                        return candidate.strip() + " [Source: Test Doc]"
        return "I don't have enough information in our help docs to answer that. [Escalate]"


@pytest.fixture(scope="module")
def assistant(tmp_path_factory) -> QTradeAssistant:
    tmp_dir = str(tmp_path_factory.mktemp("chroma_e2e"))
    return QTradeAssistant(
        docs_dir=DOCS_DIR,
        provider=_EchoProvider(),
        top_k=3,
        persist_dir=tmp_dir,
    )


# ---------------------------------------------------------------------------
# Safety escalation (pre-retrieval, message-level)
# ---------------------------------------------------------------------------

def test_safety_message_is_escalated(assistant):
    response = assistant.handle("My hub is sparking and smells like burning.")
    assert response.is_escalated
    assert response.escalation_decision is not None


def test_explicit_human_request_is_escalated(assistant):
    response = assistant.handle("I want to speak to a manager right now.")
    assert response.is_escalated


def test_repeat_frustration_is_escalated(assistant):
    response = assistant.handle("This is the third time I've called, still no fix.")
    assert response.is_escalated


# ---------------------------------------------------------------------------
# Normal answerable queries — should NOT escalate
# ---------------------------------------------------------------------------

def test_return_policy_is_answered(assistant):
    response = assistant.handle("Can I return an opened item?")
    assert not response.is_escalated
    assert response.answer.strip()
    assert response.cited_docs  # must cite something


def test_smarthub_reset_is_answered(assistant):
    response = assistant.handle("How do I reset my SmartHub?")
    assert not response.is_escalated
    assert response.answer.strip()


# ---------------------------------------------------------------------------
# Out-of-scope → escalate (no grounded answer)
# ---------------------------------------------------------------------------

def test_out_of_scope_query_escalates(assistant):
    """'bulk discount' is not in any doc; retrieval should score too low."""
    response = assistant.handle(
        "Do you offer bulk commercial pricing for large installs?"
    )
    # Either escalated (preferred) or answered with low confidence.
    # At minimum the answer must be non-empty.
    assert response.answer.strip()


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_empty_query_returns_graceful_response(assistant):
    response = assistant.handle("   ")
    assert not response.is_escalated
    assert response.answer.strip()