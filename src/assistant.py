"""

QTradeAssistant — the interface for the RAG pipeline.

Pipeline per query:
  1. Escalation pre-check (message-level safety/pattern rules)
  2. Retrieval (semantic search over indexed chunks)
  3. Escalation post-check (no-grounded-answer check)
  4. Generation (LLM answers using retrieved context)
  5. Return a structured AssistantResponse
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.escalation.rules import EscalationDecision, EscalationTrigger, evaluate_escalation
from src.generation.answer_generator import GeneratedAnswer, generate_answer
from src.llm.basellm import LLMProvider
from src.ingestion.document_loader import load_chunks_from_directory
from src.retrieval.vector_store import QTradeVectorStore, RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

@dataclass
class AssistantResponse:
    """Structured output for every query processed by QTradeAssistant."""

    query: str
    answer: str
    is_escalated: bool
    escalation_decision: EscalationDecision | None = None
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    cited_docs: tuple[str, ...] = ()

    def __str__(self) -> str:
        """Human-readable string for CLI / logging output."""
        lines = [
            f"\n{'='*60}",
            f"Query : {self.query}",
            f"{'='*60}",
        ]
        if self.is_escalated:
            lines.append("⚠  ESCALATED TO HUMAN AGENT")
            if self.escalation_decision:
                lines.append(
                    f"   Reason : {self.escalation_decision.reason_summary}"
                )
            lines.append(f"\n{self.answer}")
        else:
            lines.append(self.answer)
            if self.cited_docs:
                lines.append(f"\n📄 Sources: {', '.join(self.cited_docs)}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Handoff summary (stretch: auto-generated for escalated queries)
# ---------------------------------------------------------------------------

_HANDOFF_TEMPLATE = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HANDOFF SUMMARY — QTrade Support Bot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Customer asked  : {query}
Escalation why  : {reason}
Triggers fired  : {triggers}
Context found   : {context_summary}
Recommended action: {action}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


def _build_handoff_summary(
    query: str,
    decision: EscalationDecision,
    retrieved: list[RetrievedChunk],
) -> str:
    """
      Build a structured handoff summary for human agents when escalation is triggered.
    """
    context_summary = (
        "; ".join(f'"{r.chunk.source_doc}" (score {r.score:.2f})' for r in retrieved)
        if retrieved else "no relevant docs found"
    )
    trigger_names = ", ".join(t.name for t in decision.triggers)

    action = "unknown"
    if EscalationTrigger.SAFETY_HAZARD in decision.triggers:
        action = "Route to safety/hardware team immediately"
    elif EscalationTrigger.EXPLICIT_HUMAN_REQ in decision.triggers:
        action = "Route to live agent — customer requested human contact"
    elif EscalationTrigger.REPEAT_FRUSTRATION in decision.triggers:
        action = "Route to senior support — customer has contacted multiple times"
    elif EscalationTrigger.NO_GROUNDED_ANSWER in decision.triggers:
        action = "Route to support — question outside documented help topics"

    return _HANDOFF_TEMPLATE.format(
        query=query,
        reason=decision.reason_summary,
        triggers=trigger_names,
        context_summary=context_summary,
        action=action,
    )


# ---------------------------------------------------------------------------
# Main assistant class
# ---------------------------------------------------------------------------

class QTradeAssistant:
    """
    Orchestrates the full RAG pipeline for QTrade customer support.

    @param docs_dir    : path to the directory of .txt help doc files
    @param provider    : an LLMProvider instance (OllamaProvider or GeminiProvider)
    @param top_k       : how many chunks to retrieve per query (default: 5)
    @param persist_dir : ChromaDB persistence directory (default: ./chroma_db)
    """

    def __init__(
        self,
        docs_dir: str | Path,
        provider: LLMProvider,
        top_k: int = 5,
        persist_dir: str = "./chroma_db",
    ) -> None:
        self._provider = provider
        self._top_k = top_k
        self._store = QTradeVectorStore(persist_dir=persist_dir)

        # Build index at startup
        chunks = load_chunks_from_directory(docs_dir)
        self._store.index(chunks)
        logger.info(
            "QTradeAssistant ready — %d chunks indexed from %s",
            self._store.num_chunks,
            docs_dir,
        )

    # ------------------------------------------------------------------
    # Interface for callers (CLI, API, tests)
    # ------------------------------------------------------------------

    def handle(self, query: str) -> AssistantResponse:
        """
        Process a single customer query end-to-end.

        Returns an AssistantResponse regardless of whether the query is
        answered, escalated, or unresolvable.
        """
        query = query.strip()
        if not query:
            return AssistantResponse(
                query=query,
                answer="Please type your question and I'll do my best to help.",
                is_escalated=False,
            )

        # Step 1 — pre-check message for escalation signals
        pre_decision = evaluate_escalation(query, retrieval_succeeded=True)
        if pre_decision.should_escalate:
            retrieved = self._store.retrieve(query, top_k=self._top_k)
            handoff = _build_handoff_summary(query, pre_decision, retrieved)
            return AssistantResponse(
                query=query,
                answer=(
                    "I'm sorry to hear about this. I'm immediately connecting you "
                    "with a human support agent who can help you properly.\n\n"
                    + handoff
                ),
                is_escalated=True,
                escalation_decision=pre_decision,
                retrieved_chunks=retrieved,
            )

        # Step 2 — retrieve relevant chunks
        retrieved = self._store.retrieve(query, top_k=self._top_k)

        # Step 3 — post-check: ChromaDB returned nothing useful?
        # An empty list means the collection had no results — genuinely out of scope.
        post_decision = evaluate_escalation(query, retrieval_succeeded=bool(retrieved))
        if post_decision.should_escalate:
            handoff = _build_handoff_summary(query, post_decision, retrieved)
            return AssistantResponse(
                query=query,
                answer=(
                    "I wasn't able to find an answer to that in our help docs. "
                    "Let me connect you with a support agent.\n\n" + handoff
                ),
                is_escalated=True,
                escalation_decision=post_decision,
                retrieved_chunks=retrieved,
            )

        # Step 4 — generate cited answer
        generated: GeneratedAnswer = generate_answer(query, retrieved, self._provider)

        # Step 5 — if LLM itself says it doesn't know, escalate
        if not generated.is_grounded:
            no_answer_decision = evaluate_escalation(
                query, retrieval_succeeded=False
            )
            handoff = _build_handoff_summary(query, no_answer_decision, retrieved)
            return AssistantResponse(
                query=query,
                answer=(
                    "I don't have enough information in our help docs to answer that. "
                    "Let me connect you with a support agent.\n\n" + handoff
                ),
                is_escalated=True,
                escalation_decision=no_answer_decision,
                retrieved_chunks=retrieved,
            )

        return AssistantResponse(
            query=query,
            answer=generated.text,
            is_escalated=False,
            retrieved_chunks=retrieved,
            cited_docs=generated.cited_docs,
        )