from __future__ import annotations

import logging
import os
import re as _re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.retrieval.vector_store import RetrievedChunk
from src.llm import LLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the QTrade customer support assistant.
Answer questions ONLY using the provided context excerpts.
Rules:
1. If the context contains the answer, respond concisely and end with [Source: <doc name>].
2. If the message is a greeting or conversational (not a support question), respond warmly
   and ask how you can help. Do NOT add [Source:] or [Escalate] for these.
3. If the context does NOT contain enough information to answer a genuine support question,
   respond with exactly: "I don't have enough information in our help docs to answer that. [Escalate]"
4. Never invent facts not present in the context.
5. Be friendly, polite and professional."""


def _build_user_prompt(query: str, retrieved: list[RetrievedChunk]) -> str:
    """
      Build the user prompt for the LLM, including context from retrieved chunks.

      @param query: the customer question
      @param retrieved: list of RetrievedChunk objects, each containing a DocumentChunk and its similarity score

      @return: a string prompt to be sent to the LLM
    """

    # combine the retrieved chunks into a single context string, with each chunk prefixed by its source doc name
    context_blocks = "\n\n".join(
        f"[{r.chunk.source_doc}]\n{r.chunk.text}"
        for r in retrieved
    )
    return (
        f"Context:\n{context_blocks}\n\n"
        f"Customer question: {query}\n\n"
        f"Answer (with citation):"
    )


# ---------------------------------------------------------------------------
# Answer generator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    is_grounded: bool       # True when at least one chunk cleared the floor
    cited_docs: tuple[str, ...]


def generate_answer(
    query: str,
    retrieved: list[RetrievedChunk],
    provider: LLMProvider,
) -> GeneratedAnswer:
    """
    Generate a cited answer from *retrieved* chunks using *provider*.

    If retrieved is empty the function returns an ungrounded "I don't know"
    response without calling the LLM — saving a round-trip.
    """
    if not retrieved:
        return GeneratedAnswer(
            text=(
                "I don't have enough information in our help docs to answer that. "
                "[Escalate]"
            ),
            is_grounded=False,
            cited_docs=(),
        )

    user_prompt = _build_user_prompt(query, retrieved)
    raw_text = provider.complete(_SYSTEM_PROMPT, user_prompt)

    source_tags = _re.findall(r"\[Source:\s*([^\]]+)\]", raw_text, _re.IGNORECASE)
    cited_docs = tuple(dict.fromkeys(tag.strip() for tag in source_tags))

    is_grounded = "[Escalate]" not in raw_text

    return GeneratedAnswer(
        text=raw_text,
        is_grounded=is_grounded,
        cited_docs=cited_docs,
    )