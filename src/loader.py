from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DocumentChunk:
    """A single chunk of text extracted from a help doc."""

    chunk_id: str          # e.g. "returns_and_refunds::0"
    source_doc: str        # name of the doc for display, e.g. "Returns & Refunds"
    source_file: str       # name of the source file for filtering, e.g. "returns_and_refunds"
    text: str
    char_start: int        # offset within the original doc to help with debugging


# ---------------------------------------------------------------------------
# Chunking strategy
# ---------------------------------------------------------------------------

CHUNK_SIZE = 300        # target size for each chunk, in characters (not tokens)
CHUNK_OVERLAP = 50      # how many characters to keep as "overlap" between chunks to preserve context


def _chunk_by_sentences(text: str, size: int, overlap: int) -> list[str]:
    """
        This function splits the input text into chunks of approximately size characters,
        trying to split on sentence boundaries (., !, ?) when possible.
        It also ensures that there is an overlap of characters between consecutive chunks to preserve context.

        For example, if a chunk ends at character 300, the next chunk will start at character 250
        to include the last 50 characters of the previous chunk.

        @param text: The input text to be chunked.
        @param size: The target size for each chunk in characters.
        @param overlap: The number of characters to overlap between consecutive chunks.
        @return: A list of text chunks.
    """

    # Split the text into sentences using regex that looks for punctuation followed by whitespace.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # initialise variables to build chunks
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    # Iterate over sentences and build chunks while respecting the size and overlap constraints.
    for sentence in sentences:
        sentence_len = len(sentence)
        # If the length of the current chunk and the next sentence exceeds the target chunk size,
        # store the current chunk and start a new chunk with the sentence.
        if current_len + sentence_len > size and current:
            chunks.append(" ".join(current))
            # keep the tail for overlap
            tail: list[str] = []
            tail_len = 0
            # Walk backward through the current chunk and collect whole
            # sentences until adding another sentence would exceed the
            # overlap limit.
            for s in reversed(current):
                if tail_len + len(s) <= overlap:
                    tail.insert(0, s)
                    tail_len += len(s)
                else:
                    break
            current = tail
            current_len = tail_len
        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _parse_doc_name(raw_text: str) -> str:
    """
        Extract the document name from the first line of the raw text, which should start with "Doc:".
        For example, if the first line is "Doc: Returns & Refunds", this function will return "Returns & Refunds".
            @param raw_text: The raw text of the document, which should contain a line starting with "Doc:".
            @return: The extracted document name for display purposes.
    """
    first_line = raw_text.strip().splitlines()[0]
    if first_line.startswith("Doc:"):
        return first_line[4:].strip()
    return first_line

