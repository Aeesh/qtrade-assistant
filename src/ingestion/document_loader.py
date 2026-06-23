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
    return txt_file.stem.replace("_", " ").title() # fallback to the file stem if the "Doc:" line is missing


def load_chunks_from_directory(docs_dir: str | Path) -> list[DocumentChunk]:
    """
    go through the dcs directory and read every .txt file, and return a flat list of
    DocumentChunk objects ready for embedding.

    Each chunk has:
      - chunk_id   — stable identifier usable as a vector store key
      - source_doc — display name for citations ("Returns & Refunds")
      - source_file — stem for filtering ("returns_and_refunds")
      - text        — the chunk content
      - char_start  — byte offset of source data for debugging
    """
    docs_path = Path(docs_dir)
    if not docs_path.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {docs_path}")

    all_chunks: list[DocumentChunk] = []

    # iterate through all .txt files in the directory, sorted for consistency
    for txt_file in sorted(docs_path.glob("*.txt")):
        raw = txt_file.read_text(encoding="utf-8")
        doc_name = _parse_doc_name(raw)
        file_stem = txt_file.stem

        # strip the metadata header lines before chunking
        body_lines = [
            line for line in raw.splitlines()
            if not line.startswith("Doc:")
        ]
        body = " ".join(body_lines).strip()

        raw_chunks = _chunk_by_sentences(body, CHUNK_SIZE, CHUNK_OVERLAP)

        char_cursor = 0
        # create DocumentChunk objects for each chunk and keep track of the character offset
        for idx, chunk_text in enumerate(raw_chunks):
            all_chunks.append(
                DocumentChunk(
                    chunk_id=f"{file_stem}::{idx}",
                    source_doc=doc_name,
                    source_file=file_stem,
                    text=chunk_text,
                    char_start=char_cursor,
                )
            )
            char_cursor += len(chunk_text)

    return all_chunks


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    docs = load_chunks_from_directory(sys.argv[1] if len(sys.argv) > 1 else "help_docs")
    for chunk in docs:
        print(f"[{chunk.chunk_id}] ({chunk.source_doc})\n  {chunk.text[:80]}…\n")
    print(f"Total chunks: {len(docs)}")