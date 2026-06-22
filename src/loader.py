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

