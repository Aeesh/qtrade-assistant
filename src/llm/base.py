from __future__ import annotations

from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...

