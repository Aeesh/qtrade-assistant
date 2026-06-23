from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...

