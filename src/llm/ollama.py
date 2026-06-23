from __future__ import annotations

from src.llm.basellm import LLMProvider

import requests
import logging


logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """
    Calls a locally running Ollama instance.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            raise

