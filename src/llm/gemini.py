from __future__ import annotations

from google import genai

from src.llm import LLMProvider

import os
import time
import logging

logger = logging.getLogger(__name__)

class GeminiProvider(LLMProvider):
    """
    Calls the Gemini API (free tier, no credit card needed for basic use).
    Set GEMINI_API_KEY in the .env file.
    """

    def __init__(
        self,
        model: str = "models/gemini-3.1-flash-lite",
        api_key: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. "
                "Get a free key at https://aistudio.google.com"
            )
        self._model_name = model
        self._client = genai.Client(
            api_key=self._api_key
        )


    def complete(self, system: str, user: str) -> str:
        for attempt in range(3):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=user,
                    config={
                        "system_instruction": system,
                    },
                )

                if not response.text:
                    raise ValueError("Gemini returned empty response")

                return response.text.strip()

            except Exception as e:
                logger.warning(
                    "Gemini request failed (attempt %s/3): %s",
                    attempt + 1,
                    e,
                )

                if attempt == 2:
                    raise

                time.sleep(2 ** attempt)

