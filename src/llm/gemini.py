from __future__ import annotations

from src.llm.basellm import LLMProvider


class GeminiProvider(LLMProvider):
    """
    Calls the Gemini API (free tier, no credit card needed for basic use).
    Set GEMINI_API_KEY in the .env file.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. "
                "Get a free key at https://aistudio.google.com"
            )

    def complete(self, system: str, user: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
        }
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return (
                data["candidates"][0]["content"]["parts"][0]["text"].strip()
            )
        except (requests.RequestException, KeyError) as exc:
            logger.error("Gemini request failed: %s", exc)
            raise

