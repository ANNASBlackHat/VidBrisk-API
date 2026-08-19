"""Google Gemini LLM Client wrapper."""

import json
import os
import re
from typing import Any, Optional
from google import genai
from google.genai import types

from pipeline.config import get_settings


class GeminiLLMClient:
    """Wrapper around Google GenAI client for text generation and structured JSON output."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        self.model = model or settings.GEMINI_MODEL or "gemini-2.5-flash"
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Please provide it in your .env file or environment."
                )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generates raw text response."""
        config = None
        if system_instruction:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return response.text.strip() if response.text else ""

    def generate_json(self, prompt: str, system_instruction: Optional[str] = None) -> Any:
        """Generates JSON and parses into Python dict/list."""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        text = response.text.strip() if response.text else ""
        if not text:
            raise ValueError("Empty response received from Gemini API.")

        # Clean potential markdown wrapping if returned
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)

        return json.loads(text)
