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

    def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        schema: Optional[Any] = None,
        max_retries: int = 3,
    ) -> Any:
        """Generates JSON and parses into Python dict/list with retries for malformed output."""
        config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "temperature": 0.1,
        }
        if schema:
            config_kwargs["response_schema"] = schema
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_kwargs)
        if system_instruction:
            config.system_instruction = system_instruction

        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

            text = response.text.strip() if response.text else ""
            if not text:
                last_error = ValueError("Empty response received from Gemini API.")
                continue

            # Clean potential markdown wrapping if returned
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
            text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
            text = text.strip()

            # Attempt direct parse
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                last_error = e
                # Try to repair common issues: trailing commas, single quotes, unescaped newlines inside strings
                repaired = re.sub(r",\s*([}\]])", r"\1", text)
                # Remove stray trailing content after final closing brace
                # Find last complete JSON object
                try:
                    # Try to extract the outermost JSON object via balancing braces
                    start = text.find("{")
                    end = text.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        candidate = text[start : end + 1]
                        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                        return json.loads(candidate)
                except Exception:
                    pass
                if attempt < max_retries - 1:
                    # Slightly increase temperature for diversity on retry
                    config.temperature = min(0.4, 0.1 + 0.15 * (attempt + 1))
                    continue
                # Final attempt: try repaired version
                try:
                    return json.loads(repaired)
                except Exception:
                    pass

        if last_error:
            raise last_error
        raise ValueError("Failed to parse JSON from Gemini after retries.")
