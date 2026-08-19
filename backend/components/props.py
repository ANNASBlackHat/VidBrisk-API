"""LLM-assisted and heuristic prop extractors for motion graphics components."""

import json
import re
from typing import Any, Optional
from pydantic import BaseModel, Field
from pipeline.llm.gemini import GeminiLLMClient


class StatCardProps(BaseModel):
    """Structured props for DataAnimations/StatCard."""
    value: str = Field(..., description="Key numerical or metric value, e.g. '$25B' or '650M+' or '4%'")
    label: str = Field(..., description="Short headline label, e.g. 'Project Apollo Total Cost'")
    subtext: Optional[str] = Field(default=None, description="Optional secondary context or supporting figure")


class QuoteCardProps(BaseModel):
    """Structured props for TextAnimations/QuoteCard."""
    quote: str = Field(..., description="The main quote or abstract takeaway sentence")
    emphasis: Optional[str] = Field(default=None, description="Key phrase or punchline to highlight/bold")
    author: Optional[str] = Field(default=None, description="Speaker or attribution if mentioned, otherwise None")


class TitleProps(BaseModel):
    """Structured props for TextAnimations/Typewriter or KineticTitle."""
    text: str
    variant: str = "kinetic"


def extract_stat_props(text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Extracts concise, structured statistical props from narrative text via LLM with fallback."""
    if not text or not text.strip():
        return {"value": "0", "label": "", "subtext": None}

    llm = client or GeminiLLMClient()
    prompt = (
        "You are an expert motion graphics designer. "
        "Extract the core numerical statistic, concise label, and optional subtext from this text "
        "to display on an eye-catching video data callout card.\n\n"
        f"Input Text:\n\"{text.strip()}\"\n\n"
        "Return ONLY a JSON object matching this schema:\n"
        "{\n"
        "  \"value\": \"<short impactful number/metric, e.g. $25B, 650M, 4%>\",\n"
        "  \"label\": \"<concise label 2-5 words>\",\n"
        "  \"subtext\": \"<optional supporting detail or context, or null>\"\n"
        "}"
    )

    try:
        data = llm.generate_json(prompt=prompt, schema=StatCardProps)
        if isinstance(data, dict) and "value" in data and "label" in data:
            return {
                "value": str(data["value"]),
                "label": str(data["label"]),
                "subtext": data.get("subtext"),
            }
    except Exception:
        pass

    # Heuristic fallback if LLM is unavailable
    # Extract numbers or currency like $25 billion, 4 percent, 650 million
    match = re.search(r"(\$?\d+(?:\.\d+)?\s*(?:billion|million|thousand|percent|%|k|m|b)?|\d+%)", text, re.IGNORECASE)
    value = match.group(1).upper() if match else "DATA"
    return {
        "value": value,
        "label": text[:40] + ("..." if len(text) > 40 else ""),
        "subtext": None,
    }


def extract_quote_props(text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Extracts structured quote and emphasis phrase from narrative text."""
    if not text or not text.strip():
        return {"quote": "", "emphasis": None, "author": None}

    llm = client or GeminiLLMClient()
    prompt = (
        "Extract the main quote sentence and identify the single most impactful phrase "
        "to emphasize in bold motion typography.\n\n"
        f"Input Text:\n\"{text.strip()}\"\n\n"
        "Return ONLY a JSON object matching this schema:\n"
        "{\n"
        "  \"quote\": \"<full sentence>\",\n"
        "  \"emphasis\": \"<3-6 words to highlight>\",\n"
        "  \"author\": \"<speaker name if present, else null>\"\n"
        "}"
    )

    try:
        data = llm.generate_json(prompt=prompt, schema=QuoteCardProps)
        if isinstance(data, dict) and "quote" in data:
            return {
                "quote": str(data["quote"]),
                "emphasis": data.get("emphasis"),
                "author": data.get("author"),
            }
    except Exception:
        pass

    return {
        "quote": text.strip(),
        "emphasis": text.split(",")[-1].strip() if "," in text else text[:30],
        "author": None,
    }


def extract_title_props(text: str) -> dict[str, Any]:
    """Extracts typewriter / kinetic title props."""
    return {
        "text": text.strip(),
        "variant": "kinetic",
    }
