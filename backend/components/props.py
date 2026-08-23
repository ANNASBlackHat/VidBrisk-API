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
    numeric_value: Optional[float] = Field(default=None, description="Extracted raw float for count-up animation if applicable")
    unit: Optional[str] = Field(default=None, description="Prefix/suffix unit e.g. '$', '%', 'B', 'M'")


class QuoteCardProps(BaseModel):
    """Structured props for TextAnimations/QuoteCard."""
    quote: str = Field(..., description="The main quote or abstract takeaway sentence")
    emphasis: Optional[str] = Field(default=None, description="Key phrase or punchline to highlight/bold")
    author: Optional[str] = Field(default=None, description="Speaker or attribution if mentioned, otherwise None")


class ListCardProps(BaseModel):
    """Structured props for ListAnimations/SwipeDeck or list reveal components."""
    items: list[str] = Field(..., description="Short, punchy, standalone facts or points (under 12 words)")


class TitleProps(BaseModel):
    """Structured props for TextAnimations/Typewriter or KineticTitle."""
    text: str
    variant: str = "kinetic"


def extract_stat_props(text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Extracts concise, structured statistical props from narrative text via LLM with fallback."""
    if not text or not text.strip():
        return {"value": "0", "label": "", "subtext": None, "numeric_value": 0.0, "unit": ""}

    llm = client or GeminiLLMClient()
    prompt = (
        "You are an expert motion graphics designer. "
        "Extract the core numerical statistic, concise label, and optional subtext from this text "
        "to display on an eye-catching video data callout card with full-duration count-up animation.\n\n"
        f"Input Text:\n\"{text.strip()}\"\n\n"
        "Return ONLY a JSON object matching this schema:\n"
        "{\n"
        "  \"value\": \"<short impactful number/metric, e.g. $25B, 650M, 4%>\",\n"
        "  \"label\": \"<concise label 2-5 words>\",\n"
        "  \"subtext\": \"<optional supporting detail or context, or null>\",\n"
        "  \"numeric_value\": <float number for count-up, e.g. 25 or 650 or 4, or null>,\n"
        "  \"unit\": \"<unit string e.g. $, %, B, M or null>\"\n"
        "}"
    )

    try:
        data = llm.generate_json(prompt=prompt, schema=StatCardProps)
        if isinstance(data, dict) and "value" in data and "label" in data:
            return {
                "value": str(data["value"]),
                "label": str(data["label"]),
                "subtext": data.get("subtext"),
                "numeric_value": data.get("numeric_value"),
                "unit": data.get("unit"),
            }
    except Exception:
        pass

    # Heuristic fallback if LLM is unavailable
    # Extract numbers or currency like $25 billion, 4 percent, 650 million
    match = re.search(r"(\$?\d+(?:\.\d+)?\s*(?:billion|million|thousand|percent|%|k|m|b)?|\d+%)", text, re.IGNORECASE)
    value = match.group(1).upper() if match else "DATA"
    num_match = re.search(r"(\d+(?:\.\d+)?)", value)
    num_val = float(num_match.group(1)) if num_match else None

    return {
        "value": value,
        "label": text[:40] + ("..." if len(text) > 40 else ""),
        "subtext": None,
        "numeric_value": num_val,
        "unit": "$" if "$" in value else ("%" if "%" in value else ""),
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


def extract_list_props(text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Splits narrative text into a short list of discrete, on-screen-displayable items."""
    if not text or not text.strip():
        return {"items": []}

    llm = client or GeminiLLMClient()
    prompt = (
        "Break this narration into 3-5 short, punchy, standalone facts or points "
        "suitable for displaying one at a time on video cards. Each item must be "
        "under 12 words and make sense read in isolation.\n\n"
        f"Input Text:\n\"{text.strip()}\"\n\n"
        "Return ONLY a JSON object: {\"items\": [\"<item 1>\", \"<item 2>\", ...]}"
    )
    try:
        data = llm.generate_json(prompt=prompt, schema=ListCardProps)
        if isinstance(data, dict) and isinstance(data.get("items"), list) and data["items"]:
            items = [str(i).strip() for i in data["items"] if str(i).strip()]
            if items:
                return {"items": items[:5]}
    except Exception:
        pass

    # Deterministic fallback: split on sentence boundaries, cap at 5, trim length
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return {"items": sentences[:5] if sentences else [text.strip()[:60]]}


def extract_chat_props(text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Same extraction as extract_list_props, reshaped into alternating chat messages."""
    base = extract_list_props(text, client)
    messages = [
        {"text": item, "sender": "system" if i % 2 == 0 else "user"}
        for i, item in enumerate(base.get("items", []))
    ]
    return {"messages": messages}

