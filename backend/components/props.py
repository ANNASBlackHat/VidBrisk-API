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


class MapExplainerProps(BaseModel):
    """Structured props for GeoAnimations/MapExplainer."""
    origin: str = Field(..., description="Origin location or city name")
    destination: Optional[str] = Field(default=None, description="Destination location or city name if route")
    title: Optional[str] = Field(default=None, description="Short title header")
    subtext: Optional[str] = Field(default=None, description="Short telemetry category")
    mode: str = Field(default="route", description="'route' for travel between 2 locations, 'pin' for single location")


def extract_map_props(text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Extracts geographic route or pin properties from narrative text."""
    if not text or not text.strip():
        return {"origin": "Cape Canaveral", "destination": "Pacific Ocean", "mode": "route"}

    # Common location detection heuristics
    locs = [
        "Cape Canaveral", "Houston", "New York", "Washington DC",
        "London", "Paris", "Berlin", "Moscow", "Baikonur",
        "Tokyo", "Beijing", "Sydney", "Cairo", "Pacific Ocean", "Atlantic Ocean"
    ]
    found = [loc for loc in locs if re.search(r"\b" + re.escape(loc) + r"\b", text, re.IGNORECASE)]

    if len(found) >= 2:
        return {
            "origin": found[0],
            "destination": found[1],
            "mode": "route",
            "title": f"{found[0]} → {found[1]}",
            "subtext": "TRAJECTORY ROUTE",
        }
    elif len(found) == 1:
        return {
            "origin": found[0],
            "destination": None,
            "mode": "pin",
            "title": found[0].upper(),
            "subtext": "COORDINATE LOCK",
        }

    # Default fallback
    return {
        "origin": "Cape Canaveral",
        "destination": "Pacific Ocean",
        "mode": "route",
        "title": "MISSION TRAJECTORY",
        "subtext": "FLIGHT PATH",
    }


class AudioWaveformProps(BaseModel):
    """Structured props for AudioAnimations/AudioWaveform."""
    speaker: str = Field(default="VOICE COMM", description="Speaker name or callsign")
    title: Optional[str] = Field(default="AUDIO FEED", description="Header stream title")
    subtext: Optional[str] = Field(default="LIVE TELEMETRY", description="Category tag")
    quote: Optional[str] = Field(default=None, description="Spoken transcript or quotation")


def extract_waveform_props(text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Extracts speaker attribution and quote for an audio waveform visualizer."""
    if not text or not text.strip():
        return {
            "speaker": "VOICE COMM",
            "title": "TRANSMISSION FEED",
            "subtext": "AUDIO STREAM",
            "quote": None,
        }

    clean = text.strip()
    # Check for speaker prefix like "Speaker Name: Quote" or "Name (Title): Quote"
    m = re.match(r"^([A-Za-z0-9\s\.\(\)\-]+):\s*[\"']?(.*?)[\"']?$", clean)
    if m:
        speaker = m.group(1).strip().upper()
        quote = m.group(2).strip()
        return {
            "speaker": speaker,
            "title": "TRANSMISSION LOG",
            "subtext": "VOICE COMM",
            "quote": quote if quote else None,
        }

    return {
        "speaker": "AUDIO LOG",
        "title": "VOICE RECORDING",
        "subtext": "TELEMETRY FEED",
        "quote": clean if len(clean) < 140 else clean[:137] + "...",
    }


class KineticCaptionsProps(BaseModel):
    """Structured props for TextAnimations/KineticCaptions."""
    text: str = Field(..., description="Full spoken transcript to animate word-by-word")
    themeColor: Optional[str] = Field(default="#facc15", description="Accent highlight color")
    styleVariant: Optional[str] = Field(default="bouncy", description="Animation style variant ('bouncy' | 'karaoke')")


def extract_captions_props(text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Extracts caption props for word-by-word kinetic text animation."""
    return {
        "text": text.strip() if text else "",
        "themeColor": "#facc15",
        "styleVariant": "bouncy",
    }


# =========================================================================
# Documentary & Archival Components ("Dead Reckoning" style)
# =========================================================================

class DocumentViewerProps(BaseModel):
    """Structured props for Archival/DocumentViewer."""
    imageUrl: Optional[str] = Field(default=None, description="Path or URL to archival document scan")
    title: str = Field(default="ARCHIVAL DOCUMENT", description="Document label or archive record title")
    subtext: Optional[str] = Field(default="CONTEMPORARY SOURCE", description="Subtext or date")
    highlightText: Optional[str] = Field(default=None, description="Key phrase highlighted in document")
    zoom: float = Field(default=1.15, description="Ken Burns zoom scale factor")


def extract_document_props(text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Extracts archival document inspection props, including highlight phrases."""
    clean = text.strip() if text else ""
    # Extract highlight phrase if in quotes
    highlight = None
    m_quote = re.search(r"['\"]([^'\"]+)['\"]", clean)
    if m_quote:
        highlight = m_quote.group(1).strip()
    elif "highlight" in clean.lower():
        parts = re.split(r"highlight\s*(?:phrase|text|on)?\s*[:\-]?\s*", clean, flags=re.IGNORECASE)
        if len(parts) > 1:
            highlight = parts[1].split(".")[0].strip()

    # Extract title
    title = "ARCHIVAL RECORD"
    for keyword in ["Homeward Mail", "Times", "Lloyd's Register", "Lloyd's", "Sacramento Daily Union", "News of the World", "Verne"]:
        if keyword.lower() in clean.lower():
            title = keyword.upper()
            break

    return {
        "imageUrl": None,
        "title": title,
        "subtext": "19TH-CENTURY RECORD",
        "highlightText": highlight,
        "zoom": 1.2,
    }


class RatingCardProps(BaseModel):
    """Structured props for Evidence/RatingCard."""
    rating: str = Field(..., description="Locked rating: CONFIRMED | PROBABLE | POSSIBLE | UNSUPPORTED | DISPROVEN")
    claim: str = Field(..., description="Specific claim statement")
    test: str = Field(..., description="The evidentiary test applied to this claim")
    chapter: Optional[str] = Field(default=None, description="Chapter context")


def extract_rating_props(text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Extracts forensic evidence rating props from narration/cue text."""
    clean = text.strip() if text else ""
    rating = "UNSUPPORTED"

    # Prioritize Rating: PREFIX
    m_prefix = re.search(r"Rating:\s*(CONFIRMED|PROBABLE|POSSIBLE|UNSUPPORTED|DISPROVEN)\b", clean, re.IGNORECASE)
    if m_prefix:
        rating = m_prefix.group(1).upper()
    else:
        m_any = re.search(r"\b(CONFIRMED|PROBABLE|POSSIBLE|UNSUPPORTED|DISPROVEN)\b", clean, re.IGNORECASE)
        if m_any:
            rating = m_any.group(1).upper()

    # Extract claim and test
    claim = clean
    test = "Primary sources document it and do not contradict each other"
    if "Test:" in clean:
        parts = clean.split("Test:")
        claim_part = parts[0]
        test = parts[1].strip()
        # strip Rating: XXX -
        claim = re.sub(r"Rating:\s*[A-Z]+\s*[-:]?\s*", "", claim_part, flags=re.IGNORECASE).strip()
    else:
        claim = re.sub(r"Rating:\s*[A-Z]+\s*[-:]?\s*", "", clean, flags=re.IGNORECASE).strip()

    return {
        "rating": rating,
        "claim": claim if claim else "Claim under investigation",
        "test": test,
        "chapter": None,
    }


class SourcingCardProps(BaseModel):
    """Structured props for Evidence/SourcingCard."""
    tier: int = Field(default=1, description="Tier level 1 to 4")
    tierName: str = Field(default="PRIMARY", description="PRIMARY | SCIENTIFIC | SECONDARY | REPRINT")
    source: str = Field(..., description="Name of source, author, archive, or publication")
    date: Optional[str] = Field(default=None, description="Publication date or year")
    isIndependent: bool = Field(default=True, description="False for REPRINT tier which is not independent")


def extract_sourcing_props(text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Extracts sourcing tier and provenance details."""
    clean = text.strip() if text else ""
    tier = 1
    tier_name = "PRIMARY"
    is_independent = True

    if "REPRINT" in clean.upper() or "TIER 4" in clean.upper() or "TIER: 4" in clean.upper():
        tier = 4
        tier_name = "REPRINT"
        is_independent = False
    elif "SECONDARY" in clean.upper() or "TIER 3" in clean.upper() or "TIER: 3" in clean.upper():
        tier = 3
        tier_name = "SECONDARY"
    elif "SCIENTIFIC" in clean.upper() or "TIER 2" in clean.upper() or "TIER: 2" in clean.upper():
        tier = 2
        tier_name = "SCIENTIFIC"

    source = clean
    m_src = re.search(r"Source:\s*([^\|]+)", clean, re.IGNORECASE)
    if m_src:
        source = m_src.group(1).strip()

    return {
        "tier": tier,
        "tierName": tier_name,
        "source": source,
        "date": None,
        "isIndependent": is_independent,
    }


class MeasurementCompareProps(BaseModel):
    """Structured props for DataAnimations/MeasurementCompare."""
    title: str = Field(default="FORENSIC MEASUREMENT COMPARISON", description="Comparison title")
    steps: list[dict[str, str]] = Field(default_factory=list, description="List of measurement step dicts with 'label' and 'value'")
    comparisonType: str = Field(default="shrinkage", description="shrinkage | dispute | scale")


def extract_measurement_props(text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Extracts multi-step measurement collapse or comparative size figures."""
    clean = text.strip() if text else ""
    steps = []

    # Check for arrow sequence e.g. 19 ft -> 17 ft -> 13 ft 1 in
    if "->" in clean or "→" in clean:
        delimiter = "->" if "->" in clean else "→"
        raw_parts = [p.strip() for p in clean.split(delimiter)]
        for i, part in enumerate(raw_parts):
            # Extract measurement value like 19 ft, 13 ft 1 in
            val_match = re.search(r"(\d+(?:\s*(?:ft|m|in|kg|tons?|m\b))+.*)", part, re.IGNORECASE)
            val = val_match.group(1).strip() if val_match else part
            steps.append({
                "label": f"Stage {i+1}",
                "value": val,
            })
    elif "vs" in clean.lower():
        parts = re.split(r"\s+vs\.?\s+", clean, flags=re.IGNORECASE)
        for i, part in enumerate(parts):
            steps.append({
                "label": f"Estimate {i+1}",
                "value": part.strip(),
            })
    else:
        steps = [
            {"label": "Documented", "value": clean[:30]},
        ]

    return {
        "title": "SPECIMEN MEASUREMENT COLLAPSE",
        "steps": steps,
        "comparisonType": "shrinkage" if "shrinkage" in clean.lower() else "dispute",
    }


class ReprintChainProps(BaseModel):
    """Structured props for Evidence/ReprintChain."""
    title: str = Field(default="REPRINT TRANSMISSION CHAIN", description="Chart header")
    nodes: list[dict[str, str]] = Field(default_factory=list, description="Sequential newspaper nodes")


def extract_reprint_props(text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Extracts newspaper reprint progression nodes."""
    clean = text.strip() if text else ""
    nodes = []

    # Strip prefix
    clean_chain = re.sub(r"^.*?chain\s*(?:building)?\s*[:\-]?\s*", "", clean, flags=re.IGNORECASE)
    delimiter = "->" if "->" in clean_chain else ("→" if "→" in clean_chain else None)

    if delimiter:
        parts = [p.strip() for p in clean_chain.split(delimiter)]
        for part in parts:
            nodes.append({
                "outlet": part,
                "date": "1874",
                "location": "Archive",
            })
    else:
        default_papers = ["Homeward Mail", "The Times", "News of the World", "Sacramento Daily Union"]
        for p in default_papers:
            nodes.append({"outlet": p, "date": "1874", "location": "London / US"})

    return {
        "title": "FIVE-PAPER REPRINT TRANSMISSION",
        "nodes": nodes,
    }


class VerdictTableProps(BaseModel):
    """Structured props for Evidence/VerdictTable."""
    title: str = Field(default="FINAL INVESTIGATIVE VERDICT", description="Header title")
    claims: list[dict[str, str]] = Field(default_factory=list, description="List of claim dicts with 'claim' and 'rating'")


def extract_verdict_props(text: str, client: Optional[GeminiLLMClient] = None) -> dict[str, Any]:
    """Extracts claim-by-claim verdict table items."""
    clean = text.strip() if text else ""
    claims = []

    # Parse items like Claim [RATING] or Claim: RATING
    matches = re.findall(r"([A-Za-z0-9\s\.\-,'\"—]+?)(?:\[(CONFIRMED|PROBABLE|POSSIBLE|UNSUPPORTED|DISPROVEN)\]|:\s*(CONFIRMED|PROBABLE|POSSIBLE|UNSUPPORTED|DISPROVEN))", clean)
    if matches:
        for m in matches:
            claim_text = m[0].replace("Verdict:", "").strip(" ,;")
            rating = m[1] or m[2]
            if claim_text:
                claims.append({
                    "claim": claim_text,
                    "rating": rating,
                })
    else:
        # Default 8 claims from episode_outline
        claims = [
            {"claim": "Giant squid exists", "rating": "CONFIRMED"},
            {"claim": "Ship encountered a giant squid (Alecton)", "rating": "CONFIRMED"},
            {"claim": "Squid attacked the ship (Alecton)", "rating": "UNSUPPORTED"},
            {"claim": "Squid attacked a boat (Portugal Cove, 1873)", "rating": "PROBABLE"},
            {"claim": "Large squid in the Bay of Bengal (1874)", "rating": "POSSIBLE"},
            {"claim": "Schooner Pearl sunk by a squid", "rating": "UNSUPPORTED"},
            {"claim": "Pearl account fabricated from scratch", "rating": "POSSIBLE"},
            {"claim": "Pearl is a garbled transmission of Alecton", "rating": "POSSIBLE"},
        ]

    return {
        "title": "DEAD RECKONING — INVESTIGATIVE VERDICT",
        "claims": claims,
    }


