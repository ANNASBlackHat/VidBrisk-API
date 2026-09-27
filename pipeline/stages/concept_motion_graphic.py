"""Stage [5]: Motion Graphic Concepting.

Selects and formats concrete Remotion motion-graphics components from the 10
registered archetypes when:
1. A beat was designated as non-narrative during structuring (e.g. stat, kinetic), OR
2. A narrative beat had footage_status == INADEQUATE and fell back to motion graphics.
"""

import re
from pathlib import Path
from typing import Any, Optional

from pipeline.llm.gemini import GeminiLLMClient
from pipeline.models import Beat, BeatType

MOTION_SKILL_PATH = Path(__file__).parent.parent / "style_skills" / "motion_graphics.md"


def _load_motion_skill() -> str:
    if MOTION_SKILL_PATH.exists():
        return MOTION_SKILL_PATH.read_text(encoding="utf-8")
    return ""


def _heuristic_concept_graphic(beat: Beat, is_fallback: bool) -> tuple[BeatType, dict[str, Any]]:
    """Rule-based fallback when LLM is unavailable or for deterministic testing."""
    text = beat.text.strip()
    lower = text.lower()
    display_mode = "takeover" if is_fallback else (
        (beat.motion_props or {}).get("display_mode", "takeover")
    )

    # 1. Stat detection
    has_digits = any(c.isdigit() for c in text)
    stat_keywords = ["%", "percent", "$", "billion", "million", "thousand", "years ago", "teeth", "feet", "meters"]
    if has_digits and any(kw in lower for kw in stat_keywords):
        # Extract a candidate primary value
        number_match = re.search(r"(\$?\d+(?:,\d+)*(?:\.\d+)?%?(?:\s*(?:billion|million|thousand|ft|meters|teeth|mph))?)", text, re.IGNORECASE)
        val = number_match.group(1).upper() if number_match else "KEY STAT"
        return "stat", {
            "component": "DataAnimations/StatCard",
            "primary_value": val,
            "kicker": "KEY DATA POINT",
            "visual_type": "ring" if "%" in val else "chart",
            "subtext": text[:120],
            "display_mode": display_mode,
            "layout_recipe": "stat_over_footage" if display_mode == "overlay" else None,
        }

    # 2. Sequential list / steps detection
    if any(kw in lower for kw in ["1.", "step 1", "first,", "second,", "third,"]):
        items = [s.strip() for s in re.split(r"[;\n]|\b\d+\.\s*", text) if len(s.strip()) > 3][:4]
        if len(items) >= 2:
            return "swipe_deck", {
                "component": "ListAnimations/SwipeDeck",
                "title": "KEY TAKEAWAYS",
                "items": items,
                "display_mode": "takeover",
            }

    # 3. Direct quote / philosophical thought
    if '"' in text or "said" in lower or "stated" in lower or beat.beat_type == "quote":
        clean_q = text.strip('"')
        words = clean_q.split()
        emphasis = " ".join(words[:min(3, len(words))])
        return "quote", {
            "component": "TextAnimations/QuoteCard",
            "quote": clean_q,
            "emphasis": emphasis,
            "author": "RECORDED STATEMENT",
            "display_mode": display_mode,
            "layout_recipe": "quote_over_footage" if display_mode == "overlay" else None,
        }

    # 4. Terminal / archival typewriter
    if "mistake #" in lower or "log:" in lower or "chapter" in lower or beat.beat_type == "typewriter":
        return "typewriter", {
            "component": "TextAnimations/Typewriter",
            "text": text,
            "display_mode": "takeover",
        }

    # 5. Default high-energy kinetic text takeover
    return "kinetic", {
        "component": "TextAnimations/KineticText",
        "text": text,
        "mode": "reveal",
        "display_mode": display_mode,
    }


COMPONENT_MAP: dict[str, str] = {
    "stat": "DataAnimations/StatCard",
    "kinetic": "TextAnimations/KineticText",
    "typewriter": "TextAnimations/Typewriter",
    "quote": "TextAnimations/QuoteCard",
    "abstract": "TextAnimations/QuoteCard",
    "swipe_deck": "ListAnimations/SwipeDeck",
    "chat_bubbles": "ListAnimations/ChatBubbles",
    "split_screen": "Layouts/SplitScreen",
    "map_route": "GeoAnimations/MapExplainer",
    "audio_waveform": "AudioAnimations/AudioWaveform",
}


def concept_motion_graphic(
    beat: Beat,
    is_fallback: bool = False,
    client: Optional[GeminiLLMClient] = None,
) -> Beat:
    """Concepts or refines motion graphics properties for a beat.

    Args:
        beat: The Beat to concept motion graphics for.
        is_fallback: True if triggered because footage was INADEQUATE.
        client: Optional pre-initialised GeminiLLMClient.

    Returns:
        The updated Beat with beat_type and motion_props configured.
    """
    # If beat already has valid motion props and is not a fallback override, keep as-is
    if not is_fallback and beat.motion_props and beat.motion_props.get("component"):
        return beat

    skill_doc = _load_motion_skill()
    system_instruction = (
        f"{skill_doc}\n\n"
        "You are an expert broadcast motion designer. "
        "Your task is to select the single best Remotion motion component for this narration beat "
        "and return valid JSON with 'beat_type' and 'motion_props'."
    )

    context_prompt = (
        f"Narration text: \"{beat.text}\"\n"
        f"Visual intent: \"{beat.visual_intent}\"\n"
        f"Is fallback due to missing footage: {is_fallback}\n"
        f"Fallback reason: {beat.fallback_reason or 'None'}\n\n"
        "Return a JSON object with:\n"
        "{\n"
        "  \"beat_type\": \"stat\" | \"kinetic\" | \"typewriter\" | \"quote\" | \"swipe_deck\" | \"chat_bubbles\" | \"split_screen\" | \"map_route\" | \"audio_waveform\",\n"
        "  \"motion_props\": { ... }\n"
        "}"
    )

    try:
        llm = client or GeminiLLMClient()
        data = llm.generate_json(prompt=context_prompt, system_instruction=system_instruction)
        beat_type = data.get("beat_type")
        props = data.get("motion_props", {})
        if beat_type and props and isinstance(props, dict):
            if is_fallback:
                props["display_mode"] = "takeover"
            if not props.get("component"):
                props["component"] = COMPONENT_MAP.get(beat_type, "TextAnimations/KineticText")
            beat.beat_type = beat_type
            beat.motion_props = props
            return beat
    except Exception:
        pass

    # Heuristic fallback if LLM is unavailable or unconfigured
    h_type, h_props = _heuristic_concept_graphic(beat, is_fallback=is_fallback)
    beat.beat_type = h_type
    beat.motion_props = h_props
    return beat
