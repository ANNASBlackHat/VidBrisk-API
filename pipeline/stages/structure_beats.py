"""Stage [2]: Beat Structuring.

Segments clean narration prose into atomic beats and tags each with:
- id: unique sequential identifier (e.g. 'b1', 'b2')
- text: exact narration text for this beat
- visual_intent: rich semantic query optimized for footage search
- beat_type: 'narrative' | 'stat' | 'abstract'
- motion_props: structured properties for motion graphics (numbers, kickers, charts)
"""

from typing import Any, Optional
from pipeline.llm.gemini import GeminiLLMClient
from pipeline.models import Beat, BeatType
from pipeline.stages.clean_script import clean_script


STRUCTURE_BEATS_SYSTEM_PROMPT = """You are an expert broadcast video director and motion graphics designer.
Your task is to take spoken narration prose and segment it into atomic visual beats.

For each beat:
1. `id`: A sequential identifier like "b1", "b2", "b3", etc.
2. `text`: 1–2 sentences of spoken narration (roughly 3–7 seconds of speech).
3. `visual_intent`: A vivid, concrete, semantic search prompt describing the ideal footage or visual to accompany this beat. Focus on physical actions, environments, lighting, and subjects.
4. `beat_type`: Choose one of:
   - "narrative": Concrete storytelling, actions, physical subjects, or scenes. (Default)
   - "stat": Focuses on a standout number, percentage, dollar amount, or metric where a motion typography stat card or data chart would excel.
   - "abstract": High-level conceptual quotes, introspective thoughts, or philosophical transitions where kinetic quote cards or typewriter text are appropriate.
5. `motion_props`: (Required if beat_type is "stat" or "abstract", or when a multi-layer composition recipe applies; null otherwise):
   - Optional `layout_recipe`:
     - "stat_over_footage": Pairs standout stat motion typography with background footage or photography.
     - "quote_over_footage": Pairs key kinetic quote cards over contextual background imagery.
     - "split_screen": Pairs two distinct entities or comparative subjects side-by-side.
   - If "stat":
     - `component`: "DataAnimations/StatCard"
     - `primary_value`: String of the standout stat (e.g., "$25.4B", "650M+", "4.0%")
     - `kicker`: Short 2-4 word uppercase category label (e.g., "PROGRAM BUDGET", "GLOBAL AUDIENCE", "PERCENTAGE SHARE")
     - `visual_type`: "chart" (for financial/time trends), "ring" (for % / shares), or "bar" (for single comparisons)
     - `subtext`: 1 brief contextual sentence explaining the metric.
     - `display_mode`: "overlay" (if visual intent can pair with background footage) or "takeover" (if pure motion graphic).
   - If "abstract":
     - `component`: "TextAnimations/QuoteCard"
     - `quote`: The key quoted sentence or thought.
     - `emphasis`: 2-5 words within the quote that should receive glowing highlight styling.
     - `author`: Attributed speaker or context (e.g., "Neil Armstrong, Commander" or "Mission Overview").
     - `display_mode`: "overlay" | "takeover"

Output JSON format:
{
  "beats": [
    {
      "id": "b1",
      "text": "...",
      "visual_intent": "...",
      "beat_type": "narrative",
      "motion_props": null
    },
    {
      "id": "b2",
      "text": "...",
      "visual_intent": "...",
      "beat_type": "stat",
      "motion_props": {
        "layout_recipe": "stat_over_footage",
        "component": "DataAnimations/StatCard",
        "primary_value": "$25.4B",
        "kicker": "TOTAL PROGRAM INVESTMENT",
        "visual_type": "chart",
        "subtext": "Represented 4% of the entire federal budget at its peak.",
        "display_mode": "overlay"
      }
    }
  ]
}
"""

COMBINED_CLEAN_AND_STRUCTURE_SYSTEM_PROMPT = """You are an expert video director and script editor.
Your task is to take a raw, messy video script (which may contain visual cues, bracketed directions, headers, and sound notes), clean out all non-spoken elements, and segment the pure narration into atomic visual beats.

Rules:
1. Strip all visual notes (`[B-roll: ...]`), sound cues (`[SFX: ...]`), headers (`# Scene 1`), and speaker tags (`Narrator:`).
2. Segment the remaining narration into sequential beats of 1–2 sentences each.
3. For each beat, provide:
   - `id`: "b1", "b2", ...
   - `text`: Pure spoken narration for this beat.
   - `visual_intent`: Detailed semantic footage search prompt.
   - `beat_type`: "narrative" | "stat" | "abstract".
   - `motion_props`: Detailed structured visual props if beat_type is "stat" or "abstract" (primary_value, kicker, visual_type, quote, emphasis, author, display_mode).

Output JSON format:
{
  "beats": [
    {
      "id": "b1",
      "text": "...",
      "visual_intent": "...",
      "beat_type": "narrative",
      "motion_props": null
    }
  ]
}
"""


def _parse_beat_json(raw_beats: list[dict[str, Any]]) -> list[Beat]:
    beats: list[Beat] = []
    for idx, b in enumerate(raw_beats, start=1):
        beat_id = str(b.get("id") or f"b{idx}")
        text = str(b.get("text", "")).strip()
        visual_intent = str(b.get("visual_intent", "")).strip()
        beat_type: BeatType = b.get("beat_type", "narrative")
        if beat_type not in ("narrative", "stat", "abstract"):
            beat_type = "narrative"
        motion_props = b.get("motion_props")
        if isinstance(motion_props, dict):
            # Clean up empty strings or none
            motion_props = {k: v for k, v in motion_props.items() if v is not None}
        else:
            motion_props = None

        if text:
            beats.append(
                Beat(
                    id=beat_id,
                    text=text,
                    visual_intent=visual_intent or text,
                    beat_type=beat_type,
                    motion_props=motion_props,
                )
            )
    return beats


def structure_beats(clean_text: str, client: Optional[GeminiLLMClient] = None) -> list[Beat]:
    """Segments cleaned narration text into a structured list of Beat objects."""
    if not clean_text or not clean_text.strip():
        return []

    llm = client or GeminiLLMClient()
    prompt = f"Segment this cleaned narration into beats:\n\n{clean_text}"
    data = llm.generate_json(prompt=prompt, system_instruction=STRUCTURE_BEATS_SYSTEM_PROMPT)

    raw_beats: list[dict[str, Any]] = data.get("beats", [])
    return _parse_beat_json(raw_beats)


def clean_and_structure_beats(raw_text: str, client: Optional[GeminiLLMClient] = None) -> list[Beat]:
    """Single-pass LLM call that cleans a raw script and structures it into beats."""
    if not raw_text or not raw_text.strip():
        return []

    llm = client or GeminiLLMClient()
    prompt = f"Clean and structure this raw script into beats:\n\n{raw_text}"
    try:
        data = llm.generate_json(
            prompt=prompt,
            system_instruction=COMBINED_CLEAN_AND_STRUCTURE_SYSTEM_PROMPT,
        )
        raw_beats: list[dict[str, Any]] = data.get("beats", [])
        beats = _parse_beat_json(raw_beats)
        if beats:
            return beats
    except Exception:
        # Fallback to two-stage execution if combined pass fails
        pass

    cleaned = clean_script(raw_text, client=llm)
    return structure_beats(cleaned, client=llm)
