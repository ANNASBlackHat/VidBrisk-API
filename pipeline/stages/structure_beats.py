"""Stage [2]: Beat Structuring.

Segments clean narration prose into atomic beats and tags each with:
- id: unique sequential identifier (e.g. 'b1', 'b2')
- text: exact narration text for this beat
- visual_intent: rich semantic query optimized for footage search
- beat_type: 'narrative' | 'stat' | 'abstract'
"""

from typing import Any, Optional
from pipeline.llm.gemini import GeminiLLMClient
from pipeline.models import Beat, BeatType
from pipeline.stages.clean_script import clean_script


STRUCTURE_BEATS_SYSTEM_PROMPT = """You are an expert video director and AI video editor.
Your task is to take spoken narration prose and segment it into atomic visual beats.

For each beat:
1. `id`: A sequential identifier like "b1", "b2", "b3", etc.
2. `text`: 1–2 sentences of spoken narration (roughly 3–7 seconds of speech).
3. `visual_intent`: A vivid, concrete, semantic search prompt describing the ideal footage or visual to accompany this beat. Focus on physical actions, environments, lighting, and subjects (e.g., "astronaut floating in zero gravity inside Apollo spacecraft", "time-lapse of stock market chart plunging into red").
4. `beat_type`: Choose one of:
   - "narrative": Concrete storytelling, actions, physical subjects, or scenes. (Default)
   - "stat": Focuses on a standout number, percentage, dollar amount, or metric where a motion typography stat card would excel.
   - "abstract": High-level conceptual questions, introspective thoughts, or mood transitions where motion text/graphics are appropriate.

Output JSON format:
{
  "beats": [
    {
      "id": "b1",
      "text": "...",
      "visual_intent": "...",
      "beat_type": "narrative"
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

Output JSON format:
{
  "beats": [
    {
      "id": "b1",
      "text": "...",
      "visual_intent": "...",
      "beat_type": "narrative"
    }
  ]
}
"""


def structure_beats(clean_text: str, client: Optional[GeminiLLMClient] = None) -> list[Beat]:
    """Segments cleaned narration text into a structured list of Beat objects."""
    if not clean_text or not clean_text.strip():
        return []

    llm = client or GeminiLLMClient()
    prompt = f"Segment this cleaned narration into beats:\n\n{clean_text}"
    data = llm.generate_json(prompt=prompt, system_instruction=STRUCTURE_BEATS_SYSTEM_PROMPT)

    raw_beats: list[dict[str, Any]] = data.get("beats", [])
    beats: list[Beat] = []
    for idx, b in enumerate(raw_beats, start=1):
        beat_id = str(b.get("id") or f"b{idx}")
        text = str(b.get("text", "")).strip()
        visual_intent = str(b.get("visual_intent", "")).strip()
        beat_type: BeatType = b.get("beat_type", "narrative")
        if beat_type not in ("narrative", "stat", "abstract"):
            beat_type = "narrative"

        if text:
            beats.append(
                Beat(
                    id=beat_id,
                    text=text,
                    visual_intent=visual_intent or text,
                    beat_type=beat_type,
                )
            )

    return beats


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
        beats: list[Beat] = []
        for idx, b in enumerate(raw_beats, start=1):
            beat_id = str(b.get("id") or f"b{idx}")
            text = str(b.get("text", "")).strip()
            visual_intent = str(b.get("visual_intent", "")).strip()
            beat_type: BeatType = b.get("beat_type", "narrative")
            if beat_type not in ("narrative", "stat", "abstract"):
                beat_type = "narrative"

            if text:
                beats.append(
                    Beat(
                        id=beat_id,
                        text=text,
                        visual_intent=visual_intent or text,
                        beat_type=beat_type,
                    )
                )
        if beats:
            return beats
    except Exception:
        # Fallback to two-stage execution if combined pass fails
        pass

    cleaned = clean_script(raw_text, client=llm)
    return structure_beats(cleaned, client=llm)
