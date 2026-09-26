"""Stage [2]: Beat Structuring.

Segments clean narration prose into atomic beats and tags each with:
- id: unique sequential identifier (e.g. 'b1', 'b2')
- text: exact narration text for this beat
- visual_intent: rich semantic query optimized for footage search
- beat_type: 'narrative' | 'stat' | 'abstract'
- motion_props: structured properties for motion graphics (numbers, kickers, charts)

The system prompt is composed of three additive layers:
1. BASE_PROMPT  — fixed schema/beat-type rules (always present)
2. Style skill  — per-genre creative direction loaded from pipeline/style_skills/
3. RAG exemplars — few-shot narration↔visual pairs retrieved from beat_exemplars DB

Layers 2 and 3 are optional and fail-safe: if missing or unavailable, the base
prompt is used as-is and the job continues without interruption.
"""

from typing import Any, Optional
from pipeline.llm.gemini import GeminiLLMClient
from pipeline.models import Beat, BeatType, MoodTag
from pipeline.skills import load_style_skill
from pipeline.rag.exemplars import format_exemplars_block, retrieve_exemplars
from pipeline.stages.clean_script import clean_script


STRUCTURE_BEATS_SYSTEM_PROMPT = """You are an expert broadcast video director and motion graphics designer.
Your task is to take spoken narration prose and segment it into atomic visual beats.

For each beat:
1. `id`: A sequential identifier like "b1", "b2", "b3", etc.
2. `text`: 1–2 sentences of spoken narration (roughly 3–7 seconds of speech).
3. `visual_intent`: A vivid, concrete, semantic search prompt describing the ideal footage or visual to accompany this beat. Focus on physical actions, environments, lighting, and subjects.
4. `beat_type`: Choose one of:
   - "narrative": Concrete storytelling, actions, physical subjects, or documentary footage scenes. (Default)
   - "stat": Standout numbers, percentages, dollar amounts, metrics, or data points suitable for an animated stat card or chart.
   - "abstract": High-level conceptual quotes, philosophical thoughts, or thematic transitions.
   - "swipe_deck": Sequential 3–5 takeaways, key bullet points, or ordered steps best presented as stacked cards swiping one at a time.
   - "chat_bubbles": Conversational dialogue, exchanges, or message notifications building up in a chat thread format.
   - "kinetic": High-energy, rhythmic keywords or slogans where synchronized word-by-word kinetic typography excels.
   - "typewriter": Terminal, retro narrative, or mechanical character-by-character typewriter exposition.
   - "split_screen": Comparative concepts, dual perspectives, or before/after entities shown side-by-side.
   - "map_route": Geographic journeys, travel trajectories between 2 locations, flight paths, or city/landmark locator pins.
   - "audio_waveform": Direct speech quotes, podcast radio comms, or significant transmissions where a reactive equalizer visualizer excels.
5. `motion_props`: (Required if beat_type is not "narrative", or when a multi-layer composition recipe applies; null otherwise):
   - Optional `layout_recipe`:
     - "stat_over_footage": Pairs standout stat motion typography with background footage or photography.
     - "quote_over_footage": Pairs key kinetic quote cards over contextual background imagery.
     - "split_screen": Pairs two distinct entities or comparative subjects side-by-side.
   - If "stat":
     - `component`: "DataAnimations/StatCard"
     - `primary_value`: Standout metric string (e.g., "$25.4B", "650M+", "4.0%")
     - `kicker`: Short 2-4 word uppercase category label (e.g., "TOTAL INVESTMENT", "AUDIENCE SHARE")
     - `visual_type`: "chart" (for financial/time trends), "ring" (for % / shares), or "bar" (for comparisons)
     - `subtext`: 1 brief contextual sentence explaining the metric.
     - `display_mode`: "overlay" (if visual intent pairs with footage) or "takeover" (pure full-screen graphic).
   - If "abstract" or "quote":
     - `component`: "TextAnimations/QuoteCard"
     - `quote`: The key quoted sentence or thought.
     - `emphasis`: 2-5 words within the quote for highlight styling.
     - `author`: Attributed speaker or context (e.g., "Neil Armstrong" or "Mission Overview").
     - `display_mode`: "overlay" | "takeover"
   - If "swipe_deck":
     - `component`: "ListAnimations/SwipeDeck"
     - `items`: 3–5 short, punchy bullet points or steps extracted from the beat text (each under 12 words).
     - `title`: Short uppercase category title (e.g., "MISSION MILESTONES", "KEY FINDINGS").
     - `display_mode`: "takeover" | "overlay"
   - If "chat_bubbles":
     - `component`: "ListAnimations/ChatBubbles"
     - `messages`: Array of 2–4 messages, e.g. [{"text": "...", "sender": "system"}, {"text": "...", "sender": "user"}].
     - `title`: Header label (e.g., "MISSION CONTROL LOG", "LIVE DIALOGUE").
     - `display_mode`: "takeover" | "overlay"
   - If "kinetic":
     - `component`: "TextAnimations/KineticText"
     - `text`: Narration phrase for dynamic word reveal.
     - `mode`: "reveal" | "karaoke"
     - `display_mode`: "takeover" | "overlay"
   - If "typewriter":
     - `component`: "TextAnimations/Typewriter"
     - `text`: Narration phrase.
     - `display_mode`: "takeover"
   - If "split_screen":
     - `component`: "Layouts/SplitScreen"
     - `layout_recipe`: "split_screen"
     - `leftTitle`: Left entity title.
     - `leftContent`: Left entity summary.
     - `rightTitle`: Right entity title.
     - `rightContent`: Right entity summary.
   - If "map_route":
     - `component`: "GeoAnimations/MapExplainer"
     - `origin`: Starting city or site (e.g. "Cape Canaveral")
     - `destination`: Ending city or site (e.g. "Pacific Ocean")
     - `mode`: "route" (for 2 locations) or "pin" (for single location)
     - `title`: Header label (e.g. "FLIGHT TRAJECTORY")
     - `display_mode`: "takeover" | "overlay"
   - If "audio_waveform":
     - `component`: "AudioAnimations/AudioWaveform"
     - `speaker`: Speaker name or callsign (e.g. "NEIL ARMSTRONG")
     - `quote`: Key quote phrase
     - `title`: Category header (e.g. "MISSION VOICE FEED")
     - `display_mode`: "takeover" | "overlay"
6. `mood` (optional): one of "tense", "hopeful", "triumphant", "somber", "urgent", "neutral" — only set when the beat has a clear emotional register; omit for beats with no strong tone.

Output JSON format:
{
  "beats": [
    {
      "id": "b1",
      "text": "...",
      "visual_intent": "...",
      "beat_type": "narrative",
      "motion_props": null,
      "mood": "hopeful"
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
      },
      "mood": "triumphant"
    },
    {
      "id": "b3",
      "text": "...",
      "visual_intent": "...",
      "beat_type": "swipe_deck",
      "motion_props": {
        "component": "ListAnimations/SwipeDeck",
        "title": "KEY PHASES",
        "items": ["1. Saturn V Ignition", "2. Translunar Injection", "3. Lunar Descent"],
        "display_mode": "takeover"
      },
      "mood": "neutral"
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
   - `beat_type`: "narrative" | "stat" | "abstract" | "swipe_deck" | "chat_bubbles" | "kinetic" | "typewriter" | "split_screen" | "map_route" | "audio_waveform".
   - `motion_props`: Structured visual properties if a motion component or layout recipe applies (component, items, messages, primary_value, kicker, visual_type, quote, emphasis, author, origin, destination, speaker, display_mode).
   - `mood`: Optional emotional tone tag ("tense" | "hopeful" | "triumphant" | "somber" | "urgent" | "neutral").

Output JSON format:
{
  "beats": [
    {
      "id": "b1",
      "text": "...",
      "visual_intent": "...",
      "beat_type": "narrative",
      "motion_props": null,
      "mood": "hopeful"
    }
  ]
}
"""

VALID_BEAT_TYPES = {
    "narrative",
    "stat",
    "abstract",
    "kinetic",
    "quote",
    "typewriter",
    "swipe_deck",
    "chat_bubbles",
    "split_screen",
    "map_route",
    "audio_waveform",
}

VALID_MOODS = {
    "tense",
    "hopeful",
    "triumphant",
    "somber",
    "urgent",
    "neutral",
}


def _parse_beat_json(raw_beats: list[dict[str, Any]]) -> list[Beat]:
    beats: list[Beat] = []
    for idx, b in enumerate(raw_beats, start=1):
        beat_id = str(b.get("id") or f"b{idx}")
        text = str(b.get("text", "")).strip()
        visual_intent = str(b.get("visual_intent", "")).strip()
        beat_type: BeatType = b.get("beat_type", "narrative")
        if beat_type not in VALID_BEAT_TYPES:
            beat_type = "narrative"
        motion_props = b.get("motion_props")
        if isinstance(motion_props, dict):
            # Clean up empty strings or none
            motion_props = {k: v for k, v in motion_props.items() if v is not None}
        else:
            motion_props = None

        raw_mood = b.get("mood")
        mood: Optional[MoodTag] = raw_mood if raw_mood in VALID_MOODS else None

        if text:
            beats.append(
                Beat(
                    id=beat_id,
                    text=text,
                    visual_intent=visual_intent or text,
                    beat_type=beat_type,
                    motion_props=motion_props,
                    mood=mood,
                )
            )
    return beats


def _build_system_instruction(
    genre: Optional[str],
    channel: Optional[str],
    rag_session: Any,
    clean_text: str,
) -> str:
    """Composes the three-layer system instruction for beat structuring.

    Layer 1 (always):   STRUCTURE_BEATS_SYSTEM_PROMPT (base schema + beat types)
    Layer 2 (if found): Style skill file for the genre/channel
    Layer 3 (if avail): RAG exemplar block from beat_exemplars table

    All optional layers degrade silently — never raises.
    """
    system_instruction = STRUCTURE_BEATS_SYSTEM_PROMPT

    # Layer 2: style skill
    try:
        skill_genre = genre or channel
        style_skill = load_style_skill(skill_genre)
        if style_skill:
            system_instruction += f"\n\n{style_skill}"
    except Exception:
        pass  # skill load failure is silent

    # Layer 3: RAG exemplar block
    try:
        if rag_session is not None:
            exemplars = retrieve_exemplars(rag_session, clean_text, genre=genre, top_k=5)
            exemplar_block = format_exemplars_block(exemplars)
            if exemplar_block:
                system_instruction += f"\n\n{exemplar_block}"
    except Exception:
        pass  # RAG failure is silent

    return system_instruction


def structure_beats(
    clean_text: str,
    channel: Optional[str] = None,
    genre: Optional[str] = None,
    client: Optional[GeminiLLMClient] = None,
    rag_session: Any = None,
) -> list[Beat]:
    """Segments cleaned narration text into a structured list of Beat objects.

    Args:
        clean_text: Cleaned narration text to segment.
        channel: Optional channel/show name (maps to a style skill file).
        genre: Optional genre slug for style skill + RAG retrieval. Takes
               priority over channel when both are supplied.
        client: Optional pre-initialised GeminiLLMClient.
        rag_session: Optional SQLAlchemy Session for RAG exemplar retrieval.
                     Pass None (default) to skip RAG entirely.

    Returns:
        List of Beat objects. Returns [] for empty input.
    """
    if not clean_text or not clean_text.strip():
        return []

    llm = client or GeminiLLMClient()
    prompt = f"Segment this cleaned narration into beats:\n\n{clean_text}"
    system_instruction = _build_system_instruction(genre, channel, rag_session, clean_text)
    data = llm.generate_json(prompt=prompt, system_instruction=system_instruction)

    raw_beats: list[dict[str, Any]] = data.get("beats", [])
    return _parse_beat_json(raw_beats)


def clean_and_structure_beats(
    raw_text: str,
    channel: Optional[str] = None,
    genre: Optional[str] = None,
    client: Optional[GeminiLLMClient] = None,
    rag_session: Any = None,
) -> list[Beat]:
    """Single-pass LLM call that cleans a raw script and structures it into beats.

    Falls back to a two-stage clean→structure pass if the combined pass fails.
    Accepts the same channel/genre/rag_session args as structure_beats and
    threads them through the fallback path.
    """
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
    return structure_beats(
        cleaned,
        channel=channel,
        genre=genre,
        client=llm,
        rag_session=rag_session,
    )
