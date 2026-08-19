"""Stage [1]: Script Cleaning.

Strips visual directions, presenter tags, sound cues, headers, and formatting noise
to yield pure spoken narration prose.
"""

from typing import Optional
from pipeline.llm.gemini import GeminiLLMClient

CLEAN_SCRIPT_SYSTEM_PROMPT = """You are an expert script editor for an automated video generation pipeline.
Your task is to take a raw video script and produce ONLY the clean spoken narration text for the voiceover artist.

Rules:
1. Strip ALL visual directions, camera angles, or footage notes (e.g. `[B-roll: ocean waves]`, `[Cut to chart]`, `(Visual: astronaut on moon)`).
2. Strip ALL sound effects or music cues (e.g. `[SFX: explosion]`, `[Upbeat music starts]`, `[Pause 2s]`).
3. Strip ALL speaker tags and role labels (e.g. `Narrator:`, `Host:`, `VO:`, `Speaker 1:`).
4. Strip ALL markdown headers, scene titles, or outline markers (e.g. `# Intro`, `### Scene 1`, `1. Hook:`).
5. Preserve the exact meaning, wording, and natural conversational cadence of the spoken lines.
6. Fix minor typos or punctuation issues only if they impede natural spoken delivery.
7. Return ONLY the raw narration text. Do NOT include any preamble, markdown code fences, or explanations."""


def clean_script(raw_text: str, client: Optional[GeminiLLMClient] = None) -> str:
    """Cleans a messy script and returns pure narration prose."""
    if not raw_text or not raw_text.strip():
        return ""

    llm = client or GeminiLLMClient()
    prompt = f"Please clean this raw script into pure spoken narration:\n\n{raw_text}"
    cleaned = llm.generate_text(prompt=prompt, system_instruction=CLEAN_SCRIPT_SYSTEM_PROMPT)
    return cleaned.strip()
