import os
from typing import Any, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(prefix="/voices", tags=["Voices"])

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VOICES_DIR = os.path.join(ROOT_DIR, "data", "voices")


class VoiceItem(BaseModel):
    id: str
    name: str
    filename: str
    has_text: bool
    transcript: Optional[str] = None
    preview_url: Optional[str] = None


class VoiceListResponse(BaseModel):
    custom_voices: list[VoiceItem]
    default_instruct: str
    valid_tags: list[str]


class RecommendPromptRequest(BaseModel):
    script: str
    genre: Optional[str] = None
    channel: Optional[str] = None


class RecommendPromptResponse(BaseModel):
    recommended_prompt: str
    reasoning: str


@router.get("", response_model=VoiceListResponse)
def list_available_voices():
    """Lists available local voice cloning audio samples from data/voices/."""
    os.makedirs(VOICES_DIR, exist_ok=True)
    custom_voices = []

    for fname in sorted(os.listdir(VOICES_DIR)):
        base, ext = os.path.splitext(fname)
        if ext.lower() in (".mp3", ".wav", ".m4a", ".flac"):
            txt_path = os.path.join(VOICES_DIR, f"{base}.txt")
            has_txt = os.path.isfile(txt_path)
            transcript = None
            if has_txt:
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        transcript = f.read().strip()
                except Exception:
                    pass

            name = base.replace("_", " ").title()
            custom_voices.append(
                VoiceItem(
                    id=base,
                    name=name,
                    filename=fname,
                    has_text=has_txt,
                    transcript=transcript,
                    preview_url=f"/api/voices/{base}/audio",
                )
            )

    from pipeline.tts.omni import DEFAULT_VOICE_INSTRUCT, VALID_INSTRUCT_TAGS

    return VoiceListResponse(
        custom_voices=custom_voices,
        default_instruct=DEFAULT_VOICE_INSTRUCT,
        valid_tags=sorted(list(VALID_INSTRUCT_TAGS)),
    )


@router.api_route("/{voice_id}/audio", methods=["GET", "HEAD"])
def get_voice_audio(voice_id: str):
    """Streams the reference audio file for a voice profile."""
    os.makedirs(VOICES_DIR, exist_ok=True)
    safe_id = os.path.basename(voice_id).strip()
    matched_file = None

    for fname in os.listdir(VOICES_DIR):
        base, ext = os.path.splitext(fname)
        if base == safe_id and ext.lower() in (".mp3", ".wav", ".m4a", ".flac", ".ogg"):
            matched_file = os.path.join(VOICES_DIR, fname)
            break

    if not matched_file or not os.path.isfile(matched_file):
        raise HTTPException(status_code=404, detail=f"Voice audio for '{voice_id}' not found.")

    ext = os.path.splitext(matched_file)[1].lower()
    media_type = "audio/mpeg" if ext == ".mp3" else f"audio/{ext.lstrip('.')}"
    return FileResponse(
        path=matched_file,
        media_type=media_type,
        filename=os.path.basename(matched_file),
    )


@router.post("/recommend-prompt", response_model=RecommendPromptResponse)
def recommend_voice_prompt(req: RecommendPromptRequest):
    """Uses LLM or heuristic rules to recommend an OmniVoice prompt from video script/theme."""
    from pipeline.tts.omni import normalize_instruct

    # Try LLM recommendation if Gemini client is configured
    try:
        from pipeline.llm import generate_text

        prompt = (
            "You are a professional voice casting director for video productions.\n"
            f"Video Genre: {req.genre or 'General'}\n"
            f"Channel / Show: {req.channel or 'General'}\n"
            f"Script Excerpt: {req.script[:500]}\n\n"
            "Select the best voice characteristics using only from these allowed tags:\n"
            "Gender: male, female\n"
            "Age: child, teenager, young adult, middle-aged, elderly\n"
            "Pitch: very low pitch, low pitch, moderate pitch, high pitch, very high pitch\n"
            "Accent: american accent, british accent, australian accent, canadian accent, indian accent, japanese accent, korean accent, chinese accent, portuguese accent, russian accent\n"
            "Special: whisper\n\n"
            "Format your answer as a comma-separated list of 3-4 tags (e.g. 'male, middle-aged, low pitch, american accent') "
            "followed on a new line by a 1-sentence reasoning."
        )
        response_text = generate_text(prompt=prompt, temperature=0.3)
        lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]
        first_line = lines[0] if lines else ""
        reasoning = lines[1] if len(lines) > 1 else "Optimized for script tone and pacing."
        normalized = normalize_instruct(first_line)
        return RecommendPromptResponse(
            recommended_prompt=normalized,
            reasoning=reasoning,
        )
    except Exception:
        # Rule-based fallback if LLM is unavailable
        text_lower = (req.script + " " + (req.genre or "")).lower()
        if any(w in text_lower for w in ("ocean", "nature", "history", "documentary", "deep", "ancient")):
            return RecommendPromptResponse(
                recommended_prompt="male, middle-aged, low pitch, american accent",
                reasoning="Documentary themes benefit from deep, steady narration.",
            )
        elif any(w in text_lower for w in ("tech", "software", "product", "demo", "app", "startup")):
            return RecommendPromptResponse(
                recommended_prompt="male, young adult, moderate pitch, american accent",
                reasoning="Modern tech demos require clear, articulate, professional delivery.",
            )
        else:
            return RecommendPromptResponse(
                recommended_prompt="female, young adult, moderate pitch, american accent",
                reasoning="Natural, engaging, versatile narration suitable for broad audiences.",
            )
