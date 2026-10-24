"""Scene Director pass formulating shot scale progressions and visual rhythm."""
import logging
from typing import Optional
from pipeline.models import Beat
from pipeline.cinematic.models import SceneDirectorSheet, ShotScale

logger = logging.getLogger(__name__)

GENRE_COLOR_TEMPS = {
    "deep_sea_documentary": 6500,
    "nature_documentary": 5400,
    "space_documentary": 6200,
    "history_documentary": 3800,
    "true_crime": 4800,
    "tech_documentary": 5800,
}


def direct_scene(
    beats: list[Beat],
    genre: Optional[str] = None,
    scene_id: Optional[str] = None,
) -> SceneDirectorSheet:
    """Pre-visualizes shot scales and editorial rhythm for a scene.
    
    Applies film grammar progression (Inductive Reveal: Wide -> Medium -> Close -> Macro)
    and maps tone and color targets based on documentary genre.
    """
    sid = scene_id or (beats[0].id if beats else "scene_0")
    clean_genre = (genre or "").lower().replace("-", "_")
    target_temp = 5500
    for g_key, temp in GENRE_COLOR_TEMPS.items():
        if g_key in clean_genre or clean_genre in g_key:
            target_temp = temp
            break

    sheet = SceneDirectorSheet(
        scene_id=sid,
        tone="cinematic documentary",
        target_color_temp_k=target_temp,
        target_asl=3.5,
        visual_motif=None,
        rhythm_pattern="steady",
    )

    if not beats:
        return sheet

    # Classical progression cycle
    progression_cycle = [
        ShotScale.ELS,
        ShotScale.MS,
        ShotScale.MCU,
        ShotScale.CU,
        ShotScale.ECU,
        ShotScale.LS,  # Reset
    ]

    cycle_idx = 0
    for beat in beats:
        b_text = (beat.text or "").lower()
        b_intent = (beat.visual_intent or "").lower()

        # 1. Specialized beat types map to inserts
        if beat.beat_type in ("stat", "quote", "typewriter", "kinetic"):
            sheet.beat_shot_scales[beat.id] = ShotScale.INS
            continue

        # 2. Keyword-specific shot scale detection
        if any(w in b_intent or w in b_text for w in ["macro", "detail", "extreme close", "serial number", "gauge", "needle", "meter"]):
            sheet.beat_shot_scales[beat.id] = ShotScale.ECU
            cycle_idx += 1
        elif any(w in b_intent or w in b_text for w in ["wide", "panorama", "landscape", "vast", "trench", "abyss", "horizon"]):
            sheet.beat_shot_scales[beat.id] = ShotScale.ELS
            cycle_idx += 1
        elif any(w in b_intent or w in b_text for w in ["close-up", "close up", "face", "portrait", "artifact", "relic", "creature eyes"]):
            sheet.beat_shot_scales[beat.id] = ShotScale.CU
            cycle_idx += 1
        else:
            # 3. Follow classical progression grammar
            sheet.beat_shot_scales[beat.id] = progression_cycle[cycle_idx % len(progression_cycle)]
            cycle_idx += 1

    return sheet
