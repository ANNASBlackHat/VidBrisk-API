"""Mood to color-treatment mapping for sentiment-driven color grading."""

from typing import Any, Optional
from pipeline.models import Beat

_MOOD_TO_TREATMENT: dict[str, str] = {
    "tense": "duotone-cool",
    "urgent": "duotone-cool",
    "somber": "duotone-mono",
    "hopeful": "duotone-warm",
    "triumphant": "duotone-warm",
    "neutral": "none",
}


def mood_to_color_treatment(mood: Optional[str]) -> Optional[str]:
    """Maps a MoodTag to a frontend duotone color treatment.

    Returns:
        Treatment name (e.g. 'duotone-cool', 'duotone-warm', 'duotone-mono', 'none')
        or None if mood is not set or unrecognized.
    """
    if not mood:
        return None
    return _MOOD_TO_TREATMENT.get(mood.lower().strip())


def apply_mood_effects_to_props(
    existing_props: Optional[dict[str, Any]],
    beat: Beat,
) -> Optional[dict[str, Any]]:
    """Merges mood-derived colorTreatment into existing props if not already set.

    Respects explicit overrides:
    1. If `existing_props.effects.colorTreatment` is already set, it is kept.
    2. If `beat.motion_props.effects.colorTreatment` (or `beat.motion_props.colorTreatment`) is set,
       it is prioritized as an explicit manual override.
    3. Otherwise, if `beat.mood` maps to a treatment, sets `effects.colorTreatment`.
    """
    props = dict(existing_props) if existing_props else {}

    # Check existing_props effects
    effects = props.get("effects")
    if isinstance(effects, dict) and effects.get("colorTreatment"):
        return props

    # Check beat.motion_props for explicit effect override
    motion_props = beat.motion_props or {}
    mp_effects = motion_props.get("effects")
    if isinstance(mp_effects, dict) and mp_effects.get("colorTreatment"):
        if "effects" not in props or not isinstance(props.get("effects"), dict):
            props["effects"] = {}
        props["effects"]["colorTreatment"] = mp_effects["colorTreatment"]
        return props
    elif motion_props.get("colorTreatment"):
        if "effects" not in props or not isinstance(props.get("effects"), dict):
            props["effects"] = {}
        props["effects"]["colorTreatment"] = motion_props["colorTreatment"]
        return props

    # Apply mood mapping
    treatment = mood_to_color_treatment(beat.mood)
    if treatment:
        if "effects" not in props or not isinstance(props.get("effects"), dict):
            props["effects"] = {}
        props["effects"]["colorTreatment"] = treatment

    return props if props else None
