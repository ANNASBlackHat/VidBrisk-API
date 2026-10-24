"""Cinematic models and film-grammar primitives."""
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ShotScale(str, Enum):
    """Standard cinematic shot scale distance from camera to subject."""
    ELS = "extreme_long_shot"
    LS = "long_shot"
    MS = "medium_shot"
    MCU = "medium_close_up"
    CU = "close_up"
    ECU = "extreme_close_up"
    INS = "insert_cutaway"


class MotionVector(str, Enum):
    """Primary directional optical flow movement of the camera or dominant subject."""
    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"


class SceneDirectorSheet(BaseModel):
    """High-level cinematic vision and shot choreography for a narrative scene."""
    scene_id: str
    tone: str = Field(default="documentary cinematic", description="Emotional atmosphere of scene")
    target_color_temp_k: int = Field(default=5500, description="Target color temperature in Kelvin (3200K tungsten to 7000K cold)")
    target_asl: float = Field(default=3.5, description="Target Average Shot Length in seconds")
    visual_motif: Optional[str] = Field(default=None, description="Thematic visual anchor to recur across scene")
    beat_shot_scales: dict[str, ShotScale] = Field(default_factory=dict, description="Assigned shot scale for each beat id")
    rhythm_pattern: Literal["accelerando", "decelerando", "steady", "staccato"] = Field(
        default="steady",
        description="Pacing rhythm across the scene"
    )


class CinematicCandidateMetadata(BaseModel):
    """Perceptual and film-grammar attributes extracted for a footage chunk."""
    chunk_id: str
    shot_scale: ShotScale = ShotScale.MS
    motion_vector: MotionVector = MotionVector.STATIC
    motion_speed: float = 0.0
    dominant_color_temp_k: int = 5500
    eye_trace_center: tuple[float, float] = (0.5, 0.5)
