"""Core data models and schemas for Video Generation Pipeline."""

from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


BeatType = Literal["narrative", "stat", "abstract"]
StrategyType = Literal["single_clip", "concat_clips", "image_kenburns", "motion_text"]
TrackType = Literal["video", "text", "audio"]


class Beat(BaseModel):
    """Represents a single atomic narrative beat segmented from clean narration."""
    id: str
    text: str
    visual_intent: str
    beat_type: BeatType = "narrative"
    motion_props: Optional[dict[str, Any]] = None


class VoiceClip(BaseModel):
    """Represents generated TTS audio for a single beat."""
    beat_id: str
    audio_path: str
    duration_sec: float
    sample_rate: int = 24000


class WordTiming(BaseModel):
    """Represents precise start and end times for an individual word or segment."""
    word: str
    start: float
    end: float
    score: Optional[float] = None


class AssetItem(BaseModel):
    """Specific asset chunk or visual card component used in an AssetPlan."""
    type: Literal["video", "image", "text_card", "motion"]
    chunk_id: Optional[str] = None
    source_in: Optional[float] = None
    source_out: Optional[float] = None
    content: Optional[str] = None
    style: Optional[str] = None
    storage_path: Optional[str] = None
    storage_url: Optional[str] = None
    component_id: Optional[str] = None
    props: Optional[dict[str, Any]] = None


class AssetPlan(BaseModel):
    """Plan for how visual assets are mapped and scheduled for a beat."""
    strategy: StrategyType
    items: list[AssetItem] = Field(default_factory=list)


class CandidateChunk(BaseModel):
    """Search result from Footage Engine hydrated with metadata."""
    chunk_id: str
    media_item_id: str
    score: float
    start_ts: float
    end_ts: Optional[float] = None
    duration_sec: Optional[float] = None
    media_type: str = "video"
    provider: str = "unknown"
    storage_path: str = ""
    storage_url: str = ""
    resolution: Optional[str] = None
    orientation: str = "unknown"
    caption: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class ResolvedBeat(BaseModel):
    """Full intermediate context of a beat after TTS, alignment, and footage search."""
    beat: Beat
    voice_clip: VoiceClip
    timings: list[WordTiming] = Field(default_factory=list)
    footage_candidates: list[CandidateChunk] = Field(default_factory=list)
    asset_plan: Optional[AssetPlan] = None


# ------------------------------------------------------------------------------
# Timeline Schema (SPEC §9 & PRD §8)
# ------------------------------------------------------------------------------

class TrackItem(BaseModel):
    """Item placed on a timeline track."""
    id: str
    trackStart: float
    trackEnd: float

    # Video & Motion track specific fields
    assetId: Optional[str] = None
    sourceIn: Optional[float] = None
    sourceOut: Optional[float] = None
    assetType: Optional[Literal["video", "image", "motion"]] = None
    storagePath: Optional[str] = None
    storageUrl: Optional[str] = None
    componentId: Optional[str] = None
    props: Optional[dict[str, Any]] = None

    # Text track specific fields
    content: Optional[str] = None
    style: Optional[str] = None  # e.g., "stat-callout", "title", "caption"


class Track(BaseModel):
    """A single timeline track (e.g. video, text, audio)."""
    type: TrackType
    items: list[TrackItem] = Field(default_factory=list)


class TimelinePlan(BaseModel):
    """The master timeline plan emitted by stage [6] and consumed by render_video.py."""
    tracks: list[Track] = Field(default_factory=list)
    total_duration: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_track(self, track_type: TrackType) -> Track:
        """Finds or creates a track of the requested type."""
        for t in self.tracks:
            if t.type == track_type:
                return t
        new_track = Track(type=track_type, items=[])
        self.tracks.append(new_track)
        return new_track

    def to_dict(self) -> dict[str, Any]:
        """Serializes timeline into editor-compatible JSON dictionary."""
        return self.model_dump(exclude_none=True)
