"""Video Generation Pipeline — Experiment Phase."""

from pipeline.config import Settings, get_settings
from pipeline.models import (
    AssetItem,
    AssetPlan,
    Beat,
    BeatType,
    CandidateChunk,
    FootageCandidate,
    FootageStatus,
    ResolvedBeat,
    StrategyType,
    TimelinePlan,
    Track,
    TrackItem,
    TrackType,
    VoiceClip,
    WordTiming,
)
from pipeline.orchestrator import run_pipeline
from pipeline.stages.assemble_timeline import assemble_timeline
from pipeline.stages.clean_script import clean_script
from pipeline.stages.extract_timestamps import extract_timestamps
from pipeline.stages.resolve_footage import resolve_beat_visuals, resolve_footage
from pipeline.stages.structure_beats import clean_and_structure_beats, structure_beats
from pipeline.stages.synthesize_voice import synthesize_voice

__all__ = [
    "Settings",
    "get_settings",
    "Beat",
    "BeatType",
    "VoiceClip",
    "WordTiming",
    "AssetItem",
    "AssetPlan",
    "StrategyType",
    "CandidateChunk",
    "FootageCandidate",
    "FootageStatus",
    "ResolvedBeat",
    "TrackItem",
    "Track",
    "TrackType",
    "TimelinePlan",
    "clean_script",
    "structure_beats",
    "clean_and_structure_beats",
    "synthesize_voice",
    "extract_timestamps",
    "resolve_footage",
    "resolve_beat_visuals",
    "assemble_timeline",
    "run_pipeline",
]

__version__ = "0.1.0"
