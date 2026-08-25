"""Pydantic schemas for Job request, response, approval, and timeline serialization."""

from datetime import datetime
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field
from backend.models.job import JobStage, JobStatus


class JobCreateRequest(BaseModel):
    """Payload for submitting a new video generation run."""
    title: Optional[str] = Field(default=None, max_length=255, description="Optional title or label for the video")
    raw_input: str = Field(..., min_length=5, description="Raw messy script or article text to generate a video from")
    tts_provider: str = Field(default="kokoro", description="Voice synthesis provider: kokoro, supersonic, chatterbox, or mock")
    aligner_provider: str = Field(default="mock", description="Alignment provider: mock, easytranscriber, or whisperx")
    target_orientation: Literal["horizontal", "vertical", "square", "any"] = Field(
        default="horizontal",
        description="Target aspect ratio / orientation for footage retrieval",
    )
    auto_approve: bool = Field(
        default=True,
        description="Whether to auto-advance past Stage 2 and Stage 5 checkpoints without pausing",
    )
    single_pass_llm: bool = Field(
        default=False,
        description="Use single combined LLM prompt for Stage 1 + 2",
    )
    custom_audio_path: Optional[str] = Field(
        default=None,
        description="Optional path to uploaded pre-recorded voiceover audio file",
    )


class JobUpdateRequest(BaseModel):
    """Payload for updating job metadata."""
    title: Optional[str] = Field(default=None, max_length=255, description="New title for the video")
    custom_audio_path: Optional[str] = Field(default=None, description="Path to custom audio file")


class JobApprovalRequest(BaseModel):
    """Payload for human approval checkpoint advancement or overrides."""
    action: Literal["approve", "reject"] = Field(default="approve", description="Action: 'approve' or 'reject'")
    beats_override: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Optional edited beats list to replace Stage 2 output before continuing",
    )
    candidates_override: Optional[dict[str, list[dict[str, Any]]]] = Field(
        default=None,
        description="Optional edited candidates map to replace Stage 5 output before continuing",
    )


class JobSummaryResponse(BaseModel):
    """Lightweight summary representation of a job for listing."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: Optional[str] = None
    stage: JobStage
    status: JobStatus
    tts_provider: str
    aligner_provider: str
    target_orientation: str
    auto_approve: bool
    custom_audio_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None
    progress: Optional[dict[str, Any]] = None
    video_url: Optional[str] = None


class JobResponse(BaseModel):
    """Full detail view of a video generation job with all intermediate outputs."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: Optional[str] = None
    raw_input: str
    stage: JobStage
    status: JobStatus
    tts_provider: str
    aligner_provider: str
    target_orientation: str
    auto_approve: bool
    single_pass_llm: bool
    custom_audio_path: Optional[str] = None
    
    clean_script: Optional[str] = None
    beats: Optional[list[dict[str, Any]]] = None
    voice_clips: Optional[list[dict[str, Any]]] = None
    timings: Optional[dict[str, Any]] = None
    footage_candidates: Optional[dict[str, Any]] = None
    asset_plan: Optional[list[dict[str, Any]]] = None
    timeline: Optional[dict[str, Any]] = None
    motion_qa_thumbnails: Optional[Union[list[dict[str, Any]], dict[str, Any]]] = None
    progress: Optional[dict[str, Any]] = None
    video_url: Optional[str] = None
    
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
