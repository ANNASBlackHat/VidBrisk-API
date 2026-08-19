"""SQLAlchemy ORM model and enums for video generation jobs."""

import enum
import uuid
from sqlalchemy import JSON, Boolean, DateTime, Enum, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.db import Base, utc_now


class JobStage(str, enum.Enum):
    """Pipeline progression stages."""
    CLEANING = "cleaning"
    STRUCTURING = "structuring"
    VOICING = "voicing"
    ALIGNING = "aligning"
    RESOLVING_FOOTAGE = "resolving_footage"
    ASSEMBLING = "assembling"
    COMPILING = "compiling"
    DONE = "done"
    FAILED = "failed"


class JobStatus(str, enum.Enum):
    """Per-stage execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETE = "complete"
    FAILED = "failed"


class VideoJob(Base):
    """Represents a single video generation job row in the video_jobs table."""

    __tablename__ = "video_jobs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )
    raw_input: Mapped[str] = mapped_column(Text, nullable=False)
    
    # State Machine columns
    stage: Mapped[JobStage] = mapped_column(
        Enum(JobStage, native_enum=False, length=30),
        default=JobStage.CLEANING,
        nullable=False,
        index=True,
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False, length=30),
        default=JobStatus.PENDING,
        nullable=False,
        index=True,
    )

    # Execution Options
    tts_provider: Mapped[str] = mapped_column(String(50), default="kokoro", nullable=False)
    aligner_provider: Mapped[str] = mapped_column(String(50), default="mock", nullable=False)
    target_orientation: Mapped[str] = mapped_column(String(20), default="horizontal", nullable=False)
    auto_approve: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    single_pass_llm: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Intermediate Stage Outputs (Durable state persistence)
    clean_script: Mapped[str | None] = mapped_column(Text, nullable=True)
    beats: Mapped[list | None] = mapped_column(JSON, nullable=True)
    voice_clips: Mapped[list | None] = mapped_column(JSON, nullable=True)
    timings: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    footage_candidates: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    asset_plan: Mapped[list | None] = mapped_column(JSON, nullable=True)
    
    # Compiled renderable tracks timeline (Frontend/Remotion consumable)
    timeline: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Diagnostic & Error tracking
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<VideoJob id={self.id} stage={self.stage.value} status={self.status.value}>"
