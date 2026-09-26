"""SQLAlchemy ORM model for the beat_exemplars RAG corpus table."""

import uuid
from sqlalchemy import JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.db import Base


class BeatExemplar(Base):
    """A narration↔visual pair extracted from a high-performing reference video.

    Used as few-shot grounding context for the beat structuring LLM call.
    Rows without embeddings are silently skipped at retrieval time.
    """

    __tablename__ = "beat_exemplars"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    source_video_id: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True
    )
    source_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    narration_text: Mapped[str] = mapped_column(Text, nullable=False)
    visual_description: Mapped[str] = mapped_column(Text, nullable=False)
    # Optional heuristic beat type inferred at ingestion time
    beat_type_guess: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # Genre slug matching style skill file names (e.g. 'deep_sea_documentary')
    channel_genre: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True, default="general"
    )
    # JSON-serialised list[float] — embedding vector.
    # Nullable so rows can be inserted without embeddings if needed.
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
