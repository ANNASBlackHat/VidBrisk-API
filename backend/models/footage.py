"""Pydantic models and schemas for VidBrisk Footage Manager & Analytics."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ShotItem(BaseModel):
    id: str
    asset_id: str
    provider: str
    source_url: str
    storage_path: Optional[str] = None
    storage_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    start_ts: float = 0.0
    end_ts: Optional[float] = None
    duration_sec: float = 0.0
    resolution: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    orientation: Literal["horizontal", "vertical", "square", "unknown"] = "unknown"
    media_type: str = "video"
    motion_mean: Optional[float] = None
    motion_std: Optional[float] = None
    motion_score: Optional[float] = None  # Normalized or raw score
    is_loop: bool = False
    usage_count: int = 0
    last_used_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    caption: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    entity_id: Optional[str] = None
    entity_name: Optional[str] = None
    similarity_score: Optional[float] = None
    item_metadata: Optional[Dict[str, Any]] = None


class ShotsListResponse(BaseModel):
    items: List[ShotItem]
    total: int
    limit: int
    offset: int
    has_more: bool


class SearchRequest(BaseModel):
    query: str
    model_id: Optional[str] = "xclip-base-patch32"
    provider: Optional[str] = None
    orientation: Optional[Literal["horizontal", "vertical", "square", "all"]] = None
    min_duration: Optional[float] = None
    max_duration: Optional[float] = None
    min_motion: Optional[float] = None
    used_filter: Optional[Literal["all", "used", "unused"]] = "all"
    limit: int = 30
    offset: int = 0


class SearchResponse(BaseModel):
    results: List[ShotItem]
    total: int
    query: str
    model_id: str
    latency_ms: float


class AnalyticsSummaryResponse(BaseModel):
    total_shots: int
    total_assets: int
    total_duration_hours: float
    unused_shots_count: int
    unused_percentage: float
    providers_breakdown: Dict[str, int]
    orientations_breakdown: Dict[str, int]
    resolutions_breakdown: Dict[str, int]
    duration_histogram: List[Dict[str, Any]]
    motion_distribution: List[Dict[str, Any]]


class EmbeddingModelInfo(BaseModel):
    id: str
    name: str
    version: str
    dims: int
    is_default: bool = False
    description: str


class DuplicateMemberItem(BaseModel):
    id: str
    chunk_id: str
    similarity: float
    is_keeper: bool
    shot: ShotItem


class DuplicateGroupItem(BaseModel):
    id: str
    tier: Literal["exact", "near", "similar"]
    status: Literal["pending", "merged", "kept", "archived"]
    suggested_keeper_id: Optional[str] = None
    created_at: Optional[datetime] = None
    members: List[DuplicateMemberItem]


class DuplicateGroupsResponse(BaseModel):
    groups: List[DuplicateGroupItem]
    total: int
    tier: Optional[str] = None


class DuplicateResolutionRequest(BaseModel):
    action: Literal["merge", "keep", "archive"]
    keeper_id: Optional[str] = None
    notes: Optional[str] = None


class ClusterItem(BaseModel):
    id: str
    model_id: str
    label: str
    centroid_x: float
    centroid_y: float
    size: int


class MapPointItem(BaseModel):
    id: str
    shot_id: str
    x: float
    y: float
    cluster_id: Optional[str] = None
    cluster_label: Optional[str] = None
    provider: str
    resolution: Optional[str] = None
    orientation: str
    duration_sec: float
    motion_score: Optional[float] = None
    title: Optional[str] = None
    thumbnail_url: Optional[str] = None
    storage_url: Optional[str] = None


class MapDataResponse(BaseModel):
    model_id: str
    points: List[MapPointItem]
    clusters: List[ClusterItem]
    total: int


class EntityItem(BaseModel):
    id: str
    name: str
    aliases: List[str]
    entity_type: str
    notes: Optional[str] = None
    linked_shots_count: int = 0
    created_at: Optional[datetime] = None


class EntityCreateRequest(BaseModel):
    name: str
    aliases: List[str] = Field(default_factory=list)
    entity_type: str = "other"
    notes: Optional[str] = None


class EntityLinkRequest(BaseModel):
    entity_id: str


# --- Phase 4: Selection Debugger, Eval Harness, and Curation Queue ---

class DebuggerCandidateItem(BaseModel):
    chunk_id: str
    media_item_id: Optional[str] = None
    provider: str = "unknown"
    title: Optional[str] = None
    thumbnail_url: Optional[str] = None
    storage_url: Optional[str] = None
    score: float = 0.0
    motion_mean: Optional[float] = None
    motion_score: Optional[float] = None
    duration_sec: float = 0.0
    resolution: Optional[str] = None
    orientation: str = "unknown"
    is_selected: bool = False


class DebuggerBeatItem(BaseModel):
    id: str
    text: str
    visual_intent: Optional[str] = None
    beat_type: str = "narrative"
    footage_status: str = "pending"
    selected_clip_id: Optional[str] = None
    requery_count: int = 0
    requery_reason: Optional[str] = None
    fallback_reason: Optional[str] = None
    candidates: List[DebuggerCandidateItem] = Field(default_factory=list)


class DebuggerJobResponse(BaseModel):
    id: str
    title: Optional[str] = None
    stage: str
    status: str
    created_at: Optional[datetime] = None
    beats: List[DebuggerBeatItem] = Field(default_factory=list)


class DebuggerOverrideRequest(BaseModel):
    beat_id: str
    chunk_id: str


class EvalQueryItem(BaseModel):
    query: str
    model_a_candidates: List[DebuggerCandidateItem] = Field(default_factory=list)
    model_b_candidates: List[DebuggerCandidateItem] = Field(default_factory=list)
    top1_agreement: bool = False
    overlap_count: int = 0
    latency_a_ms: float = 0.0
    latency_b_ms: float = 0.0


class EvalCompareRequest(BaseModel):
    queries: List[str]
    model_a: str = "xclip-base-patch32"
    model_b: str = "Qwen/Qwen3-VL-Embedding-2B"
    top_k: int = 5


class EvalCompareResponse(BaseModel):
    model_a: str
    model_b: str
    total_queries: int
    top1_agreement_rate: float
    average_overlap_rate: float
    avg_latency_a_ms: float
    avg_latency_b_ms: float
    queries: List[EvalQueryItem] = Field(default_factory=list)


class CurationQueueItem(BaseModel):
    shot: ShotItem
    curation_status: str = "pending"
    tags: List[str] = Field(default_factory=list)


class CurationActionRequest(BaseModel):
    status: Literal["approved", "rejected", "pending"]
    tags: Optional[List[str]] = None


