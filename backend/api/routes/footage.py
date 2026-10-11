"""FastAPI route handlers for VidBrisk Footage Manager & Analytics."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.api.dependencies import get_db
from backend.models.footage import (
    AnalyticsSummaryResponse,
    CurationActionRequest,
    CurationQueueItem,
    DebuggerBeatItem,
    DebuggerCandidateItem,
    DebuggerJobResponse,
    DebuggerOverrideRequest,
    DuplicateGroupsResponse,
    DuplicateResolutionRequest,
    EmbeddingModelInfo,
    EntityCreateRequest,
    EntityItem,
    EntityLinkRequest,
    EvalCompareRequest,
    EvalCompareResponse,
    MapDataResponse,
    SearchRequest,
    SearchResponse,
    ShotItem,
    ShotsListResponse,
)
from backend.repository_footage import (
    get_analytics_summary,
    get_available_models,
    get_shot_by_id,
    get_shots,
    search_shots,
)

router = APIRouter(prefix="/footage", tags=["Footage Manager"])


@router.get("/shots", response_model=ShotsListResponse, summary="Browse shots with multi-facet filters")
def list_shots_endpoint(
    provider: Optional[str] = Query(None, description="Filter by provider (e.g. youtube, pexels, pixabay)"),
    orientation: Optional[str] = Query(None, description="horizontal, vertical, square, all"),
    min_duration: Optional[float] = Query(None, ge=0.0, description="Minimum duration in seconds"),
    max_duration: Optional[float] = Query(None, ge=0.0, description="Maximum duration in seconds"),
    min_motion: Optional[float] = Query(None, ge=0.0, description="Minimum motion score"),
    used: Optional[str] = Query("all", description="all, used, unused"),
    sort: Optional[str] = Query("newest", description="newest, oldest, duration_desc, duration_asc, motion_desc, usage_desc"),
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Returns paginated shots with provider, resolution, orientation, and motion metadata."""
    return get_shots(
        db=db,
        provider=provider,
        orientation=orientation,
        min_duration=min_duration,
        max_duration=max_duration,
        min_motion=min_motion,
        used_filter=used,
        sort_by=sort,
        limit=limit,
        offset=offset,
    )


@router.get("/shots/{shot_id}", response_model=ShotItem, summary="Get individual shot details")
def get_shot_endpoint(
    shot_id: str,
    db: Session = Depends(get_db),
):
    """Retrieves full details for a single shot, including parent asset metadata."""
    shot = get_shot_by_id(db, shot_id)
    if not shot:
        raise HTTPException(status_code=404, detail=f"Shot with id '{shot_id}' not found.")
    return shot


@router.post("/search", response_model=SearchResponse, summary="Semantic vector search across shots")
def search_footage_endpoint(
    body: SearchRequest,
    db: Session = Depends(get_db),
):
    """Performs semantic search using vector similarity and returns ranked shots."""
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
    return search_shots(db=db, req=body)


@router.get("/analytics/summary", response_model=AnalyticsSummaryResponse, summary="Get library analytics summary")
def get_analytics_summary_endpoint(
    db: Session = Depends(get_db),
):
    """Returns library-wide analytics: shot count, total hours, source breakdown, motion distributions."""
    return get_analytics_summary(db)


@router.get("/models", response_model=List[EmbeddingModelInfo], summary="List registered embedding models")
def list_embedding_models_endpoint():
    """Lists available vector embedding models and their dimensionalities."""
    return get_available_models()


@router.get("/duplicates", response_model=DuplicateGroupsResponse, summary="Get duplicate clusters by tier")
def get_duplicates_endpoint(
    tier: Optional[str] = Query(None, description="exact, near, similar, all"),
    status: Optional[str] = Query("pending", description="pending, merged, kept, archived, all"),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Returns duplicate groups with side-by-side member clips, similarities, and suggested keepers."""
    from backend.repository_footage import get_duplicate_groups
    return get_duplicate_groups(db=db, tier=tier, status=status, limit=limit)


@router.post("/duplicates/detect", summary="Trigger auto-detection of duplicates")
def trigger_detect_duplicates_endpoint(
    max_groups: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Scans library chunks for exact and near duplicate clusters."""
    from backend.repository_footage import auto_detect_exact_duplicates
    created = auto_detect_exact_duplicates(db=db, max_groups=max_groups)
    return {"status": "ok", "groups_created": created}


@router.post("/duplicates/{group_id}/resolve", summary="Resolve a duplicate group")
def resolve_duplicate_endpoint(
    group_id: str,
    body: DuplicateResolutionRequest,
    db: Session = Depends(get_db),
):
    """Resolves duplicate group with merge, keep, or archive action."""
    from backend.repository_footage import resolve_duplicate_group
    success = resolve_duplicate_group(
        db=db,
        group_id=group_id,
        action=body.action,
        keeper_id=body.keeper_id,
        notes=body.notes,
    )
    if not success:
        raise HTTPException(status_code=404, detail=f"Duplicate group '{group_id}' not found.")
    return {"status": "ok", "group_id": group_id, "action": body.action}


@router.get("/map", response_model=MapDataResponse, summary="Get 2D embedding map projection & clusters")
def get_map_endpoint(
    model: str = Query("xclip-base-patch32", description="Embedding model ID"),
    db: Session = Depends(get_db),
):
    """Returns 2D projected scatter coordinates and labeled clusters for the embedding map."""
    from backend.repository_footage import get_map_data
    return get_map_data(db=db, model_id=model)


@router.post("/map/recompute", summary="Recompute 2D embedding projection")
def recompute_map_endpoint(
    model: str = Query("xclip-base-patch32", description="Embedding model ID"),
    max_points: int = Query(150, ge=10, le=500),
    db: Session = Depends(get_db),
):
    """Recomputes 2D dimensionality reduction and topic clusters."""
    from backend.repository_footage import recompute_map_projection
    count = recompute_map_projection(db=db, model_id=model, max_points=max_points)
    return {"status": "ok", "model_id": model, "projected_points": count}


@router.get("/entities", response_model=List[EntityItem], summary="List named subjects")
def list_entities_endpoint(
    db: Session = Depends(get_db),
):
    """Lists controlled vocabulary entities with linked shot counts."""
    from backend.repository_footage import list_entities_with_counts
    return list_entities_with_counts(db=db)


@router.post("/entities", response_model=EntityItem, summary="Create a new subject entity")
def create_entity_endpoint(
    body: EntityCreateRequest,
    db: Session = Depends(get_db),
):
    """Registers a new subject entity with aliases and type."""
    from backend.repository_footage import create_entity
    item = create_entity(
        db=db,
        name=body.name,
        aliases=body.aliases,
        entity_type=body.entity_type,
        notes=body.notes,
    )
    if not item:
        raise HTTPException(status_code=500, detail="Failed to create entity.")
    return item


@router.get("/entities/{entity_id}/suggestions", response_model=List[ShotItem], summary="Get suggested shots for entity")
def get_entity_suggestions_endpoint(
    entity_id: str,
    limit: int = Query(12, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Proposes matching unlinked shots based on entity name and aliases."""
    from backend.repository_footage import get_entity_suggested_shots
    return get_entity_suggested_shots(db=db, entity_id=entity_id, limit=limit)


@router.post("/shots/{shot_id}/entities", summary="Link a shot to an entity")
def link_shot_entity_endpoint(
    shot_id: str,
    body: EntityLinkRequest,
    db: Session = Depends(get_db),
):
    """Links a shot's parent media item to an entity."""
    from backend.repository_footage import link_shot_to_entity
    success = link_shot_to_entity(db=db, shot_id=shot_id, entity_id=body.entity_id)
    if not success:
        raise HTTPException(status_code=404, detail="Shot or entity not found.")
    return {"status": "ok", "shot_id": shot_id, "entity_id": body.entity_id}


# =========================================================================
# Phase 4: Selection Debugger, Eval Harness, Usage Tracking & Curation Queue
# =========================================================================

@router.get("/debugger/jobs", summary="List recent jobs for selection debugging")
def list_debugger_jobs_endpoint(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Lists video generation jobs available to debug footage selections."""
    from backend.repository_footage import list_debugger_jobs
    return list_debugger_jobs(db=db, limit=limit)


@router.get("/debugger/{job_id}", response_model=DebuggerJobResponse, summary="Get full selection debug info for a job")
def get_job_debugger_endpoint(
    job_id: str,
    db: Session = Depends(get_db),
):
    """Retrieves beat-by-beat selection details, keywords, and candidate pool scores."""
    from backend.repository_footage import get_job_selection_debug
    res = get_job_selection_debug(db=db, job_id=job_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return res


@router.post("/debugger/{job_id}/override", summary="Override a beat's chosen footage candidate")
def override_debugger_selection_endpoint(
    job_id: str,
    body: DebuggerOverrideRequest,
    db: Session = Depends(get_db),
):
    """Overrides the selected clip for a specific beat in a video job."""
    from backend.repository_footage import override_job_beat_selection
    ok = override_job_beat_selection(db=db, job_id=job_id, beat_id=body.beat_id, chunk_id=body.chunk_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Failed to apply selection override.")
    return {"status": "ok", "job_id": job_id, "beat_id": body.beat_id, "chunk_id": body.chunk_id}


@router.post("/eval/compare", response_model=EvalCompareResponse, summary="Compare model retrieval on test queries")
def run_eval_compare_endpoint(
    body: EvalCompareRequest,
    db: Session = Depends(get_db),
):
    """Runs comparative evaluation between Model A and Model B across test queries."""
    from backend.repository_footage import run_eval_comparison
    return run_eval_comparison(
        db=db,
        queries=body.queries,
        model_a=body.model_a,
        model_b=body.model_b,
        top_k=body.top_k,
    )


@router.get("/curation/queue", summary="Get uncurated or triaged shots for curation")
def get_curation_queue_endpoint(
    status: str = Query("pending", description="pending, approved, rejected"),
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Returns shots for rapid keyboard-driven triage."""
    from backend.repository_footage import get_curation_queue
    return get_curation_queue(db=db, status=status, limit=limit, offset=offset)


@router.post("/curation/{chunk_id}", summary="Set shot curation verdict and tags")
def update_curation_endpoint(
    chunk_id: str,
    body: CurationActionRequest,
    db: Session = Depends(get_db),
):
    """Updates a shot's curation status (approved/rejected) and tags."""
    from backend.repository_footage import update_shot_curation
    ok = update_shot_curation(db=db, chunk_id=chunk_id, action=body)
    if not ok:
        raise HTTPException(status_code=404, detail="Shot not found.")
    return {"status": "ok", "chunk_id": chunk_id, "curation_status": body.status}


