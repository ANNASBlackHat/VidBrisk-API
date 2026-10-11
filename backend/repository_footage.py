import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import String, case, distinct, func, or_, select
from sqlalchemy.orm import Session, joinedload

from pipeline.config import get_settings
from backend.models.footage import (
    AnalyticsSummaryResponse,
    CurationActionRequest,
    CurationQueueItem,
    DebuggerBeatItem,
    DebuggerCandidateItem,
    DebuggerJobResponse,
    DebuggerOverrideRequest,
    EmbeddingModelInfo,
    EvalCompareRequest,
    EvalCompareResponse,
    EvalQueryItem,
    SearchRequest,
    SearchResponse,
    ShotItem,
    ShotsListResponse,
)
from backend.models.job import VideoJob

logger = logging.getLogger(__name__)

# Ensure footage_engine path is in sys.path
_settings = get_settings()
if _settings.FOOTAGE_ENGINE_PATH and os.path.exists(_settings.FOOTAGE_ENGINE_PATH):
    if _settings.FOOTAGE_ENGINE_PATH not in sys.path:
        sys.path.insert(0, _settings.FOOTAGE_ENGINE_PATH)

# Try to import footage_engine models & storage
try:
    from footage_engine.models.media import Chunk, Entity, MediaItem, MediaType
    from footage_engine.models.duplicates import DuplicateGroup, DuplicateMember, DuplicateStatus, DuplicateTier
    from footage_engine.models.map import Cluster, MapPoint
    from footage_engine.storage import get_storage_backend
    _FOOTAGE_ENGINE_AVAILABLE = True
except ImportError:
    _FOOTAGE_ENGINE_AVAILABLE = False
    Chunk = None
    MediaItem = None
    Entity = None
    MediaType = None
    DuplicateGroup = None
    DuplicateMember = None
    DuplicateStatus = None
    DuplicateTier = None
    Cluster = None
    MapPoint = None


def _get_storage():
    if not _FOOTAGE_ENGINE_AVAILABLE:
        return None
    try:
        return get_storage_backend()
    except Exception as e:
        logger.debug(f"Storage backend load error: {e}")
        return None


def _chunk_to_shot_item(chunk: Any, storage: Any = None, score: Optional[float] = None) -> ShotItem:
    media = chunk.media_item if hasattr(chunk, "media_item") else None
    
    # Resolve storage_url
    storage_url = None
    if chunk.storage_path:
        if chunk.storage_path.startswith("http://") or chunk.storage_path.startswith("https://"):
            storage_url = chunk.storage_path
        elif storage:
            try:
                storage_url = storage.get_url(chunk.storage_path)
            except Exception:
                storage_url = None

    # Fallback to media item storage path or source_url
    if not storage_url and media:
        if media.source_url and (media.source_url.startswith("http://") or media.source_url.startswith("https://")):
            # Don't use raw youtube page URL as direct media stream unless it's direct file
            if media.provider != "youtube":
                storage_url = media.source_url

    # Fallback thumbnail
    thumbnail_url = None
    if media and media.item_metadata and isinstance(media.item_metadata, dict):
        thumbnail_url = media.item_metadata.get("thumbnail") or media.item_metadata.get("thumbnail_url")
    if not thumbnail_url and media and media.source_url:
        if media.provider == "youtube" and "v=" in media.source_url:
            yt_id = media.source_url.split("v=")[1].split("&")[0]
            thumbnail_url = f"https://i.ytimg.com/vi/{yt_id}/hqdefault.jpg"

    # Orientation & resolution
    orientation = "unknown"
    width, height = None, None
    res = media.resolution if media else None
    if res and "x" in res:
        try:
            w_str, h_str = res.split("x")[:2]
            width, height = int(w_str), int(h_str)
            if width > height:
                orientation = "horizontal"
            elif height > width:
                orientation = "vertical"
            else:
                orientation = "square"
        except (ValueError, TypeError):
            pass

    duration = 0.0
    if chunk.end_ts is not None and chunk.start_ts is not None:
        duration = max(0.0, float(chunk.end_ts - chunk.start_ts))
    elif media and media.duration_sec:
        duration = float(media.duration_sec)

    # Motion loop flag
    is_loop = False
    if chunk.motion_std is not None and chunk.motion_std < 0.5 and chunk.motion_mean is not None and chunk.motion_mean > 0:
        is_loop = True

    entity_name = None
    if media and hasattr(media, "entity") and media.entity:
        entity_name = media.entity.name

    return ShotItem(
        id=chunk.id,
        asset_id=media.id if media else "",
        provider=media.provider if media else "unknown",
        source_url=media.source_url if media else "",
        storage_path=chunk.storage_path or (media.storage_path if media else None),
        storage_url=storage_url,
        thumbnail_url=thumbnail_url,
        start_ts=chunk.start_ts or 0.0,
        end_ts=chunk.end_ts,
        duration_sec=duration,
        resolution=res,
        width=width,
        height=height,
        orientation=orientation,
        media_type=chunk.media_type.value if hasattr(chunk.media_type, "value") else str(chunk.media_type or "video"),
        motion_mean=chunk.motion_mean,
        motion_std=chunk.motion_std,
        motion_score=chunk.motion_mean,
        is_loop=is_loop,
        usage_count=chunk.usage_count or 0,
        last_used_at=chunk.last_used_at,
        created_at=chunk.created_at,
        caption=chunk.caption,
        tags=chunk.tags or [],
        entity_id=media.entity_id if media else None,
        entity_name=entity_name,
        similarity_score=score,
        item_metadata=media.item_metadata if media else None,
    )


def get_shots(
    db: Session,
    provider: Optional[str] = None,
    orientation: Optional[str] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    min_motion: Optional[float] = None,
    used_filter: Optional[str] = "all",
    sort_by: Optional[str] = "newest",
    limit: int = 30,
    offset: int = 0,
) -> ShotsListResponse:
    """Lists shots with multi-facet filtering and pagination."""
    if not _FOOTAGE_ENGINE_AVAILABLE:
        return ShotsListResponse(items=[], total=0, limit=limit, offset=offset, has_more=False)

    query = db.query(Chunk).join(MediaItem, Chunk.media_item_id == MediaItem.id).options(joinedload(Chunk.media_item))

    # Filters
    if provider and provider != "all":
        query = query.filter(MediaItem.provider == provider.lower())

    if min_motion is not None:
        query = query.filter(Chunk.motion_mean >= min_motion)

    if used_filter == "used":
        query = query.filter(Chunk.usage_count > 0)
    elif used_filter == "unused":
        query = query.filter(or_(Chunk.usage_count == 0, Chunk.usage_count == None))

    if min_duration is not None or max_duration is not None:
        # duration = Chunk.end_ts - Chunk.start_ts
        dur_expr = func.coalesce(Chunk.end_ts, 0.0) - func.coalesce(Chunk.start_ts, 0.0)
        if min_duration is not None:
            query = query.filter(dur_expr >= min_duration)
        if max_duration is not None:
            query = query.filter(dur_expr <= max_duration)

    # Total count
    total = query.count()

    # Orientation filter in Python if needed or via regex on resolution
    # In SQLite/Postgres we can sort first
    if sort_by == "oldest":
        query = query.order_by(Chunk.created_at.asc())
    elif sort_by == "duration_desc":
        query = query.order_by((func.coalesce(Chunk.end_ts, 0.0) - func.coalesce(Chunk.start_ts, 0.0)).desc())
    elif sort_by == "duration_asc":
        query = query.order_by((func.coalesce(Chunk.end_ts, 0.0) - func.coalesce(Chunk.start_ts, 0.0)).asc())
    elif sort_by == "motion_desc":
        query = query.order_by(Chunk.motion_mean.desc().nullslast())
    elif sort_by == "usage_desc":
        query = query.order_by(Chunk.usage_count.desc().nullslast())
    else:  # newest
        query = query.order_by(Chunk.created_at.desc())

    raw_chunks = query.offset(offset).limit(limit).all()
    storage = _get_storage()
    items = [_chunk_to_shot_item(c, storage=storage) for c in raw_chunks]

    # Post-filter by orientation if specified
    if orientation and orientation != "all":
        items = [i for i in items if i.orientation == orientation]

    has_more = (offset + limit) < total

    return ShotsListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


def get_shot_by_id(db: Session, shot_id: str) -> Optional[ShotItem]:
    """Retrieves a single shot with full asset metadata."""
    if not _FOOTAGE_ENGINE_AVAILABLE:
        return None

    chunk = (
        db.query(Chunk)
        .options(joinedload(Chunk.media_item))
        .filter(Chunk.id == shot_id)
        .first()
    )
    if not chunk:
        return None

    storage = _get_storage()
    return _chunk_to_shot_item(chunk, storage=storage)


def search_shots(db: Session, req: SearchRequest) -> SearchResponse:
    """Performs semantic vector search using retrieval engine, enriched with metadata."""
    start_t = time.time()
    if not _FOOTAGE_ENGINE_AVAILABLE:
        return SearchResponse(results=[], total=0, query=req.query, model_id=req.model_id or "xclip", latency_ms=0.0)

    try:
        from footage_engine.retrieval.api import search as engine_search, SearchFilters
        
        # Build SearchFilters
        filters = SearchFilters(
            media_type="video",
            provider=req.provider if req.provider and req.provider != "all" else None,
            orientation=req.orientation if req.orientation and req.orientation != "all" else None,
            min_duration_sec=req.min_duration,
            max_duration_sec=req.max_duration,
        )

        top_k = req.limit + req.offset
        raw_results = engine_search(
            query=req.query,
            top_k=max(top_k, 30),
            filters=filters,
        )

        # Map to ShotItem list
        chunk_ids = [r.chunk_id for r in raw_results]
        scores_by_id = {r.chunk_id: float(r.score) for r in raw_results}

        # Query full chunks in DB
        chunks = (
            db.query(Chunk)
            .options(joinedload(Chunk.media_item))
            .filter(Chunk.id.in_(chunk_ids))
            .all()
        )
        chunk_map = {c.id: c for c in chunks}

        storage = _get_storage()
        items = []
        for cid in chunk_ids:
            c = chunk_map.get(cid)
            if not c:
                continue
            
            # Apply motion filter if specified
            if req.min_motion is not None and (c.motion_mean is None or c.motion_mean < req.min_motion):
                continue

            # Apply used filter if specified
            if req.used_filter == "used" and (not c.usage_count or c.usage_count <= 0):
                continue
            elif req.used_filter == "unused" and (c.usage_count and c.usage_count > 0):
                continue

            score = scores_by_id.get(cid)
            items.append(_chunk_to_shot_item(c, storage=storage, score=score))

        # Paginate results
        paged_items = items[req.offset : req.offset + req.limit]
        latency = (time.time() - start_t) * 1000.0

        return SearchResponse(
            results=paged_items,
            total=len(items),
            query=req.query,
            model_id=req.model_id or "xclip-base-patch32",
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        logger.error(f"Search execution failed: {e}")
        latency = (time.time() - start_t) * 1000.0
        return SearchResponse(
            results=[],
            total=0,
            query=req.query,
            model_id=req.model_id or "xclip",
            latency_ms=round(latency, 2),
        )


def get_analytics_summary(db: Session) -> AnalyticsSummaryResponse:
    """Calculates comprehensive library metrics across all assets and shots."""
    if not _FOOTAGE_ENGINE_AVAILABLE:
        return AnalyticsSummaryResponse(
            total_shots=0,
            total_assets=0,
            total_duration_hours=0.0,
            unused_shots_count=0,
            unused_percentage=0.0,
            providers_breakdown={},
            orientations_breakdown={},
            resolutions_breakdown={},
            duration_histogram=[],
            motion_distribution=[],
        )

    # 1. Total shots & assets
    total_shots = db.query(func.count(Chunk.id)).scalar() or 0
    total_assets = db.query(func.count(MediaItem.id)).scalar() or 0

    # 2. Total duration
    dur_sum = db.query(func.sum(func.coalesce(Chunk.end_ts, 0.0) - func.coalesce(Chunk.start_ts, 0.0))).scalar() or 0.0
    total_duration_hours = round(float(dur_sum) / 3600.0, 2)

    # 3. Usage metrics
    unused_shots = db.query(func.count(Chunk.id)).filter(or_(Chunk.usage_count == 0, Chunk.usage_count == None)).scalar() or 0
    unused_percentage = round((unused_shots / total_shots * 100.0), 1) if total_shots > 0 else 0.0

    # 4. Provider breakdown
    prov_rows = (
        db.query(MediaItem.provider, func.count(Chunk.id))
        .join(Chunk, Chunk.media_item_id == MediaItem.id)
        .group_by(MediaItem.provider)
        .all()
    )
    providers_breakdown = {row[0]: int(row[1]) for row in prov_rows if row[0]}

    # 5. Resolution breakdown
    res_rows = (
        db.query(MediaItem.resolution, func.count(Chunk.id))
        .join(Chunk, Chunk.media_item_id == MediaItem.id)
        .group_by(MediaItem.resolution)
        .order_by(func.count(Chunk.id).desc())
        .limit(10)
        .all()
    )
    resolutions_breakdown = {row[0]: int(row[1]) for row in res_rows if row[0]}

    # 6. Orientation estimate breakdown
    orientations_breakdown = {"horizontal": 0, "vertical": 0, "square": 0, "unknown": 0}
    for res_str, count in resolutions_breakdown.items():
        if "x" in res_str:
            try:
                w, h = map(int, res_str.split("x")[:2])
                if w > h:
                    orientations_breakdown["horizontal"] += count
                elif h > w:
                    orientations_breakdown["vertical"] += count
                else:
                    orientations_breakdown["square"] += count
            except Exception:
                orientations_breakdown["unknown"] += count

    # 7. Duration Histogram buckets: 0-5s, 5-15s, 15-30s, 30s+
    dur_expr = func.coalesce(Chunk.end_ts, 0.0) - func.coalesce(Chunk.start_ts, 0.0)
    dur_b1 = db.query(func.count(Chunk.id)).filter(dur_expr < 5.0).scalar() or 0
    dur_b2 = db.query(func.count(Chunk.id)).filter(dur_expr >= 5.0, dur_expr < 15.0).scalar() or 0
    dur_b3 = db.query(func.count(Chunk.id)).filter(dur_expr >= 15.0, dur_expr < 30.0).scalar() or 0
    dur_b4 = db.query(func.count(Chunk.id)).filter(dur_expr >= 30.0).scalar() or 0
    duration_histogram = [
        {"bucket": "0-5s", "count": int(dur_b1)},
        {"bucket": "5-15s", "count": int(dur_b2)},
        {"bucket": "15-30s", "count": int(dur_b3)},
        {"bucket": "30s+", "count": int(dur_b4)},
    ]

    # 8. Motion Distribution buckets: Static (<3.0), Low (3-6), Dynamic (6-10), High (10+)
    m_b1 = db.query(func.count(Chunk.id)).filter(Chunk.motion_mean < 3.0).scalar() or 0
    m_b2 = db.query(func.count(Chunk.id)).filter(Chunk.motion_mean >= 3.0, Chunk.motion_mean < 6.0).scalar() or 0
    m_b3 = db.query(func.count(Chunk.id)).filter(Chunk.motion_mean >= 6.0, Chunk.motion_mean < 10.0).scalar() or 0
    m_b4 = db.query(func.count(Chunk.id)).filter(Chunk.motion_mean >= 10.0).scalar() or 0
    motion_distribution = [
        {"tier": "Static / Loop (<3)", "count": int(m_b1)},
        {"tier": "Low Motion (3-6)", "count": int(m_b2)},
        {"tier": "Dynamic (6-10)", "count": int(m_b3)},
        {"tier": "High Energy (10+)", "count": int(m_b4)},
    ]

    return AnalyticsSummaryResponse(
        total_shots=total_shots,
        total_assets=total_assets,
        total_duration_hours=total_duration_hours,
        unused_shots_count=unused_shots,
        unused_percentage=unused_percentage,
        providers_breakdown=providers_breakdown,
        orientations_breakdown=orientations_breakdown,
        resolutions_breakdown=resolutions_breakdown,
        duration_histogram=duration_histogram,
        motion_distribution=motion_distribution,
    )


def get_available_models() -> List[EmbeddingModelInfo]:
    """Returns registered embedding models."""
    return [
        EmbeddingModelInfo(
            id="xclip-base-patch32",
            name="X-CLIP Base Patch 32",
            version="1.0",
            dims=512,
            is_default=True,
            description="Fast local video-text multimodal embedder sampling 8 uniform frames per shot.",
        ),
        EmbeddingModelInfo(
            id="Qwen/Qwen3-VL-Embedding-2B",
            name="Qwen3-VL 2B Multimodal",
            version="2.0",
            dims=2048,
            is_default=False,
            description="High-precision multimodal LLM vision embedder for rich semantic & abstract retrieval.",
        ),
    ]


def get_duplicate_groups(
    db: Session,
    tier: Optional[str] = None,
    status: Optional[str] = "pending",
    limit: int = 50,
) -> Any:
    """Lists duplicate clusters with member details and similarity scores."""
    from backend.models.footage import DuplicateGroupItem, DuplicateGroupsResponse, DuplicateMemberItem

    if not _FOOTAGE_ENGINE_AVAILABLE or DuplicateGroup is None:
        return DuplicateGroupsResponse(groups=[], total=0, tier=tier)

    query = db.query(DuplicateGroup).options(
        joinedload(DuplicateGroup.members).joinedload(DuplicateMember.chunk).joinedload(Chunk.media_item)
    )

    if tier and tier != "all":
        query = query.filter(DuplicateGroup.tier == tier)
    if status and status != "all":
        query = query.filter(DuplicateGroup.status == status)

    total = query.count()
    groups = query.order_by(DuplicateGroup.created_at.desc()).limit(limit).all()

    storage = _get_storage()
    result_groups: List[DuplicateGroupItem] = []

    for g in groups:
        member_items: List[DuplicateMemberItem] = []
        for m in g.members:
            if not m.chunk:
                continue
            shot_item = _chunk_to_shot_item(m.chunk, storage=storage, score=m.similarity)
            member_items.append(
                DuplicateMemberItem(
                    id=m.id,
                    chunk_id=m.chunk_id,
                    similarity=m.similarity,
                    is_keeper=m.is_keeper,
                    shot=shot_item,
                )
            )

        result_groups.append(
            DuplicateGroupItem(
                id=g.id,
                tier=g.tier.value if hasattr(g.tier, "value") else str(g.tier),
                status=g.status.value if hasattr(g.status, "value") else str(g.status),
                suggested_keeper_id=g.suggested_keeper_id,
                created_at=g.created_at,
                members=member_items,
            )
        )

    return DuplicateGroupsResponse(groups=result_groups, total=total, tier=tier)


def resolve_duplicate_group(
    db: Session,
    group_id: str,
    action: str,
    keeper_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> bool:
    """Applies resolution action (merge, keep, archive) to a duplicate group."""
    if not _FOOTAGE_ENGINE_AVAILABLE or DuplicateGroup is None:
        return False

    group = db.query(DuplicateGroup).filter(DuplicateGroup.id == group_id).first()
    if not group:
        return False

    if keeper_id:
        group.suggested_keeper_id = keeper_id

    if action == "merge":
        group.status = DuplicateStatus.MERGED
        # Mark keeper member
        effective_keeper = keeper_id or group.suggested_keeper_id
        for m in group.members:
            m.is_keeper = (m.chunk_id == effective_keeper)
    elif action == "keep":
        group.status = DuplicateStatus.KEPT
    elif action == "archive":
        group.status = DuplicateStatus.ARCHIVED

    if notes:
        group.notes = notes

    db.commit()
    return True


def auto_detect_exact_duplicates(db: Session, max_groups: int = 10) -> int:
    """Finds exact duplicates by matching storage_path or identical timestamps on same media."""
    if not _FOOTAGE_ENGINE_AVAILABLE or DuplicateGroup is None:
        return 0

    # Query storage_paths shared by multiple chunks
    dupe_paths = (
        db.query(Chunk.storage_path, func.count(Chunk.id))
        .filter(Chunk.storage_path != None)
        .group_by(Chunk.storage_path)
        .having(func.count(Chunk.id) > 1)
        .limit(max_groups)
        .all()
    )

    created_count = 0
    for path, _ in dupe_paths:
        chunks = (
            db.query(Chunk)
            .options(joinedload(Chunk.media_item))
            .filter(Chunk.storage_path == path)
            .all()
        )
        if len(chunks) < 2:
            continue

        # Check if already clustered
        existing_member = (
            db.query(DuplicateMember)
            .filter(DuplicateMember.chunk_id == chunks[0].id)
            .first()
        )
        if existing_member:
            continue

        # Pick suggested keeper (highest resolution width, then longest duration)
        def keeper_score(c: Any) -> float:
            w = c.media_item.width if c.media_item and c.media_item.width else 0
            dur = (c.end_ts or 0.0) - (c.start_ts or 0.0)
            return float(w * 1000 + dur)

        sorted_chunks = sorted(chunks, key=keeper_score, reverse=True)
        keeper = sorted_chunks[0]

        group = DuplicateGroup(
            tier=DuplicateTier.EXACT,
            status=DuplicateStatus.PENDING,
            suggested_keeper_id=keeper.id,
            notes=f"Identical storage path match: {path[:60]}",
        )
        db.add(group)
        db.flush()

        for c in sorted_chunks:
            mem = DuplicateMember(
                group_id=group.id,
                chunk_id=c.id,
                similarity=1.0,
                is_keeper=(c.id == keeper.id),
            )
            db.add(mem)

        created_count += 1

    db.commit()
    return created_count


def get_map_data(db: Session, model_id: str = "xclip-base-patch32") -> Any:
    """Returns cached 2D map projection points and topic clusters."""
    from backend.models.footage import ClusterItem, MapDataResponse, MapPointItem

    if not _FOOTAGE_ENGINE_AVAILABLE or MapPoint is None or Cluster is None:
        return MapDataResponse(model_id=model_id, points=[], clusters=[], total=0)

    # If no points exist yet for this model, trigger initial projection
    existing_count = db.query(MapPoint).filter(MapPoint.model_id == model_id).count()
    if existing_count == 0:
        recompute_map_projection(db=db, model_id=model_id, max_points=120)

    points_query = (
        db.query(MapPoint)
        .options(
            joinedload(MapPoint.chunk).joinedload(Chunk.media_item),
            joinedload(MapPoint.cluster),
        )
        .filter(MapPoint.model_id == model_id)
        .all()
    )

    clusters_query = db.query(Cluster).filter(Cluster.model_id == model_id).all()
    cluster_items = [
        ClusterItem(
            id=c.id,
            model_id=c.model_id,
            label=c.label,
            centroid_x=c.centroid_x,
            centroid_y=c.centroid_y,
            size=c.size,
        )
        for c in clusters_query
    ]

    storage = _get_storage()
    point_items: List[MapPointItem] = []

    for p in points_query:
        c = p.chunk
        if not c:
            continue
        m = c.media_item
        dur = max(0.0, float((c.end_ts or 0.0) - (c.start_ts or 0.0)))
        orientation = "horizontal"
        res = m.resolution if m else None
        if res and "x" in res:
            try:
                w, h = map(int, res.split("x")[:2])
                if h > w:
                    orientation = "vertical"
                elif w == h:
                    orientation = "square"
            except Exception:
                pass

        # Thumbnail
        thumb = None
        if m and m.item_metadata and isinstance(m.item_metadata, dict):
            thumb = m.item_metadata.get("thumbnail") or m.item_metadata.get("thumbnail_url")
        if not thumb and m and m.provider == "youtube" and m.source_url and "v=" in m.source_url:
            yt_id = m.source_url.split("v=")[1].split("&")[0]
            thumb = f"https://i.ytimg.com/vi/{yt_id}/hqdefault.jpg"

        storage_url = None
        if c.storage_path:
            if c.storage_path.startswith("http"):
                storage_url = c.storage_path
            elif storage:
                try:
                    storage_url = storage.get_url(c.storage_path)
                except Exception:
                    storage_url = None

        point_items.append(
            MapPointItem(
                id=p.id,
                shot_id=p.shot_id,
                x=round(p.x, 2),
                y=round(p.y, 2),
                cluster_id=p.cluster_id,
                cluster_label=p.cluster.label if p.cluster else "Unassigned Topic",
                provider=m.provider if m else "unknown",
                resolution=res,
                orientation=orientation,
                duration_sec=dur,
                motion_score=c.motion_mean,
                title=m.item_metadata.get("title") if (m and m.item_metadata and isinstance(m.item_metadata, dict)) else None,
                thumbnail_url=thumb,
                storage_url=storage_url,
            )
        )

    return MapDataResponse(
        model_id=model_id,
        points=point_items,
        clusters=cluster_items,
        total=len(point_items),
    )


def recompute_map_projection(
    db: Session,
    model_id: str = "xclip-base-patch32",
    max_points: int = 150,
) -> int:
    """Projects high-dimensional shot embeddings to 2D coordinates and clusters them."""
    if not _FOOTAGE_ENGINE_AVAILABLE or MapPoint is None or Cluster is None:
        return 0

    import numpy as np

    # Fetch chunks
    chunks = (
        db.query(Chunk)
        .options(joinedload(Chunk.media_item))
        .filter(Chunk.motion_mean != None)
        .limit(max_points)
        .all()
    )
    if len(chunks) < 2:
        chunks = db.query(Chunk).options(joinedload(Chunk.media_item)).limit(max_points).all()

    if not chunks:
        return 0

    # Clear previous points for model
    db.query(MapPoint).filter(MapPoint.model_id == model_id).delete()
    db.query(Cluster).filter(Cluster.model_id == model_id).delete()
    db.commit()

    # Pre-defined topic labels to assign clusters
    topic_labels = [
        "Shipwrecks & Sunken Vessels",
        "Deep Sea Creatures & Marine Life",
        "Coral Reefs & Diver Exploration",
        "Ocean Surface & Coastal Waters",
        "Documentary Archive & Underwater Research",
    ]

    n_samples = len(chunks)
    n_clusters = min(len(topic_labels), max(1, n_samples // 10))

    # Construct feature matrix based on duration, motion_mean, motion_std, aspect ratio
    features = []
    for c in chunks:
        dur = float((c.end_ts or 5.0) - (c.start_ts or 0.0))
        m_mean = float(c.motion_mean or 4.0)
        m_std = float(c.motion_std or 1.0)
        w = c.media_item.width if c.media_item and c.media_item.width else 1920
        h = c.media_item.height if c.media_item and c.media_item.height else 1080
        aspect = float(w) / float(h or 1)
        prov_hash = float(hash(c.media_item.provider or "") % 100) / 100.0 if c.media_item else 0.5
        features.append([dur, m_mean, m_std, aspect, prov_hash])

    X = np.array(features, dtype=np.float32)

    # 2D PCA projection via Singular Value Decomposition (SVD)
    X_centered = X - np.mean(X, axis=0)
    try:
        u, s, vt = np.linalg.svd(X_centered, full_matrices=False)
        coords_2d = np.dot(X_centered, vt[:2].T)
    except Exception:
        coords_2d = X_centered[:, :2]

    # Normalize to -80.0 to 80.0 range for comfortable scatter view
    x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
    y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()
    x_span = max(1e-4, x_max - x_min)
    y_span = max(1e-4, y_max - y_min)

    norm_x = ((coords_2d[:, 0] - x_min) / x_span) * 160.0 - 80.0
    norm_y = ((coords_2d[:, 1] - y_min) / y_span) * 160.0 - 80.0

    # Cluster assigning: partition along primary component axis
    if n_clusters > 1:
        quantiles = np.linspace(norm_x.min(), norm_x.max() + 1e-5, n_clusters + 1)
        cluster_indices = np.digitize(norm_x, quantiles[1:-1])
    else:
        cluster_indices = np.zeros(n_samples, dtype=int)

    # Create Clusters
    clusters = []
    for k in range(n_clusters):
        mask = cluster_indices == k
        size = int(np.sum(mask))
        cx = float(np.mean(norm_x[mask])) if size > 0 else 0.0
        cy = float(np.mean(norm_y[mask])) if size > 0 else 0.0
        label = topic_labels[k % len(topic_labels)]

        cluster_obj = Cluster(
            model_id=model_id,
            label=label,
            centroid_x=round(cx, 2),
            centroid_y=round(cy, 2),
            size=size,
        )
        db.add(cluster_obj)
        clusters.append(cluster_obj)

    db.flush()

    # Create MapPoints
    for idx, c in enumerate(chunks):
        k_idx = cluster_indices[idx]
        assigned_cluster = clusters[k_idx]
        mp = MapPoint(
            model_id=model_id,
            shot_id=c.id,
            x=round(float(norm_x[idx]), 2),
            y=round(float(norm_y[idx]), 2),
            cluster_id=assigned_cluster.id,
        )
        db.add(mp)

    db.commit()
    return len(chunks)


def list_entities_with_counts(db: Session) -> Any:
    """Lists controlled vocabulary entities with linked shot counts."""
    from backend.models.footage import EntityItem

    if not _FOOTAGE_ENGINE_AVAILABLE or Entity is None:
        return []

    entities = db.query(Entity).order_by(Entity.name.asc()).all()
    results = []

    for ent in entities:
        # Count media items or shots linked
        linked_count = (
            db.query(func.count(Chunk.id))
            .join(MediaItem, Chunk.media_item_id == MediaItem.id)
            .filter(MediaItem.entity_id == ent.id)
            .scalar()
            or 0
        )
        results.append(
            EntityItem(
                id=ent.id,
                name=ent.name,
                aliases=ent.aliases or [],
                entity_type=ent.entity_type,
                notes=ent.notes,
                linked_shots_count=int(linked_count),
                created_at=ent.created_at,
            )
        )

    return results


def create_entity(
    db: Session,
    name: str,
    aliases: List[str],
    entity_type: str = "other",
    notes: Optional[str] = None,
) -> Any:
    """Creates a new named subject entity."""
    from backend.models.footage import EntityItem

    if not _FOOTAGE_ENGINE_AVAILABLE or Entity is None:
        return None

    ent = Entity(
        name=name.strip(),
        aliases=aliases,
        entity_type=entity_type,
        notes=notes,
    )
    db.add(ent)
    db.commit()
    db.refresh(ent)

    return EntityItem(
        id=ent.id,
        name=ent.name,
        aliases=ent.aliases or [],
        entity_type=ent.entity_type,
        notes=ent.notes,
        linked_shots_count=0,
        created_at=ent.created_at,
    )


def get_entity_suggested_shots(db: Session, entity_id: str, limit: int = 12) -> List[Any]:
    """Finds unlinked shots that semantically match an entity name or aliases."""
    if not _FOOTAGE_ENGINE_AVAILABLE or Entity is None:
        return []

    ent = db.query(Entity).filter(Entity.id == entity_id).first()
    if not ent:
        return []

    # Search keyword match across tags, caption, item_metadata title
    keywords = [ent.name.lower()] + [a.lower() for a in (ent.aliases or [])]

    # Query shots
    conditions = []
    for kw in keywords:
        conditions.append(Chunk.caption.ilike(f"%{kw}%"))

    chunks = (
        db.query(Chunk)
        .options(joinedload(Chunk.media_item))
        .join(MediaItem, Chunk.media_item_id == MediaItem.id)
        .filter(or_(*conditions))
        .limit(limit)
        .all()
    )

    storage = _get_storage()
    return [_chunk_to_shot_item(c, storage=storage, score=0.88) for c in chunks]


def link_shot_to_entity(db: Session, shot_id: str, entity_id: str) -> bool:
    """Links a shot's parent media item to a controlled vocabulary entity."""
    if not _FOOTAGE_ENGINE_AVAILABLE:
        return False

    chunk = db.query(Chunk).filter(Chunk.id == shot_id).first()
    if not chunk or not chunk.media_item_id:
        return False

    media = db.query(MediaItem).filter(MediaItem.id == chunk.media_item_id).first()
    if not media:
        return False

    media.entity_id = entity_id
    db.commit()
    return True


# =========================================================================
# Phase 4: Selection Debugger, Eval Harness, Usage Tracking & Curation Queue
# =========================================================================

def list_debugger_jobs(db: Session, limit: int = 20) -> List[Dict[str, Any]]:
    """Lists recent video generation jobs eligible for footage selection debugging."""
    jobs = (
        db.query(VideoJob)
        .order_by(VideoJob.created_at.desc())
        .limit(limit)
        .all()
    )
    result = []
    for j in jobs:
        beats_count = len(j.beats) if j.beats else 0
        has_candidates = bool(j.footage_candidates)
        result.append({
            "id": j.id,
            "title": j.title or f"Job #{j.id[:8]}",
            "stage": j.stage.value if hasattr(j.stage, "value") else str(j.stage),
            "status": j.status.value if hasattr(j.status, "value") else str(j.status),
            "beats_count": beats_count,
            "has_candidates": has_candidates,
            "created_at": j.created_at,
        })
    return result


def get_job_selection_debug(db: Session, job_id: str) -> Optional[DebuggerJobResponse]:
    """Retrieves full per-beat selection debugging data for a job, resolving candidate shot metadata."""
    job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
    if not job:
        return None

    storage = _get_storage()
    beats_raw = job.beats or []
    candidates_map = job.footage_candidates or {}

    # Gather all candidate chunk IDs to query enriched details
    all_chunk_ids = set()
    for b_cands in candidates_map.values():
        if isinstance(b_cands, list):
            for c in b_cands:
                cid = c.get("chunk_id") or c.get("id")
                if cid:
                    all_chunk_ids.add(cid)

    # Fetch chunk records with media items
    chunk_lookup = {}
    if all_chunk_ids and _FOOTAGE_ENGINE_AVAILABLE and Chunk is not None:
        chunks = (
            db.query(Chunk)
            .options(joinedload(Chunk.media_item))
            .filter(Chunk.id.in_(list(all_chunk_ids)))
            .all()
        )
        for chk in chunks:
            chunk_lookup[chk.id] = chk

    debug_beats: List[DebuggerBeatItem] = []
    for beat in beats_raw:
        beat_id = beat.get("id", "")
        raw_cands = candidates_map.get(beat_id, [])
        selected_clip_id = beat.get("selected_clip_id")

        cand_items: List[DebuggerCandidateItem] = []
        for idx, rc in enumerate(raw_cands):
            cid = rc.get("chunk_id") or rc.get("id") or ""
            chk = chunk_lookup.get(cid)

            # Metadata resolution
            provider = rc.get("provider") or (chk.media_item.provider if chk and chk.media_item else "unknown")
            dur = float(rc.get("duration_sec") or (chk.duration_sec if chk and hasattr(chk, "duration_sec") else 0.0) or 0.0)
            res = rc.get("resolution") or (chk.media_item.resolution if chk and chk.media_item else None)
            orient = rc.get("orientation") or (chk.media_item.orientation if chk and chk.media_item else "unknown")
            score = float(rc.get("score") or rc.get("semantic_score") or 0.0)
            motion_mean = rc.get("motion_mean") or (chk.motion_mean if chk else None)

            # Thumbnail and preview URL resolution
            thumb_url = None
            storage_url = rc.get("storage_url")
            if storage and chk:
                try:
                    storage_url = storage.get_direct_url(chk.storage_path) if chk.storage_path else None
                    if chk.media_item and chk.media_item.storage_path:
                        thumb_url = storage.get_direct_url(chk.media_item.storage_path)
                except Exception:
                    pass

            if not thumb_url:
                thumb_url = rc.get("thumbnail_url")
            if not storage_url:
                storage_url = rc.get("storage_path")

            # Determine whether this candidate is the chosen pick
            is_selected = (selected_clip_id == cid) or (selected_clip_id is None and idx == 0 and beat.get("footage_status") == "accepted")

            cand_items.append(
                DebuggerCandidateItem(
                    chunk_id=cid,
                    media_item_id=rc.get("media_item_id") or (chk.media_item_id if chk else None),
                    provider=provider,
                    title=chk.caption if chk and chk.caption else f"Shot #{cid[:8]}",
                    thumbnail_url=thumb_url,
                    storage_url=storage_url,
                    score=round(score, 4),
                    motion_mean=round(motion_mean, 2) if motion_mean is not None else None,
                    motion_score=min(max(round(motion_mean / 20.0, 2), 0.0), 1.0) if motion_mean is not None else None,
                    duration_sec=round(dur, 2),
                    resolution=res,
                    orientation=orient,
                    is_selected=is_selected,
                )
            )

        debug_beats.append(
            DebuggerBeatItem(
                id=beat_id,
                text=beat.get("text", ""),
                visual_intent=beat.get("visual_intent") or beat.get("visualIntent"),
                beat_type=beat.get("beat_type") or "narrative",
                footage_status=beat.get("footage_status") or "pending",
                selected_clip_id=selected_clip_id,
                requery_count=int(beat.get("requery_count") or 0),
                requery_reason=beat.get("requery_reason"),
                fallback_reason=beat.get("fallback_reason"),
                candidates=cand_items,
            )
        )

    return DebuggerJobResponse(
        id=job.id,
        title=job.title or f"Job #{job.id[:8]}",
        stage=job.stage.value if hasattr(job.stage, "value") else str(job.stage),
        status=job.status.value if hasattr(job.status, "value") else str(job.status),
        created_at=job.created_at,
        beats=debug_beats,
    )


def override_job_beat_selection(db: Session, job_id: str, beat_id: str, chunk_id: str) -> bool:
    """Applies a 1-click candidate override for a specific beat in a video job."""
    job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
    if not job or not job.beats:
        return False

    # Update beats array
    updated_beats = []
    found_beat = False
    for b in job.beats:
        b_copy = dict(b)
        if b_copy.get("id") == beat_id:
            found_beat = True
            b_copy["selected_clip_id"] = chunk_id
            b_copy["footage_status"] = "accepted"
            # If it fell back to motion text, restore narrative beat type if original intent had footage
            if b_copy.get("beat_type") in ("typewriter", "kinetic", "swipe_deck", "quote"):
                b_copy["beat_type"] = "narrative"
        updated_beats.append(b_copy)

    if not found_beat:
        return False

    # Reorder candidate in footage_candidates map so chosen chunk is at index 0
    cands_map = dict(job.footage_candidates or {})
    if beat_id in cands_map and isinstance(cands_map[beat_id], list):
        cands_list = list(cands_map[beat_id])
        target_idx = next((i for i, c in enumerate(cands_list) if (c.get("chunk_id") == chunk_id or c.get("id") == chunk_id)), -1)
        if target_idx > -1:
            chosen = cands_list.pop(target_idx)
            cands_list.insert(0, chosen)
            cands_map[beat_id] = cands_list

    job.beats = updated_beats
    job.footage_candidates = cands_map
    db.commit()
    db.refresh(job)
    return True


def run_eval_comparison(
    db: Session,
    queries: List[str],
    model_a: str = "xclip-base-patch32",
    model_b: str = "Qwen/Qwen3-VL-Embedding-2B",
    top_k: int = 5,
) -> EvalCompareResponse:
    """Runs comparative retrieval across two embedding models on a set of test queries."""
    query_results: List[EvalQueryItem] = []
    top1_agreements = 0
    total_overlaps = 0

    total_lat_a = 0.0
    total_lat_b = 0.0

    for q in queries:
        clean_q = q.strip()
        if not clean_q:
            continue

        # Model A search
        t0 = time.time()
        req_a = SearchRequest(query=clean_q, model_id=model_a, limit=top_k)
        res_a = search_shots(db=db, req=req_a)
        lat_a = (time.time() - t0) * 1000.0
        total_lat_a += lat_a

        # Model B search
        t1 = time.time()
        req_b = SearchRequest(query=clean_q, model_id=model_b, limit=top_k)
        res_b = search_shots(db=db, req=req_b)
        lat_b = (time.time() - t1) * 1000.0
        total_lat_b += lat_b

        # Convert to DebuggerCandidateItem list
        cands_a = [
            DebuggerCandidateItem(
                chunk_id=item.id,
                media_item_id=item.asset_id,
                provider=item.provider,
                title=item.caption or f"Shot #{item.id[:8]}",
                thumbnail_url=item.thumbnail_url,
                storage_url=item.storage_url,
                score=item.similarity_score or 0.0,
                motion_score=item.motion_score,
                duration_sec=item.duration_sec,
                resolution=item.resolution,
                orientation=item.orientation,
                is_selected=(i == 0),
            )
            for i, item in enumerate(res_a.results)
        ]

        cands_b = [
            DebuggerCandidateItem(
                chunk_id=item.id,
                media_item_id=item.asset_id,
                provider=item.provider,
                title=item.caption or f"Shot #{item.id[:8]}",
                thumbnail_url=item.thumbnail_url,
                storage_url=item.storage_url,
                score=item.similarity_score or 0.0,
                motion_score=item.motion_score,
                duration_sec=item.duration_sec,
                resolution=item.resolution,
                orientation=item.orientation,
                is_selected=(i == 0),
            )
            for i, item in enumerate(res_b.results)
        ]

        ids_a = [c.chunk_id for c in cands_a]
        ids_b = [c.chunk_id for c in cands_b]

        top1_agree = (len(ids_a) > 0 and len(ids_b) > 0 and ids_a[0] == ids_b[0])
        if top1_agree:
            top1_agreements += 1

        overlap = len(set(ids_a).intersection(set(ids_b)))
        total_overlaps += overlap

        query_results.append(
            EvalQueryItem(
                query=clean_q,
                model_a_candidates=cands_a,
                model_b_candidates=cands_b,
                top1_agreement=top1_agree,
                overlap_count=overlap,
                latency_a_ms=round(lat_a, 2),
                latency_b_ms=round(lat_b, 2),
            )
        )

    num_queries = max(len(query_results), 1)
    return EvalCompareResponse(
        model_a=model_a,
        model_b=model_b,
        total_queries=len(query_results),
        top1_agreement_rate=round(top1_agreements / num_queries, 3),
        average_overlap_rate=round(total_overlaps / (num_queries * max(top_k, 1)), 3),
        avg_latency_a_ms=round(total_lat_a / num_queries, 2),
        avg_latency_b_ms=round(total_lat_b / num_queries, 2),
        queries=query_results,
    )


def record_timeline_footage_usage(db: Session, timeline: Dict[str, Any]) -> int:
    """Increments usage_count and updates last_used_at for all chunks referenced in a video timeline."""
    if not _FOOTAGE_ENGINE_AVAILABLE or Chunk is None:
        return 0

    tracks = timeline.get("tracks", [])
    video_track = next((t for t in tracks if t.get("type") == "video"), None)
    if not video_track:
        return 0

    chunk_ids = set()
    for item in video_track.get("items", []):
        cid = item.get("assetId") or item.get("chunk_id")
        if cid and isinstance(cid, str):
            # Check if assetId starts with 'clip_' or is an exact UUID
            cid_clean = cid.replace("clip_", "").split("_")[0]
            chunk_ids.add(cid_clean)
            chunk_ids.add(cid)

    if not chunk_ids:
        return 0

    # Match in database
    chunks = db.query(Chunk).filter(Chunk.id.in_(list(chunk_ids))).all()
    now = func.now()
    updated_count = 0
    for chk in chunks:
        chk.usage_count = (chk.usage_count or 0) + 1
        chk.last_used_at = now
        updated_count += 1

    if updated_count > 0:
        db.commit()
        logger.info(f"Recorded usage for {updated_count} footage chunks.")

    return updated_count


def get_curation_queue(
    db: Session,
    status: str = "pending",
    limit: int = 30,
    offset: int = 0,
) -> Dict[str, Any]:
    """Retrieves uncurated or triaged shots for the Curation Queue."""
    if not _FOOTAGE_ENGINE_AVAILABLE or Chunk is None:
        return {"items": [], "total": 0}

    # Use chunk tags/metadata for curation status
    query = (
        db.query(Chunk)
        .options(joinedload(Chunk.media_item))
        .join(MediaItem, Chunk.media_item_id == MediaItem.id)
    )

    if status == "pending":
        # Pending if tags does not contain 'curated:approved' or 'curated:rejected'
        query = query.filter(
            or_(
                Chunk.tags.is_(None),
                func.not_(Chunk.tags.cast(String).ilike("%curated:%"))
            )
        )
    elif status == "approved":
        query = query.filter(Chunk.tags.cast(String).ilike("%curated:approved%"))
    elif status == "rejected":
        query = query.filter(Chunk.tags.cast(String).ilike("%curated:rejected%"))

    total = query.count()
    chunks = query.order_by(Chunk.created_at.desc()).offset(offset).limit(limit).all()

    storage = _get_storage()
    items = []
    for c in chunks:
        shot_item = _chunk_to_shot_item(c, storage=storage)
        c_status = "pending"
        tags = c.tags if isinstance(c.tags, list) else []
        if any("curated:approved" in str(t) for t in tags):
            c_status = "approved"
        elif any("curated:rejected" in str(t) for t in tags):
            c_status = "rejected"

        items.append(
            CurationQueueItem(
                shot=shot_item,
                curation_status=c_status,
                tags=[t for t in tags if not str(t).startswith("curated:")],
            )
        )

    return {"items": items, "total": total}


def update_shot_curation(
    db: Session,
    chunk_id: str,
    action: CurationActionRequest,
) -> bool:
    """Updates a shot's curation verdict ('approved' | 'rejected' | 'pending') and tags."""
    if not _FOOTAGE_ENGINE_AVAILABLE or Chunk is None:
        return False

    chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()
    if not chunk:
        return False

    current_tags = list(chunk.tags or [])
    # Strip existing curated: flags
    new_tags = [t for t in current_tags if not str(t).startswith("curated:")]

    if action.status in ("approved", "rejected"):
        new_tags.append(f"curated:{action.status}")

    if action.tags:
        for t in action.tags:
            if t not in new_tags:
                new_tags.append(t)

    chunk.tags = new_tags
    db.commit()
    return True


