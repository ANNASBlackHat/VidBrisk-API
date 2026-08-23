"""Multi-layer visual composition recipes for timeline assembly."""

from typing import Optional
from pipeline.models import (
    AssetItem,
    AssetPlan,
    Beat,
    CandidateChunk,
    Layer,
    StrategyType,
    TrackItem,
    VoiceClip,
)


def plan_split_screen(
    beat: Beat,
    voice_clip: VoiceClip,
    candidates: list[CandidateChunk],
    track_cursor: float,
) -> tuple[AssetPlan, list[TrackItem], list[TrackItem]]:
    """Composes two visual candidates side-by-side (split-left and split-right).
    
    Returns:
        (asset_plan, video_track_items, text_track_items)
    """
    vo_duration = max(0.1, round(voice_clip.duration_sec, 2))
    beat_start = round(track_cursor, 2)
    beat_end = round(track_cursor + vo_duration, 2)

    cand_left = candidates[0] if len(candidates) > 0 else None
    cand_right = candidates[1] if len(candidates) > 1 else cand_left

    if not cand_left:
        cand_left = CandidateChunk(
            chunk_id=f"chk_{beat.id}_left",
            media_item_id="placeholder_left",
            score=1.0,
            start_ts=0.0,
            duration_sec=vo_duration,
            media_type="video",
        )
        cand_right = CandidateChunk(
            chunk_id=f"chk_{beat.id}_right",
            media_item_id="placeholder_right",
            score=1.0,
            start_ts=0.0,
            duration_sec=vo_duration,
            media_type="video",
        )

    # 1. Left layer (z=0, layout="split-left")
    left_type = "image" if cand_left.media_type in ("image", "photo") else "video"
    left_s_in = round(cand_left.start_ts, 2)
    left_s_out = round(left_s_in + vo_duration, 2)

    layer_left = Layer(
        role="background",
        z=0,
        type=left_type,
        layout="split-left",
        chunk_id=cand_left.chunk_id,
        source_in=left_s_in,
        source_out=left_s_out,
        storage_path=cand_left.storage_path,
        storage_url=cand_left.storage_url,
    )
    item_left = TrackItem(
        id=f"clip_{beat.id}_split_l",
        assetId=cand_left.chunk_id,
        trackStart=beat_start,
        trackEnd=beat_end,
        zIndex=0,
        layerRole="background",
        layout="split-left",
        sourceIn=left_s_in,
        sourceOut=left_s_out,
        assetType=left_type,
        storagePath=cand_left.storage_path,
        storageUrl=cand_left.storage_url,
    )

    # 2. Right layer (z=1, layout="split-right")
    right_type = "image" if cand_right.media_type in ("image", "photo") else "video"
    right_s_in = round(cand_right.start_ts, 2)
    right_s_out = round(right_s_in + vo_duration, 2)

    layer_right = Layer(
        role="background",
        z=1,
        type=right_type,
        layout="split-right",
        chunk_id=cand_right.chunk_id,
        source_in=right_s_in,
        source_out=right_s_out,
        storage_path=cand_right.storage_path,
        storage_url=cand_right.storage_url,
    )
    item_right = TrackItem(
        id=f"clip_{beat.id}_split_r",
        assetId=cand_right.chunk_id,
        trackStart=beat_start,
        trackEnd=beat_end,
        zIndex=1,
        layerRole="background",
        layout="split-right",
        sourceIn=right_s_in,
        sourceOut=right_s_out,
        assetType=right_type,
        storagePath=cand_right.storage_path,
        storageUrl=cand_right.storage_url,
    )

    plan = AssetPlan(
        strategy="split_screen",
        items=[
            AssetItem(
                type=left_type,
                chunk_id=cand_left.chunk_id,
                source_in=left_s_in,
                source_out=left_s_out,
                storage_path=cand_left.storage_path,
                storage_url=cand_left.storage_url,
            ),
            AssetItem(
                type=right_type,
                chunk_id=cand_right.chunk_id,
                source_in=right_s_in,
                source_out=right_s_out,
                storage_path=cand_right.storage_path,
                storage_url=cand_right.storage_url,
            ),
        ],
        layers=[layer_left, layer_right],
    )

    return plan, [item_left, item_right], []


def plan_stat_over_footage(
    beat: Beat,
    voice_clip: VoiceClip,
    candidates: list[CandidateChunk],
    track_cursor: float,
) -> tuple[AssetPlan, list[TrackItem], list[TrackItem]]:
    """Composes background footage (layer 0) with a motion StatCard overlay (layer 1)."""
    vo_duration = max(0.1, round(voice_clip.duration_sec, 2))
    beat_start = round(track_cursor, 2)
    beat_end = round(track_cursor + vo_duration, 2)

    motion_props = beat.motion_props or {}
    comp_id = motion_props.get("component", "DataAnimations/StatCard")

    # Layer 1: Overlay StatCard
    overlay_layer = Layer(
        role="overlay",
        z=1,
        type="motion",
        layout="overlay-lower-third",
        component_id=comp_id,
        content=beat.text,
        style="stat-callout",
        props=motion_props,
    )
    overlay_item = TrackItem(
        id=f"txt_{beat.id}",
        trackStart=beat_start,
        trackEnd=beat_end,
        zIndex=1,
        layerRole="overlay",
        layout="overlay-lower-third",
        content=beat.text,
        style="stat-callout",
        componentId=comp_id,
        props=motion_props,
    )

    if not candidates:
        overlay_layer.layout = "takeover"
        overlay_layer.z = 0
        overlay_item.layout = "takeover"
        overlay_item.zIndex = 0
        plan = AssetPlan(
            strategy="motion_text",
            items=[
                AssetItem(
                    type="motion",
                    content=beat.text,
                    style="stat-callout",
                    component_id=comp_id,
                    props=motion_props,
                )
            ],
            layers=[overlay_layer],
        )
        return plan, [], [overlay_item]

    # Layer 0: Background footage
    top_cand = candidates[0]
    asset_type = "image" if top_cand.media_type in ("image", "photo") else "video"
    source_in = round(top_cand.start_ts, 2)
    source_out = round(source_in + vo_duration, 2)

    bg_layer = Layer(
        role="background",
        z=0,
        type=asset_type,
        layout="full",
        chunk_id=top_cand.chunk_id,
        source_in=source_in,
        source_out=source_out,
        storage_path=top_cand.storage_path,
        storage_url=top_cand.storage_url,
    )
    video_item = TrackItem(
        id=f"clip_{beat.id}_bg",
        assetId=top_cand.chunk_id,
        trackStart=beat_start,
        trackEnd=beat_end,
        zIndex=0,
        layerRole="background",
        layout="full",
        sourceIn=source_in,
        sourceOut=source_out,
        assetType=asset_type,
        storagePath=top_cand.storage_path,
        storageUrl=top_cand.storage_url,
    )

    plan = AssetPlan(
        strategy="stat_over_footage",
        items=[
            AssetItem(
                type=asset_type,
                chunk_id=top_cand.chunk_id,
                source_in=source_in,
                source_out=source_out,
                storage_path=top_cand.storage_path,
                storage_url=top_cand.storage_url,
            ),
            AssetItem(
                type="motion",
                content=beat.text,
                style="stat-callout",
                component_id=comp_id,
                props=motion_props,
            ),
        ],
        layers=[bg_layer, overlay_layer],
    )

    return plan, [video_item], [overlay_item]


def plan_quote_over_footage(
    beat: Beat,
    voice_clip: VoiceClip,
    candidates: list[CandidateChunk],
    track_cursor: float,
) -> tuple[AssetPlan, list[TrackItem], list[TrackItem]]:
    """Composes background footage (layer 0) with a motion QuoteCard overlay (layer 1)."""
    vo_duration = max(0.1, round(voice_clip.duration_sec, 2))
    beat_start = round(track_cursor, 2)
    beat_end = round(track_cursor + vo_duration, 2)

    motion_props = beat.motion_props or {}
    comp_id = motion_props.get("component", "TextAnimations/QuoteCard")

    # Layer 1: Overlay QuoteCard
    overlay_layer = Layer(
        role="overlay",
        z=1,
        type="motion",
        layout="takeover",
        component_id=comp_id,
        content=beat.text,
        style="abstract-card",
        props=motion_props,
    )
    overlay_item = TrackItem(
        id=f"txt_{beat.id}",
        trackStart=beat_start,
        trackEnd=beat_end,
        zIndex=1,
        layerRole="overlay",
        layout="takeover",
        content=beat.text,
        style="abstract-card",
        componentId=comp_id,
        props=motion_props,
    )

    if not candidates:
        overlay_layer.z = 0
        overlay_item.zIndex = 0
        plan = AssetPlan(
            strategy="motion_text",
            items=[
                AssetItem(
                    type="motion",
                    content=beat.text,
                    style="abstract-card",
                    component_id=comp_id,
                    props=motion_props,
                )
            ],
            layers=[overlay_layer],
        )
        return plan, [], [overlay_item]

    top_cand = candidates[0]
    asset_type = "image" if top_cand.media_type in ("image", "photo") else "video"
    source_in = round(top_cand.start_ts, 2)
    source_out = round(source_in + vo_duration, 2)

    bg_layer = Layer(
        role="background",
        z=0,
        type=asset_type,
        layout="full",
        chunk_id=top_cand.chunk_id,
        source_in=source_in,
        source_out=source_out,
        storage_path=top_cand.storage_path,
        storage_url=top_cand.storage_url,
    )
    video_item = TrackItem(
        id=f"clip_{beat.id}_bg",
        assetId=top_cand.chunk_id,
        trackStart=beat_start,
        trackEnd=beat_end,
        zIndex=0,
        layerRole="background",
        layout="full",
        sourceIn=source_in,
        sourceOut=source_out,
        assetType=asset_type,
        storagePath=top_cand.storage_path,
        storageUrl=top_cand.storage_url,
    )

    plan = AssetPlan(
        strategy="quote_over_footage",
        items=[
            AssetItem(
                type=asset_type,
                chunk_id=top_cand.chunk_id,
                source_in=source_in,
                source_out=source_out,
                storage_path=top_cand.storage_path,
                storage_url=top_cand.storage_url,
            ),
            AssetItem(
                type="motion",
                content=beat.text,
                style="abstract-card",
                component_id=comp_id,
                props=motion_props,
            ),
        ],
        layers=[bg_layer, overlay_layer],
    )

    return plan, [video_item], [overlay_item]
