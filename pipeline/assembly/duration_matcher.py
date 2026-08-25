from typing import Optional
from pipeline.assembly.effects_mapping import apply_mood_effects_to_props
from pipeline.assembly.recipes import (
    plan_quote_over_footage,
    plan_split_screen,
    plan_stat_over_footage,
)
from pipeline.models import (
    AssetItem,
    AssetPlan,
    Beat,
    CandidateChunk,
    Layer,
    StrategyType,
    TrackItem,
    VoiceClip,
    WordTiming,
)


def plan_beat_assets(
    beat: Beat,
    voice_clip: VoiceClip,
    candidates: list[CandidateChunk],
    track_cursor: float,
) -> tuple[AssetPlan, list[TrackItem], list[TrackItem]]:
    """Determines the visual asset plan and produces video & text TrackItems for a single beat.

    Returns:
        (asset_plan, video_track_items, text_track_items)
    """
    vo_duration = max(0.1, round(voice_clip.duration_sec, 2))
    beat_start = round(track_cursor, 2)
    beat_end = round(track_cursor + vo_duration, 2)

    motion_props = beat.motion_props or {}
    layout_recipe = motion_props.get("layout_recipe")

    # --------------------------------------------------------------------------
    # Case 0: Explicit Multi-Layer Layout Recipes or Overlay Modes
    # --------------------------------------------------------------------------
    if layout_recipe == "split_screen":
        return plan_split_screen(beat, voice_clip, candidates, track_cursor)
    elif layout_recipe == "stat_over_footage" or (
        beat.beat_type == "stat" and candidates and motion_props.get("display_mode") == "overlay"
    ):
        return plan_stat_over_footage(beat, voice_clip, candidates, track_cursor)
    elif layout_recipe == "quote_over_footage" or (
        beat.beat_type == "abstract" and candidates and motion_props.get("display_mode") == "overlay"
    ):
        return plan_quote_over_footage(beat, voice_clip, candidates, track_cursor)

    # --------------------------------------------------------------------------
    # Case 1: 'stat' or 'abstract' beats -> motion_text strategy
    # --------------------------------------------------------------------------
    if beat.beat_type == "stat":
        motion_props = beat.motion_props or {}
        comp_id = motion_props.get("component", "DataAnimations/StatCard")
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
            layers=[
                Layer(
                    role="overlay",
                    z=0,
                    type="motion",
                    layout="takeover",
                    component_id=comp_id,
                    content=beat.text,
                    style="stat-callout",
                    props=motion_props,
                )
            ],
        )
        text_item = TrackItem(
            id=f"txt_{beat.id}",
            trackStart=beat_start,
            trackEnd=beat_end,
            zIndex=0,
            layerRole="overlay",
            layout="takeover",
            content=beat.text,
            style="stat-callout",
            componentId=comp_id,
            props=motion_props,
        )
        return plan, [], [text_item]

    if (
        beat.beat_type
        in (
            "abstract",
            "kinetic",
            "quote",
            "typewriter",
            "swipe_deck",
            "chat_bubbles",
        )
        or not candidates
    ):
        motion_props = beat.motion_props or {}
        comp_id = motion_props.get("component")
        if not comp_id:
            if beat.beat_type == "swipe_deck":
                comp_id = "ListAnimations/SwipeDeck"
            elif beat.beat_type == "chat_bubbles":
                comp_id = "ListAnimations/ChatBubbles"
            elif beat.beat_type == "kinetic":
                comp_id = "TextAnimations/KineticText"
            elif beat.beat_type == "typewriter":
                comp_id = "TextAnimations/Typewriter"
            else:
                comp_id = "TextAnimations/QuoteCard"

        # Determine style identifier
        if "SwipeDeck" in comp_id:
            style = "swipe-deck"
        elif "ChatBubbles" in comp_id:
            style = "chat-reveal"
        elif "KineticText" in comp_id or "Typewriter" in comp_id:
            style = "kinetic-title"
        elif "StatCard" in comp_id:
            style = "stat-callout"
        else:
            style = "quote-card"

        layout_role = motion_props.get("display_mode", "takeover")
        if layout_role not in ("full", "takeover", "overlay", "overlay-lower-third"):
            layout_role = "takeover"

        plan = AssetPlan(
            strategy="motion_text",
            items=[
                AssetItem(
                    type="motion",
                    content=beat.text,
                    style=style,
                    component_id=comp_id,
                    props=motion_props,
                )
            ],
            layers=[
                Layer(
                    role="overlay",
                    z=0,
                    type="motion",
                    layout=layout_role,
                    component_id=comp_id,
                    content=beat.text,
                    style=style,
                    props=motion_props,
                )
            ],
        )
        text_item = TrackItem(
            id=f"txt_{beat.id}",
            trackStart=beat_start,
            trackEnd=beat_end,
            zIndex=0,
            layerRole="overlay",
            layout=layout_role,
            content=beat.text,
            style=style,
            componentId=comp_id,
            props=motion_props,
        )
        return plan, [], [text_item]

    # --------------------------------------------------------------------------
    # Case 2: Top candidate is an Image -> image_kenburns strategy
    # --------------------------------------------------------------------------
    top_cand = candidates[0]
    if top_cand.media_type in ("image", "photo"):
        bg_props = apply_mood_effects_to_props(None, beat)
        plan = AssetPlan(
            strategy="image_kenburns",
            items=[
                AssetItem(
                    type="image",
                    chunk_id=top_cand.chunk_id,
                    source_in=0.0,
                    source_out=vo_duration,
                    storage_path=top_cand.storage_path,
                    storage_url=top_cand.storage_url,
                    props=bg_props,
                )
            ],
            layers=[
                Layer(
                    role="background",
                    z=0,
                    type="image",
                    layout="full",
                    chunk_id=top_cand.chunk_id,
                    source_in=0.0,
                    source_out=vo_duration,
                    storage_path=top_cand.storage_path,
                    storage_url=top_cand.storage_url,
                    props=bg_props,
                )
            ],
        )
        video_item = TrackItem(
            id=f"clip_{beat.id}_img",
            assetId=top_cand.chunk_id,
            trackStart=beat_start,
            trackEnd=beat_end,
            zIndex=0,
            layerRole="background",
            layout="full",
            sourceIn=0.0,
            sourceOut=vo_duration,
            assetType="image",
            storagePath=top_cand.storage_path,
            storageUrl=top_cand.storage_url,
            props=bg_props,
        )
        return plan, [video_item], []

    # --------------------------------------------------------------------------
    # Case 3: Video clip long enough -> single_clip (crop/center window)
    # --------------------------------------------------------------------------
    cand_duration = top_cand.duration_sec or (
        top_cand.end_ts - top_cand.start_ts if top_cand.end_ts else vo_duration
    )

    if cand_duration >= vo_duration:
        # Center the window within candidate bounds if possible
        surplus = cand_duration - vo_duration
        offset = surplus / 2.0
        source_in = round(top_cand.start_ts + offset, 2)
        source_out = round(source_in + vo_duration, 2)
        bg_props = apply_mood_effects_to_props(None, beat)

        plan = AssetPlan(
            strategy="single_clip",
            items=[
                AssetItem(
                    type="video",
                    chunk_id=top_cand.chunk_id,
                    source_in=source_in,
                    source_out=source_out,
                    storage_path=top_cand.storage_path,
                    storage_url=top_cand.storage_url,
                    props=bg_props,
                )
            ],
            layers=[
                Layer(
                    role="background",
                    z=0,
                    type="video",
                    layout="full",
                    chunk_id=top_cand.chunk_id,
                    source_in=source_in,
                    source_out=source_out,
                    storage_path=top_cand.storage_path,
                    storage_url=top_cand.storage_url,
                    props=bg_props,
                )
            ],
        )
        video_item = TrackItem(
            id=f"clip_{beat.id}",
            assetId=top_cand.chunk_id,
            trackStart=beat_start,
            trackEnd=beat_end,
            zIndex=0,
            layerRole="background",
            layout="full",
            sourceIn=source_in,
            sourceOut=source_out,
            assetType="video",
            storagePath=top_cand.storage_path,
            storageUrl=top_cand.storage_url,
            props=bg_props,
        )
        return plan, [video_item], []

    # --------------------------------------------------------------------------
    # Case 4: Video clip shorter than VO duration -> concat_clips fallback
    # --------------------------------------------------------------------------
    video_items: list[TrackItem] = []
    plan_items: list[AssetItem] = []
    plan_layers: list[Layer] = []
    remaining_duration = vo_duration
    current_track_pos = beat_start
    bg_props = apply_mood_effects_to_props(None, beat)

    for idx, cand in enumerate(candidates):
        if remaining_duration <= 0.05:
            break

        c_dur = cand.duration_sec or (cand.end_ts - cand.start_ts if cand.end_ts else remaining_duration)
        use_dur = min(c_dur, remaining_duration)
        s_in = round(cand.start_ts, 2)
        s_out = round(s_in + use_dur, 2)
        t_start = round(current_track_pos, 2)
        t_end = round(t_start + use_dur, 2)

        asset_type = "image" if cand.media_type in ("image", "photo") else "video"

        video_items.append(
            TrackItem(
                id=f"clip_{beat.id}_{idx+1}",
                assetId=cand.chunk_id,
                trackStart=t_start,
                trackEnd=t_end,
                zIndex=0,
                layerRole="background",
                layout="full",
                sourceIn=s_in,
                sourceOut=s_out,
                assetType=asset_type,
                storagePath=cand.storage_path,
                storageUrl=cand.storage_url,
                props=bg_props,
            )
        )
        plan_items.append(
            AssetItem(
                type=asset_type,
                chunk_id=cand.chunk_id,
                source_in=s_in,
                source_out=s_out,
                storage_path=cand.storage_path,
                storage_url=cand.storage_url,
                props=bg_props,
            )
        )
        plan_layers.append(
            Layer(
                role="background",
                z=0,
                type=asset_type,
                layout="full",
                chunk_id=cand.chunk_id,
                source_in=s_in,
                source_out=s_out,
                storage_path=cand.storage_path,
                storage_url=cand.storage_url,
                props=bg_props,
            )
        )

        current_track_pos += use_dur
        remaining_duration -= use_dur

    # If still remaining deficit, extend last item (or loop/hold) to cover full VO
    if remaining_duration > 0.05 and video_items:
        video_items[-1].trackEnd = beat_end
        if plan_items:
            plan_items[-1].source_out = round(
                (plan_items[-1].source_out or 0.0) + remaining_duration, 2
            )
        if plan_layers:
            plan_layers[-1].source_out = round(
                (plan_layers[-1].source_out or 0.0) + remaining_duration, 2
            )

    plan = AssetPlan(strategy="concat_clips", items=plan_items, layers=plan_layers)
    return plan, video_items, []
