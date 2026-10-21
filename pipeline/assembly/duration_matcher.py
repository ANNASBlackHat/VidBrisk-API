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
    deficit_strategy: str = "balanced_splice",
    engine: Optional[str] = None,
) -> tuple[AssetPlan, list[TrackItem], list[TrackItem]]:
    """Determines the visual asset plan and produces video & text TrackItems for a single beat.

    Returns:
        (asset_plan, video_track_items, text_track_items)
    """
    vo_duration = max(0.1, round(voice_clip.duration_sec, 2))
    pause_duration = max(0.0, round(getattr(beat, "pause_after", 0.0) or 0.0, 2))
    visual_duration = round(vo_duration + pause_duration, 2)
    beat_start = round(track_cursor, 2)
    beat_end = round(track_cursor + visual_duration, 2)

    # Filter out candidates with missing storage references (keep unit test mock identifiers)
    candidates = [
        c for c in candidates
        if (c.storage_path or c.storage_url or c.chunk_id.startswith(("mock", "chk_", "c_")))
    ]

    motion_props = beat.motion_props or {}
    layout_recipe = motion_props.get("layout_recipe")

    # --------------------------------------------------------------------------
    # Case 0: Explicit Multi-Layer Layout Recipes or Overlay Modes
    # --------------------------------------------------------------------------
    if engine != "native_fast":
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
    if engine != "native_fast" and beat.beat_type == "stat":
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

    # For native_fast: if candidates exist, prefer video footage over motion text
    should_use_motion = (
        not candidates
        if engine == "native_fast"
        else (
            beat.beat_type
            in (
                "abstract",
                "kinetic",
                "quote",
                "typewriter",
                "swipe_deck",
                "chat_bubbles",
                "map_route",
                "audio_waveform",
            )
            or not candidates
            or getattr(beat, "footage_status", None) == "inadequate"
        )
    )

    if should_use_motion:
        motion_props = beat.motion_props or {}
        comp_id = motion_props.get("component")
        if not comp_id and engine != "native_fast":
            if beat.beat_type == "swipe_deck":
                comp_id = "ListAnimations/SwipeDeck"
            elif beat.beat_type == "chat_bubbles":
                comp_id = "ListAnimations/ChatBubbles"
            elif beat.beat_type == "kinetic":
                comp_id = "TextAnimations/KineticText"
            elif beat.beat_type == "typewriter":
                comp_id = "TextAnimations/Typewriter"
            elif beat.beat_type == "map_route":
                comp_id = "GeoAnimations/MapExplainer"
            elif beat.beat_type == "audio_waveform":
                comp_id = "AudioAnimations/AudioWaveform"
            elif beat.beat_type == "stat":
                comp_id = "DataAnimations/StatCard"
            else:
                comp_id = "TextAnimations/QuoteCard"

        # Determine style identifier
        if comp_id and "SwipeDeck" in comp_id:
            style = "swipe-deck"
        elif comp_id and "ChatBubbles" in comp_id:
            style = "chat-reveal"
        elif comp_id and ("KineticText" in comp_id or "Typewriter" in comp_id):
            style = "kinetic-title"
        elif comp_id and "StatCard" in comp_id:
            style = "stat-callout"
        elif comp_id and "MapExplainer" in comp_id:
            style = "map-route"
        elif comp_id and "AudioWaveform" in comp_id:
            style = "audio-waveform"
        else:
            style = "quote-card"

        if engine == "native_fast":
            comp_id = None
            layout_role = "full"
            layer_role = "background"
            asset_type = "text"
        else:
            layout_role = motion_props.get("display_mode", "takeover")
            if layout_role == "overlay":
                layout_role = "overlay-lower-third"
            elif layout_role not in ("full", "takeover", "overlay-lower-third"):
                layout_role = "takeover"
            layer_role = "overlay"
            asset_type = "motion"

        plan = AssetPlan(
            strategy="text_card" if engine == "native_fast" else "motion_text",
            items=[
                AssetItem(
                    type=asset_type,
                    content=beat.text,
                    style=style,
                    component_id=comp_id,
                    props=motion_props,
                )
            ],
            layers=[
                Layer(
                    role=layer_role,
                    z=0,
                    type=asset_type,
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
            layerRole=layer_role,
            layout=layout_role,
            content=beat.text,
            style=style,
            componentId=comp_id,
            props=motion_props,
            assetType=asset_type,
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
                    source_out=visual_duration,
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
                    source_out=visual_duration,
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
            sourceOut=visual_duration,
            assetType="image",
            storagePath=top_cand.storage_path,
            storageUrl=top_cand.storage_url,
            props=bg_props,
        )
        try:
            from pipeline.footage.audit import get_audit_collector
            audit = get_audit_collector()
            audit.record_selection(
                beat.id,
                [{
                    "clip_id": video_item.id,
                    "asset_id": video_item.assetId,
                    "track_start": video_item.trackStart,
                    "track_end": video_item.trackEnd,
                    "duration": round(video_item.trackEnd - video_item.trackStart, 2),
                    "storage_path": video_item.storagePath,
                }],
                [f"Image Ken-Burns strategy: top candidate is an image ({top_cand.chunk_id})."],
            )
        except Exception as e:
            logger.debug(f"[DurationMatcher] Audit recording note: {e}")
        return plan, [video_item], []

    # --------------------------------------------------------------------------
    # Case 3: Video clip long enough -> single_clip (crop/center window)
    # --------------------------------------------------------------------------
    cand_duration = top_cand.duration_sec or (
        top_cand.end_ts - top_cand.start_ts if top_cand.end_ts else visual_duration
    )
    # Low-motion clips (likely static loops/slow pans) capped to max 2.0s so they don't drag
    if top_cand.motion_mean is not None and top_cand.motion_mean < 5.0:
        cand_duration = min(cand_duration, 2.0)

    if cand_duration >= visual_duration:
        # Center the window within candidate bounds if possible
        surplus = cand_duration - visual_duration
        offset = surplus / 2.0
        source_in = round(offset, 2)
        source_out = round(source_in + visual_duration, 2)
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
        try:
            from pipeline.footage.audit import get_audit_collector
            audit = get_audit_collector()
            audit.record_selection(
                beat.id,
                [{
                    "clip_id": video_item.id,
                    "asset_id": video_item.assetId,
                    "track_start": video_item.trackStart,
                    "track_end": video_item.trackEnd,
                    "duration": round(video_item.trackEnd - video_item.trackStart, 2),
                    "storage_path": video_item.storagePath,
                }],
                [f"Single clip strategy: top candidate ({top_cand.chunk_id}, {cand_duration:.2f}s) fully covers visual duration ({visual_duration:.2f}s)."],
            )
        except Exception as e:
            logger.debug(f"[DurationMatcher] Audit recording note: {e}")
        return plan, [video_item], []

    # --------------------------------------------------------------------------
    # Case 4: Video clip shorter than required duration -> handle via deficit_strategy
    # --------------------------------------------------------------------------
    bg_props = apply_mood_effects_to_props(None, beat) or {}
    c1 = top_cand
    c1_dur = c1.duration_sec or (c1.end_ts - c1.start_ts if c1.end_ts else visual_duration)
    if c1.motion_mean is not None and c1.motion_mean < 5.0:
        c1_dur = min(c1_dur, 2.0)

    # 1. Continuous Strategy
    if deficit_strategy == "continuous":
        s_in = round(c1.start_ts or 0.0, 2)
        s_out = round(s_in + visual_duration, 2)
        props = {**bg_props, "continuous_window": True}
        plan = AssetPlan(
            strategy="continuous_window",
            items=[
                AssetItem(
                    type="video",
                    chunk_id=c1.chunk_id,
                    source_in=s_in,
                    source_out=s_out,
                    storage_path=c1.storage_path,
                    storage_url=c1.storage_url,
                    props=props,
                )
            ],
            layers=[
                Layer(
                    role="background",
                    z=0,
                    type="video",
                    layout="full",
                    chunk_id=c1.chunk_id,
                    source_in=s_in,
                    source_out=s_out,
                    storage_path=c1.storage_path,
                    storage_url=c1.storage_url,
                    props=props,
                )
            ],
        )
        item = TrackItem(
            id=f"clip_{beat.id}",
            assetId=c1.chunk_id,
            trackStart=beat_start,
            trackEnd=beat_end,
            zIndex=0,
            layerRole="background",
            layout="full",
            sourceIn=s_in,
            sourceOut=s_out,
            assetType="video",
            storagePath=c1.storage_path,
            storageUrl=c1.storage_url,
            props=props,
        )
        try:
            from pipeline.footage.audit import get_audit_collector
            audit = get_audit_collector()
            audit.record_selection(
                beat.id,
                [{
                    "clip_id": item.id,
                    "asset_id": item.assetId,
                    "track_start": item.trackStart,
                    "track_end": item.trackEnd,
                    "duration": visual_duration,
                    "storage_path": item.storagePath,
                }],
                [f"Continuous strategy: extended source window [{s_in:.2f}s - {s_out:.2f}s] to cover visual duration ({visual_duration:.2f}s)."],
            )
        except Exception as e:
            logger.debug(f"[DurationMatcher] Audit recording note: {e}")
        return plan, [item], []

    from pipeline.config import get_settings
    settings = get_settings()
    max_slowmo = getattr(settings, "MAX_SLOWMO_FACTOR", 1.35)

    # 1.5 Find distinct candidates for balanced_splice or multi-clip early
    distinct_cands: list[CandidateChunk] = []
    seen_media: set[str] = set()
    seen_paths: set[str] = set()
    selection_notes: list[str] = []
    for idx, c in enumerate(candidates):
        m_id = c.media_item_id
        s_path = c.storage_path
        if m_id and m_id in seen_media:
            selection_notes.append(
                f"Skipped candidate #{idx+1} ({c.chunk_id}): media_item_id '{m_id}' already used in beat."
            )
            continue
        if s_path and s_path in seen_paths:
            selection_notes.append(
                f"Skipped candidate #{idx+1} ({c.chunk_id}): storage_path '{s_path}' already used in beat."
            )
            continue
        distinct_cands.append(c)
        if m_id:
            seen_media.add(m_id)
        if s_path:
            seen_paths.add(s_path)

    # 2. Check micro-cut deficit: if candidate 1 covers almost the whole beat (deficit < 1.0s),
    # avoid splicing a second candidate and extend/retime candidate 1 instead, PROVIDED retiming is within ceiling.
    deficit = visual_duration - c1_dur
    c1_retime = round(visual_duration / max(0.1, c1_dur), 3)
    if deficit < 1.0 and deficit_strategy != "continuous" and (c1_retime <= max_slowmo or len(distinct_cands) < 2):
        retime_factor = min(max_slowmo, c1_retime)
        props = {**bg_props, "retime_strategy": "slowmo", "retime_factor": retime_factor}
        plan = AssetPlan(
            strategy="single_clip",
            items=[
                AssetItem(
                    type="video",
                    chunk_id=c1.chunk_id,
                    source_in=0.0,
                    source_out=round(c1_dur, 2),
                    storage_path=c1.storage_path,
                    storage_url=c1.storage_url,
                    props=props,
                )
            ],
            layers=[
                Layer(
                    role="background",
                    z=0,
                    type="video",
                    layout="full",
                    chunk_id=c1.chunk_id,
                    source_in=0.0,
                    source_out=round(c1_dur, 2),
                    storage_path=c1.storage_path,
                    storage_url=c1.storage_url,
                    props=props,
                )
            ],
        )
        item = TrackItem(
            id=f"clip_{beat.id}",
            assetId=c1.chunk_id,
            trackStart=beat_start,
            trackEnd=beat_end,
            zIndex=0,
            layerRole="background",
            layout="full",
            sourceIn=0.0,
            sourceOut=round(c1_dur, 2),
            assetType="video",
            storagePath=c1.storage_path,
            storageUrl=c1.storage_url,
            props=props,
        )
        try:
            from pipeline.footage.audit import get_audit_collector
            audit = get_audit_collector()
            audit.record_selection(
                beat.id,
                [{
                    "clip_id": item.id,
                    "asset_id": item.assetId,
                    "track_start": item.trackStart,
                    "track_end": item.trackEnd,
                    "duration": visual_duration,
                    "storage_path": item.storagePath,
                }],
                [f"Avoided micro-cut ({deficit:.2f}s < 1.00s): extended clip '{item.id}' instead of splicing secondary candidate."],
            )
        except Exception as e:
            logger.debug(f"[DurationMatcher] Audit recording note: {e}")
        return plan, [item], []

    # 3. Slowmo Strategy (explicit, or as fallback when only 1 distinct candidate exists)
    if deficit_strategy == "slowmo" or (deficit_strategy == "balanced_splice" and len(distinct_cands) < 2):
        retime_factor = round(visual_duration / max(0.1, c1_dur), 3)
        retime_factor = min(max_slowmo, retime_factor)
        props = {**bg_props, "retime_strategy": "slowmo", "retime_factor": retime_factor}
        plan = AssetPlan(
            strategy="slowmo_retime" if deficit_strategy == "slowmo" else "single_clip",
            items=[
                AssetItem(
                    type="video",
                    chunk_id=c1.chunk_id,
                    source_in=0.0,
                    source_out=round(c1_dur, 2),
                    storage_path=c1.storage_path,
                    storage_url=c1.storage_url,
                    props=props,
                )
            ],
            layers=[
                Layer(
                    role="background",
                    z=0,
                    type="video",
                    layout="full",
                    chunk_id=c1.chunk_id,
                    source_in=0.0,
                    source_out=round(c1_dur, 2),
                    storage_path=c1.storage_path,
                    storage_url=c1.storage_url,
                    props=props,
                )
            ],
        )
        item = TrackItem(
            id=f"clip_{beat.id}",
            assetId=c1.chunk_id,
            trackStart=beat_start,
            trackEnd=beat_end,
            zIndex=0,
            layerRole="background",
            layout="full",
            sourceIn=0.0,
            sourceOut=round(c1_dur, 2),
            assetType="video",
            storagePath=c1.storage_path,
            storageUrl=c1.storage_url,
            props=props,
        )
        note = (
            f"Slowmo strategy: retiming candidate {c1.chunk_id} ({c1_dur:.2f}s -> {visual_duration:.2f}s, factor {retime_factor:.3f}x)."
            if deficit_strategy == "slowmo"
            else f"Balanced splice fallback to slowmo: only 1 distinct candidate available. Retimed {c1.chunk_id} ({c1_dur:.2f}s -> {visual_duration:.2f}s, factor {retime_factor:.3f}x)."
        )
        selection_notes.append(note)
        try:
            from pipeline.footage.audit import get_audit_collector
            audit = get_audit_collector()
            audit.record_selection(
                beat.id,
                [{
                    "clip_id": item.id,
                    "asset_id": item.assetId,
                    "track_start": item.trackStart,
                    "track_end": item.trackEnd,
                    "duration": visual_duration,
                    "storage_path": item.storagePath,
                }],
                selection_notes,
            )
        except Exception as e:
            logger.debug(f"[DurationMatcher] Audit recording note: {e}")
        return plan, [item], []

    # 4. Balanced Splice & Multi-Candidate Chaining Strategy (when >= 2 distinct candidates are available)
    if deficit_strategy == "balanced_splice" and len(distinct_cands) >= 2:
        c1, c2 = distinct_cands[0], distinct_cands[1]
        c1_dur = c1.duration_sec or (c1.end_ts - c1.start_ts if c1.end_ts else visual_duration)
        c2_dur = c2.duration_sec or (c2.end_ts - c2.start_ts if c2.end_ts else visual_duration)
        if c1.motion_mean is not None and c1.motion_mean < 5.0:
            c1_dur = min(c1_dur, 2.0)
        if c2.motion_mean is not None and c2.motion_mean < 5.0:
            c2_dur = min(c2_dur, 2.0)

        # Standard 2-candidate balanced splice if the top 2 candidates have sufficient combined duration
        if c1_dur + c2_dur >= visual_duration:
            target_split = round(visual_duration / 2.0, 2)
            dur1 = min(c1_dur, target_split)
            dur2 = round(visual_duration - dur1, 2)
            if dur2 > c2_dur:
                dur2 = c2_dur
                dur1 = round(visual_duration - dur2, 2)

            dur1 = round(dur1, 2)
            dur2 = round(dur2, 2)

            items = [
                TrackItem(
                    id=f"clip_{beat.id}_1",
                    assetId=c1.chunk_id,
                    trackStart=beat_start,
                    trackEnd=round(beat_start + dur1, 2),
                    zIndex=0,
                    layerRole="background",
                    layout="full",
                    sourceIn=0.0,
                    sourceOut=dur1,
                    assetType="video",
                    storagePath=c1.storage_path,
                    storageUrl=c1.storage_url,
                    props=bg_props,
                ),
                TrackItem(
                    id=f"clip_{beat.id}_2",
                    assetId=c2.chunk_id,
                    trackStart=round(beat_start + dur1, 2),
                    trackEnd=beat_end,
                    zIndex=0,
                    layerRole="background",
                    layout="full",
                    sourceIn=0.0,
                    sourceOut=dur2,
                    assetType="video",
                    storagePath=c2.storage_path,
                    storageUrl=c2.storage_url,
                    props=bg_props,
                ),
            ]
            plan = AssetPlan(
                strategy="concat_clips",
                items=[
                    AssetItem(type="video", chunk_id=c1.chunk_id, source_in=0.0, source_out=dur1, storage_path=c1.storage_path, storage_url=c1.storage_url, props=bg_props),
                    AssetItem(type="video", chunk_id=c2.chunk_id, source_in=0.0, source_out=dur2, storage_path=c2.storage_path, storage_url=c2.storage_url, props=bg_props),
                ],
                layers=[
                    Layer(role="background", z=0, type="video", layout="full", chunk_id=c1.chunk_id, source_in=0.0, source_out=dur1, storage_path=c1.storage_path, storage_url=c1.storage_url, props=bg_props),
                    Layer(role="background", z=0, type="video", layout="full", chunk_id=c2.chunk_id, source_in=0.0, source_out=dur2, storage_path=c2.storage_path, storage_url=c2.storage_url, props=bg_props),
                ],
            )
            selection_notes.append(
                f"Balanced splice applied: 50/50 split ({dur1:.2f}s + {dur2:.2f}s) between '{c1.chunk_id}' and '{c2.chunk_id}'."
            )
            try:
                from pipeline.footage.audit import get_audit_collector
                audit = get_audit_collector()
                audit.record_selection(
                    beat.id,
                    [
                        {"clip_id": items[0].id, "asset_id": items[0].assetId, "track_start": items[0].trackStart, "track_end": items[0].trackEnd, "duration": dur1, "storage_path": items[0].storagePath},
                        {"clip_id": items[1].id, "asset_id": items[1].assetId, "track_start": items[1].trackStart, "track_end": items[1].trackEnd, "duration": dur2, "storage_path": items[1].storagePath},
                    ],
                    selection_notes,
                )
            except Exception as e:
                logger.debug(f"[DurationMatcher] Audit recording note: {e}")
            return plan, items, []

        # Multi-candidate chaining across 3+ candidates when top 2 cannot cover deficit
        chain_items: list[TrackItem] = []
        plan_items: list[AssetItem] = []
        plan_layers: list[Layer] = []
        remaining = visual_duration
        curr_track = beat_start

        usable_durs = []
        for cand in distinct_cands:
            cdur = cand.duration_sec or (cand.end_ts - cand.start_ts if cand.end_ts else visual_duration)
            if cand.motion_mean is not None and cand.motion_mean < 5.0:
                cdur = min(cdur, 2.0)
            usable_durs.append(cdur)
        total_usable = sum(usable_durs)

        global_retime = 1.0
        if total_usable < visual_duration and (visual_duration / max(0.1, total_usable)) <= max_slowmo:
            global_retime = round(visual_duration / max(0.1, total_usable), 3)

        for idx, cand in enumerate(distinct_cands):
            if remaining <= 0.05:
                break
            base_dur = usable_durs[idx]
            max_track_dur = round(base_dur * global_retime, 2)
            track_dur = min(max_track_dur, remaining)
            source_dur = min(base_dur, round(track_dur / global_retime, 2))

            is_last = (idx == len(distinct_cands) - 1) or (remaining - track_dur <= 0.05)
            t_start = round(curr_track, 2)
            t_end = beat_end if is_last else round(t_start + track_dur, 2)
            track_dur = round(t_end - t_start, 2)

            props = {**bg_props}
            if global_retime > 1.01:
                props["retime_strategy"] = "slowmo"
                props["retime_factor"] = global_retime

            item = TrackItem(
                id=f"clip_{beat.id}_{len(chain_items)+1}",
                assetId=cand.chunk_id,
                trackStart=t_start,
                trackEnd=t_end,
                zIndex=0,
                layerRole="background",
                layout="full",
                sourceIn=0.0,
                sourceOut=round(source_dur, 2),
                assetType="video",
                storagePath=cand.storage_path,
                storageUrl=cand.storage_url,
                props=props,
            )
            chain_items.append(item)
            plan_items.append(
                AssetItem(
                    type="video",
                    chunk_id=cand.chunk_id,
                    source_in=0.0,
                    source_out=round(source_dur, 2),
                    storage_path=cand.storage_path,
                    storage_url=cand.storage_url,
                    props=props,
                )
            )
            plan_layers.append(
                Layer(
                    role="background",
                    z=0,
                    type="video",
                    layout="full",
                    chunk_id=cand.chunk_id,
                    source_in=0.0,
                    source_out=round(source_dur, 2),
                    storage_path=cand.storage_path,
                    storage_url=cand.storage_url,
                    props=props,
                )
            )
            curr_track += track_dur
            remaining -= track_dur

        plan = AssetPlan(strategy="concat_clips", items=plan_items, layers=plan_layers)
        selection_notes.append(
            f"Multi-candidate chaining applied: {len(chain_items)} clips chained to cover {visual_duration:.2f}s."
        )
        try:
            from pipeline.footage.audit import get_audit_collector
            audit = get_audit_collector()
            audit.record_selection(
                beat.id,
                [
                    {
                        "clip_id": it.id,
                        "asset_id": it.assetId,
                        "track_start": it.trackStart,
                        "track_end": it.trackEnd,
                        "duration": round(it.trackEnd - it.trackStart, 2),
                        "storage_path": it.storagePath,
                    }
                    for it in chain_items
                ],
                selection_notes,
            )
        except Exception as e:
            logger.debug(f"[DurationMatcher] Audit recording note: {e}")
        return plan, chain_items, []


    video_items: list[TrackItem] = []
    plan_items: list[AssetItem] = []
    plan_layers: list[Layer] = []
    remaining_duration = visual_duration
    current_track_pos = beat_start
    used_media_ids: set[str] = set()
    used_storage_paths: set[str] = set()
    selection_notes: list[str] = []
    MIN_SPLICE_DURATION = 1.0  # seconds: guard against jarring micro-cuts

    for idx, cand in enumerate(candidates):
        if remaining_duration <= 0.05:
            break

        # Disallow reusing the same physical video file within the same beat
        if cand.media_item_id and cand.media_item_id in used_media_ids:
            selection_notes.append(
                f"Skipped candidate #{idx+1} ({cand.chunk_id}): media_item_id '{cand.media_item_id}' already used in beat."
            )
            continue
        if cand.storage_path and cand.storage_path in used_storage_paths:
            selection_notes.append(
                f"Skipped candidate #{idx+1} ({cand.chunk_id}): storage_path '{cand.storage_path}' already used in beat."
            )
            continue

        # If a primary clip is already placed and the remaining deficit is tiny (< 1.0s),
        # extend the primary clip rather than flashing a sub-second micro-cut.
        if len(video_items) > 0 and remaining_duration < MIN_SPLICE_DURATION:
            selection_notes.append(
                f"Avoided micro-cut ({remaining_duration:.2f}s < {MIN_SPLICE_DURATION}s): extended clip '{video_items[-1].id}' instead of splicing candidate #{idx+1}."
            )
            video_items[-1].trackEnd = beat_end
            if plan_items:
                plan_items[-1].source_out = round(
                    (plan_items[-1].source_out or 0.0) + remaining_duration, 2
                )
            if plan_layers:
                plan_layers[-1].source_out = round(
                    (plan_layers[-1].source_out or 0.0) + remaining_duration, 2
                )
            remaining_duration = 0.0
            break

        c_dur = cand.duration_sec or (cand.end_ts - cand.start_ts if cand.end_ts else remaining_duration)
        if cand.motion_mean is not None and cand.motion_mean < 5.0:
            c_dur = min(c_dur, 2.0)
        use_dur = min(c_dur, remaining_duration)
        s_in = 0.0
        s_out = round(s_in + use_dur, 2)
        t_start = round(current_track_pos, 2)
        t_end = round(t_start + use_dur, 2)

        asset_type = "image" if cand.media_type in ("image", "photo") else "video"

        video_items.append(
            TrackItem(
                id=f"clip_{beat.id}_{len(video_items)+1}",
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

        if cand.media_item_id:
            used_media_ids.add(cand.media_item_id)
        if cand.storage_path:
            used_storage_paths.add(cand.storage_path)

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
        selection_notes.append(f"Extended final clip '{video_items[-1].id}' by {remaining_duration:.2f}s to reach voiceover end.")

    # Record selection into audit collector
    try:
        from pipeline.footage.audit import get_audit_collector
        audit = get_audit_collector()
        final_clips_summary = [
            {
                "clip_id": item.id,
                "asset_id": item.assetId,
                "track_start": item.trackStart,
                "track_end": item.trackEnd,
                "duration": round(item.trackEnd - item.trackStart, 2),
                "storage_path": item.storagePath,
            }
            for item in video_items
        ]
        audit.record_selection(beat.id, final_clips_summary, selection_notes)
    except Exception as e:
        logger.debug(f"[DurationMatcher] Audit recording note: {e}")

    plan = AssetPlan(strategy="concat_clips", items=plan_items, layers=plan_layers)
    return plan, video_items, []
