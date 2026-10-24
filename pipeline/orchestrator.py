"""Pipeline Orchestrator — Chains stages [1] through [6] end-to-end."""

import os
from typing import Optional
from pipeline.alignment import AlignerProvider, get_aligner_provider
from pipeline.audio import slice_audio_for_beats
from pipeline.footage.resolver import FootageResolver
from pipeline.llm.gemini import GeminiLLMClient
from pipeline.models import Beat, CandidateChunk, TimelinePlan, VoiceClip, WordTiming
from pipeline.stages.assemble_timeline import assemble_timeline
from pipeline.stages.clean_script import clean_script
from pipeline.stages.extract_timestamps import extract_timestamps
from pipeline.stages.resolve_footage import resolve_beat_visuals, resolve_footage
from pipeline.stages.structure_beats import clean_and_structure_beats, structure_beats
from pipeline.tts import TTSProvider, get_tts_provider


def run_pipeline(
    raw_script: str,
    tts_provider: Optional[TTSProvider] = None,
    aligner_provider: Optional[AlignerProvider] = None,
    footage_resolver: Optional[FootageResolver] = None,
    llm_client: Optional[GeminiLLMClient] = None,
    output_json_path: Optional[str] = "output/timeline.json",
    audio_output_dir: Optional[str] = "output/audio",
    single_pass_llm: bool = False,
    target_orientation: Optional[str] = "horizontal",
    custom_audio_path: Optional[str] = None,
    footage_provider: Optional[str] = None,
    channel: Optional[str] = None,
    genre: Optional[str] = None,
    voice: Optional[str] = None,
    engine: str = "remotion",
    deficit_strategy: Optional[str] = None,
    cinematic: bool = False,
) -> TimelinePlan:
    """Executes stages [1] to [6] end-to-end on raw script text.

    Returns:
        TimelinePlan ready for rendering or editor ingestion.
    """
    if not raw_script or not raw_script.strip():
        raise ValueError("Cannot run pipeline on empty raw script.")

    print("▶ [1 & 2] Cleaning script and structuring beats...")
    if single_pass_llm:
        try:
            beats = clean_and_structure_beats(
                raw_script,
                channel=channel,
                genre=genre,
                client=llm_client,
                engine=engine,
            )
        except TypeError:
            try:
                beats = clean_and_structure_beats(raw_script, client=llm_client, engine=engine)
            except TypeError:
                try:
                    beats = clean_and_structure_beats(raw_script, engine=engine)
                except TypeError:
                    beats = clean_and_structure_beats(raw_script)
    else:
        cleaned_text = clean_script(raw_script, client=llm_client)
        try:
            beats = structure_beats(
                cleaned_text,
                channel=channel,
                genre=genre,
                client=llm_client,
                engine=engine,
            )
        except TypeError:
            try:
                beats = structure_beats(cleaned_text, client=llm_client, engine=engine)
            except TypeError:
                try:
                    beats = structure_beats(cleaned_text, engine=engine)
                except TypeError:
                    beats = structure_beats(cleaned_text)

    if not beats:
        raise ValueError("Failed to extract any beats from the input script.")

    print(f"  ✓ Extracted {len(beats)} beats:")
    for b in beats:
        print(f"    - [{b.id}] ({b.beat_type}) {b.text[:40]}... -> intent: {b.visual_intent[:30]}...")

    # Auto-detect genre if not explicitly provided
    effective_genre = genre
    if not effective_genre:
        from pipeline.genre_detector import detect_genre
        text_for_detection = raw_script if single_pass_llm else cleaned_text
        effective_genre = detect_genre(text_for_detection, client=llm_client)
        if effective_genre and effective_genre != "default":
            print(f"  ↳ Auto-detected genre: '{effective_genre}'")

    from pipeline.genre_anchor import get_genre_anchor
    context_anchor = get_genre_anchor(effective_genre, channel)
    if context_anchor:
        print(f"  ↳ Visual domain anchor: '{context_anchor}'")

    tts = tts_provider or get_tts_provider()
    aligner = aligner_provider or get_aligner_provider()
    resolver = footage_resolver or FootageResolver()


    from pipeline.footage.audit import FootageDecisionAuditCollector, set_audit_collector
    audit = FootageDecisionAuditCollector()
    set_audit_collector(audit)
    for b in beats:
        audit.get_or_create(b.id, script_text=b.text, beat_type=b.beat_type, visual_intent=b.visual_intent)

    voice_clips: list[VoiceClip] = []
    timings_map: dict[str, list[WordTiming]] = {}
    candidates_map: dict[str, list[CandidateChunk]] = {}

    # Cross-beat deduplication tracking
    selected_chunk_ids: list[str] = []
    selected_media_ids: list[str] = []

    if custom_audio_path and os.path.exists(custom_audio_path):
        print(f"\n▶ [3 & 4] Aligning & slicing pre-recorded custom audio from '{custom_audio_path}'...")
        voice_clips, timings_dict = slice_audio_for_beats(
            audio_path=custom_audio_path,
            beats=beats,
            aligner=aligner,
            output_dir=audio_output_dir or "output/audio",
        )
        for b_id, info in timings_dict.items():
            timings_map[b_id] = [
                WordTiming(word=w["word"], start=w.get("rel_start", w["start"]), end=w.get("rel_end", w["end"]))
                for w in info.get("words", [])
            ]

        print("\n▶ [5] Processing per-beat footage resolution...")
        for idx, beat in enumerate(beats, start=1):
            print(f"  • Beat {idx}/{len(beats)} [{beat.id}]:")
            # Exclude already used chunks and sliding 2-beat media window
            recent_media = [m for m in selected_media_ids[-2:] if m]
            try:
                updated_beat, candidates = resolve_beat_visuals(
                    beat=beat,
                    resolver=resolver,
                    top_k=5,
                    target_orientation=target_orientation,
                    provider=footage_provider,
                    exclude_chunk_ids=selected_chunk_ids or None,
                    exclude_source_video_ids=recent_media or None,
                    engine=engine,
                    context_anchor=context_anchor,
                    genre=effective_genre,
                )
                beats[idx - 1] = updated_beat
            except Exception as e:
                print(f"    - Footage resolution note: {e} (using fallback)")
                candidates = []
            candidates_map[beat.id] = candidates
            if candidates:
                top_cand = candidates[0]
                if top_cand.chunk_id:
                    selected_chunk_ids.append(top_cand.chunk_id)
                if top_cand.media_item_id:
                    selected_media_ids.append(top_cand.media_item_id)
            print(f"    - Footage candidates resolved: {len(candidates)} match(es) (status: {getattr(beats[idx - 1].footage_status, 'value', str(beats[idx - 1].footage_status))})")
    else:
        print("\n▶ [3, 4] Processing per-beat synthesis and alignment...")
        for idx, beat in enumerate(beats, start=1):
            print(f"  • Beat {idx}/{len(beats)} [{beat.id}]:")

            # Stage 3: Synthesize voice
            clip = tts_engine = tts.synthesize(
                text=beat.text,
                voice=voice,
                output_path=os.path.join(audio_output_dir, f"{beat.id}.wav") if audio_output_dir else None,
            )
            voice_clip = VoiceClip(
                beat_id=beat.id,
                audio_path=clip.audio_path,
                duration_sec=clip.duration_sec,
                sample_rate=clip.sample_rate,
            )
            voice_clips.append(voice_clip)
            print(f"    - Voice synthesized: {voice_clip.duration_sec:.2f}s ({voice_clip.audio_path})")

            # Stage 4: Extract timestamps
            beat_timings = extract_timestamps(voice_clip=voice_clip, beat=beat, aligner=aligner)
            timings_map[beat.id] = beat_timings
            print(f"    - Timestamps aligned: {len(beat_timings)} words")

        if cinematic:
            print("\n▶ [5] Processing cinematic scene directing & global Viterbi footage sequence optimization...")
            from pipeline.stages.resolve_footage import resolve_scene_visuals_cinematic
            cinematic_results = resolve_scene_visuals_cinematic(
                beats=beats,
                resolver=resolver,
                genre=effective_genre,
                top_k=5,
                target_orientation=target_orientation,
                provider=footage_provider,
                engine=engine,
                deficit_strategy=deficit_strategy or "virtual_camera",
            )
            for idx, (updated_b, cands) in enumerate(cinematic_results):
                beats[idx] = updated_b
                candidates_map[updated_b.id] = cands
                print(f"  • Beat {idx+1}/{len(beats)} [{updated_b.id}]: selected clip '{updated_b.selected_clip_id}' ({len(cands)} candidates)")
                audit.log_beat_summary(updated_b.id)
        else:
            print("\n▶ [5] Processing per-beat footage resolution...")
            for idx, beat in enumerate(beats, start=1):
                print(f"  • Beat {idx}/{len(beats)} [{beat.id}]:")
                recent_media = [m for m in selected_media_ids[-2:] if m]
                try:
                    updated_beat, candidates = resolve_beat_visuals(
                        beat=beat,
                        resolver=resolver,
                        top_k=5,
                        target_orientation=target_orientation,
                        provider=footage_provider,
                        exclude_chunk_ids=selected_chunk_ids or None,
                        exclude_source_video_ids=recent_media or None,
                        engine=engine,
                        context_anchor=context_anchor,
                        genre=effective_genre,
                    )
                    beats[idx - 1] = updated_beat
                except Exception as e:
                    print(f"    - Footage resolution note: {e} (using fallback)")
                    candidates = []
                candidates_map[beat.id] = candidates
                if candidates:
                    top_cand = candidates[0]
                    if top_cand.chunk_id and top_cand.chunk_id not in selected_chunk_ids:
                        selected_chunk_ids.append(top_cand.chunk_id)
                    if top_cand.media_item_id and top_cand.media_item_id not in selected_media_ids:
                        selected_media_ids.append(top_cand.media_item_id)
                print(f"    - Footage candidates resolved: {len(candidates)} match(es) (status: {getattr(beats[idx - 1].footage_status, 'value', str(beats[idx - 1].footage_status))})")
                audit.log_beat_summary(beat.id)

    # Stage 6: Assemble timeline
    print("\n▶ [6] Assembling timeline plan and gap-filling...")
    timeline = assemble_timeline(
        beats=beats,
        voice_clips=voice_clips,
        timings=timings_map,
        footage_candidates=candidates_map,
        output_json_path=output_json_path,
        deficit_strategy=deficit_strategy,
        engine=engine,
    )
    print(f"  ✓ Timeline generated: {timeline.total_duration:.2f}s total duration across {len(timeline.tracks)} tracks.")
    if output_json_path:
        print(f"  ✓ Exported timeline to {output_json_path}")
        base, ext = os.path.splitext(output_json_path)
        audit_path = f"{base}_audit.json"
        audit.save_json(audit_path)
        print(f"  ✓ Exported footage decision audit to {audit_path}")

    return timeline
