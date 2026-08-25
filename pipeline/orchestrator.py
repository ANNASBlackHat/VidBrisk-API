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
from pipeline.stages.resolve_footage import resolve_footage
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
) -> TimelinePlan:
    """Executes stages [1] to [6] end-to-end on raw script text.

    Returns:
        TimelinePlan ready for rendering or editor ingestion.
    """
    if not raw_script or not raw_script.strip():
        raise ValueError("Cannot run pipeline on empty raw script.")

    print("▶ [1 & 2] Cleaning script and structuring beats...")
    if single_pass_llm:
        beats = clean_and_structure_beats(raw_script, client=llm_client)
    else:
        cleaned_text = clean_script(raw_script, client=llm_client)
        beats = structure_beats(cleaned_text, client=llm_client)

    if not beats:
        raise ValueError("Failed to extract any beats from the input script.")

    print(f"  ✓ Extracted {len(beats)} beats:")
    for b in beats:
        print(f"    - [{b.id}] ({b.beat_type}) {b.text[:40]}... -> intent: {b.visual_intent[:30]}...")

    tts = tts_provider or get_tts_provider()
    aligner = aligner_provider or get_aligner_provider()
    resolver = footage_resolver or FootageResolver()

    voice_clips: list[VoiceClip] = []
    timings_map: dict[str, list[WordTiming]] = {}
    candidates_map: dict[str, list[CandidateChunk]] = {}

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
            try:
                candidates = resolve_footage(
                    beat=beat,
                    resolver=resolver,
                    top_k=5,
                    target_orientation=target_orientation,
                )
            except Exception as e:
                print(f"    - Footage resolution note: {e} (using fallback)")
                candidates = []
            candidates_map[beat.id] = candidates
            print(f"    - Footage candidates resolved: {len(candidates)} match(es)")
    else:
        print("\n▶ [3, 4, 5] Processing per-beat synthesis, alignment, and footage resolution...")
        for idx, beat in enumerate(beats, start=1):
            print(f"  • Beat {idx}/{len(beats)} [{beat.id}]:")

            # Stage 3: Synthesize voice
            clip = tts_engine = tts.synthesize(
                text=beat.text,
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

            # Stage 5: Resolve footage
            try:
                candidates = resolve_footage(
                    beat=beat,
                    resolver=resolver,
                    top_k=5,
                    target_orientation=target_orientation,
                )
            except Exception as e:
                print(f"    - Footage resolution note: {e} (using fallback)")
                candidates = []
            candidates_map[beat.id] = candidates
            print(f"    - Footage candidates resolved: {len(candidates)} match(es)")

    # Stage 6: Assemble timeline
    print("\n▶ [6] Assembling timeline plan and gap-filling...")
    timeline = assemble_timeline(
        beats=beats,
        voice_clips=voice_clips,
        timings=timings_map,
        footage_candidates=candidates_map,
        output_json_path=output_json_path,
    )
    print(f"  ✓ Timeline generated: {timeline.total_duration:.2f}s total duration across {len(timeline.tracks)} tracks.")
    if output_json_path:
        print(f"  ✓ Exported to {output_json_path}")

    return timeline
