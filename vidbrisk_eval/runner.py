"""Runner engine for VidBrisk evaluation harness."""

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from pipeline.config import get_settings
from pipeline.footage.resolver import FootageResolver
from pipeline.llm.gemini import GeminiLLMClient
from pipeline.models import Beat, CandidateChunk, FootageStatus
from pipeline.stages.resolve_footage import resolve_beat_visuals
from pipeline.stages.structure_beats import structure_beats

logger = logging.getLogger("vidbrisk_eval")


def get_git_commit_hash() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_evaluation(
    cases_file: str | Path,
    config_file: str | Path,
    mode: str = "A",
    beats_source: str = "original",
    limit_beats: Optional[int] = None,
    tag: str = "baseline",
    cache_selection: bool = False,
    output_base_dir: str | Path = "eval/runs",
) -> Path:
    """Executes headless evaluation run across cases and saves outputs."""
    start_time = time.time()
    cfg = load_yaml(config_file)
    cases_manifest = load_yaml(cases_file)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    run_dir_name = f"{timestamp}_{tag}"
    run_dir = Path(output_base_dir) / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Snapshot config into run directory
    config_snapshot_path = run_dir / "config.yaml"
    with open(config_snapshot_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

    resolver_cfg = cfg.get("resolver", {})
    resolver = FootageResolver(
        worker_enabled=resolver_cfg.get("worker_enabled", True),
        worker_timeout_sec=resolver_cfg.get("worker_timeout_sec", 15),
        worker_backend=resolver_cfg.get("worker_backend", "qwen"),
    )

    llm_client = GeminiLLMClient()
    top_k = cfg.get("retrieval", {}).get("top_k", 10)
    target_orientation = resolver_cfg.get("target_orientation", "horizontal")
    semantic_threshold = cfg.get("semantic_threshold", 0.50)
    motion_floor = cfg.get("motion_floor", 5.0)
    max_requeries = cfg.get("max_requeries", 2)

    cache_dir = Path("eval/cache/selection")
    if cache_selection:
        cache_dir.mkdir(parents=True, exist_ok=True)

    results_file = run_dir / "results.jsonl"
    all_results = []
    total_beats_evaluated = 0

    cases_list = cases_manifest.get("cases", [])
    logger.info(f"Starting run '{run_dir_name}' in Mode {mode} (beats: {beats_source}) across {len(cases_list)} cases.")

    with open(results_file, "w", encoding="utf-8") as rf:
        for c_entry in cases_list:
            c_file = Path(c_entry.get("file", c_entry.get("path", "")))
            if not c_file.exists():
                logger.warning(f"Case file {c_file} not found. Skipping.")
                continue

            case_data = json.loads(c_file.read_text(encoding="utf-8"))
            case_id = case_data["case_id"]
            video_id = case_data.get("video_id", "")
            topic = case_data.get("topic", "general")
            beats = case_data.get("beats", [])

            if limit_beats is not None:
                beats = beats[:limit_beats]

            logger.info(f"Processing case '{case_id}' ({len(beats)} beats)...")

            # In Mode B, exclude all media items originating from this case's video
            excluded_media_ids = [video_id] if mode.upper() == "B" and video_id else []
            if mode.upper() == "B":
                # Collect all original source video IDs present in this case's clips
                for b in case_data.get("beats", []):
                    for oc in b.get("original_clips", []):
                        if oc.get("source_video_id"):
                            excluded_media_ids.append(oc["source_video_id"])
                excluded_media_ids = sorted(list(set(excluded_media_ids)))

            if beats_source == "regenerate":
                # Apply style skill overrides if configured
                skill_path = cfg.get("skills", {}).get("footage_selection")
                skill_content = ""
                if skill_path and Path(skill_path).exists():
                    skill_content = Path(skill_path).read_text(encoding="utf-8")

                if beats_source == "regenerate":
                    full_script = " ".join(b.get("vo_text", "") for b in beats)
                    structured = structure_beats(clean_text=full_script, genre=topic, client=llm_client)
                    eval_beats = []
                    for s_idx, sb in enumerate(structured):
                        eval_beats.append(
                            {
                                "beat_id": sb.id,
                                "vo_text": sb.text,
                                "visual_intent": sb.visual_intent,
                                "beat_type": sb.beat_type,
                                "motion_props": sb.motion_props,
                                "original_clips": beats[s_idx].get("original_clips", []) if s_idx < len(beats) else [],
                            }
                        )
                else:
                    eval_beats = beats

                for b_idx, b in enumerate(eval_beats):
                    total_beats_evaluated += 1
                    b_id = b.get("beat_id", f"b{b_idx+1}")
                    vo_text = b.get("vo_text", "")
                    visual_intent = b.get("visual_intent") or vo_text
                    original_clips = b.get("original_clips", [])

                    # If an experimental skill is loaded, inject skill guidance into visual_intent query
                    if skill_content and "Cinematic Maritime" in skill_content:
                        visual_intent = f"{visual_intent} underwater naval exploration cinematic"

                    beat_obj = Beat(
                        id=b_id,
                        text=vo_text,
                        visual_intent=visual_intent,
                        beat_type=b.get("beat_type", "narrative"),
                        motion_props=b.get("motion_props"),
                    )

                # Selection execution (with caching support if requested)
                cached = False
                cache_key = f"{mode}_{case_id}_{b_id}_{top_k}.json"
                cache_file = cache_dir / cache_key if cache_selection else None

                if cache_selection and cache_file and cache_file.exists():
                    payload = json.loads(cache_file.read_text(encoding="utf-8"))
                    cached = True
                else:
                    updated_beat, candidates = resolve_beat_visuals(
                        beat=beat_obj,
                        resolver=resolver,
                        top_k=top_k,
                        target_orientation=target_orientation,
                        semantic_threshold=semantic_threshold,
                        motion_floor=motion_floor,
                        max_requeries=max_requeries,
                        llm_client=llm_client,
                        exclude_source_video_ids=excluded_media_ids or None,
                    )

                    cands_dump = [c.model_dump() for c in candidates]
                    selected_cand = cands_dump[0] if cands_dump else None

                    payload = {
                        "case_id": case_id,
                        "topic": topic,
                        "beat_id": b_id,
                        "vo_text": vo_text,
                        "visual_intent": updated_beat.visual_intent,
                        "query_used": updated_beat.visual_intent,
                        "beat_type": updated_beat.beat_type,
                        "footage_status": getattr(updated_beat.footage_status, "value", str(updated_beat.footage_status)),
                        "requery_count": updated_beat.requery_count,
                        "requery_reason": updated_beat.requery_reason,
                        "fallback_reason": updated_beat.fallback_reason,
                        "original_clips": original_clips,
                        "top_candidates": cands_dump,
                        "selected_clip": selected_cand,
                        "mode": mode.upper(),
                        "excluded_source_video_ids": excluded_media_ids,
                    }

                    if cache_selection and cache_file:
                        cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

                rf.write(json.dumps(payload) + "\n")
                all_results.append(payload)

    elapsed_time = time.time() - start_time
    meta = {
        "git_commit": get_git_commit_hash(),
        "timestamp": timestamp,
        "mode": mode.upper(),
        "beats_source": beats_source,
        "tag": tag,
        "total_cases": len(cases_list),
        "total_beats": total_beats_evaluated,
        "elapsed_seconds": round(elapsed_time, 2),
        "config_file": str(config_file),
        "cases_file": str(cases_file),
    }

    meta_file = run_dir / "meta.json"
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        f"Completed run '{run_dir_name}' in {elapsed_time:.2f}s. Results saved to {results_file}."
    )
    return run_dir
