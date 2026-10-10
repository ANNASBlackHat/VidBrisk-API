"""Evaluation metrics calculation and aggregation for VidBrisk."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from vidbrisk_eval.dynamism import compute_clip_dynamism
from vidbrisk_eval.judge import evaluate_with_judge

logger = logging.getLogger("vidbrisk_eval.metrics")


def compute_repeat_penalty(
    beat_index: int,
    selected_clip_id: str,
    selected_media_id: str,
    history: list[tuple[str, str]],
    window_n: int = 5,
) -> bool:
    """Returns True if the same clip or media item was reused within the last window_n beats."""
    if not selected_clip_id:
        return False
    recent = history[-window_n:]
    for past_cid, past_mid in recent:
        if past_cid == selected_clip_id or (past_mid and past_mid == selected_media_id):
            return True
    return False


def calculate_metrics_for_run(
    run_dir: str | Path,
    repeat_window: int = 5,
    run_judge: bool = True,
    run_dynamism: bool = True,
) -> dict[str, Any]:
    """Reads results.jsonl in run_dir, computes per-beat metrics, updates results.jsonl, and writes metrics.json."""
    run_path = Path(run_dir)
    results_file = run_path / "results.jsonl"
    if not results_file.exists():
        raise FileNotFoundError(f"results.jsonl not found in {run_dir}")

    lines = [json.loads(l) for l in results_file.read_text(encoding="utf-8").strip().splitlines() if l.strip()]

    updated_rows = []
    case_history: dict[str, list[tuple[str, str]]] = defaultdict(list)

    # Metric accumulators
    case_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    topic_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_metrics: list[dict[str, Any]] = []

    for row in lines:
        case_id = row["case_id"]
        topic = row.get("topic", "general")
        vo_text = row.get("vo_text", "")
        visual_intent = row.get("visual_intent", "")
        tier = row.get("tier")
        top_candidates = row.get("top_candidates", [])
        original_clips = row.get("original_clips", [])
        selected_clip = row.get("selected_clip") or {}

        orig_chunk_ids = set(c.get("clip_id") for c in original_clips if c.get("clip_id"))
        orig_media_ids = set(c.get("source_video_id") for c in original_clips if c.get("source_video_id"))
        top_chunk_ids = [c.get("chunk_id") for c in top_candidates]
        top_media_ids = [c.get("media_item_id") for c in top_candidates]

        # 1. hit@k (Mode A mainly, sanity reported in B)
        def is_hit(chunk_id: str, media_id: str) -> bool:
            return (chunk_id in orig_chunk_ids) or (media_id in orig_media_ids)

        hit_1 = is_hit(top_chunk_ids[0], top_media_ids[0]) if top_candidates else False
        hit_3 = any(is_hit(top_chunk_ids[i], top_media_ids[i]) for i in range(min(3, len(top_candidates))))
        hit_5 = any(is_hit(top_chunk_ids[i], top_media_ids[i]) for i in range(min(5, len(top_candidates))))
        hit_10 = any(is_hit(top_chunk_ids[i], top_media_ids[i]) for i in range(min(10, len(top_candidates))))

        # 2. Similarity (cosine similarity from search score)
        sim_score = float(selected_clip.get("score") or 0.0)

        # 3. Dynamism
        dynamism_res = {"dynamism_score": 0.0, "looping_or_static": False}
        if run_dynamism and selected_clip:
            storage_url = selected_clip.get("storage_url") or selected_clip.get("storage_path") or ""
            start_ts = float(selected_clip.get("start_ts") or 0.0)
            dur = float(selected_clip.get("duration_sec") or 4.0)
            dynamism_res = compute_clip_dynamism(storage_url, start_ts=start_ts, duration_sec=dur)

        # 4. Repeat penalty
        sel_cid = selected_clip.get("chunk_id", "")
        sel_mid = selected_clip.get("media_item_id", "")
        is_repeat = compute_repeat_penalty(
            beat_index=len(case_history[case_id]),
            selected_clip_id=sel_cid,
            selected_media_id=sel_mid,
            history=case_history[case_id],
            window_n=repeat_window,
        )
        case_history[case_id].append((sel_cid, sel_mid))

        # 5. LLM Judge
        judge_res = {"relevance_to_vo": 3, "visual_interest": 3, "tier_fit": 3, "reason": ""}
        if run_judge and selected_clip:
            judge_res = evaluate_with_judge(
                vo_text=vo_text,
                visual_intent=visual_intent,
                tier=tier,
                selected_clip=selected_clip,
            )

        beat_metrics = {
            "hit_1": hit_1,
            "hit_3": hit_3,
            "hit_5": hit_5,
            "hit_10": hit_10,
            "sim": sim_score,
            "dynamism": dynamism_res.get("dynamism_score", 0.0),
            "is_static": dynamism_res.get("looping_or_static", False),
            "is_repeat": is_repeat,
            "judge_relevance": judge_res.get("relevance_to_vo", 3),
            "judge_visual_interest": judge_res.get("visual_interest", 3),
            "judge_tier_fit": judge_res.get("tier_fit", 3),
            "judge_reason": judge_res.get("reason", ""),
            "low_confidence": row.get("footage_status") in ("inadequate", "requeried"),
            "low_score": sim_score < 0.50,
        }

        row["metrics"] = beat_metrics
        updated_rows.append(row)

        case_metrics[case_id].append(beat_metrics)
        topic_metrics[topic].append(beat_metrics)
        all_metrics.append(beat_metrics)

    # Re-write results.jsonl with enriched metrics
    with open(results_file, "w", encoding="utf-8") as f:
        for r in updated_rows:
            f.write(json.dumps(r) + "\n")

    # Aggregate summaries
    def aggregate(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(metrics_list)
        if n == 0:
            return {}
        return {
            "count": n,
            "hit_at_1": round(sum(1 for m in metrics_list if m["hit_1"]) / n, 4),
            "hit_at_3": round(sum(1 for m in metrics_list if m["hit_3"]) / n, 4),
            "hit_at_5": round(sum(1 for m in metrics_list if m["hit_5"]) / n, 4),
            "hit_at_10": round(sum(1 for m in metrics_list if m["hit_10"]) / n, 4),
            "mean_sim": round(sum(m["sim"] for m in metrics_list) / n, 4),
            "mean_dynamism": round(sum(m["dynamism"] for m in metrics_list) / n, 2),
            "static_rate": round(sum(1 for m in metrics_list if m["is_static"]) / n, 4),
            "repeat_rate": round(sum(1 for m in metrics_list if m["is_repeat"]) / n, 4),
            "judge_relevance": round(sum(m["judge_relevance"] for m in metrics_list) / n, 2),
            "judge_visual_interest": round(sum(m["judge_visual_interest"] for m in metrics_list) / n, 2),
            "judge_tier_fit": round(sum(m["judge_tier_fit"] for m in metrics_list) / n, 2),
            "coverage_low_confidence_rate": round(sum(1 for m in metrics_list if m["low_confidence"]) / n, 4),
            "coverage_low_score_rate": round(sum(1 for m in metrics_list if m["low_score"]) / n, 4),
        }

    summary = {
        "overall": aggregate(all_metrics),
        "by_case": {cid: aggregate(c_list) for cid, c_list in case_metrics.items()},
        "by_topic": {top: aggregate(t_list) for top, t_list in topic_metrics.items()},
    }

    metrics_file = run_path / "metrics.json"
    metrics_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Metrics aggregated and written to {metrics_file}")
    return summary
