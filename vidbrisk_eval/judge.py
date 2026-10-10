"""LLM Judge metric computation for footage selection quality."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

from pipeline.llm.gemini import GeminiLLMClient

logger = logging.getLogger("vidbrisk_eval.judge")

RUBRIC_VERSION = "v1.0"
JUDGE_CACHE_DIR = Path("eval/cache/judge")

JUDGE_PROMPT_TEMPLATE = """You are an expert video director evaluating footage selection for a video beat.

BEAT CONTEXT:
- Voiceover Narration: "{vo_text}"
- Beat Visual Intent: "{visual_intent}"
- Match Tier: {tier}

SELECTED FOOTAGE CANDIDATE:
- Media ID: {clip_id}
- Caption: "{caption}"
- Tags: {tags}
- Provider: {provider}

Evaluate the chosen footage on a scale from 1 to 5 for each criterion:
1. `relevance_to_vo` (1-5): How well does this footage depict the topic, semantics, and narrative of the narration?
2. `visual_interest` (1-5): Is this visual compelling, dynamic, and cinematic (5) vs generic, dull, or static stock (1)?
3. `tier_fit` (1-5): Does the candidate fit the requested visual intent and mood?

Provide a concise 1-sentence explanation of your assessment.

Return JSON in this exact format:
{{
  "relevance_to_vo": <int 1-5>,
  "visual_interest": <int 1-5>,
  "tier_fit": <int 1-5>,
  "reason": "<one sentence explanation>"
}}
"""


def evaluate_with_judge(
    vo_text: str,
    visual_intent: str,
    tier: Optional[str],
    selected_clip: dict[str, Any],
    client: Optional[GeminiLLMClient] = None,
) -> dict[str, Any]:
    """Runs deterministic LLM judge evaluation on the selected clip for a beat."""
    clip_id = selected_clip.get("chunk_id") or selected_clip.get("media_item_id") or "unknown"
    caption = selected_clip.get("caption") or visual_intent
    tags = selected_clip.get("tags") or []
    provider = selected_clip.get("provider") or "unknown"

    JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(
        f"{RUBRIC_VERSION}_{vo_text}_{clip_id}_{caption}".encode()
    ).hexdigest()
    cache_file = JUDGE_CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    llm = client or GeminiLLMClient()
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        vo_text=vo_text,
        visual_intent=visual_intent,
        tier=tier or "narrative",
        clip_id=clip_id,
        caption=caption,
        tags=tags,
        provider=provider,
    )

    try:
        res = llm.generate_json(
            prompt=prompt,
            system_instruction="You are an objective media evaluation director. Return valid JSON only.",
        )
        judge_res = {
            "relevance_to_vo": int(res.get("relevance_to_vo", 3)),
            "visual_interest": int(res.get("visual_interest", 3)),
            "tier_fit": int(res.get("tier_fit", 3)),
            "reason": str(res.get("reason", "")),
            "rubric_version": RUBRIC_VERSION,
        }
        cache_file.write_text(json.dumps(judge_res, indent=2), encoding="utf-8")
        return judge_res
    except Exception as e:
        logger.warning(f"Judge evaluation failed for {clip_id}: {e}")
        return {
            "relevance_to_vo": 3,
            "visual_interest": 3,
            "tier_fit": 3,
            "reason": "Judge evaluation failed, defaulted to neutral 3.",
            "rubric_version": RUBRIC_VERSION,
        }
