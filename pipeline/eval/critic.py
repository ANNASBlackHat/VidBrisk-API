"""Automated Post-Run Video Quality Critic & QA Inspector.

Analyzes completed video generation runs by inspecting:
1. `timeline.json` (Tracks, timings, clip placements, asset IDs, fallback cards).
2. `timeline_audit.json` (Footage candidate rankings, scores, deduplication actions, requeries).
3. Spoken script narration vs. visual intent vs. selected footage metadata.

Produces:
- Deterministic heuristic checks (repetition, micro-cuts, fallback takeovers, pool starvation).
- LLM Semantic Evaluation (Gemini Flash critique of narrative-to-visual fit).
- Categorized recommendations: [CODE], [ASSETS], [PROMPTING].
- Markdown & JSON reports.
- Formatted summary for Telegram notifications.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class VideoQualityCritic:
    """Evaluates completed video timeline and audit trail to diagnose quality issues."""

    def __init__(
        self,
        timeline_path: str,
        audit_path: Optional[str] = None,
        title: Optional[str] = None,
        genre: Optional[str] = None,
        gemini_client: Optional[Any] = None,
    ):
        self.timeline_path = timeline_path
        self.audit_path = audit_path or self._infer_audit_path(timeline_path)
        self.title = title or "Video Generation Run"
        self.genre = genre or "general"
        self._gemini = gemini_client

        self.timeline_data: Dict[str, Any] = {}
        self.audit_data: Dict[str, Any] = {}
        self._load_data()

    def _infer_audit_path(self, timeline_path: str) -> Optional[str]:
        candidates = [
            timeline_path.replace("_timeline.json", "_timeline_audit.json"),
            timeline_path.replace(".json", "_audit.json"),
            os.path.join(os.path.dirname(timeline_path), "timeline_audit.json"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        return None

    def _load_data(self):
        if os.path.isfile(self.timeline_path):
            try:
                with open(self.timeline_path, "r", encoding="utf-8") as f:
                    self.timeline_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load timeline from {self.timeline_path}: {e}")

        if self.audit_path and os.path.isfile(self.audit_path):
            try:
                with open(self.audit_path, "r", encoding="utf-8") as f:
                    self.audit_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load audit data from {self.audit_path}: {e}")

    def run_deterministic_analysis(self) -> Dict[str, Any]:
        """Runs rule-based checks across timeline and audit logs."""
        findings: Dict[str, Any] = {
            "repetition": [],
            "pacing": [],
            "fallbacks": [],
            "candidate_starvation": [],
            "low_confidence_matches": [],
            "summary_stats": {},
        }

        tracks = self.timeline_data.get("tracks", [])
        video_items: List[Dict[str, Any]] = []
        text_items: List[Dict[str, Any]] = []
        audio_items: List[Dict[str, Any]] = []

        for t in tracks:
            t_type = t.get("type")
            if t_type == "video":
                video_items = t.get("items", [])
            elif t_type == "text":
                text_items = t.get("items", [])
            elif t_type == "audio":
                audio_items = t.get("items", [])

        total_duration = float(self.timeline_data.get("total_duration", 0.0))
        findings["summary_stats"] = {
            "total_duration_sec": total_duration,
            "video_segments": len(video_items),
            "text_cards": len(text_items),
            "audio_clips": len(audio_items),
        }

        # 1. Repetition Analysis
        asset_sequence: List[str] = []
        asset_occurrences: Dict[str, int] = {}
        for item in video_items:
            # Asset key can be storagePath, storageUrl, or assetId
            asset_key = item.get("storagePath") or item.get("storageUrl") or item.get("assetId") or "unknown"
            asset_sequence.append(asset_key)
            asset_occurrences[asset_key] = asset_occurrences.get(asset_key, 0) + 1

        # Check consecutive repeats
        for i in range(len(asset_sequence) - 1):
            if asset_sequence[i] == asset_sequence[i + 1] and asset_sequence[i] != "unknown":
                findings["repetition"].append({
                    "type": "consecutive_repeat",
                    "segment_index_1": i + 1,
                    "segment_index_2": i + 2,
                    "asset": asset_sequence[i],
                    "message": f"Segments {i + 1} and {i + 2} reused identical footage asset consecutively.",
                })

        # Check excessive overall reuse (> 2 times in short videos)
        for asset, count in asset_occurrences.items():
            if count > 2 and asset != "unknown":
                findings["repetition"].append({
                    "type": "excessive_reuse",
                    "asset": asset,
                    "count": count,
                    "message": f"Asset '{asset[:30]}...' was reused {count} times across the video.",
                })

        # 2. Pacing & Retiming Analysis
        for idx, item in enumerate(video_items, 1):
            t_start = float(item.get("trackStart", 0.0))
            t_end = float(item.get("trackEnd", 0.0))
            dur = t_end - t_start
            if dur > 0 and dur < 1.5:
                findings["pacing"].append({
                    "segment_index": idx,
                    "duration_sec": round(dur, 2),
                    "issue": "micro_cut",
                    "message": f"Segment {idx} is very short ({dur:.2f}s < 1.5s), creating a rushed cut.",
                })
            elif dur > 12.0:
                findings["pacing"].append({
                    "segment_index": idx,
                    "duration_sec": round(dur, 2),
                    "issue": "lingering_shot",
                    "message": f"Segment {idx} is prolonged ({dur:.2f}s > 12s), which may cause visual fatigue.",
                })

        # 3. Fallbacks & Low-Confidence Checks (using timeline metadata or audit)
        audit_beats = self.audit_data.get("beats", {})
        metadata_beats = self.timeline_data.get("metadata", {}).get("resolved_beats", [])

        # Check beats from metadata
        for rb in metadata_beats:
            b_info = rb.get("beat", {})
            b_id = b_info.get("id", "unknown")
            status = b_info.get("footage_status", "")
            selected_clip = b_info.get("selected_clip_id")
            candidates = b_info.get("footage_candidates", [])

            if status in ("inadequate", "fallback") or not selected_clip:
                findings["fallbacks"].append({
                    "beat_id": b_id,
                    "text": b_info.get("text", "")[:60],
                    "status": status,
                    "message": f"Beat '{b_id}' fell back to text card / motion graphic because footage was inadequate.",
                })

            if len(candidates) < 2 and status == "accepted":
                findings["candidate_starvation"].append({
                    "beat_id": b_id,
                    "candidate_count": len(candidates),
                    "message": f"Beat '{b_id}' had only {len(candidates)} candidate(s) in the search pool.",
                })

            # Check semantic scores
            for c in candidates:
                if c.get("clip_id") == selected_clip:
                    score = float(c.get("semantic_score", 1.0))
                    if score < 0.60:
                        findings["low_confidence_matches"].append({
                            "beat_id": b_id,
                            "score": score,
                            "message": f"Beat '{b_id}' selected a clip with low semantic score ({score:.2f} < 0.60).",
                        })

        return findings

    def run_llm_critique(self, deterministic_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Runs Gemini LLM semantic critique on script-to-footage cohesion and failure diagnosis."""
        # Check if Gemini client is available
        client = self._gemini
        if client is None:
            try:
                from pipeline.llm.gemini import GeminiLLMClient
                client = GeminiLLMClient()
            except Exception as e:
                logger.debug(f"Gemini client unavailable for critique: {e}")
                return self._fallback_critique(deterministic_findings)

        # Prepare beat summary for prompt
        beats_summary: List[Dict[str, Any]] = []
        audit_beats = self.audit_data.get("beats", {})
        metadata_beats = self.timeline_data.get("metadata", {}).get("resolved_beats", [])

        for rb in metadata_beats:
            b_info = rb.get("beat", {})
            b_id = b_info.get("id", "")
            sel_id = b_info.get("selected_clip_id", "")
            audit_entry = audit_beats.get(b_id, {})

            chosen_cap = ""
            chosen_tags = []
            for hit in audit_entry.get("fused_candidates", []) + audit_entry.get("vector_hits", []):
                if hit.get("chunk_id") == sel_id or hit.get("media_item_id") == sel_id:
                    chosen_cap = hit.get("caption_snippet", "")
                    chosen_tags = hit.get("tags", [])
                    break

            beats_summary.append({
                "beat_id": b_id,
                "narration_text": b_info.get("text", ""),
                "visual_intent": b_info.get("visual_intent", ""),
                "selected_clip_id": sel_id,
                "clip_caption": chosen_cap,
                "clip_tags": chosen_tags[:6],
                "status": b_info.get("footage_status", "accepted"),
            })

        prompt = f"""You are a professional video editor and documentary director conducting an automated post-run quality audit on a video generation pipeline.

Video Title: {self.title}
Genre: {self.genre}
Total Duration: {deterministic_findings.get('summary_stats', {}).get('total_duration_sec', 0.0):.1f}s

Script Beats & Selected Footage:
{json.dumps(beats_summary, indent=2)}

Rule-Based Diagnostic Findings:
{json.dumps(deterministic_findings, indent=2)}

Analyze this video generation run thoroughly.
1. Evaluate whether the selected footage matches the semantic mood and idea of each beat.
2. If there are repeated clips, explain likely reasons (e.g. database asset shortage, strict search filters).
3. If any beats fell back to text cards or had low confidence, diagnose why.
4. Categorize actionable fixes into:
   - "code_fixes": Algorithm, threshold, or retiming logic improvements.
   - "asset_needs": Specific footage/assets missing from the database that should be ingested.
   - "prompt_adjustments": Script or visual intent prompting adjustments.

Return your response strictly in the following JSON format:
{{
  "overall_grade": "A|B|C|D",
  "quality_score": 85,
  "executive_summary": "1-2 sentence overall assessment of visual cohesion and flow.",
  "key_issues": [
    "Specific issue 1 with beat reference",
    "Specific issue 2"
  ],
  "root_causes": [
    "Cause of repetition or fallback 1",
    "Cause 2"
  ],
  "recommendations": {{
    "code_fixes": ["Concrete code / config suggestions"],
    "asset_needs": ["Missing footage topics to ingest into database"],
    "prompt_adjustments": ["Prompting or beat structuring adjustments"]
  }}
}}"""

        try:
            critique_json = client.generate_json(
                prompt=prompt,
                system_instruction="You are an expert AI video editor and diagnostic quality auditor. Return strict JSON only.",
            )
            if isinstance(critique_json, dict) and "overall_grade" in critique_json:
                return critique_json
        except Exception as e:
            logger.warning(f"Gemini LLM critique generation failed: {e}")

        return self._fallback_critique(deterministic_findings)

    def _fallback_critique(self, deterministic_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Provides a rule-based critique when LLM is unavailable."""
        reps = len(deterministic_findings.get("repetition", []))
        fallbacks = len(deterministic_findings.get("fallbacks", []))
        pacings = len(deterministic_findings.get("pacing", []))
        total_issues = reps * 2 + fallbacks * 3 + pacings

        if total_issues == 0:
            grade = "A"
            score = 95
            summary = "Excellent visual variety and pacing with zero repetition or fallback cards."
        elif total_issues <= 2:
            grade = "B+"
            score = 85
            summary = "Good overall presentation with minor pacing or candidate repetition."
        elif total_issues <= 5:
            grade = "B-"
            score = 75
            summary = "Fair presentation; some footage repetition or fallback takeovers observed."
        else:
            grade = "C"
            score = 65
            summary = "Multiple visual flow issues detected including repeated footage and fallbacks."

        key_issues = []
        for r in deterministic_findings.get("repetition", []):
            key_issues.append(r.get("message", "Repeated footage asset"))
        for f in deterministic_findings.get("fallbacks", []):
            key_issues.append(f.get("message", "Fallback card used"))
        for p in deterministic_findings.get("pacing", []):
            key_issues.append(p.get("message", "Pacing issue"))

        return {
            "overall_grade": grade,
            "quality_score": score,
            "executive_summary": summary,
            "key_issues": key_issues or ["No critical issues detected."],
            "root_causes": [
                "Repetitions occur when candidate pool has limited distinct assets for the topic.",
                "Fallbacks trigger when semantic scores or motion scores fall below configured thresholds.",
            ],
            "recommendations": {
                "code_fixes": [
                    "Consider setting FOOTAGE_ALLOW_LOW_MOTION_CAP=True for static documentary scenes.",
                    "Review balanced_splice deficit retiming for short duration clips.",
                ],
                "asset_needs": [
                    "Ingest additional B-roll matching visual intent keywords.",
                ],
                "prompt_adjustments": [
                    "Ensure beat visual intents describe specific subjects rather than generic environments.",
                ],
            },
        }

    def generate_full_report(self) -> Dict[str, Any]:
        """Runs complete deterministic + LLM evaluation and compiles output reports."""
        det_findings = self.run_deterministic_analysis()
        llm_critique = self.run_llm_critique(det_findings)

        report = {
            "run_id": os.path.basename(self.timeline_path).replace("_timeline.json", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": self.title,
            "genre": self.genre,
            "evaluation": llm_critique,
            "deterministic_findings": det_findings,
        }

        # Save JSON Report
        json_path = self.timeline_path.replace("_timeline.json", "_critique.json")
        if json_path == self.timeline_path:
            json_path = self.timeline_path.replace(".json", "_critique.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            report["critique_json_path"] = json_path
        except Exception as e:
            logger.warning(f"Could not save critique JSON to {json_path}: {e}")

        # Save Markdown Report
        md_path = json_path.replace(".json", ".md")
        md_content = self.render_markdown_report(report)
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            report["critique_md_path"] = md_path
        except Exception as e:
            logger.warning(f"Could not save critique Markdown to {md_path}: {e}")

        # Generate Telegram Snippet
        report["telegram_summary"] = self.format_telegram_summary(report)

        return report

    def render_markdown_report(self, report: Dict[str, Any]) -> str:
        """Renders comprehensive Markdown report for developer inspection."""
        eval_data = report.get("evaluation", {})
        det = report.get("deterministic_findings", {})
        stats = det.get("summary_stats", {})

        recs = eval_data.get("recommendations", {})
        code_fixes = recs.get("code_fixes", [])
        asset_needs = recs.get("asset_needs", [])
        prompt_adjs = recs.get("prompt_adjustments", [])

        lines = [
            f"# 🎬 Video Generation Quality Audit: {self.title}",
            "",
            f"- **Date**: `{report.get('timestamp')}`",
            f"- **Genre**: `{self.genre}`",
            f"- **Overall Grade**: **{eval_data.get('overall_grade', 'N/A')}** ({eval_data.get('quality_score', 0)}/100)",
            f"- **Total Duration**: {stats.get('total_duration_sec', 0.0):.1f}s | **Cuts**: {stats.get('video_segments', 0)} video, {stats.get('text_cards', 0)} cards",
            "",
            "## 📌 Executive Summary",
            eval_data.get("executive_summary", "No summary provided."),
            "",
            "## ⚠️ Key Findings & Issues",
        ]

        issues = eval_data.get("key_issues", [])
        if issues:
            for iss in issues:
                lines.append(f"- {iss}")
        else:
            lines.append("- *No visual or pacing issues detected.*")

        lines.extend([
            "",
            "## 🔍 Root-Cause Analysis",
        ])
        causes = eval_data.get("root_causes", [])
        if causes:
            for c in causes:
                lines.append(f"- {c}")
        else:
            lines.append("- *All footage matches adhered to target thresholds.*")

        lines.extend([
            "",
            "## 💡 Actionable Recommendations",
            "",
            "### 🔧 Code / Algorithm Tweaks",
        ])
        if code_fixes:
            for cf in code_fixes:
                lines.append(f"- {cf}")
        else:
            lines.append("- *No code adjustments needed.*")

        lines.extend([
            "",
            "### 📦 Missing Footage Assets (Database Needs)",
        ])
        if asset_needs:
            for an in asset_needs:
                lines.append(f"- {an}")
        else:
            lines.append("- *Asset coverage was sufficient.*")

        lines.extend([
            "",
            "### ✍️ Prompting & Beat Structuring",
        ])
        if prompt_adjs:
            for pa in prompt_adjs:
                lines.append(f"- {pa}")
        else:
            lines.append("- *Script structuring was optimal.*")

        return "\n".join(lines)

    def format_telegram_summary(self, report: Dict[str, Any]) -> str:
        """Formats compact, high-impact Telegram message (under 800 chars)."""
        eval_data = report.get("evaluation", {})
        grade = eval_data.get("overall_grade", "B")
        score = eval_data.get("quality_score", 80)
        
        # Badge mapping
        badge = "🟢" if score >= 85 else ("🟡" if score >= 70 else "🔴")

        issues = eval_data.get("key_issues", [])[:2]
        issue_lines = "\n".join([f"• {iss}" for iss in issues]) if issues else "• Visual flow clean"

        recs = eval_data.get("recommendations", {})
        code_fix = recs.get("code_fixes", ["No code fixes needed"])[0]
        asset_need = recs.get("asset_needs", ["Asset coverage sufficient"])[0]

        summary = (
            f"📊 *Quality Audit*: {self.title}\n"
            f"Rating: {badge} *{grade}* ({score}/100)\n\n"
            f"*Findings*:\n{issue_lines}\n\n"
            f"*Suggestions*:\n"
            f"🔧 `[CODE]`: {code_fix}\n"
            f"📦 `[ASSETS]`: {asset_need}"
        )
        return summary


def run_post_generation_critic(
    timeline_path: str,
    audit_path: Optional[str] = None,
    title: Optional[str] = None,
    genre: Optional[str] = None,
    skip: bool = False,
) -> Optional[Dict[str, Any]]:
    """Convenience entry point to run critic if not skipped by config or environment."""
    if skip:
        logger.info("ℹ️ AI Post-Run Critic skipped via config / environment.")
        return None

    # Check environment variable overrides
    if os.environ.get("SKIP_AI_CRITIC", "").lower() in ("1", "true", "yes"):
        logger.info("ℹ️ AI Post-Run Critic skipped via SKIP_AI_CRITIC environment variable.")
        return None

    if os.environ.get("ENABLE_AI_CRITIC", "").lower() in ("0", "false", "no"):
        logger.info("ℹ️ AI Post-Run Critic disabled via ENABLE_AI_CRITIC environment variable.")
        return None

    logger.info("🔍 Running Post-Generation AI Quality Critic & Inspector...")
    try:
        critic = VideoQualityCritic(
            timeline_path=timeline_path,
            audit_path=audit_path,
            title=title,
            genre=genre,
        )
        return critic.generate_full_report()
    except Exception as e:
        logger.warning(f"⚠️ Post-generation critic encountered an error (non-fatal): {e}", exc_info=True)
        return None
