"""Contact sheet generator for VidBrisk evaluation runs."""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("vidbrisk_eval.contact_sheet")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>VidBrisk Eval Contact Sheet — {tag}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      margin: 0;
      padding: 24px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #334155;
      padding-bottom: 16px;
      margin-bottom: 24px;
    }}
    h1 {{ margin: 0; font-size: 24px; color: #38bdf8; }}
    .meta {{ color: #94a3b8; font-size: 14px; }}
    .controls {{
      display: flex;
      gap: 12px;
      margin-bottom: 24px;
      background: #1e293b;
      padding: 12px 16px;
      border-radius: 8px;
    }}
    .controls select, .controls input {{
      background: #0f172a;
      color: #f8fafc;
      border: 1px solid #475569;
      padding: 6px 12px;
      border-radius: 6px;
    }}
    .summary-card {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .stat-box {{
      background: #1e293b;
      padding: 12px;
      border-radius: 8px;
      text-align: center;
    }}
    .stat-value {{ font-size: 20px; font-weight: bold; color: #38bdf8; }}
    .stat-label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; }}

    .beat-card {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 8px;
      margin-bottom: 16px;
      padding: 16px;
      display: grid;
      grid-template-columns: 280px 1fr 1fr;
      gap: 20px;
    }}
    .beat-card.static-flag {{ border-left: 4px solid #f59e0b; }}
    .beat-card.repeat-flag {{ border-left: 4px solid #ef4444; }}
    .beat-card.low-sim {{ border-left: 4px solid #ec4899; }}

    .beat-info {{ display: flex; flex-direction: column; gap: 8px; }}
    .beat-id {{ font-size: 12px; font-weight: bold; color: #38bdf8; text-transform: uppercase; }}
    .vo-text {{ font-size: 14px; font-weight: 500; line-height: 1.4; }}
    .visual-intent {{ font-size: 12px; color: #94a3b8; font-style: italic; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }}
    .chip {{
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 12px;
      background: #334155;
      color: #cbd5e1;
    }}
    .chip.warn {{ background: #78350f; color: #fde68a; }}
    .chip.alert {{ background: #7f1d1d; color: #fecaca; }}
    .chip.good {{ background: #064e3b; color: #a7f3d0; }}

    .video-preview {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .video-title {{ font-size: 12px; font-weight: 600; color: #94a3b8; text-transform: uppercase; }}
    video, .img-preview {{
      width: 100%;
      height: 160px;
      object-fit: cover;
      border-radius: 6px;
      background: #000;
    }}
    .clip-meta {{ font-size: 11px; color: #64748b; }}

    .candidates-list {{
      display: flex;
      gap: 8px;
      margin-top: 8px;
    }}
    .cand-thumb {{
      width: 80px;
      height: 50px;
      object-fit: cover;
      border-radius: 4px;
      border: 1px solid #475569;
    }}
  </style>
</head>
<body>

  <header>
    <div>
      <h1>VidBrisk Eval Contact Sheet — {tag}</h1>
      <div class="meta">Mode: {mode} | Beats: {beats_source} | Commit: {git_commit} | Generated: {timestamp}</div>
    </div>
  </header>

  <div class="summary-card">
    <div class="stat-box"><div class="stat-value">{total_beats}</div><div class="stat-label">Total Beats</div></div>
    <div class="stat-box"><div class="stat-value">{mean_sim}</div><div class="stat-label">Mean Sim</div></div>
    <div class="stat-box"><div class="stat-value">{hit_1_pct}%</div><div class="stat-label">Hit@1</div></div>
    <div class="stat-box"><div class="stat-value">{judge_rel} / 5</div><div class="stat-label">Judge Relevance</div></div>
    <div class="stat-box"><div class="stat-value">{static_rate}%</div><div class="stat-label">Static Rate</div></div>
    <div class="stat-box"><div class="stat-value">{repeat_rate}%</div><div class="stat-label">Repeat Rate</div></div>
  </div>

  <div class="controls">
    <label>Filter Case:
      <select id="caseFilter" onchange="filterBeats()">
        <option value="all">All Cases</option>
        {case_options}
      </select>
    </label>
    <label>Filter Flag:
      <select id="flagFilter" onchange="filterBeats()">
        <option value="all">All Beats</option>
        <option value="static">Static / Low Motion</option>
        <option value="repeat">Repeated</option>
        <option value="low_score">Low Similarity Score (&lt;0.5)</option>
      </select>
    </label>
  </div>

  <div id="beatsContainer">
    {beat_cards_html}
  </div>

  <script>
    function filterBeats() {{
      const c = document.getElementById("caseFilter").value;
      const f = document.getElementById("flagFilter").value;
      const cards = document.querySelectorAll(".beat-card");

      cards.forEach(card => {{
        const cardCase = card.getAttribute("data-case");
        const isStatic = card.getAttribute("data-static") === "true";
        const isRepeat = card.getAttribute("data-repeat") === "true";
        const isLowScore = card.getAttribute("data-low-score") === "true";

        let caseMatch = (c === "all" || cardCase === c);
        let flagMatch = true;
        if (f === "static") flagMatch = isStatic;
        if (f === "repeat") flagMatch = isRepeat;
        if (f === "low_score") flagMatch = isLowScore;

        if (caseMatch && flagMatch) {{
          card.style.display = "grid";
        }} else {{
          card.style.display = "none";
        }}
      }});
    }}
  </script>
</body>
</html>
"""


def generate_contact_sheet(run_dir: str | Path) -> Path:
    """Generates standalone static contact_sheet.html for a given evaluation run."""
    run_path = Path(run_dir)
    results_file = run_path / "results.jsonl"
    metrics_file = run_path / "metrics.json"
    meta_file = run_path / "meta.json"

    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
    overall = metrics.get("overall", {})

    lines = [json.loads(l) for l in results_file.read_text(encoding="utf-8").strip().splitlines() if l.strip()]

    distinct_cases = sorted(list(set(r.get("case_id", "") for r in lines)))
    case_options = "".join(f'<option value="{c}">{c}</option>' for c in distinct_cases)

    cards_html = []
    for r in lines:
        case_id = r.get("case_id", "")
        b_id = r.get("beat_id", "")
        vo_text = r.get("vo_text", "")
        visual_intent = r.get("visual_intent", "")
        m = r.get("metrics", {})
        engine_pick = r.get("selected_clip") or {}
        orig_clips = r.get("original_clips", [])
        orig_pick = orig_clips[0] if orig_clips else {}

        engine_url = engine_pick.get("storage_url") or engine_pick.get("storage_path") or ""
        engine_score = engine_pick.get("score", 0.0)
        is_static = m.get("is_static", False)
        is_repeat = m.get("is_repeat", False)
        is_low_score = m.get("low_score", False)

        flag_classes = []
        if is_static:
            flag_classes.append("static-flag")
        if is_repeat:
            flag_classes.append("repeat-flag")
        if is_low_score:
            flag_classes.append("low-sim")

        chips = []
        chips.append(f'<span class="chip">Sim: {engine_score:.3f}</span>')
        chips.append(f'<span class="chip">Dynamism: {m.get("dynamism", 0.0)}</span>')
        chips.append(f'<span class="chip good">Judge: {m.get("judge_relevance", 3)}/5</span>')
        if is_static:
            chips.append('<span class="chip warn">Static</span>')
        if is_repeat:
            chips.append('<span class="chip alert">Repeat</span>')
        if is_low_score:
            chips.append('<span class="chip warn">Low Score</span>')

        card = f"""
        <div class="beat-card {' '.join(flag_classes)}"
             data-case="{case_id}"
             data-static="{str(is_static).lower()}"
             data-repeat="{str(is_repeat).lower()}"
             data-low-score="{str(is_low_score).lower()}">
          <div class="beat-info">
            <div class="beat-id">{case_id} — {b_id}</div>
            <div class="vo-text">"{vo_text}"</div>
            <div class="visual_intent">Query: {visual_intent}</div>
            <div class="chips">
              {''.join(chips)}
            </div>
            {f'<div style="font-size: 11px; color: #a5b4fc; margin-top: 6px;">Judge: {m.get("judge_reason", "")}</div>' if m.get("judge_reason") else ''}
          </div>

          <div class="video-preview">
            <div class="video-title">Engine Pick (Selected)</div>
            {f'<video src="{engine_url}" controls muted loop preload="metadata"></video>' if engine_url.endswith('.mp4') or engine_url.endswith('.webm') else f'<img class="img-preview" src="{engine_url}">' if engine_url else '<div class="img-preview">No Media</div>'}
            <div class="clip-meta">ID: {engine_pick.get("chunk_id", "none")} | Provider: {engine_pick.get("provider", "unknown")}</div>
          </div>

          <div class="video-preview">
            <div class="video-title">Original Pick</div>
            <div style="background: #020617; height: 160px; border-radius: 6px; padding: 12px; font-size: 12px; color: #94a3b8; display: flex; flex-direction: column; justify-content: center;">
              <div><strong>Original Clip ID:</strong> {orig_pick.get("clip_id", "N/A")}</div>
              <div><strong>Source Video ID:</strong> {orig_pick.get("source_video_id", "N/A")}</div>
              <div><strong>Span:</strong> {orig_pick.get("start", 0.0)}s - {orig_pick.get("end", 0.0)}s</div>
            </div>
            <div class="clip-meta">Matches: {'Hit in Top-K' if m.get('hit_10') else 'No Hit in Top-10'}</div>
          </div>
        </div>
        """
        cards_html.append(card)

    rendered_html = HTML_TEMPLATE.format(
        tag=meta.get("tag", "run"),
        mode=meta.get("mode", "A"),
        beats_source=meta.get("beats_source", "original"),
        git_commit=meta.get("git_commit", "head")[:8],
        timestamp=meta.get("timestamp", ""),
        total_beats=overall.get("count", len(lines)),
        mean_sim=overall.get("mean_sim", 0.0),
        hit_1_pct=round(overall.get("hit_at_1", 0.0) * 100, 1),
        judge_rel=overall.get("judge_relevance", 0.0),
        static_rate=round(overall.get("static_rate", 0.0) * 100, 1),
        repeat_rate=round(overall.get("repeat_rate", 0.0) * 100, 1),
        case_options=case_options,
        beat_cards_html="".join(cards_html),
    )

    out_file = run_path / "contact_sheet.html"
    out_file.write_text(rendered_html, encoding="utf-8")
    logger.info(f"Contact sheet generated at: {out_file}")
    return out_file
