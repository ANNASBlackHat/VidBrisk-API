"""Comparison page generator and human pairwise labelling engine for VidBrisk eval."""

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("vidbrisk_eval.compare")

COMPARE_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>VidBrisk Eval Comparison: {tagA} vs {tagB}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      margin: 0;
      padding: 24px;
    }}
    header {{
      border-bottom: 1px solid #334155;
      padding-bottom: 16px;
      margin-bottom: 24px;
    }}
    h1 {{ margin: 0 0 8px 0; font-size: 22px; color: #38bdf8; }}
    .meta {{ font-size: 13px; color: #94a3b8; }}

    .diff-table {{
      width: 100%;
      border-collapse: collapse;
      background: #1e293b;
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 24px;
    }}
    .diff-table th, .diff-table td {{
      padding: 10px 16px;
      text-align: left;
      border-bottom: 1px solid #334155;
      font-size: 13px;
    }}
    .diff-table th {{ background: #0f172a; color: #94a3b8; text-transform: uppercase; font-size: 11px; }}
    .delta-pos {{ color: #4ade80; font-weight: bold; }}
    .delta-neg {{ color: #f87171; font-weight: bold; }}

    .changed-beat-card {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 8px;
      margin-bottom: 20px;
      padding: 16px;
    }}
    .beat-header {{
      display: flex;
      justify-content: space-between;
      margin-bottom: 12px;
      border-bottom: 1px solid #334155;
      padding-bottom: 8px;
    }}
    .beat-title {{ font-size: 14px; font-weight: bold; color: #38bdf8; }}
    .beat-vo {{ font-size: 14px; margin-bottom: 16px; }}

    .options-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }}
    .option-box {{
      background: #0f172a;
      padding: 12px;
      border-radius: 6px;
      border: 1px solid #334155;
    }}
    .option-label {{ font-size: 12px; font-weight: bold; color: #94a3b8; margin-bottom: 8px; text-transform: uppercase; }}
    video, .img-preview {{
      width: 100%;
      height: 180px;
      object-fit: cover;
      border-radius: 4px;
      background: #000;
    }}

    .vote-controls {{
      display: flex;
      gap: 10px;
      justify-content: center;
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid #334155;
    }}
    .vote-btn {{
      background: #334155;
      color: #f8fafc;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 500;
      transition: background 0.15s;
    }}
    .vote-btn:hover {{ background: #475569; }}
    .vote-btn.selected {{ background: #0284c7; color: #fff; }}
    .status-msg {{ font-size: 12px; color: #38bdf8; text-align: center; margin-top: 8px; }}
  </style>
</head>
<body>

  <header>
    <h1>A/B Comparison: {tagA} vs {tagB}</h1>
    <div class="meta">Changed beats: {changed_count} / {total_beats} total beats | Blinded A/B Pairwise Evaluation</div>
  </header>

  <table class="diff-table">
    <thead>
      <tr>
        <th>Metric</th>
        <th>{tagA}</th>
        <th>{tagB}</th>
        <th>Delta (B - A)</th>
      </tr>
    </thead>
    <tbody>
      {metrics_diff_rows}
    </tbody>
  </table>

  <h2>Changed Beats ({changed_count})</h2>
  <div id="changedBeats">
    {beats_html}
  </div>

  <div style="text-align: center; margin-top: 24px;">
    <button class="vote-btn" style="background: #10b981;" onclick="downloadLabels()">Export Labels JSON</button>
  </div>

  <script>
    const labelsData = [];

    async function vote(beatId, choice, opt1Run, opt2Run, opt1Clip, opt2Clip) {{
      const record = {{
        run_a: "{tagA}",
        run_b: "{tagB}",
        beat_id: beatId,
        choice: choice,
        chosen_run: choice === "1" ? opt1Run : choice === "2" ? opt2Run : choice,
        clip_1: opt1Clip,
        clip_2: opt2Clip,
        timestamp: new Date().toISOString()
      }};

      labelsData.push(record);

      // Attempt to save to local API server if running
      try {{
        await fetch("http://localhost:8000/labels", {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify(record)
        }});
      }} catch (e) {{
        console.log("Local label server not running, saving in memory for download.");
      }}

      // Highlight selected button
      const btns = document.querySelectorAll(`[data-beat="${{beatId}}"]`);
      btns.forEach(b => b.classList.remove("selected"));
      const activeBtn = document.getElementById(`btn_${{beatId}}_${{choice}}`);
      if (activeBtn) activeBtn.classList.add("selected");
      const statusEl = document.getElementById(`status_${{beatId}}`);
      if (statusEl) statusEl.innerText = `Recorded: ${{choice === "1" ? "Option 1" : choice === "2" ? "Option 2" : choice}}`;
    }}

    function downloadLabels() {{
      const blob = new Blob([JSON.stringify(labelsData, null, 2)], {{type: "application/json"}});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "labels_{tagA}_vs_{tagB}.json";
      a.click();
    }}
  </script>
</body>
</html>
"""


def generate_comparison_page(run_a_dir: str | Path, run_b_dir: str | Path, output_file: Optional[Path] = None) -> Path:
    """Generates compare_<A>_vs_<B>.html showing metric deltas and changed beats side-by-side."""
    path_a = Path(run_a_dir)
    path_b = Path(run_b_dir)

    tag_a = path_a.name
    tag_b = path_b.name

    metrics_a = json.loads((path_a / "metrics.json").read_text(encoding="utf-8")).get("overall", {})
    metrics_b = json.loads((path_b / "metrics.json").read_text(encoding="utf-8")).get("overall", {})

    rows_a = {json.loads(l)["case_id"] + "_" + json.loads(l)["beat_id"]: json.loads(l) for l in (path_a / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()}
    rows_b = {json.loads(l)["case_id"] + "_" + json.loads(l)["beat_id"]: json.loads(l) for l in (path_b / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()}

    # Metric deltas table
    metric_keys = ["mean_sim", "hit_at_1", "hit_at_3", "judge_relevance", "judge_visual_interest", "mean_dynamism", "static_rate", "repeat_rate"]
    diff_rows = []
    for k in metric_keys:
        val_a = metrics_a.get(k, 0.0)
        val_b = metrics_b.get(k, 0.0)
        delta = val_b - val_a
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        delta_class = "delta-pos" if (delta > 0 and "rate" not in k) or (delta < 0 and "rate" in k) else "delta-neg" if delta != 0 else ""
        diff_rows.append(f"""
        <tr>
          <td><strong>{k}</strong></td>
          <td>{val_a}</td>
          <td>{val_b}</td>
          <td class="{delta_class}">{delta_str}</td>
        </tr>
        """)

    # Find changed beats
    changed_beats_html = []
    common_keys = sorted(list(set(rows_a.keys()) & set(rows_b.keys())))

    for k in common_keys:
        ra = rows_a[k]
        rb = rows_b[k]
        pick_a = ra.get("selected_clip") or {}
        pick_b = rb.get("selected_clip") or {}

        cid_a = pick_a.get("chunk_id")
        cid_b = pick_b.get("chunk_id")

        if cid_a != cid_b:
            # Blind randomized order
            options = [("A", pick_a, tag_a), ("B", pick_b, tag_b)]
            random.seed(k)  # Deterministic shuffle per beat
            random.shuffle(options)

            opt1_letter, opt1_clip, opt1_run = options[0]
            opt2_letter, opt2_clip, opt2_run = options[1]

            url1 = opt1_clip.get("storage_url") or opt1_clip.get("storage_path") or ""
            url2 = opt2_clip.get("storage_url") or opt2_clip.get("storage_path") or ""

            card = f"""
            <div class="changed-beat-card">
              <div class="beat-header">
                <span class="beat-title">{ra.get("case_id")} — {ra.get("beat_id")}</span>
                <span style="font-size: 12px; color: #94a3b8;">Query: {ra.get("visual_intent")}</span>
              </div>
              <div class="beat-vo">"{ra.get("vo_text")}"</div>

              <div class="options-grid">
                <div class="option-box">
                  <div class="option-label">Option 1</div>
                  {f'<video src="{url1}" controls muted loop preload="metadata"></video>' if url1.endswith('.mp4') or url1.endswith('.webm') else f'<img class="img-preview" src="{url1}">' if url1 else '<div class="img-preview">No Media</div>'}
                  <div style="font-size: 11px; color: #64748b; margin-top: 6px;">Sim: {opt1_clip.get("score", 0.0):.3f}</div>
                </div>

                <div class="option-box">
                  <div class="option-label">Option 2</div>
                  {f'<video src="{url2}" controls muted loop preload="metadata"></video>' if url2.endswith('.mp4') or url2.endswith('.webm') else f'<img class="img-preview" src="{url2}">' if url2 else '<div class="img-preview">No Media</div>'}
                  <div style="font-size: 11px; color: #64748b; margin-top: 6px;">Sim: {opt2_clip.get("score", 0.0):.3f}</div>
                </div>
              </div>

              <div class="vote-controls">
                <button id="btn_{k}_1" data-beat="{k}" class="vote-btn" onclick="vote('{k}', '1', '{opt1_run}', '{opt2_run}', '{opt1_clip.get("chunk_id")}', '{opt2_clip.get("chunk_id")}')">Option 1 is Better</button>
                <button id="btn_{k}_2" data-beat="{k}" class="vote-btn" onclick="vote('{k}', '2', '{opt1_run}', '{opt2_run}', '{opt1_clip.get("chunk_id")}', '{opt2_clip.get("chunk_id")}')">Option 2 is Better</button>
                <button id="btn_{k}_tie" data-beat="{k}" class="vote-btn" onclick="vote('{k}', 'tie', '{opt1_run}', '{opt2_run}', '{opt1_clip.get("chunk_id")}', '{opt2_clip.get("chunk_id")}')">Tie</button>
                <button id="btn_{k}_both_bad" data-beat="{k}" class="vote-btn" onclick="vote('{k}', 'both_bad', '{opt1_run}', '{opt2_run}', '{opt1_clip.get("chunk_id")}', '{opt2_clip.get("chunk_id")}')">Both Bad</button>
              </div>
              <div id="status_{k}" class="status-msg"></div>
            </div>
            """
            changed_beats_html.append(card)

    rendered = COMPARE_HTML_TEMPLATE.format(
        tagA=tag_a,
        tagB=tag_b,
        total_beats=len(common_keys),
        changed_count=len(changed_beats_html),
        metrics_diff_rows="".join(diff_rows),
        beats_html="".join(changed_beats_html) if changed_beats_html else "<p style='color: #94a3b8;'>No picks changed between runs!</p>",
    )

    out_path = output_file or Path(f"eval/compare_{tag_a}_vs_{tag_b}.html")
    out_path.write_text(rendered, encoding="utf-8")
    logger.info(f"Comparison page generated at: {out_path}")
    return out_path
