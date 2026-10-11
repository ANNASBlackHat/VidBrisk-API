"""Localhost labelling server and reporting for VidBrisk evaluation."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("vidbrisk_eval.labels")

LABELS_FILE = Path("eval/labels.jsonl")

app = FastAPI(title="VidBrisk Eval Labelling Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LabelRecord(BaseModel):
    run_a: str
    run_b: str
    beat_id: str
    choice: str  # "1", "2", "tie", "both_bad"
    chosen_run: str
    clip_1: str
    clip_2: str
    timestamp: str


@app.post("/labels")
def save_label(record: LabelRecord):
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.model_dump()) + "\n")
    return {"status": "saved", "beat_id": record.beat_id}


def report_labels(labels_file: Path = LABELS_FILE) -> dict[str, Any]:
    """Computes and prints win rates per run pair and judge-vs-human agreement."""
    if not labels_file.exists():
        print(f"No labels file found at {labels_file}")
        return {}

    lines = [json.loads(l) for l in labels_file.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    if not lines:
        print("No labels recorded yet.")
        return {}

    pair_stats: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_labels = len(lines)

    for r in lines:
        run_a = r.get("run_a", "")
        run_b = r.get("run_b", "")
        choice = r.get("choice", "")
        chosen_run = r.get("chosen_run", "")

        pair = (run_a, run_b)
        pair_stats[pair]["total"] += 1

        if chosen_run == run_a:
            pair_stats[pair]["win_a"] += 1
        elif chosen_run == run_b:
            pair_stats[pair]["win_b"] += 1
        elif choice == "tie":
            pair_stats[pair]["tie"] += 1
        elif choice == "both_bad":
            pair_stats[pair]["both_bad"] += 1

    print("\n" + "=" * 50)
    print("HUMAN PAIRWISE LABELLING & JUDGE CALIBRATION REPORT")
    print("=" * 50)
    print(f"Total labels recorded: {total_labels}\n")

    report_data = {}
    for (ra, rb), stats in pair_stats.items():
        tot = stats["total"]
        win_a_pct = (stats["win_a"] / tot) * 100
        win_b_pct = (stats["win_b"] / tot) * 100
        tie_pct = (stats["tie"] / tot) * 100
        bad_pct = (stats["both_bad"] / tot) * 100

        print(f"Comparison: {ra} vs {rb}")
        print(f" - Total evaluations: {tot}")
        print(f" - {ra} Win Rate: {win_a_pct:.1f}% ({stats['win_a']})")
        print(f" - {rb} Win Rate: {win_b_pct:.1f}% ({stats['win_b']})")
        print(f" - Tie Rate: {tie_pct:.1f}% ({stats['tie']})")
        print(f" - Both Bad Rate: {bad_pct:.1f}% ({stats['both_bad']})")

        # Compute judge agreement
        decisive_human = 0
        judge_agreements = 0
        judge_contradictions = 0

        # Load run results to get judge scores
        run_a_path = Path("eval/runs") / ra / "results.jsonl"
        run_b_path = Path("eval/runs") / rb / "results.jsonl"
        if not run_a_path.exists():
            # Try searching by exact name or substring
            matches_a = list(Path("eval/runs").glob(f"*{ra}*"))
            run_a_path = matches_a[0] / "results.jsonl" if matches_a else run_a_path
        if not run_b_path.exists():
            matches_b = list(Path("eval/runs").glob(f"*{rb}*"))
            run_b_path = matches_b[0] / "results.jsonl" if matches_b else run_b_path

        scores_a = {}
        scores_b = {}
        if run_a_path.exists() and run_b_path.exists():
            for line in run_a_path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    item = json.loads(line)
                    comp_id = f"{item.get('case_id')}_{item.get('beat_id')}"
                    scores_a[item.get("beat_id")] = item.get("metrics") or {}
                    scores_a[comp_id] = item.get("metrics") or {}
            for line in run_b_path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    item = json.loads(line)
                    comp_id = f"{item.get('case_id')}_{item.get('beat_id')}"
                    scores_b[item.get("beat_id")] = item.get("metrics") or {}
                    scores_b[comp_id] = item.get("metrics") or {}

            for r in lines:
                if (r.get("run_a"), r.get("run_b")) == (ra, rb) or (r.get("run_b"), r.get("run_a")) == (ra, rb):
                    ch = r.get("chosen_run")
                    bid = r.get("beat_id")
                    if ch in (ra, rb) and bid in scores_a and bid in scores_b:
                        decisive_human += 1
                        ja_rel = scores_a[bid].get("judge_relevance", 0) + scores_a[bid].get("judge_visual_interest", 0)
                        jb_rel = scores_b[bid].get("judge_relevance", 0) + scores_b[bid].get("judge_visual_interest", 0)
                        if ja_rel > jb_rel and ch == ra:
                            judge_agreements += 1
                        elif jb_rel > ja_rel and ch == rb:
                            judge_agreements += 1
                        elif ja_rel == jb_rel:
                            # Judge saw both candidates as equally good (tied)
                            pass
                        else:
                            judge_contradictions += 1

        agreement_pct = (judge_agreements / decisive_human * 100) if decisive_human > 0 else 0.0
        tie_judge_pct = ((decisive_human - judge_agreements - judge_contradictions) / decisive_human * 100) if decisive_human > 0 else 0.0
        print(f" - Judge vs Human Agreement: {agreement_pct:.1f}% ({judge_agreements}/{decisive_human} decisive pairs)")
        print(f" - Judge Indifference (Tied Score): {tie_judge_pct:.1f}% ({decisive_human - judge_agreements - judge_contradictions}/{decisive_human} pairs)")
        print(f" - Judge Contradiction: {(judge_contradictions / decisive_human * 100) if decisive_human > 0 else 0.0:.1f}% ({judge_contradictions}/{decisive_human} pairs)")
        print("-" * 50)

        report_data[f"{ra}_vs_{rb}"] = {
            "total": tot,
            "win_a_pct": round(win_a_pct, 1),
            "win_b_pct": round(win_b_pct, 1),
            "tie_pct": round(tie_pct, 1),
            "both_bad_pct": round(bad_pct, 1),
            "judge_agreement_pct": round(agreement_pct, 1),
        }

    return report_data
