"""CLI entry point for vidbrisk_eval."""

import argparse
import logging
import sys
from pathlib import Path

from vidbrisk_eval.runner import run_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vidbrisk_eval")


def main():
    parser = argparse.ArgumentParser(prog="vidbrisk_eval", description="VidBrisk Footage Selection Evaluation Harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: run
    run_parser = subparsers.add_parser("run", help="Run headless footage evaluation")
    run_parser.add_argument("--cases", default="eval/golden.yaml", help="Path to golden cases YAML")
    run_parser.add_argument("--config", default="eval/configs/baseline.yaml", help="Path to experiment config YAML")
    run_parser.add_argument("--mode", choices=["A", "B"], default="A", help="Mode A (recall) or Mode B (substitution)")
    run_parser.add_argument("--beats", choices=["original", "regenerate"], default="original", help="Beat source")
    run_parser.add_argument("--limit-beats", type=int, default=None, help="Max beats per case to evaluate")
    run_parser.add_argument("--tag", default="baseline", help="Tag for the run output directory")
    run_parser.add_argument("--cache-selection", action="store_true", help="Cache selection API calls")

    args = parser.parse_args()

    # Subcommand: compare
    compare_parser = subparsers.add_parser("compare", help="Compare two evaluation runs")
    compare_parser.add_argument("run_a", help="Path to run A directory")
    compare_parser.add_argument("run_b", help="Path to run B directory")
    compare_parser.add_argument("--out", default=None, help="Output HTML file path")

    # Subcommand: serve-labels
    subparsers.add_parser("serve-labels", help="Start local server to collect labels")

    # Subcommand: report-labels
    subparsers.add_parser("report-labels", help="Report pairwise labelling win rates and stats")

    args = parser.parse_args()

    if args.command == "run":
        run_dir = run_evaluation(
            cases_file=args.cases,
            config_file=args.config,
            mode=args.mode,
            beats_source=args.beats,
            limit_beats=args.limit_beats,
            tag=args.tag,
            cache_selection=args.cache_selection,
        )
        from vidbrisk_eval.metrics import calculate_metrics_for_run
        from vidbrisk_eval.contact_sheet import generate_contact_sheet
        metrics = calculate_metrics_for_run(run_dir)
        sheet = generate_contact_sheet(run_dir)
        print(f"Run completed successfully: {run_dir}")
        print(f"Contact sheet: {sheet}")
        print("Overall metrics:", json.dumps(metrics.get("overall", {}), indent=2))

    elif args.command == "compare":
        from vidbrisk_eval.compare import generate_comparison_page
        out_p = Path(args.out) if args.out else None
        page = generate_comparison_page(args.run_a, args.run_b, output_file=out_p)
        print(f"Comparison page generated: {page}")

    elif args.command == "serve-labels":
        import uvicorn
        from vidbrisk_eval.labels import app
        print("Starting label server on http://localhost:8000...")
        uvicorn.run(app, host="127.0.0.1", port=8000)

    elif args.command == "report-labels":
        from vidbrisk_eval.labels import report_labels
        report_labels()


if __name__ == "__main__":
    main()
