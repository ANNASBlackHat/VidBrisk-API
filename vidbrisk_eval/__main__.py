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
        metrics = calculate_metrics_for_run(run_dir)
        print(f"Run completed successfully: {run_dir}")
        print("Overall metrics:", json.dumps(metrics.get("overall", {}), indent=2))


if __name__ == "__main__":
    main()
