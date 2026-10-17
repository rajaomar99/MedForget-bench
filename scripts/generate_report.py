"""
scripts/generate_report.py
--------------------------
CLI entry point for aggregating unlearning experiment results and generating
publication-quality comparison tables and figures.

Usage:
    python scripts/generate_report.py --results_dir results/exp001_classwise_pathmnist
    python scripts/generate_report.py --results_dir results/exp002_random_subset_pathmnist --output_dir reports/subset
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path for direct CLI execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medforget.report import BenchmarkReport


def main() -> None:
    # Ensure UTF-8 output on Windows consoles to prevent charmap encoding errors
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="MedForget-bench Report Generator — Statistical Aggregation & Academic Visualizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        "--results-dir",
        type=str,
        required=True,
        dest="results_dir",
        help="Path to directory containing per-run JSON result files (e.g., results/exp001_classwise_pathmnist)",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Optional custom output directory. Defaults to <results_dir>/reports",
    )

    args = parser.parse_args()

    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"Error: Results directory '{results_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print(" MedForget-bench — Publication Benchmark Report Generator")
    print("=" * 70)
    print(f"Loading results from : {results_path}")

    try:
        reporter = BenchmarkReport(results_dir=results_path, output_dir=args.output_dir)
        artifacts = reporter.generate_all()

        print("\n" + "=" * 70)
        print(" Summary Comparison Table (Mean +/- Std Across Seeds)")
        print("=" * 70)

        # Print markdown table safely across terminal encodings
        summary_df = artifacts["summary_table"]
        try:
            print(summary_df.to_markdown())
        except UnicodeEncodeError:
            # Fallback replacing Unicode arrows and plusminus with ASCII equivalents
            ascii_text = (
                summary_df.to_markdown()
                .replace("↑", "[^]")
                .replace("↓", "[v]")
                .replace("±", "+/-")
            )
            print(ascii_text)

        print("\n" + "=" * 70)
        print(" Generated Publication Artifacts:")
        print("=" * 70)
        print(f"  [CSV Table]       : {artifacts['csv_path']}")
        print(f"  [Markdown Table]  : {artifacts['md_path']}")
        print(f"  [Figure 1 (Bar)]  : {artifacts['figure_retain_forget']}")
        print(f"  [Figure 2 (Pareto): {artifacts['figure_pareto']}")
        print(f"  [Figure 3 (MIA)]   : {artifacts['figure_mia']}")
        print("=" * 70)
        print("Successfully generated all benchmark tables and figures.\n")

    except Exception as e:
        print(f"\nError generating report: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
