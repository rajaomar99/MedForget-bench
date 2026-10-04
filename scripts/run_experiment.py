"""
scripts/run_experiment.py
--------------------------
CLI entry point for running a full unlearning experiment.

Usage:
    python scripts/run_experiment.py --config configs/experiments/exp001_classwise_pathmnist.yaml
"""

import argparse
from medforget.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run a MedForget-bench unlearning experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment YAML config file",
    )
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
