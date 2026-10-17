"""
medforget/report.py
-------------------
Benchmark report generator and publication-quality visualization engine for MedForget-bench.

Aggregates per-run JSON results across seeds and unlearning methods into:
1. Long-format and wide-format Pandas DataFrames.
2. Statistical comparison tables (mean ± std) exported to CSV and Markdown.
3. Publication-grade figures (300 DPI) for academic papers and thesis presentations:
   - Grouped bar chart: Retain vs. Forget Accuracy (with error bars).
   - Pareto trade-off scatter plot: Forget Accuracy vs. Efficiency Ratio.
   - Privacy evaluation chart: Membership Inference Attack (MIA) with chance baseline.

Adheres strictly to the MedForget Engineering Roadmap Specification (Section 10).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/CLI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("medforget.report")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Styling & Color Palette Configuration ──────────────────────────────────────
# Professional color palette suitable for academic journals (MICCAI / NeurIPS / ICLR)
PALETTE = {
    "exact_retrain": "#1f4e78",     # Deep Academic Navy (Gold Standard)
    "naive_finetune": "#d95f02",    # Warm Amber / Rust (Weak Baseline)
    "gradient_ascent": "#2e7d32",   # Forest Green (Target Parameter-efficient)
    "retain": "#1f4e78",            # Blue for utility/retention
    "forget": "#c0392b",            # Crimson for forgetting
    "test": "#546e7a",              # Slate Gray for generalization
    "mia": "#6a1b9a",               # Deep Violet for privacy risk
    "chance_line": "#d32f2f",       # Red dashed reference line
}

METHOD_PRETTY_NAMES = {
    "exact_retrain": "Exact Retrain (Oracle)",
    "naive_finetune": "Naive Fine-Tune",
    "gradient_ascent": "Gradient Ascent",
}

METRIC_LABELS = {
    "retain_acc": "Retain Accuracy (↑)",
    "test_acc": "Test Accuracy (↑)",
    "forget_acc": "Forget Accuracy (↓)",
    "mia": "MIA Vulnerability (↓)",
    "efficiency": "Efficiency Ratio (↓)",
}


def _apply_academic_plot_style() -> None:
    """Configures global matplotlib parameters for clean, publication-grade figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "figure.autolayout": True,
    })


def _despine(ax: plt.Axes) -> None:
    """Removes top and right spines from an axis for clean academic presentation."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")


class BenchmarkReport:
    """Core reporting engine for aggregating, analyzing, and plotting benchmark results.

    Attributes:
        results_dir (Path): Directory containing raw per-run result JSON files.
        output_dir (Path): Directory where tables and figures will be saved.
        raw_df (pd.DataFrame): Raw records loaded from JSON files.
        long_df (pd.DataFrame): Long-format records (experiment, method, seed, metric, value).
    """

    def __init__(self, results_dir: Union[str, Path], output_dir: Optional[Union[str, Path]] = None):
        """Initializes the report generator.

        Args:
            results_dir: Path to directory containing result JSON files.
            output_dir: Path to output directory. Defaults to `results_dir / 'reports'`.
        """
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory '{self.results_dir}' does not exist.")

        self.output_dir = Path(output_dir) if output_dir else self.results_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.raw_df = self._load_json_files()
        self.long_df = self._create_long_format()

    def _load_json_files(self) -> pd.DataFrame:
        """Loads and validates all JSON result files from the results directory.

        Returns:
            pd.DataFrame: Structured DataFrame of all experiment runs.
        """
        json_files = sorted(self.results_dir.glob("*.json"))
        if not json_files:
            raise ValueError(f"No result JSON files found in '{self.results_dir}'.")

        records: List[Dict] = []
        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Ensure required benchmark keys are present
                if "method" in data and "seed" in data:
                    records.append(data)
                else:
                    logger.warning(f"Skipping '{file_path.name}': Missing 'method' or 'seed' keys.")
            except Exception as e:
                logger.error(f"Error reading '{file_path.name}': {e}")

        if not records:
            raise ValueError(f"No valid result records could be parsed from '{self.results_dir}'.")

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} runs across methods: {df['method'].unique().tolist()}")
        return df

    def _create_long_format(self) -> pd.DataFrame:
        """Transforms raw results into long-format per Roadmap Section 10 Spec.

        Columns: [experiment, method, seed, metric_name, value]
        """
        metadata_cols = ["experiment", "method", "seed"]
        available_meta = [c for c in metadata_cols if c in self.raw_df.columns]
        metric_cols = [c for c in self.raw_df.columns if c not in available_meta and c != "time_seconds"]

        long_df = pd.melt(
            self.raw_df,
            id_vars=available_meta,
            value_vars=metric_cols,
            var_name="metric_name",
            value_name="value",
        )
        return long_df

    def get_summary_table(self, format_as_str: bool = True) -> pd.DataFrame:
        """Computes mean ± std aggregated over random seeds for each method and metric.

        Args:
            format_as_str: If True, returns wide DataFrame where cells are 'mean ± std'.
                           If False, returns MultiIndex DataFrame with numeric mean and std.

        Returns:
            pd.DataFrame: Aggregated summary table.
        """
        metrics = ["retain_acc", "test_acc", "forget_acc", "mia", "efficiency"]
        # Filter only metrics that exist in the dataframe
        avail_metrics = [m for m in metrics if m in self.raw_df.columns]

        # Enforce canonical ordering of methods
        canonical_order = ["exact_retrain", "naive_finetune", "gradient_ascent"]
        methods_in_data = [m for m in canonical_order if m in self.raw_df["method"].unique()]
        # Add any unexpected methods to the end
        for m in self.raw_df["method"].unique():
            if m not in methods_in_data:
                methods_in_data.append(m)

        summary = self.raw_df.groupby("method")[avail_metrics].agg(["mean", "std"])
        summary = summary.reindex(methods_in_data)

        if not format_as_str:
            return summary

        # Format as publication-ready string: 'mean ± std'
        formatted_df = pd.DataFrame(index=summary.index)
        for metric in avail_metrics:
            col_name = METRIC_LABELS.get(metric, metric)
            means = summary[(metric, "mean")]
            stds = summary[(metric, "std")].fillna(0.0)

            formatted_df[col_name] = [
                f"{m:.4f} ± {s:.4f}" for m, s in zip(means, stds)
            ]

        # Pretty-print method names in index
        formatted_df.index = [METHOD_PRETTY_NAMES.get(m, m) for m in formatted_df.index]
        formatted_df.index.name = "Unlearning Method"
        return formatted_df

    def export_tables(self) -> Tuple[Path, Path]:
        """Exports statistical comparison tables to CSV and Markdown files.

        Returns:
            Tuple[Path, Path]: (csv_path, markdown_path)
        """
        summary_table = self.get_summary_table(format_as_str=True)
        raw_numeric_summary = self.get_summary_table(format_as_str=False)

        csv_path = self.output_dir / "comparison_table.csv"
        raw_csv_path = self.output_dir / "summary_statistics.csv"
        md_path = self.output_dir / "comparison_table.md"

        # 1. Export CSVs
        summary_table.to_csv(csv_path)
        raw_numeric_summary.to_csv(raw_csv_path)

        # 2. Export clean GitHub-flavored Markdown
        md_content = [
            f"# Benchmark Comparison Table — {self.raw_df['experiment'].iloc[0]}",
            "",
            "> Evaluated across seeds: " + ", ".join(map(str, sorted(self.raw_df["seed"].unique()))),
            "> Format: **Mean ± Std** ($N=3$). Arrows indicate optimization direction: (↑) Higher is better, (↓) Lower is better.",
            "",
            summary_table.to_markdown(),
            "",
            "### Interpretation Notes:",
            "- **Exact Retrain (Oracle)**: Gold standard reference; completely isolates and trains exclusively on retain data.",
            "- **Naive Fine-Tune**: Fast but weak baseline; demonstrates catastrophic retention of forget-set memory.",
            "- **Gradient Ascent**: Selective parameter-efficient unlearning; balances computational speedup against retention stability.",
        ]

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content) + "\n")

        logger.info(f"Comparison table exported to CSV: {csv_path}")
        logger.info(f"Comparison table exported to Markdown: {md_path}")
        return csv_path, md_path

    def plot_retain_vs_forget(self) -> Path:
        """Generates a grouped bar chart comparing Retain Accuracy vs. Forget Accuracy.

        Shows retention of valid domain knowledge alongside selective unlearning depth.
        """
        _apply_academic_plot_style()
        summary = self.raw_df.groupby("method")[["retain_acc", "forget_acc"]].agg(["mean", "std"])

        canonical_order = ["exact_retrain", "naive_finetune", "gradient_ascent"]
        methods = [m for m in canonical_order if m in summary.index]
        summary = summary.reindex(methods)

        x = np.arange(len(methods))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

        retain_means = summary[("retain_acc", "mean")]
        retain_stds = summary[("retain_acc", "std")].fillna(0.0)

        forget_means = summary[("forget_acc", "mean")]
        forget_stds = summary[("forget_acc", "std")].fillna(0.0)

        bars1 = ax.bar(
            x - width / 2,
            retain_means,
            width,
            yerr=retain_stds,
            capsize=4,
            label="Retain Accuracy (Utility)",
            color=PALETTE["retain"],
            edgecolor="#153655",
            alpha=0.92,
            zorder=3,
        )

        bars2 = ax.bar(
            x + width / 2,
            forget_means,
            width,
            yerr=forget_stds,
            capsize=4,
            label="Forget Accuracy (Forgetting Efficacy)",
            color=PALETTE["forget"],
            edgecolor="#8a2015",
            alpha=0.92,
            zorder=3,
        )

        # Value annotations atop bars
        for bar in bars1:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#1f4e78",
                zorder=5,
            )

        for bar in bars2:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#c0392b",
                zorder=5,
            )

        ax.set_ylabel("Accuracy Score [0.0 – 1.0]", fontsize=11, fontweight="bold")
        ax.set_title("Utility Retention vs. Forgetting Efficacy Across Methods", fontsize=12.5, fontweight="bold", pad=32)
        ax.set_xticks(x)
        # Wrap long tick labels to prevent horizontal text collisions
        ax.set_xticklabels(
            [METHOD_PRETTY_NAMES.get(m, m).replace(" (Oracle)", "\n(Oracle)") for m in methods],
            fontsize=10.5,
            fontweight="normal",
        )
        ax.set_ylim(0, 1.15)

        # Position legend at the top center above the plot area so it never covers right-side bars
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#cccccc",
            fontsize=9.5,
        )

        _despine(ax)
        plt.tight_layout()

        out_path = self.output_dir / "retain_vs_forget_comparison.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved Retain vs. Forget plot: {out_path}")
        return out_path

    def plot_pareto_efficiency(self) -> Path:
        """Generates a Pareto trade-off scatter plot: Forget Accuracy vs. Efficiency Ratio.

        Illustrates the trade-off between unlearning speedup and unlearning thoroughness.
        """
        _apply_academic_plot_style()
        summary = self.raw_df.groupby("method")[["efficiency", "forget_acc"]].agg(["mean", "std"])

        canonical_order = ["exact_retrain", "naive_finetune", "gradient_ascent"]
        methods = [m for m in canonical_order if m in summary.index]
        summary = summary.reindex(methods)

        fig, ax = plt.subplots(figsize=(8.0, 5.5), dpi=300)

        for method in methods:
            eff_mean = summary.loc[method, ("efficiency", "mean")]
            eff_std = summary.loc[method, ("efficiency", "std")]
            forg_mean = summary.loc[method, ("forget_acc", "mean")]
            forg_std = summary.loc[method, ("forget_acc", "std")]

            color = PALETTE.get(method, "#333333")
            label = METHOD_PRETTY_NAMES.get(method, method)

            ax.errorbar(
                eff_mean,
                forg_mean,
                xerr=eff_std,
                yerr=forg_std,
                fmt="o",
                markersize=10,
                color=color,
                ecolor=color,
                capsize=5,
                elinewidth=1.8,
                label=label,
                zorder=4,
            )

            # Dynamic annotation offset to place labels in wide open space without touching axes or neighboring dots
            if method == "exact_retrain":
                # In subset (high acc), position safely below the dot; in classwise (zero acc), position safely above the zero line
                offset = (-65, -30) if forg_mean > 0.5 else (-65, 14)
            elif method == "gradient_ascent":
                # In subset (high acc), position below dot; in classwise (zero acc), position above dot
                offset = (12, -22) if forg_mean > 0.5 else (12, 10)
            else:  # naive_finetune
                offset = (12, -8) if forg_mean > 0.95 else (12, 6)

            ax.annotate(
                f"{label}\n({eff_mean:.2f}x, {forg_mean:.2f} acc)",
                xy=(eff_mean, forg_mean),
                xytext=offset,
                textcoords="offset points",
                fontsize=9.5,
                fontweight="bold",
                color=color,
            )

        ax.set_xlabel("Efficiency Ratio (Time rel. to Exact Retrain) [Lower is Faster]", fontsize=11, fontweight="bold")
        ax.set_ylabel("Forget-set Accuracy [Lower is Stronger Unlearning]", fontsize=11, fontweight="bold")
        ax.set_title("Pareto Frontier: Unlearning Completeness vs. Computational Cost", fontsize=13, fontweight="bold", pad=14)

        # Highlight optimal corner (low efficiency ratio, low forget accuracy)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlim(-0.05, 1.15)
        ax.set_ylim(-0.05, 1.10)

        # Place legend outside on the right so it NEVER obstructs data points or titles
        ax.legend(
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#cccccc",
            fontsize=9.5,
        )

        _despine(ax)
        plt.tight_layout()

        out_path = self.output_dir / "efficiency_vs_forgetting_pareto.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved Pareto efficiency plot: {out_path}")
        return out_path

    def plot_mia_privacy(self) -> Path:
        """Generates a bar chart evaluating privacy risk via Membership Inference Attack (MIA).

        Displays the Loss-Threshold attack vulnerability with the 0.5 chance line.
        """
        _apply_academic_plot_style()
        summary = self.raw_df.groupby("method")[["mia"]].agg(["mean", "std"])

        canonical_order = ["exact_retrain", "naive_finetune", "gradient_ascent"]
        methods = [m for m in canonical_order if m in summary.index]
        summary = summary.reindex(methods)

        fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=300)

        x = np.arange(len(methods))
        means = summary[("mia", "mean")]
        stds = summary[("mia", "std")].fillna(0.0)
        bar_colors = [PALETTE.get(m, "#444444") for m in methods]

        bars = ax.bar(
            x,
            means,
            width=0.45,
            yerr=stds,
            capsize=5,
            color=bar_colors,
            alpha=0.9,
            edgecolor="#222222",
            zorder=3,
        )

        # Draw chance baseline (0.50 threshold)
        ax.axhline(
            0.50,
            color=PALETTE["chance_line"],
            linestyle="--",
            linewidth=1.8,
            label="Random Guessing Baseline (Attack Failure = 0.50)",
            zorder=4,
        )

        for i, bar in enumerate(bars):
            h = bar.get_height()
            s = stds.iloc[i]
            # When the bar height is right near the 0.50 baseline (e.g., Naive Fine-Tune in exp001),
            # place the text cleanly at y=0.535 with a subtle background badge so neither the error bar
            # nor the red dashed baseline cuts through it.
            if 0.38 <= h <= 0.52:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, 0.535),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#222222",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="#ffffff",
                        edgecolor="#dddddd",
                        linewidth=0.8,
                        alpha=0.95,
                    ),
                    zorder=5,
                )
            else:
                y_text = h + s + 0.015
                ax.annotate(
                    f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, y_text),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#222222",
                    zorder=5,
                )

        ax.set_ylabel("MIA Attack Accuracy", fontsize=11, fontweight="bold")
        ax.set_title("Privacy Leakage Risk: Membership Inference Attack (MIA)", fontsize=12.5, fontweight="bold", pad=32)
        ax.set_xticks(x)
        # Wrap long tick labels to prevent horizontal text collisions
        ax.set_xticklabels(
            [METHOD_PRETTY_NAMES.get(m, m).replace(" (Oracle)", "\n(Oracle)") for m in methods],
            fontsize=10.5,
            fontweight="normal",
        )
        ax.set_ylim(0, 0.78)

        # Position legend at the top center above the plot area so it never covers any bars or values
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            frameon=True,
            facecolor="#ffffff",
            edgecolor="#cccccc",
            fontsize=9.5,
        )

        _despine(ax)
        plt.tight_layout()

        out_path = self.output_dir / "mia_privacy_risk.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved MIA privacy plot: {out_path}")
        return out_path

    def generate_all(self) -> Dict[str, Union[Path, pd.DataFrame]]:
        """Runs the complete reporting pipeline: exports tables and generates all plots.

        Returns:
            Dict: Mapping of generated artifacts (tables, figures, paths).
        """
        logger.info(f"Generating full benchmark report for: {self.results_dir}")
        csv_path, md_path = self.export_tables()
        fig_retain = self.plot_retain_vs_forget()
        fig_pareto = self.plot_pareto_efficiency()
        fig_mia = self.plot_mia_privacy()

        return {
            "summary_table": self.get_summary_table(format_as_str=True),
            "csv_path": csv_path,
            "md_path": md_path,
            "figure_retain_forget": fig_retain,
            "figure_pareto": fig_pareto,
            "figure_mia": fig_mia,
        }
