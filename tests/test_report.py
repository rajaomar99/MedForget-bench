"""
tests/test_report.py
--------------------
Sanity test suite for medforget/report.py.
Compatible with both pytest and python's built-in unittest module.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from medforget.report import BenchmarkReport


class TestBenchmarkReport(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        methods = ["exact_retrain", "naive_finetune", "gradient_ascent"]
        seeds = [42, 123, 456]

        for seed in seeds:
            for method in methods:
                data = {
                    "experiment": "test_exp",
                    "method": method,
                    "seed": seed,
                    "time_seconds": 100.0 if method != "exact_retrain" else 600.0,
                    "retain_acc": 0.95,
                    "test_acc": 0.85,
                    "forget_acc": 0.05 if method != "naive_finetune" else 0.90,
                    "mia": 0.05 if method != "naive_finetune" else 0.45,
                    "efficiency": 0.166 if method != "exact_retrain" else 1.0,
                }
                file_path = self.test_dir / f"{method}_seed{seed}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_results_success(self):
        reporter = BenchmarkReport(results_dir=self.test_dir)
        self.assertEqual(len(reporter.raw_df), 9)
        self.assertEqual(
            set(reporter.raw_df["method"].unique()),
            {"exact_retrain", "naive_finetune", "gradient_ascent"}
        )
        self.assertEqual(set(reporter.raw_df["seed"].unique()), {42, 123, 456})

    def test_long_format_structure(self):
        reporter = BenchmarkReport(results_dir=self.test_dir)
        long_df = reporter.long_df

        self.assertIsInstance(long_df, pd.DataFrame)
        for col in ["experiment", "method", "seed", "metric_name", "value"]:
            self.assertIn(col, long_df.columns)

    def test_summary_table_formatting(self):
        reporter = BenchmarkReport(results_dir=self.test_dir)
        summary = reporter.get_summary_table(format_as_str=True)

        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), 3)
        for col in summary.columns:
            self.assertTrue(summary[col].str.contains("±").all())

    def test_export_tables(self):
        out_dir = self.test_dir / "custom_reports"
        reporter = BenchmarkReport(results_dir=self.test_dir, output_dir=out_dir)
        csv_path, md_path = reporter.export_tables()

        self.assertTrue(csv_path.exists())
        self.assertTrue(md_path.exists())
        self.assertGreater(csv_path.stat().st_size, 0)
        self.assertGreater(md_path.stat().st_size, 0)

        content = md_path.read_text(encoding="utf-8")
        self.assertIn("# Benchmark Comparison Table", content)
        self.assertIn("Exact Retrain (Oracle)", content)

    def test_generate_figures(self):
        out_dir = self.test_dir / "custom_reports"
        reporter = BenchmarkReport(results_dir=self.test_dir, output_dir=out_dir)
        artifacts = reporter.generate_all()

        p1 = artifacts["figure_retain_forget"]
        p2 = artifacts["figure_pareto"]
        p3 = artifacts["figure_mia"]

        self.assertTrue(p1.exists() and p1.stat().st_size > 0)
        self.assertTrue(p2.exists() and p2.stat().st_size > 0)
        self.assertTrue(p3.exists() and p3.stat().st_size > 0)

    def test_missing_directory_raises_error(self):
        non_existent = self.test_dir / "does_not_exist"
        with self.assertRaises(FileNotFoundError):
            BenchmarkReport(results_dir=non_existent)

    def test_empty_directory_raises_error(self):
        empty_dir = self.test_dir / "empty_dir"
        empty_dir.mkdir()
        with self.assertRaises(ValueError):
            BenchmarkReport(results_dir=empty_dir)


if __name__ == "__main__":
    unittest.main()
