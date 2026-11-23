from __future__ import annotations

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DataIntegrityTests(unittest.TestCase):
    def test_metrics_json_exists_and_matches_baseline(self):
        metrics_file = ROOT / "results" / "metrics.json"
        self.assertTrue(metrics_file.exists(), "results/metrics.json must exist")
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("ridge", data)
        self.assertAlmostEqual(data["ridge"]["spearman"], 0.6297, places=3)
        self.assertIn("mlp_mean5seed", data)
        self.assertAlmostEqual(data["mlp_mean5seed"]["spearman"], 0.6166, places=3)

    def test_model_comparison_exists(self):
        comp_file = ROOT / "results" / "model_comparison.json"
        self.assertTrue(comp_file.exists(), "results/model_comparison.json must exist")
        with open(comp_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("models", data)
        self.assertIn("esm2_650M", data["models"])
        self.assertIn("esm2_35M", data["models"])
        self.assertIn("saprot_35m_seqonly", data["models"])
        self.assertIn("protbert_bfd", data["models"])

    def test_learning_curve_exists(self):
        lc_file = ROOT / "results" / "learning_curve.json"
        self.assertTrue(lc_file.exists(), "results/learning_curve.json must exist")
        with open(lc_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("curve", data)
        self.assertEqual(len(data["curve"]), 10)


if __name__ == "__main__":
    unittest.main()
