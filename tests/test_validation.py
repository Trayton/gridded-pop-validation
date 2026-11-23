"""Unit tests for gridded_pop_validation using deterministic synthetic benchmarks."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gridded_pop_validation import (
    accuracy_summary,
    check_raster_anomalies,
    check_window_alignment,
    evaluate_by_group,
    evaluate_population_change,
    group_kfold_indices,
    log_linear_calibrate,
    proportional_calibrate,
    snap_bounds_to_grid,
    theil_mse_decomposition,
)
from gridded_pop_validation.cli import main as cli_main


class TestMetrics(unittest.TestCase):
    def test_theil_decomposition_sums_to_mse(self):
        rng = np.random.default_rng(42)
        obs = rng.uniform(500.0, 50000.0, size=100)
        pred = 1.08 * obs + rng.normal(300.0, 1500.0, size=100)
        dec = theil_mse_decomposition(obs, pred)
        total = dec["bias_component"] + dec["variance_component"] + dec["covariance_component"]
        self.assertAlmostEqual(total, dec["mse"], places=6)
        self.assertAlmostEqual(
            dec["bias_share"] + dec["variance_share"] + dec["covariance_share"],
            1.0,
            places=9,
        )

    def test_accuracy_summary_perfect_fit(self):
        obs = [100.0, 250.0, 1000.0, 4200.0]
        res = accuracy_summary(obs, obs)
        self.assertEqual(res["n"], 4)
        self.assertAlmostEqual(res["mae"], 0.0, places=12)
        self.assertAlmostEqual(res["rmse"], 0.0, places=12)
        self.assertAlmostEqual(res["r2"], 1.0, places=12)
        self.assertAlmostEqual(res["pearson_r"], 1.0, places=12)
        self.assertAlmostEqual(res["spearman_rho"], 1.0, places=12)

    def test_evaluate_by_group(self):
        obs = [1000, 2000, 3000, 4000]
        pred = [1100, 2200, 2700, 3600]
        grp = ["growing", "growing", "shrinking", "shrinking"]
        out = evaluate_by_group(obs, pred, grp)
        self.assertEqual(set(out.keys()), {"growing", "shrinking"})
        self.assertAlmostEqual(out["growing"]["total_relative_bias_pct"], 10.0, places=9)
        self.assertAlmostEqual(out["shrinking"]["total_relative_bias_pct"], -10.0, places=9)

    def test_evaluate_population_change(self):
        o0 = [1000, 1000, 1000, 1000]
        o1 = [1200, 1100, 800, 900]     # +, +, -, -
        p0 = [1000, 1000, 1000, 1000]
        p1 = [1150, 950, 850, 950]      # +, -, -, - (3/4 direction match; both shrinking detected)
        chg = evaluate_population_change(o0, o1, p0, p1)
        self.assertAlmostEqual(chg["direction_agreement_pct"], 75.0, places=9)
        self.assertAlmostEqual(chg["shrinkage_recall_pct"], 100.0, places=9)
        self.assertAlmostEqual(chg["shrinkage_precision_pct"], 200.0 / 3.0, places=9)


class TestRasterAudit(unittest.TestCase):
    def test_window_alignment_and_snap(self):
        # 100m north-up grid with origin (0, 1000)
        transform = (100.0, 0.0, 0.0, 0.0, -100.0, 1000.0)
        aligned = check_window_alignment(transform, (100.0, 200.0, 500.0, 800.0))
        self.assertTrue(aligned["is_pixel_aligned"])

        shifted = check_window_alignment(transform, (100.0, 200.0, 500.0, 840.0))
        self.assertFalse(shifted["is_pixel_aligned"])
        self.assertAlmostEqual(abs(shifted["dy_top_pixels"]), 0.4, places=9)
        self.assertAlmostEqual(abs(shifted["dy_top_units"]), 40.0, places=9)

        snapped, window = snap_bounds_to_grid(transform, (120.0, 230.0, 480.0, 840.0))
        self.assertEqual(snapped, (100.0, 200.0, 500.0, 900.0))
        self.assertEqual(window, (1, 1, 4, 7))
        self.assertTrue(check_window_alignment(transform, snapped)["is_pixel_aligned"])

    def test_raster_anomalies(self):
        arr = np.full((6, 6), 33400.0, dtype=np.float64)
        arr[0, 0] = -5.0
        arr[0, 1] = np.nan
        arr[0, 2] = -9999.0
        rep = check_raster_anomalies(arr, nodata=-9999.0, constant_tile_threshold=16)
        self.assertEqual(rep["negative_cells"], 1)
        self.assertEqual(rep["nan_cells"], 1)
        self.assertTrue(rep["has_constant_fill_warning"])
        self.assertEqual(rep["max_identical_positive_cells"], 33)


class TestCalibrationAndCLI(unittest.TestCase):
    def test_proportional_calibrate_conservation_and_zero_fallback(self):
        vals = [10.0, 30.0, 0.0, 0.0]
        units = ["A", "A", "B", "B"]
        census = {"A": 200.0, "B": 50.0}
        cal, diag = proportional_calibrate(vals, units, census)
        np.testing.assert_allclose(cal, [50.0, 150.0, 25.0, 25.0])
        self.assertEqual(diag["zero_grid_fallback_units"], ["B"])

    def test_log_linear_calibrate_reduces_bias(self):
        rng = np.random.default_rng(7)
        obs = rng.uniform(1000.0, 20000.0, size=60)
        pred = 0.75 * obs ** 1.04
        cal, params = log_linear_calibrate(pred[:40], obs[:40], pred[40:])
        raw_rmse = accuracy_summary(obs[40:], pred[40:])["rmse"]
        cal_rmse = accuracy_summary(obs[40:], cal)["rmse"]
        self.assertLess(cal_rmse, raw_rmse)
        self.assertGreater(params["beta"], 0.0)

    def test_group_kfold_no_group_leakage(self):
        groups = ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P5", "P6"]
        splits = group_kfold_indices(groups, n_splits=3)
        self.assertEqual(len(splits), 3)
        seen_val = set()
        for tr, va in splits:
            tr_g = {groups[i] for i in tr}
            va_g = {groups[i] for i in va}
            self.assertEqual(tr_g & va_g, set())
            seen_val.update(va.tolist())
        self.assertEqual(seen_val, set(range(len(groups))))

    def test_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_csv = Path(tmp) / "sample.csv"
            out_json = Path(tmp) / "report.json"
            with in_csv.open("w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["unit_id", "group", "census_t0", "census_pop", "grid_t0", "gridded_pop"])
                w.writerow(["U1", "urban", 1000, 1200, 980, 1150])
                w.writerow(["U2", "urban", 2000, 2300, 2050, 2250])
                w.writerow(["U3", "rural", 1500, 1300, 1520, 1450])
                w.writerow(["U4", "rural", 1800, 1600, 1780, 1650])
            rc = cli_main([
                str(in_csv),
                "--group-col", "group",
                "--obs-t0-col", "census_t0",
                "--pred-t0-col", "grid_t0",
                "-o", str(out_json),
            ])
            self.assertEqual(rc, 0)
            rep = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(rep["overall"]["n"], 4)
            self.assertIn("urban", rep["by_group"])
            self.assertIn("rural", rep["by_group"])
            self.assertAlmostEqual(rep["change"]["direction_agreement_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
