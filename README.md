# gridded-pop-validation

Lightweight Python toolkit for **accuracy validation, MSE error decomposition, sub-pixel alignment auditing, and census-anchored bias calibration** of gridded population datasets (e.g., WorldPop, LandScan, GPW).

## Motivation

Global gridded population products are widely used in climate adaptation, public-service accessibility, and sustainability studies, yet their sub-regional accuracy often degrades in rapidly urbanizing or shrinking areas. Common error sources include dasymetric allocation mismatch, intercensal extrapolation lag, uniform-fill raster tiles, and sub-pixel clipping drift when extracting regional windows.

This package provides transparent, dependency-light (NumPy + Python standard library) utilities to diagnose, decompose, and calibrate these errors reproducibly.

## Features

- **Level Accuracy & Theil MSE Decomposition (`metrics.py`)**:
  - `accuracy_summary`: MAE, RMSE, MAPE, MedAPE, sMAPE, mean log error, RMSLE, $R^2$, Pearson $r$, Spearman $\rho$, aggregate relative bias.
  - `theil_mse_decomposition`: Exact decomposition of MSE into mean bias, variance difference, and unsystematic covariance shares:
    $$\text{MSE} = (\bar{\hat{y}} - \bar{y})^2 + (s_{\hat{y}} - s_y)^2 + 2(1 - r)s_{\hat{y}}s_y$$
  - `evaluate_by_group`: Stratified evaluation across urban/rural or growing/shrinking groups.
  - `evaluate_population_change`: Intercensal change correlation, log growth-rate error, directional agreement, and shrinkage recall/precision.
- **Raster Alignment & Integrity Auditing (`raster_audit.py`)**:
  - `check_window_alignment`: Detects sub-pixel offsets between clipping bounding boxes and parent raster affine transforms.
  - `snap_bounds_to_grid`: Snaps clipping bounds outward to exact integer pixel windows.
  - `check_raster_anomalies`: Flags negative counts, non-finite values, and suspicious uniform-fill positive blocks.
- **Calibration & Spatial Block CV (`calibration.py`)**:
  - `proportional_calibrate`: Enforces exact census total conservation per administrative unit with zero-grid fallback diagnostics.
  - `log_linear_calibrate`: Log-linear bias adjustment with Duan smearing retransformation.
  - `group_kfold_indices`: Deterministic group-based K-fold splits to prevent spatial leakage.
- **Command-Line Interface (`cli.py`)**: Run `gridded-pop-validate` directly on CSV benchmark tables to output JSON reports.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+ and NumPy.

## Quick Start

```python
from gridded_pop_validation import (
    accuracy_summary,
    check_window_alignment,
    evaluate_population_change,
    proportional_calibrate,
    snap_bounds_to_grid,
)

# 1. Level accuracy and MSE error decomposition
obs = [12000, 8500, 43000, 6200]
pred = [13100, 9200, 40500, 7100]
summary = accuracy_summary(obs, pred)
print(f"RMSE={summary['rmse']:.1f}, MedAPE={summary['medape_pct']:.2f}%")

# 2. Detect sub-pixel clipping drift before zonal statistics
transform = (1000.0, 0.0, 0.0, 0.0, -1000.0, 50000.0)
audit = check_window_alignment(transform, (1000.0, 2000.0, 9000.0, 12400.0))
if not audit["is_pixel_aligned"]:
    snapped_bounds, pixel_window = snap_bounds_to_grid(transform, (1000.0, 2000.0, 9000.0, 12400.0))

# 3. Census-anchored proportional calibration
calibrated, diag = proportional_calibrate(
    grid_values=[120.0, 280.0, 50.0, 150.0],
    unit_ids=["county_a", "county_a", "county_b", "county_b"],
    census_totals={"county_a": 500.0, "county_b": 180.0},
)
```

### CLI Usage

```bash
gridded-pop-validate data.csv \
  --obs-col census_pop \
  --pred-col gridded_pop \
  --group-col stratum \
  --obs-t0-col census_2010 \
  --pred-t0-col grid_2010 \
  -o validation_report.json
```

## Running Tests

```bash
pytest -v
# or using the standard library test runner:
python3 -m unittest discover -s tests -v
```

## License

MIT License. See [LICENSE](LICENSE).
