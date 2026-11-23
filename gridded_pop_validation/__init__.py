"""Accuracy validation, error decomposition, alignment auditing, and bias calibration
for gridded population datasets.
"""

from .metrics import (
    accuracy_summary,
    evaluate_by_group,
    evaluate_population_change,
    theil_mse_decomposition,
)
from .raster_audit import (
    check_raster_anomalies,
    check_window_alignment,
    snap_bounds_to_grid,
)
from .calibration import (
    group_kfold_indices,
    log_linear_calibrate,
    proportional_calibrate,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "accuracy_summary",
    "evaluate_by_group",
    "evaluate_population_change",
    "theil_mse_decomposition",
    "check_raster_anomalies",
    "check_window_alignment",
    "snap_bounds_to_grid",
    "group_kfold_indices",
    "log_linear_calibrate",
    "proportional_calibrate",
]
