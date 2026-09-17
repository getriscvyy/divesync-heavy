#!/usr/bin/env python3
"""
compute_rmse.py

Walks a directory of DiveSync Heavy trial folders (e.g. DSH-RL-01, DSH-RL-03,
DSH-RL-05, PID, ...), each containing a processed.csv with columns:

    time_s, actuator_mm, actuator_setpoint_mm, depth_filtered_m,
    depth_setpoint_m, motor_cmd

and computes, for each trial:
  - RMSE of depth tracking (depth_filtered_m vs depth_setpoint_m)
  - % of samples within +/- TOLERANCE_M of the setpoint (default 0.05 m)

Usage:
    python compute_rmse.py [path_to_divesync-heavy-final-data]

If no path is given, defaults to "./divesync-heavy-final-data".
"""

import csv
import math
import sys
from pathlib import Path

# Tolerance band (meters) for the "% time within setpoint" metric
TOLERANCE_M = 0.05


def compute_depth_metrics(csv_path: Path):
    """Return (rmse, pct_within_tol, n_samples) for depth_filtered_m vs
    depth_setpoint_m in csv_path.
    Returns (None, None, 0) if the file is missing required columns or has
    no usable rows.
    """
    squared_errors = []
    within_tol_count = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        required = {"depth_filtered_m", "depth_setpoint_m"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            return None, None, 0

        for row in reader:
            try:
                depth = float(row["depth_filtered_m"])
                setpoint = float(row["depth_setpoint_m"])
            except (ValueError, TypeError):
                # Skip blank/malformed rows
                continue
            error = depth - setpoint
            squared_errors.append(error ** 2)
            if abs(error) <= TOLERANCE_M:
                within_tol_count += 1

    if not squared_errors:
        return None, None, 0

    n = len(squared_errors)
    mse = sum(squared_errors) / n
    rmse = math.sqrt(mse)
    pct_within_tol = 100.0 * within_tol_count / n
    return rmse, pct_within_tol, n


def main():
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("divesync-heavy-final-data")

    if not base_dir.is_dir():
        print(f"Error: '{base_dir}' is not a directory.")
        sys.exit(1)

    trial_dirs = sorted(p for p in base_dir.iterdir() if p.is_dir())

    if not trial_dirs:
        print(f"No trial subfolders found in '{base_dir}'.")
        sys.exit(1)

    results = []
    for trial_dir in trial_dirs:
        csv_path = trial_dir / "processed.csv"
        if not csv_path.exists():
            print(f"[skip] {trial_dir.name}: no processed.csv found")
            continue

        rmse, pct_within_tol, n = compute_depth_metrics(csv_path)
        if rmse is None:
            print(f"[skip] {trial_dir.name}: processed.csv missing required columns or no valid rows")
            continue

        results.append((trial_dir.name, rmse, pct_within_tol, n))

    if not results:
        print("No valid trials processed.")
        sys.exit(1)

    # Print summary table
    name_width = max(len(name) for name, _, _, _ in results)
    name_width = max(name_width, len("Trial"))
    tol_header = f"% within {TOLERANCE_M:.2f}m"

    print()
    print(f"{'Trial':<{name_width}}  {'RMSE (m)':>10}  {tol_header:>14}  {'Samples':>8}")
    print(f"{'-'*name_width}  {'-'*10}  {'-'*14}  {'-'*8}")
    for name, rmse, pct_within_tol, n in results:
        print(f"{name:<{name_width}}  {rmse:>10.5f}  {pct_within_tol:>13.1f}%  {n:>8d}")
    print()


if __name__ == "__main__":
    main()
