"""
plot_gen_table.py

Read CSV files produced by gen_table.py and create one scatter plot per CSV.
X axis: var_between (variance of element-wise difference between hidden states)
Y axis: f1_diff (f1 difference as computed in gen_table)

Usage:
  # plot a single CSV
  python plot_gen_table.py --csv /path/to/gen_table_results_blend_vs_full_prefill.csv

  # or plot all comparison CSVs in a directory
  python plot_gen_table.py --dir /path/to/exports --out-dir /path/to/plots

The script writes PNG files (one per CSV) next to the CSV or in --out-dir.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import statistics

import matplotlib
# Use Agg backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def read_xy_from_csv(path: Path):
    x = []
    y = []
    # Backwards-compatible simple reader kept for callers that expect f1 plotting.
    # We now provide a more flexible per-metric reader below; keep this as a
    # thin wrapper for the 'f1' metric so existing code paths continue to work.
    return read_metric_from_csv(path, metric_key="f1")


def read_metric_from_csv(path: Path, metric_key: str = "f1"):
    """Read var_between and <metric>_diff columns from CSV and apply filters.

    Filters applied:
    - If the CSV file name is of the form <left>_vs_<right>.csv and one side is
      'full_prefill', then any row where that side's absolute metric value is 0
      will be dropped (user requested: drop samples where full_prefill metric is 0).
    - Only rows where <metric>_diff > 0 are kept (i.e., the comparison metric is
      larger than full_prefill for that metric when diff is computed as right-left
      or left-right depending on the CSV generation).
    """
    metric_diff_col = f"{metric_key}_diff"
    metric_left_col = f"{metric_key}_left"
    metric_right_col = f"{metric_key}_right"

    x = []
    y = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "var_between" not in fieldnames or metric_diff_col not in fieldnames:
            raise ValueError(f"CSV {path} missing required columns 'var_between' or '{metric_diff_col}'")

        # Determine whether the CSV compares against full_prefill by inspecting
        # the filename pattern left_vs_right. If so, note which side corresponds
        # to full_prefill so we can lookup the appropriate per-metric column.
        stem = path.stem
        full_side = None
        if "_vs_" in stem:
            left, right = stem.split("_vs_", 1)
            if left == "full_prefill":
                full_side = "left"
            elif right == "full_prefill":
                full_side = "right"

        # select which per-metric column to check for full_prefill value
        full_metric_col = None
        if full_side == "left":
            full_metric_col = metric_left_col if metric_left_col in fieldnames else None
        elif full_side == "right":
            full_metric_col = metric_right_col if metric_right_col in fieldnames else None

        for row in reader:
            try:
                xv = float(row.get("var_between", "nan"))
                yv = float(row.get(metric_diff_col, "nan"))
            except Exception:
                continue
            if not math.isfinite(xv) or not math.isfinite(yv):
                continue

            # If we know which column corresponds to full_prefill for this metric,
            # drop rows where full_prefill's metric value is exactly 0.
            if full_metric_col is not None:
                try:
                    full_val = float(row.get(full_metric_col, "nan"))
                except Exception:
                    continue
                if not math.isfinite(full_val):
                    continue
                if full_val == 0.0:
                    continue

            # Only keep rows where the metric difference is positive.
            if not (yv > 0.0):
                continue

            x.append(xv)
            y.append(yv)

    return np.array(x, dtype=float), np.array(y, dtype=float)
    return np.array(x, dtype=float), np.array(y, dtype=float)


def plot_xy(x: np.ndarray, y: np.ndarray, out_path: Path, title: Optional[str] = None):
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=20, alpha=0.7)
    plt.xlabel("var_between (element-wise variance)")
    plt.ylabel("f1_diff")
    # allow callers to override ylabel via title or by passing a title explicitly
    if title:
        plt.title(title)

    # add linear fit if possible
    if x.size >= 2:
        try:
            coeffs = np.polyfit(x, y, 1)
            xs = np.linspace(np.min(x), np.max(x), 100)
            ys = np.polyval(coeffs, xs)
            plt.plot(xs, ys, color="C1", linestyle="--", label=f"fit: y={coeffs[0]:.3g}x+{coeffs[1]:.3g}")
            plt.legend()
        except Exception:
            pass

    # small layout tweak and save
    plt.grid(alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()


def find_csvs_in_dir(d: Path):
    # match files that look like gen_table outputs: *_vs_*.csv
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix == ".csv" and "_vs_" in p.stem])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", dest="dir", default="./hidden_states_compares", type=Path, help="Directory containing CSVs to plot")
    parser.add_argument("--out-dir", dest="out_dir", default="./pic", type=Path, help="Directory to write PDFs into (overrides default pic folder)")
    args = parser.parse_args(argv)

    files = []
    if args.dir:
        files.extend(find_csvs_in_dir(args.dir))
    if not files:
        print("No CSV files specified. Use --csv or --dir.")
        return 2

    for csv_path in files:
        try:
            # create plots for f1, precision and recall
            metrics = [
                ("f1", "f1", "f1_diff", "F1"),
                ("prec", "prec", "prec_diff", "Precision"),
                ("rec", "rec", "rec_diff", "Recall"),
            ]
        except Exception as e:
            print(f"Skipping {csv_path}: {e}", file=sys.stderr)
            continue
        # derive out path: default is ./pic in current working directory (create if missing)
        if args.out_dir:
            out_dir = args.out_dir
        else:
            out_dir = Path.cwd() / "pic"
        out_dir.mkdir(parents=True, exist_ok=True)

        comp_name = csv_path.stem
        # For each metric, try to read and plot; skip if the CSV lacks required columns
        any_plots = 0
        for metric_key, short_key, diff_col, pretty in metrics:
            try:
                x, y = read_metric_from_csv(csv_path, metric_key=metric_key)
            except Exception as e:
                # missing columns or parse error => skip this metric
                print(f"Skipping metric {pretty} for {csv_path}: {e}", file=sys.stderr)
                continue
            if x.size == 0:
                print(f"No valid points for {pretty} in {csv_path}, skipping.")
                continue

            out_file = out_dir / f"{comp_name}_{short_key}.pdf"
            title = f"{comp_name.replace('_', ' ')} — {pretty}"
            # set ylabel to the metric diff name
            plt_ylabel = f"{pretty} diff"
            # plot with same helper but set ylabel before saving
            plt.figure(figsize=(6, 4))
            plt.scatter(x, y, s=20, alpha=0.7)
            plt.xlabel("var_between (element-wise variance)")
            plt.ylabel(plt_ylabel)
            plt.title(title)

            if x.size >= 2:
                try:
                    coeffs = np.polyfit(x, y, 1)
                    xs = np.linspace(np.min(x), np.max(x), 100)
                    ys = np.polyval(coeffs, xs)
                    plt.plot(xs, ys, color="C1", linestyle="--", label=f"fit: y={coeffs[0]:.3g}x+{coeffs[1]:.3g}")
                    plt.legend()
                except Exception:
                    pass

            plt.grid(alpha=0.2)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_file), dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Wrote plot {out_file} ({x.size} points)")
            any_plots += 1

        if any_plots == 0:
            print(f"No plots created for {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
