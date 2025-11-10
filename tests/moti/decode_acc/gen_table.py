"""
gen_table.py

Scan exported last-layer hidden-state files (PyTorch .pt saved by
`export_last_layer_hidden_states`) and compute per-sample variance and
differences in f1/precision/recall for the following comparisons:

- blend_<id> vs full_prefill_<id>
- full_reuse_<id> vs full_prefill_<id>
- blend_<id> vs full_reuse_<id>

Output is a CSV with one row per sample per comparison.

Usage:
    python gen_table.py --export-dir /path/to/hidden_states_exports --out results.csv

If not provided, --export-dir defaults to <REPO_ROOT>/hidden_states_exports and
--out defaults to gen_table_results.csv in the repo root.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Any

FNAME_RE = re.compile(r"(?P<prefix>blend|full_prefill|full_reuse)_(?P<sample>\d+)_last_layer_hidden_states(?:_.*)?\.pt$")


def load_export(path: Path) -> Dict[str, Any]:
    d = torch.load(str(path), map_location="cpu")
    return d


def find_exports(export_dir: Path) -> Dict[Tuple[str, str], Path]:
    """Return mapping (prefix, sample_id) -> path"""
    res = {}
    if not export_dir.exists():
        print(f"Export dir {export_dir} does not exist", file=sys.stderr)
        return res
    for p in export_dir.iterdir():
        if not p.is_file():
            continue
        m = FNAME_RE.search(p.name)
        if not m:
            continue
        prefix = m.group("prefix")
        sample = m.group("sample")
        res[(prefix, sample)] = p
    return res


def compare_pair(path_a: Path, path_b: Path) -> Dict[str, Any]:
    """Load two exported files and return raw metrics and variances.

    The caller decides the direction of differences (which minus which).
    """
    print(f"Comparing a:{path_a} and b:{path_b}")
    a = load_export(path_a)
    b = load_export(path_b)

    ha = a.get("last_layer_hidden_states")
    hb = b.get("last_layer_hidden_states")

    # Compute a distance/similarity metric between the two hidden states.
    # Previously we used cosine similarity; replace with Euclidean distance
    # (L2 norm of the elementwise difference) as requested.
    # print(f"ha shape: {ha.shape}, hb shape: {hb.shape}")
    # take last token's hidden vector and compute L2 distance directly
    ha = ha[-1, :]
    hb = hb[-1, :]
    # print(f"ha last token shape: {ha.shape}, hb last token shape: {hb.shape}")
    var_between = float(torch.norm(ha - hb, p=2).item())
    # var_between = F.cosine_similarity(ha.unsqueeze(0), hb.unsqueeze(0)).item()

    f1_a = float(a.get("f1") or 0.0)
    f1_b = float(b.get("f1") or 0.0)
    prec_a = float(a.get("precision") or 0.0)
    prec_b = float(b.get("precision") or 0.0)
    rec_a = float(a.get("recall") or 0.0)
    rec_b = float(b.get("recall") or 0.0)

    return {
        "var_between": var_between,
        "f1_a": f1_a,
        "f1_b": f1_b,
        "prec_a": prec_a,
        "prec_b": prec_b,
        "rec_a": rec_a,
        "rec_b": rec_b,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", dest="export_dir", type=Path, default="./hidden_states_exports")
    parser.add_argument("--out-dir", dest="out_dir", type=Path, default="./hidden_states_compares")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    export_dir = args.export_dir
    out_dir = args.out_dir

    exports = find_exports(export_dir)
    if not exports:
        print(f"No exports found in {export_dir}", file=sys.stderr)
        return 2

    # collect sample ids present for each prefix
    samples = set(sample for (_, sample) in exports.keys())

    # We compute three comparisons. For each we define how to compute the
    # metric differences (f1/precision/recall) according to your request:
    # - blend vs full_prefill : diff = full_prefill - blend
    # - full_reuse vs full_prefill : diff = full_prefill - full_reuse
    # - blend vs full_reuse : diff = blend - full_reuse
    comparisons = [
        {"left": "blend", "right": "full_prefill", "diff_order": "right_minus_left"},
        {"left": "full_reuse", "right": "full_prefill", "diff_order": "right_minus_left"},
        {"left": "blend", "right": "full_reuse", "diff_order": "left_minus_right"},
    ]

    # Determine output directory and base name. If --out is a .csv file name,
    # use its parent directory and stem as base; otherwise treat --out as a directory.
    out_dir.mkdir(parents=True, exist_ok=True)

    any_written = 0
    for comp_spec in comparisons:
        left = comp_spec["left"]
        right = comp_spec["right"]
        diff_order = comp_spec.get("diff_order", "right_minus_left")

        rows_comp = []
        for sample in sorted(samples, key=lambda s: int(s)):
            key_l = (left, sample)
            key_r = (right, sample)
            if key_l not in exports or key_r not in exports:
                if not args.quiet:
                    print(f"Skipping sample {sample} for comparison {left} vs {right}: missing file")
                continue
            path_l = exports[key_l]
            path_r = exports[key_r]
            try:
                comp = compare_pair(path_l, path_r)
            except Exception as e:
                print(f"Error comparing {path_l} and {path_r}: {e}", file=sys.stderr)
                continue

            # Compute diffs in the requested direction for the metrics (f1/prec/rec)
            if diff_order == "right_minus_left":
                f1_diff = comp["f1_b"] - comp["f1_a"]
                prec_diff = comp["prec_b"] - comp["prec_a"]
                rec_diff = comp["rec_b"] - comp["rec_a"]
            else:  # left_minus_right
                f1_diff = comp["f1_a"] - comp["f1_b"]
                prec_diff = comp["prec_a"] - comp["prec_b"]
                rec_diff = comp["rec_a"] - comp["rec_b"]

            # Variance between two hidden states (element-wise variance of difference)
            var_between = comp.get("var_between", float("nan"))

            rows_comp.append({
                "sample": sample,
                "var_between": var_between,
                "f1_left": comp["f1_a"],
                "f1_right": comp["f1_b"],
                "f1_diff": f1_diff,
                "prec_left": comp["prec_a"],
                "prec_right": comp["prec_b"],
                "prec_diff": prec_diff,
                "rec_left": comp["rec_a"],
                "rec_right": comp["rec_b"],
                "rec_diff": rec_diff,
            })

        # write per-comparison CSV
        if rows_comp:
            # Explicit header order (no 'comparison' field — the comparison is in the file name)
            fieldnames = [
                "sample",
                "var_between",
                "f1_left",
                "f1_right",
                "f1_diff",
                "prec_left",
                "prec_right",
                "prec_diff",
                "rec_left",
                "rec_right",
                "rec_diff",
            ]
            out_file = out_dir / f"{left}_vs_{right}.csv"
            with out_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows_comp:
                    writer.writerow(r)
            any_written += len(rows_comp)
            print(f"Wrote {len(rows_comp)} rows to {out_file}")
        else:
            if not args.quiet:
                print(f"No rows for comparison {left} vs {right}")

    if any_written == 0:
        print("No comparison rows produced.")


if __name__ == "__main__":
    raise SystemExit(main())
