"""Compare concatenated chunk q/k for a layer with a full prefill q/k.

Usage:
    python compare_chunks_prefill.py --chunks-dir /path/to/attn_exports \
        --prefill /path/to/prefill_layer0.pt --out /tmp/compare_out

The script:
- Finds chunk .pt files in --chunks-dir matching `layer{layer}` and containing `.pt`.
- Loads q/k from each chunk and concatenates along dim=0.
- Loads the provided prefill .pt file and compares q/k (elementwise) with the concatenated tensors.
- Computes per-token L2 distance and cosine similarity (flattened per-token features).
- Writes a results file `compare_results.pt` into --out.

This is a standalone utility inspired by functions in
`read_attention_matrix_topk_indices.py` and `row_select_consistent.py`.
"""
from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import numpy as np

# Regex to extract chunk ordering index from filename
SORT_INDEX_RE = re.compile(r"chunk[_-]?(\d+)|_(\d+)_", flags=re.IGNORECASE)


def _extract_index_from_name(name: str) -> Optional[int]:
    m = SORT_INDEX_RE.search(name)
    if not m:
        return None
    for g in m.groups():
        if g is None:
            continue
        try:
            return int(g)
        except Exception:
            continue
    return None


def find_chunk_pt_files(directory: Path, layer: int, name_contains: Optional[str] = None) -> List[Path]:
    layer_marker = f"layer{layer}"
    name_contains_lc = (name_contains or "").lower()
    candidates: List[Path] = []
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".pt":
            continue
        nl = p.name.lower()
        if layer_marker not in nl:
            continue
        if name_contains_lc and name_contains_lc not in nl:
            continue
        candidates.append(p)

    # sort by extracted integer index when available
    indexed = []
    for p in candidates:
        idx = _extract_index_from_name(p.name)
        if idx is None:
            # fallback to large index to push to the end but still stable
            idx = 10 ** 9
        indexed.append((idx, p.name, p))
    indexed.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in indexed]


def load_and_concat_qk(paths: List[Path]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Load q/k from each path and concat along dim=0.
    Returns (q_all, k_all, chunk_lens) where chunk_lens is list of token counts per chunk.
    """
    q_list = []
    k_list = []
    chunk_lens: List[int] = []
    for p in paths:
        try:
            d = torch.load(str(p))
        except Exception as e:
            print(f"Failed to load {p}: {e}")
            continue
        if not isinstance(d, dict):
            print(f"Skipping {p}: not a dict")
            continue
        q = d.get("q")
        k = d.get("k")
        if q is None or k is None:
            print(f"Skipping {p}: missing q or k")
            continue
        q_t = q if isinstance(q, torch.Tensor) else torch.as_tensor(q)
        k_t = k if isinstance(k, torch.Tensor) else torch.as_tensor(k)
        q_t = q_t.to(torch.float32).cpu()
        k_t = k_t.to(torch.float32).cpu()
        q_list.append(q_t)
        k_list.append(k_t)
        # infer chunk token count from q's first dimension if possible
        try:
            chunk_lens.append(int(q_t.shape[0]))
        except Exception:
            chunk_lens.append(0)

    if not q_list:
        return torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32), chunk_lens

    # try to concatenate along dim=0
    try:
        q_all = torch.cat(q_list, dim=0)
        k_all = torch.cat(k_list, dim=0)
    except Exception:
        # if concat fails, stack along new dim
        q_all = torch.stack(q_list, dim=0)
        k_all = torch.stack(k_list, dim=0)
    return q_all, k_all, chunk_lens


def flatten_per_token(t: torch.Tensor) -> torch.Tensor:
    """Flatten per-token features into shape (num_tokens, feat_dim).
    If tensor has shape (num_tokens, ...), collapse trailing dims.
    """
    if t.ndim == 0:
        return t.view(-1, 1)
    if t.ndim == 1:
        return t.view(-1, 1)
    # assume first dim indexes tokens
    num_tokens = t.shape[0]
    feat = int(np.prod(t.shape[1:]))
    return t.view(num_tokens, feat)


def l2_per_token(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute L2 norm per token between a and b (both must have same shape (T, D))."""
    diff = a - b
    return torch.norm(diff, dim=1)


def cosine_per_token(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity per token between a and b (shape (T, D))."""
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.sum(a_norm * b_norm, dim=1)


def summarize(t: torch.Tensor) -> dict:
    arr = t.detach().cpu().numpy()
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "count": int(arr.size),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", type=str, required=True)
    parser.add_argument("--prefill", type=str, required=True, help="Full prefill .pt file to compare against")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--name-contains", type=str, default=None)
    parser.add_argument("--out", type=str, default="./compare_out")
    args = parser.parse_args(argv)

    chunks_dir = Path(args.chunks_dir)
    prefill = Path(args.prefill)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = find_chunk_pt_files(chunks_dir, args.layer, args.name_contains)
    print(f"Found {len(paths)} chunk files for layer {args.layer} in {chunks_dir}")
    if not paths:
        return 2

    q_all, k_all, chunk_lens = load_and_concat_qk(paths)
    print(f"Concatenated q_all shape: {q_all.shape}, k_all shape: {k_all.shape}")

    try:
        prefill_d = torch.load(str(prefill))
    except Exception as e:
        print(f"Failed to load prefill file {prefill}: {e}")
        return 3

    q_prefill = prefill_d.get("q")
    k_prefill = prefill_d.get("k")
    if q_prefill is None or k_prefill is None:
        print("Prefill file missing 'q' or 'k'")
        return 4
    q_prefill = q_prefill.to(torch.float32).cpu() if isinstance(q_prefill, torch.Tensor) else torch.as_tensor(q_prefill, dtype=torch.float32)
    k_prefill = k_prefill.to(torch.float32).cpu() if isinstance(k_prefill, torch.Tensor) else torch.as_tensor(k_prefill, dtype=torch.float32)

    # Try to align shapes. We expect q_all and q_prefill to have same first-dim (num tokens)
    if q_all.shape[0] != q_prefill.shape[0]:
        print(f"Warning: token count mismatch: q_all {q_all.shape[0]} vs prefill {q_prefill.shape[0]}")
        min_t = min(q_all.shape[0], q_prefill.shape[0])
        print(f"Truncating to min tokens {min_t}")
        q_all = q_all[:min_t]
        q_prefill = q_prefill[:min_t]
        k_all = k_all[:min_t]
        k_prefill = k_prefill[:min_t]

    # Flatten per token
    a_q = flatten_per_token(q_all)
    b_q = flatten_per_token(q_prefill)
    a_k = flatten_per_token(k_all)
    b_k = flatten_per_token(k_prefill)

    if a_q.shape != b_q.shape:
        print(f"Error: flattened q shapes differ: {a_q.shape} vs {b_q.shape}")
        return 5

    if a_k.shape != b_k.shape:
        print(f"Warning: flattened k shapes differ: {a_k.shape} vs {b_k.shape}; attempting to align by truncation")
        mink = min(a_k.shape[0], b_k.shape[0])
        a_k = a_k[:mink]
        b_k = b_k[:mink]

    l2_q = l2_per_token(a_q, b_q)
    cos_q = cosine_per_token(a_q, b_q)
    l2_k = l2_per_token(a_k, b_k)
    cos_k = cosine_per_token(a_k, b_k)

    results = {
        "q_all_shape": tuple(q_all.shape),
        "k_all_shape": tuple(k_all.shape),
        "chunk_lens": chunk_lens,
        "l2_q_summary": summarize(l2_q),
        "cos_q_summary": summarize(cos_q),
        "l2_k_summary": summarize(l2_k),
        "cos_k_summary": summarize(cos_k),
        "l2_q_per_token": l2_q.cpu(),
        "cos_q_per_token": cos_q.cpu(),
    }

    out_file = out_dir / "compare_results.pt"
    torch.save(results, str(out_file))
    print(f"Wrote results to {out_file}")
    # Also print short summaries
    print("q L2 summary:", results["l2_q_summary"]) 
    print("q cos summary:", results["cos_q_summary"]) 
    print("k L2 summary:", results["l2_k_summary"]) 
    print("k cos summary:", results["cos_k_summary"]) 

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
