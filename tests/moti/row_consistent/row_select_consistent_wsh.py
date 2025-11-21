import torch
import os
import argparse
from pathlib import Path
import re
from typing import List, Optional, Tuple, Union

def load_chunk_files_for_layer(directory: Union[str, Path], layer: int,
                               name_contains: Optional[str] = None,
                               sort_index_regex: str = r"chunk[_-]?(\d+)|_(\d+)_") -> List[Tuple[Path, dict]]:
    """Load .pt chunk files for a specific layer from `directory`.

    This function is robust to directories that contain other files. It filters
    for files with a ``.pt`` suffix and that contain the marker ``layer{layer}``.
    An optional ``name_contains`` substring can be supplied to further narrow
    matches. The returned list is sorted by an integer index extracted from the
    filename using ``sort_index_regex`` when possible; otherwise filenames are
    sorted lexicographically.

    Args:
        directory: path to directory containing .pt files.
        layer: the layer number to filter by (looks for 'layer{layer}' in filename).
        name_contains: optional substring filenames must contain (case-insensitive).
        sort_index_regex: regex (as a string) with a capturing group for an
            integer used to sort chunk files (for example a pattern that
            matches ``chunk_<index>`` or ``_<index>_``). Multiple capture
            groups are supported; the first matched numeric group is used.

    Returns:
        A list of (Path, loaded_object) tuples for the matching .pt files.
    """
    dirp = Path(directory)
    results: List[Tuple[Path, dict]] = []
    if not dirp.exists():
        print(f"Directory does not exist: {dirp}")
        return results

    layer_marker = f"layer{layer}"
    name_contains_lc = (name_contains or "").lower()

    candidates = []
    for p in dirp.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".pt":
            continue
        name_lc = p.name.lower()
        # Must contain the layer marker
        if layer_marker not in name_lc:
            continue
        if name_contains_lc and name_contains_lc not in name_lc:
            continue
        candidates.append(p)

    if not candidates:
        return results

    # Try to sort by an integer index extracted from the filename using regex
    def _extract_index(p: Path) -> Optional[int]:
        m = re.search(sort_index_regex, p.name, flags=re.IGNORECASE)
        if not m:
            return None
        # find first numeric capture group
        for g in m.groups():
            if g is None:
                continue
            try:
                return int(g)
            except Exception:
                continue
        return None

    # build list of (index_or_inf, path) to sort; missing index -> large number
    indexed = []
    for p in candidates:
        idx = _extract_index(p)
        if idx is None:
            continue
        indexed.append((idx, p))

    indexed.sort(key=lambda t: (t[0], t[1].name))

    for _, p in indexed:
        try:
            data = torch.load(str(p))
            results.append((p, data))
        except Exception as e:
            print(f"Failed to load {p}: {e}")

    return results

def extract_chunk_len_from_chunks(chunks: List[Tuple[Path, dict]]) -> List[Tuple[Path, int]]:
    """Extract chunk lengths from loaded chunk list.

    Args:
        chunks: list of (Path, data_dict) as returned by
            ``load_chunk_files_for_layer``.

    Returns:
        A list of tuples (Path, chunk_length).
    """
    out: List[Tuple[Path, int]] = []
    for p, data in chunks:
        if not isinstance(data, dict):
            print(f"Skipping {p}: loaded object is not a dict (type={type(data)})")
            continue

        meta = data.get("meta", {}) if isinstance(data.get("meta", {}), dict) else {}

        out.append((p, meta.get("num_tokens", -1)))

    return out

def row_attention_select(output_dir, prefill_file: Path, chunks: List[Tuple[Path, int]]):
    """Select token to recompute based on row attention.
    Args:
        output_dir: directory to save output index files.
        prefill_file: optional Path to a .pt file containing full prefill
            q/k tensors and metadata.
        chunks: optional list of (Path, chunk_length) tuples representing
            loaded chunk files for a specific layer.
    
    Returns:
        None
    
    Note: This function saves all layers' selection indices in `output_dir`
    """
    if prefill_file is None:
        print("No prefill_file provided; skipping row attention selection.")
        return
    # get layer number from prefill_file name
    m = re.search(r"layer[_-]?(\d+)", prefill_file.name, flags=re.IGNORECASE)
    layer_num = -1
    if m:
        try:
            layer_num = int(m.group(1))
        except Exception:
            pass
    if layer_num < 0:
        print(f"Could not determine layer number from prefill_file name: {prefill_file}")
        return
    print(f"Processing layer {layer_num} from prefill file: {prefill_file}")
    prefill_data = torch.load(prefill_file)
    print(f"Loaded prefill file: {prefill_file}")
    q = prefill_data.get("q", None).to(torch.float32)
    k = prefill_data.get("k", None).to(torch.float32)
    meta = prefill_data.get("meta", {})
    if q is None or k is None:
        print(f"Prefill file {prefill_file} missing 'q' or 'k' tensors; skipping.")
        return
    print(f"q shape: {q.shape}, k shape: {k.shape}")
    if not isinstance(meta, dict):
        print(f"Prefill file {prefill_file} has invalid 'meta' field; expected dict, got {type(meta)}")
        return
    
    # Try to use metadata-driven reshape if possible
    h_dim = meta.get("h_dim")
    num_tokens = meta.get("num_tokens")
    num_kv_heads = meta.get("num_kv_heads")
    num_queries_per_kv = meta.get("num_queries_per_kv")
    scaling = meta.get("scaling")
    print(f"Metadata: h_dim={h_dim}, num_tokens={num_tokens}, num_kv_heads={num_kv_heads}, num_queries_per_kv={num_queries_per_kv}, scaling={scaling}")

    attention_scores = None

    if None not in (h_dim, num_kv_heads, num_queries_per_kv) and q.ndim >= 2 and k.ndim >= 2:
        # attempt the same reshaping logic as older code paths
        try:
            q_matrix = q.view(q.shape[0], num_kv_heads, num_queries_per_kv, h_dim).transpose(0, 2)
            k_matrix = k[:, :, None, :].expand(k.shape[0], num_kv_heads, num_queries_per_kv, h_dim).transpose(0, 2)

            # print(f"q_matrix shape after reshape: {q_matrix.shape}, k_matrix shape after reshape: {k_matrix.shape}")

            attn_weights = torch.matmul(q_matrix, k_matrix.transpose(2, 3)) / scaling
            # print(f"attn_weights shape: {attn_weights.shape}\nattn_weights: {attn_weights[0, 0, :, :]}")

            query_len, key_len = attn_weights.shape[-2], attn_weights.shape[-1]
            # causal lower-triangular mask as in original code
            mask = torch.tril(torch.ones(query_len, key_len, device=attn_weights.device))
            mask = mask.masked_fill(mask == 0, float('-inf'))
            attn_weights = attn_weights + mask
            # print(f"after mask, attn_weights shape: {attn_weights.shape}\nattn_weights: {attn_weights[0, 0, :, :]}")

            attention_scores = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            # print(f"dtype after softmax: {attention_scores.dtype}")
            # print(f"after softmax, attn_weights shape: {attention_scores.shape}\nattn_weights: {attention_scores[0, 0, :, :]}")
        except Exception as e:
            print(f"Metadata-based reshape failed: {e}")

    if attention_scores is None:
        print("Unable to compute attention: q/k tensors have unsupported shapes and metadata was insufficient.")
        return

    attn_score = attention_scores[0, 0, :, :]
    # attn_score = torch.mean(attention_scores, dim=0)  # average over groups if present
    # attn_score = torch.nn.functional.normalize(attn_score, p=1.0, dim=-1)  # ensure rows sum to 1
    print(f"shape of attn_score: {attn_score.shape}\nattn_score: {attn_score}")
    # generate the per-chunk mask: for query tokens belonging to chunk i,
    # allow attention only to key tokens from chunk i and later chunks.
    # Build chunk lengths from per-chunk metadata when available, otherwise
    # infer from per-chunk 'q' tensors when possible.
    chunk_lens: List[int] = []
    for _, len in chunks:
        chunk_lens.append(len)

    total_chunk_tokens = sum(chunk_lens)
    query_len = attn_score.shape[-2]
    key_len = attn_score.shape[-1]

    if total_chunk_tokens == 0:
        # If we couldn't determine chunk sizes, try to fallback to using
        # the full sequence and skip per-chunk masking.
        print("Warning: could not determine per-chunk token counts; skipping per-chunk mask generation")
        attention_scores_copy = attn_score.clone()
    else:
        # Build a 2D mask (query_len x key_len) where mask[q,k]=1 if
        # key index k belongs to the same or a later chunk than query q.
        mask2d = torch.zeros((query_len, key_len), dtype=attn_score.dtype, device=attn_score.device)

        # compute chunk start offsets
        starts = []
        s = 0
        for l in chunk_lens:
            starts.append(s)
            s += l
        # print(f"Chunk lengths: {chunk_lens}, starts: {starts}")

        # apply per-chunk rules: for chunk i, queries in [starts[i], starts[i]+l)
        # can attend to keys in [starts[i], end)
        pos = 0
        for i, l in enumerate(chunk_lens):
            if l <= 0:
                continue
            q_start = pos
            q_end = min(pos + l, query_len)
            k_start = starts[i]
            if q_start >= q_end:
                pos += l
                continue
            if k_start >= key_len:
                # nothing to allow for this chunk
                pos += l
                continue
            mask2d[q_start:q_end, k_start:key_len] = 1.0
            pos += l

        # If chunks cover fewer tokens than the attention matrix, allow the
        # remaining queries to attend to remaining keys (conservative)
        if pos < query_len:
            mask2d[pos:query_len, min(pos, key_len):key_len] = 1.0

        print(f"Mask broadcast shape: {mask2d.shape}")
        # print(f"attn_score: {attn_score}\nmask: {mask2d}")
        attention_scores_copy = attn_score * mask2d

    # Aggregate attention mass per query index across all leading dims
    # attention_scores_copy shape may be (Q, K) or (T, H, Q, K) etc.; query dim is -2
    per_query_sum = attention_scores_copy.sum(dim=-1)

    # Ensure per_query_sum is 1-D (length = number of queries)
    if per_query_sum.dim() != 1:
        per_query_sum = per_query_sum.view(-1)

    # find query indices where total allowed attention mass < threshold
    threshold = 0.8
    low_idx = torch.where(per_query_sum < threshold)[0]
    low_idx, _ = torch.sort(low_idx)
    print(f"Layer {layer_num}: selected {low_idx.numel()} rows with attention sum < {threshold} out of {query_len} total queries.")
    # torch.set_printoptions(profile="full")
    # print(f"Indices with low attention sum (<{threshold}), num: {low_idx.numel()}: \n{low_idx}")
    # torch.set_printoptions(profile="default")

    fname = f"row_selection_indices_layer{layer_num}.pt"
    out_path = os.path.join(output_dir, fname)
    torch.save({"row_selection_indices": low_idx}, out_path)

def col_attention_select(output_dir, topk, prefill_file: Path, chunks: List[Tuple[Path, int]]):
    """Select token to recompute based on column attention.
    Args:
        output_dir: directory to save output index files.
        prefill_file: optional Path to a .pt file containing full prefill
            q/k tensors and metadata.
        chunks: optional list of (Path, chunk_length) tuples representing
            loaded chunk files for a specific layer.
    
    Returns:
        None
    
    Note: This function saves all layers' selection indices in `output_dir`
    """
    if prefill_file is None:
        print("No prefill_file provided; skipping row attention selection.")
        return
    # get layer number from prefill_file name
    m = re.search(r"layer[_-]?(\d+)", prefill_file.name, flags=re.IGNORECASE)
    layer_num = -1
    if m:
        try:
            layer_num = int(m.group(1))
        except Exception:
            pass
    if layer_num < 0:
        print(f"Could not determine layer number from prefill_file name: {prefill_file}")
        return
    print(f"Processing layer {layer_num} from prefill file: {prefill_file}")
    prefill_data = torch.load(prefill_file)
    print(f"Loaded prefill file: {prefill_file}")
    q = prefill_data.get("q", None).to(torch.float32)
    k = prefill_data.get("k", None).to(torch.float32)
    meta = prefill_data.get("meta", {})
    if q is None or k is None:
        print(f"Prefill file {prefill_file} missing 'q' or 'k' tensors; skipping.")
        return
    print(f"q shape: {q.shape}, k shape: {k.shape}")
    if not isinstance(meta, dict):
        print(f"Prefill file {prefill_file} has invalid 'meta' field; expected dict, got {type(meta)}")
        return
    
    # Try to use metadata-driven reshape if possible
    h_dim = meta.get("h_dim")
    num_tokens = meta.get("num_tokens")
    num_kv_heads = meta.get("num_kv_heads")
    num_queries_per_kv = meta.get("num_queries_per_kv")
    scaling = meta.get("scaling")
    print(f"Metadata: h_dim={h_dim}, num_tokens={num_tokens}, num_kv_heads={num_kv_heads}, num_queries_per_kv={num_queries_per_kv}, scaling={scaling}")

    attention_scores = None

    if None not in (h_dim, num_kv_heads, num_queries_per_kv) and q.ndim >= 2 and k.ndim >= 2:
        # attempt the same reshaping logic as older code paths
        try:
            q_matrix = q.view(q.shape[0], num_kv_heads, num_queries_per_kv, h_dim).transpose(0, 2)
            k_matrix = k[:, :, None, :].expand(k.shape[0], num_kv_heads, num_queries_per_kv, h_dim).transpose(0, 2)

            # print(f"q_matrix shape after reshape: {q_matrix.shape}, k_matrix shape after reshape: {k_matrix.shape}")

            attn_weights = torch.matmul(q_matrix, k_matrix.transpose(2, 3)) / scaling
            # print(f"attn_weights shape: {attn_weights.shape}\nattn_weights: {attn_weights[0, 0, :, :]}")

            query_len, key_len = attn_weights.shape[-2], attn_weights.shape[-1]
            # causal lower-triangular mask as in original code
            mask = torch.tril(torch.ones(query_len, key_len, device=attn_weights.device))
            mask = mask.masked_fill(mask == 0, float('-inf'))
            attn_weights = attn_weights + mask
            # print(f"after mask, attn_weights shape: {attn_weights.shape}\nattn_weights: {attn_weights[0, 0, :, :]}")

            attention_scores = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            # print(f"dtype after softmax: {attention_scores.dtype}")
            # print(f"after softmax, attn_weights shape: {attention_scores.shape}\nattn_weights: {attention_scores[0, 0, :, :]}")
        except Exception as e:
            print(f"Metadata-based reshape failed: {e}")

    if attention_scores is None:
        print("Unable to compute attention: q/k tensors have unsupported shapes and metadata was insufficient.")
        return

    # !!! use new algorithm to compute row attention selection !!!
    # sum all head and group dims to get (Q, K) attention score matrix
    attn_score = torch.sum(attention_scores, dim=[0, 1])
    query_len = attn_score.shape[-2]
    key_len = attn_score.shape[-1]
    print(f"shape of attn_score: {attn_score.shape}\nattn_score: {attn_score}")
    per_query_sum = attn_score.sum(dim=-2)

    # Ensure per_query_sum is 1-D (length = number of queries)
    if per_query_sum.dim() != 1:
        per_query_sum = per_query_sum.view(-1)

    # !!! use topk selection based on attention mass !!!
    k = max(1, int(topk * query_len))
    top_values, top_indices = torch.topk(per_query_sum, k, largest=True)
    topk_sum = top_values.sum()
    print(f"Layer {layer_num}: Sum of top-{topk} per_query_sum: {topk_sum.item()}")
    print(f"Layer {layer_num}: selected top {k} based on attention mass out of {query_len} total queries.")
    torch.set_printoptions(profile="full")
    print(f"Top-{topk} Indices and per_query_sum (desc):")
    for idx, val in zip(top_indices.tolist(), top_values.tolist()):
        print(f"{int(idx)}: {float(val):.6f}")
    torch.set_printoptions(profile="default")

    fname = f"row_selection_indices_layer{layer_num}.pt"
    out_path = os.path.join(output_dir, fname)
    torch.save({"row_selection_indices": top_indices}, out_path)


def row_attention_select_no_mask(output_dir, topk, prefill_file: Path, chunks: List[Tuple[Path, int]]):
    """Select token to recompute based on column attention.
    Args:
        output_dir: directory to save output index files.
        prefill_file: optional Path to a .pt file containing full prefill
            q/k tensors and metadata.
        chunks: optional list of (Path, chunk_length) tuples representing
            loaded chunk files for a specific layer.
    
    Returns:
        None
    
    Note: This function saves all layers' selection indices in `output_dir`
    """
    if prefill_file is None:
        print("No prefill_file provided; skipping row attention selection.")
        return
    # get layer number from prefill_file name
    m = re.search(r"layer[_-]?(\d+)", prefill_file.name, flags=re.IGNORECASE)
    layer_num = -1
    if m:
        try:
            layer_num = int(m.group(1))
        except Exception:
            pass
    if layer_num < 0:
        print(f"Could not determine layer number from prefill_file name: {prefill_file}")
        return
    print(f"Processing layer {layer_num} from prefill file: {prefill_file}")
    prefill_data = torch.load(prefill_file)
    print(f"Loaded prefill file: {prefill_file}")
    q = prefill_data.get("q", None).to(torch.float32)
    k = prefill_data.get("k", None).to(torch.float32)
    meta = prefill_data.get("meta", {})
    if q is None or k is None:
        print(f"Prefill file {prefill_file} missing 'q' or 'k' tensors; skipping.")
        return
    print(f"q shape: {q.shape}, k shape: {k.shape}")
    if not isinstance(meta, dict):
        print(f"Prefill file {prefill_file} has invalid 'meta' field; expected dict, got {type(meta)}")
        return
    
    # Try to use metadata-driven reshape if possible
    h_dim = meta.get("h_dim")
    num_tokens = meta.get("num_tokens")
    num_kv_heads = meta.get("num_kv_heads")
    num_queries_per_kv = meta.get("num_queries_per_kv")
    scaling = meta.get("scaling")
    print(f"Metadata: h_dim={h_dim}, num_tokens={num_tokens}, num_kv_heads={num_kv_heads}, num_queries_per_kv={num_queries_per_kv}, scaling={scaling}")

    attention_scores = None

    if None not in (h_dim, num_kv_heads, num_queries_per_kv) and q.ndim >= 2 and k.ndim >= 2:
        # attempt the same reshaping logic as older code paths
        try:
            q_matrix = q.view(q.shape[0], num_kv_heads, num_queries_per_kv, h_dim).transpose(0, 2)
            k_matrix = k[:, :, None, :].expand(k.shape[0], num_kv_heads, num_queries_per_kv, h_dim).transpose(0, 2)

            # print(f"q_matrix shape after reshape: {q_matrix.shape}, k_matrix shape after reshape: {k_matrix.shape}")

            attn_weights = torch.matmul(q_matrix, k_matrix.transpose(2, 3)) / scaling
            # print(f"attn_weights shape: {attn_weights.shape}\nattn_weights: {attn_weights[0, 0, :, :]}")

            query_len, key_len = attn_weights.shape[-2], attn_weights.shape[-1]
            # causal lower-triangular mask as in original code
            mask = torch.tril(torch.ones(query_len, key_len, device=attn_weights.device))
            mask = mask.masked_fill(mask == 0, float('-inf'))
            attn_weights = attn_weights + mask
            # print(f"after mask, attn_weights shape: {attn_weights.shape}\nattn_weights: {attn_weights[0, 0, :, :]}")

            attention_scores = torch.nn.functional.softmax(attn_weights, dim=-2, dtype=torch.float32)
            # print(f"dtype after softmax: {attention_scores.dtype}")
            # print(f"after softmax, attn_weights shape: {attention_scores.shape}\nattn_weights: {attention_scores[0, 0, :, :]}")
        except Exception as e:
            print(f"Metadata-based reshape failed: {e}")

    if attention_scores is None:
        print("Unable to compute attention: q/k tensors have unsupported shapes and metadata was insufficient.")
        return

    # !!! use new algorithm to compute row attention selection !!!
    # sum all head and group dims to get (Q, K) attention score matrix
    attn_score = torch.sum(attention_scores, dim=[0, 1])
    query_len = attn_score.shape[-2]
    key_len = attn_score.shape[-1]
    print(f"shape of attn_score: {attn_score.shape}\nattn_score: {attn_score}")
    per_query_sum = attn_score.sum(dim=-1)

    # Ensure per_query_sum is 1-D (length = number of queries)
    if per_query_sum.dim() != 1:
        per_query_sum = per_query_sum.view(-1)

    # !!! use topk selection based on attention mass !!!
    k = max(1, int(topk * query_len))
    top_values, top_indices = torch.topk(per_query_sum, k, largest=True)
    topk_sum = top_values.sum()
    print(f"Layer {layer_num}: Sum of top-{topk} per_query_sum: {topk_sum.item()}")
    print(f"Layer {layer_num}: selected top {k} based on attention mass out of {query_len} total queries.")
    torch.set_printoptions(profile="full")
    print(f"Top-{topk} Indices and per_query_sum (desc):")
    for idx, val in zip(top_indices.tolist(), top_values.tolist()):
        print(f"{int(idx)}: {float(val):.6f}")
    torch.set_printoptions(profile="default")

    fname = f"row_selection_indices_layer{layer_num}.pt"
    out_path = os.path.join(output_dir, fname)
    torch.save({"row_selection_indices": top_indices}, out_path)


def row_attention_select_new(output_dir, topk, prefill_file: Path, chunks: List[Tuple[Path, int]]):
    """Select token to recompute based on row attention.
    Args:
        output_dir: directory to save output index files.
        prefill_file: optional Path to a .pt file containing full prefill
            q/k tensors and metadata.
        chunks: optional list of (Path, chunk_length) tuples representing
            loaded chunk files for a specific layer.
    
    Returns:
        None
    
    Note: This function saves all layers' selection indices in `output_dir`
    """
    if prefill_file is None:
        print("No prefill_file provided; skipping row attention selection.")
        return
    # get layer number from prefill_file name
    m = re.search(r"layer[_-]?(\d+)", prefill_file.name, flags=re.IGNORECASE)
    layer_num = -1
    if m:
        try:
            layer_num = int(m.group(1))
        except Exception:
            pass
    if layer_num < 0:
        print(f"Could not determine layer number from prefill_file name: {prefill_file}")
        return
    print(f"Processing layer {layer_num} from prefill file: {prefill_file}")
    prefill_data = torch.load(prefill_file)
    print(f"Loaded prefill file: {prefill_file}")
    q = prefill_data.get("q", None).to(torch.float32)
    k = prefill_data.get("k", None).to(torch.float32)
    meta = prefill_data.get("meta", {})
    if q is None or k is None:
        print(f"Prefill file {prefill_file} missing 'q' or 'k' tensors; skipping.")
        return
    print(f"q shape: {q.shape}, k shape: {k.shape}")
    if not isinstance(meta, dict):
        print(f"Prefill file {prefill_file} has invalid 'meta' field; expected dict, got {type(meta)}")
        return
    
    # Try to use metadata-driven reshape if possible
    h_dim = meta.get("h_dim")
    num_tokens = meta.get("num_tokens")
    num_kv_heads = meta.get("num_kv_heads")
    num_queries_per_kv = meta.get("num_queries_per_kv")
    scaling = meta.get("scaling")
    print(f"Metadata: h_dim={h_dim}, num_tokens={num_tokens}, num_kv_heads={num_kv_heads}, num_queries_per_kv={num_queries_per_kv}, scaling={scaling}")

    attention_scores = None

    if None not in (h_dim, num_kv_heads, num_queries_per_kv) and q.ndim >= 2 and k.ndim >= 2:
        # attempt the same reshaping logic as older code paths
        try:
            q_matrix = q.view(q.shape[0], num_kv_heads, num_queries_per_kv, h_dim).transpose(0, 2)
            k_matrix = k[:, :, None, :].expand(k.shape[0], num_kv_heads, num_queries_per_kv, h_dim).transpose(0, 2)

            # print(f"q_matrix shape after reshape: {q_matrix.shape}, k_matrix shape after reshape: {k_matrix.shape}")

            attn_weights = torch.matmul(q_matrix, k_matrix.transpose(2, 3)) / scaling
            # print(f"attn_weights shape: {attn_weights.shape}\nattn_weights: {attn_weights[0, 0, :, :]}")

            query_len, key_len = attn_weights.shape[-2], attn_weights.shape[-1]
            # causal lower-triangular mask as in original code
            mask = torch.tril(torch.ones(query_len, key_len, device=attn_weights.device))
            mask = mask.masked_fill(mask == 0, float('-inf'))
            attn_weights = attn_weights + mask
            # print(f"after mask, attn_weights shape: {attn_weights.shape}\nattn_weights: {attn_weights[0, 0, :, :]}")

            attention_scores = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            # print(f"dtype after softmax: {attention_scores.dtype}")
            # print(f"after softmax, attn_weights shape: {attention_scores.shape}\nattn_weights: {attention_scores[0, 0, :, :]}")
        except Exception as e:
            print(f"Metadata-based reshape failed: {e}")

    if attention_scores is None:
        print("Unable to compute attention: q/k tensors have unsupported shapes and metadata was insufficient.")
        return

    # !!! use new algorithm to compute row attention selection !!!
    # sum all head and group dims to get (Q, K) attention score matrix
    attn_score = torch.sum(attention_scores, dim=[0, 1])
    query_len = attn_score.shape[-2]
    key_len = attn_score.shape[-1]
    print(f"shape of attn_score: {attn_score.shape}\nattn_score: {attn_score}")
    # generate the per-chunk mask: for query tokens belonging to chunk i,
    # allow attention only to key tokens from chunk i and later chunks.
    # Build chunk lengths from per-chunk metadata when available, otherwise
    # infer from per-chunk 'q' tensors when possible.
    chunk_lens: List[int] = []
    for _, len in chunks:
        chunk_lens.append(len)

    total_chunk_tokens = sum(chunk_lens)

    if total_chunk_tokens == 0:
        # If we couldn't determine chunk sizes, try to fallback to using
        # the full sequence and skip per-chunk masking.
        print("Warning: could not determine per-chunk token counts; skipping per-chunk mask generation")
        attention_scores_copy = attn_score.clone()
    else:
        # !!! we need to mask the reuse parts, and use the cross part to cal topk !!!
        mask2d = torch.ones((query_len, key_len), dtype=attn_score.dtype, device=attn_score.device)

        # compute chunk start offsets
        starts = []
        s = 0
        for l in chunk_lens:
            starts.append(s)
            s += l
        # print(f"Chunk lengths: {chunk_lens}, starts: {starts}")

        # apply per-chunk rules: for chunk i, queries in [starts[i], starts[i]+l)
        # can attend to keys in [starts[i], end)
        pos = 0
        for i, l in enumerate(chunk_lens):
            if l <= 0:
                continue
            q_start = pos
            q_end = min(pos + l, query_len)
            k_start = starts[i]
            if q_start >= q_end:
                pos += l
                continue
            if k_start >= key_len:
                # nothing to allow for this chunk
                pos += l
                continue
            # !!! we need to mask the reuse parts, and use the cross part to cal topk !!!
            mask2d[q_start:q_end, k_start:key_len] = 0.0
            pos += l

        # If chunks cover fewer tokens than the attention matrix, allow the
        # remaining queries to attend to remaining keys (conservative)
        if pos < query_len:
            mask2d[pos:query_len, min(pos, key_len):key_len] = 1.0

        print(f"Mask broadcast shape: {mask2d.shape}")
        # print(f"attn_score: {attn_score}\nmask: {mask2d}")
        attention_scores_copy = attn_score * mask2d

    # Aggregate attention mass per query index across all leading dims
    # attention_scores_copy shape may be (Q, K) or (T, H, Q, K) etc.; query dim is -2
    per_query_sum = attention_scores_copy.sum(dim=-1)

    # Ensure per_query_sum is 1-D (length = number of queries)
    if per_query_sum.dim() != 1:
        per_query_sum = per_query_sum.view(-1)

    # !!! use topk selection based on attention mass !!!
    k = max(1, int(topk * query_len))
    top_values, top_indices = torch.topk(per_query_sum, k, largest=True)
    topk_sum = top_values.sum()
    print(f"Layer {layer_num}: Sum of top-{topk} per_query_sum: {topk_sum.item()}")
    print(f"Layer {layer_num}: selected top {k} based on attention mass out of {query_len} total queries.")
    torch.set_printoptions(profile="full")
    print(f"Top-{topk} Indices and per_query_sum (desc):")
    for idx, val in zip(top_indices.tolist(), top_values.tolist()):
        print(f"{int(idx)}: {float(val):.6f}")
    torch.set_printoptions(profile="default")

    fname = f"row_selection_indices_layer{layer_num}.pt"
    out_path = os.path.join(output_dir, fname)
    torch.save({"row_selection_indices": top_indices}, out_path)
    return per_query_sum

def row_attention_select_filter(output_dir, topk, prefill_file: Path, chunks: List[Tuple[Path, int]]):
    """Select token to recompute based on row attention.
    Args:
        output_dir: directory to save output index files.
        prefill_file: optional Path to a .pt file containing full prefill
            q/k tensors and metadata.
        chunks: optional list of (Path, chunk_length) tuples representing
            loaded chunk files for a specific layer.
    
    Returns:
        None
    
    Note: This function saves all layers' selection indices in `output_dir`
    """
    if prefill_file is None:
        print("No prefill_file provided; skipping row attention selection.")
        return
    # get layer number from prefill_file name
    m = re.search(r"layer[_-]?(\d+)", prefill_file.name, flags=re.IGNORECASE)
    layer_num = -1
    if m:
        try:
            layer_num = int(m.group(1))
        except Exception:
            pass
    if layer_num < 0:
        print(f"Could not determine layer number from prefill_file name: {prefill_file}")
        return
    print(f"Processing layer {layer_num} from prefill file: {prefill_file}")
    prefill_data = torch.load(prefill_file)
    print(f"Loaded prefill file: {prefill_file}")
    q = prefill_data.get("q", None).to(torch.float32)
    k = prefill_data.get("k", None).to(torch.float32)
    meta = prefill_data.get("meta", {})
    if q is None or k is None:
        print(f"Prefill file {prefill_file} missing 'q' or 'k' tensors; skipping.")
        return
    print(f"q shape: {q.shape}, k shape: {k.shape}")
    if not isinstance(meta, dict):
        print(f"Prefill file {prefill_file} has invalid 'meta' field; expected dict, got {type(meta)}")
        return
    
    # Try to use metadata-driven reshape if possible
    h_dim = meta.get("h_dim")
    num_tokens = meta.get("num_tokens")
    num_kv_heads = meta.get("num_kv_heads")
    num_queries_per_kv = meta.get("num_queries_per_kv")
    scaling = meta.get("scaling")
    print(f"Metadata: h_dim={h_dim}, num_tokens={num_tokens}, num_kv_heads={num_kv_heads}, num_queries_per_kv={num_queries_per_kv}, scaling={scaling}")

    attention_scores = None

    if None not in (h_dim, num_kv_heads, num_queries_per_kv) and q.ndim >= 2 and k.ndim >= 2:
        # attempt the same reshaping logic as older code paths
        try:
            q_matrix = q.view(q.shape[0], num_kv_heads, num_queries_per_kv, h_dim).transpose(0, 2)
            k_matrix = k[:, :, None, :].expand(k.shape[0], num_kv_heads, num_queries_per_kv, h_dim).transpose(0, 2)

            # print(f"q_matrix shape after reshape: {q_matrix.shape}, k_matrix shape after reshape: {k_matrix.shape}")

            attn_weights = torch.matmul(q_matrix, k_matrix.transpose(2, 3)) / scaling
            # print(f"attn_weights shape: {attn_weights.shape}\nattn_weights: {attn_weights[0, 0, :, :]}")

            query_len, key_len = attn_weights.shape[-2], attn_weights.shape[-1]
            # causal lower-triangular mask as in original code
            mask = torch.tril(torch.ones(query_len, key_len, device=attn_weights.device))
            mask = mask.masked_fill(mask == 0, float('-inf'))
            attn_weights = attn_weights + mask
            # print(f"after mask, attn_weights shape: {attn_weights.shape}\nattn_weights: {attn_weights[0, 0, :, :]}")

            attention_scores = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            # print(f"dtype after softmax: {attention_scores.dtype}")
            # print(f"after softmax, attn_weights shape: {attention_scores.shape}\nattn_weights: {attention_scores[0, 0, :, :]}")
        except Exception as e:
            print(f"Metadata-based reshape failed: {e}")

    if attention_scores is None:
        print("Unable to compute attention: q/k tensors have unsupported shapes and metadata was insufficient.")
        return

    # !!! use new algorithm to compute row attention selection !!!
    # sum all head and group dims to get (Q, K) attention score matrix
    print(f"shape of attention_scores before sum: {attention_scores.shape}")
    attn_score = torch.sum(attention_scores, dim=[0, 1])
    query_len = attn_score.shape[-2]
    key_len = attn_score.shape[-1]
    print(f"shape of attn_score: {attn_score.shape}\nattn_score: {attn_score}")
    # generate the per-chunk mask: for query tokens belonging to chunk i,
    # allow attention only to key tokens from chunk i and later chunks.
    # Build chunk lengths from per-chunk metadata when available, otherwise
    # infer from per-chunk 'q' tensors when possible.
    chunk_lens: List[int] = []
    for _, len in chunks:
        chunk_lens.append(len)

    total_chunk_tokens = sum(chunk_lens)

    if total_chunk_tokens == 0:
        # If we couldn't determine chunk sizes, try to fallback to using
        # the full sequence and skip per-chunk masking.
        print("Warning: could not determine per-chunk token counts; skipping per-chunk mask generation")
        attention_scores_copy = attn_score.clone()
    else:
        # !!! we need to mask the reuse parts, and use the cross part to cal topk !!!
        mask2d = torch.ones((query_len, key_len), dtype=attn_score.dtype, device=attn_score.device)

        # compute chunk start offsets
        starts = []
        s = 0
        for l in chunk_lens:
            starts.append(s)
            s += l
        # print(f"Chunk lengths: {chunk_lens}, starts: {starts}")

        # apply per-chunk rules: for chunk i, queries in [starts[i], starts[i]+l)
        # can attend to keys in [starts[i], end)
        pos = 0
        for i, l in enumerate(chunk_lens):
            if l <= 0:
                continue
            q_start = pos
            q_end = min(pos + l, query_len)
            k_start = starts[i]
            if q_start >= q_end:
                pos += l
                continue
            if k_start >= key_len:
                # nothing to allow for this chunk
                pos += l
                continue
            # !!! we need to mask the reuse parts, and use the cross part to cal topk !!!
            mask2d[q_start:q_end, k_start:key_len] = 0.0
            pos += l

        # If chunks cover fewer tokens than the attention matrix, allow the
        # remaining queries to attend to remaining keys (conservative)
        if pos < query_len:
            mask2d[pos:query_len, min(pos, key_len):key_len] = 1.0

        print(f"Mask broadcast shape: {mask2d.shape}")
        attention_scores_copy = attn_score * mask2d
        # torch.set_printoptions(profile="full")
        # print(f"attn_score: {attention_scores_copy[2014,:]}\nmask: {mask2d[2014, :]}")
        # torch.set_printoptions(profile="default")

    # Aggregate attention mass per query index across all leading dims
    # attention_scores_copy shape may be (Q, K) or (T, H, Q, K) etc.; query dim is -2
    per_query_sum = attention_scores_copy.sum(dim=-1)
    per_key_sum = attention_scores_copy.sum(dim=-2)
    per_token_sum = per_query_sum + per_key_sum

    # # Ensure per_query_sum is 1-D (length = number of queries)
    # if per_query_sum.dim() != 1:
    #     per_query_sum = per_query_sum.view(-1)

    # !!! use topk selection based on attention mass !!!
    k = max(1, int(topk * query_len))
    top_values, top_indices = torch.topk(per_token_sum, k, largest=True)
    # threshold = num_kv_heads * num_queries_per_kv * 0.8
    # mask = per_token_sum[top_indices] >= threshold
    # top_indices = top_indices[mask]
    # top_values = top_values[mask]
    topk_sum = top_values.sum()
    torch.set_printoptions(profile="full")
    print(f"Layer {layer_num}: Sum of top-{topk} per_query_sum: {topk_sum.item()}")
    print(f"Layer {layer_num}: selected top {top_indices.shape[0]} based on attention mass out of {query_len} total queries.")
    print(f"Top-{topk} Indices and per_query_sum (desc):")
    for idx, val in zip(top_indices.tolist(), top_values.tolist()):
        print(f"{int(idx)}: {float(val):.6f}")
    torch.set_printoptions(profile="default")

    fname = f"row_selection_indices_layer{layer_num}.pt"
    out_path = os.path.join(output_dir, fname)
    torch.save({"row_selection_indices": top_indices}, out_path)

def main():
    parser = argparse.ArgumentParser(description="Read, process, and visualize .pt files containing attention data.")
    parser.add_argument("--data", type=str, required=True, help="Directory containing .pt files.")
    parser.add_argument("--out", type=str, required=True, help="Output directory for indices saving.")
    parser.add_argument("--topk", type=float, default=0.4, help="Fraction of top attention rows to select.")
    args = parser.parse_args()

    directory = args.data
    output_dir = args.out
    topk = args.topk

    if not os.path.exists(directory):
        print(f"Input directory does not exist: {directory}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # select the .pt files start with 'prefill'
    pt_files = [f for f in os.listdir(directory) if f.startswith("prefill") and f.endswith(".pt")]
    if not pt_files:
        print(f"No .pt files found in directory: {directory}")
        return

    print(f"Found {len(pt_files)} .pt files in directory: {directory}\n")

    # Use the chunk loader to collect and order chunk files for this layer.
    chunks = load_chunk_files_for_layer(directory, 0)
    chunks = extract_chunk_len_from_chunks(chunks)
    print(f"Found {len(chunks)} chunk files for layer 0 in {directory}:\n{chunks}")
    per_query_sum_per_total = None
    for pt_file in pt_files:
        prefill_path = Path(directory) / pt_file
        print(f"\nProcessing prefill file: {prefill_path}")
        # row_attention_select(output_dir, prefill_path, chunks)
        # col_attention_select(output_dir, topk, prefill_path, chunks)
        # row_attention_select_no_mask(output_dir, topk, prefill_path, chunks)
        per_query_sum_per_layer = row_attention_select_new(output_dir, topk, prefill_path, chunks)
        if per_query_sum_per_total is None:
            per_query_sum_per_total = per_query_sum_per_layer
        else:
            per_query_sum_per_total += per_query_sum_per_layer
        print(f"per_query_sum_per_layer shape: {per_query_sum_per_layer.shape}")
        # row_attention_select_filter(output_dir, topk, prefill_path, chunks)
    
    # total_values, total_indices = torch.sort(per_query_sum_per_total, descending=True)
    # Print total
    # print(f"total_indices: {total_indices}", f"total_values: {total_values}")
    # INSERT_YOUR_CODE
    import matplotlib.pyplot as plt
    per_query_sum_per_total = per_query_sum_per_total / 48

    # Calculate document end indices from chunks
    chunk_lens = [length for _, length in chunks]
    doc_end_indices = []
    cumulative = 0
    for length in chunk_lens:
        if length > 0:
            cumulative += length
            doc_end_indices.append(cumulative - 1)  # -1 because end index is inclusive
    
    # Truncate doc_end_indices to valid range
    valid_length = len(per_query_sum_per_total)
    doc_end_indices = [idx for idx in doc_end_indices if idx < valid_length]
    print(f"doc_end_indices (truncated to valid range < {valid_length}): {doc_end_indices}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(per_query_sum_per_total.cpu().numpy(), marker='.', linestyle='None', markersize=2)
    
    # Add vertical lines at document end indices
    for doc_end_idx in doc_end_indices:
        plt.axvline(x=doc_end_idx, color='r', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.title("per_query_sum_per_total vs. Index")
    plt.xlabel("Index")
    plt.ylabel("per_query_sum_per_total")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("per_query_sum_per_total_plot.png")

    # load all row selection indices files and calculate the selection similarity
    print("\nLoading row selection indices files for similarity computation...")
    selection_files = [f for f in os.listdir(output_dir) if f.startswith("row_selection_indices_layer") and f.endswith(".pt")]
    print(f"Found {len(selection_files)} selection index files in {output_dir}:\n{selection_files}")
    selection_indices = {}
    for sel_file in selection_files:
        layer_match = re.search(r"layer[_-]?(\d+)", sel_file, flags=re.IGNORECASE)
        if not layer_match:
            continue
        layer_num = int(layer_match.group(1))
        sel_path = Path(output_dir) / sel_file
        print(f"Loading selection indices for layer {layer_num} from {sel_path}")
        data = torch.load(sel_path)
        indices = data.get("row_selection_indices", None)
        if indices is None:
            print(f"failed to load {sel_path}: 'row_selection_indices' key not found.")
            return
        selection_indices[layer_num] = set(indices.tolist())
    # compute pairwise Jaccard similarity between layers
    layer_nums = sorted(selection_indices.keys())
    print(f"\nPairwise Jaccard similarity between layers{layer_nums} selection indices:")
    # num_total = len(selection_indices[layer_nums[0]])
    # intersection_all = selection_indices[layer_nums[0]].copy()
    for i in range(len(layer_nums)-1):
        layer_i = layer_nums[i]
        layer_j = layer_nums[i+1]
        set_i = selection_indices[layer_i]
        set_j = selection_indices[layer_j]
        # intersection_all = intersection_all.intersection(set_j)
        intersection = len(set_i.intersection(set_j))
        # union = len(set_i.union(set_j))
        jaccard_sim = intersection / len(set_i)
        print(f"Layers {layer_i} len {len(set_i)} & {layer_j} len {len(set_j)}: similarity = {jaccard_sim:.4f}")
    # print(f"Overall intersection size across all layers: {len(intersection_all)} out of {num_total} total tokens. similarity = {len(intersection_all)/num_total:.4f}")


if __name__ == "__main__":
    main()