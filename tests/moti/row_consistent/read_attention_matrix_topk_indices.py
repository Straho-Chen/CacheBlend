import torch
import os
import argparse
from pathlib import Path
import re
from typing import List, Optional, Tuple, Union
from vllm.model_executor.layers.rotary_embedding import get_rope

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


def extract_qk_from_chunks(chunks: List[Tuple[Path, dict]]) -> List[Tuple[Path, torch.Tensor, torch.Tensor, dict]]:
    """Extract q and k tensors and metadata from loaded chunk list.

    Args:
        chunks: list of (Path, data_dict) as returned by
            ``load_chunk_files_for_layer``.

    Returns:
        A list of tuples (Path, q_tensor, k_tensor, meta_dict). Entries that
        do not contain both 'q' and 'k' are skipped with a printed warning.

    Notes:
        - Returned q/k tensors are moved to CPU and converted to float32 to
          make downstream processing consistent and avoid device mismatches.
        - meta_dict is ``data.get('meta', {})`` and may be empty.
    """
    out: List[Tuple[Path, torch.Tensor, torch.Tensor, dict]] = []
    for p, data in chunks:
        if not isinstance(data, dict):
            print(f"Skipping {p}: loaded object is not a dict (type={type(data)})")
            continue

        q = data.get("q")
        k = data.get("k")
        meta = data.get("meta", {}) if isinstance(data.get("meta", {}), dict) else {}

        if q is None or k is None:
            print(f"Skipping {p}: missing 'q' or 'k' in data")
            continue

        try:
            # ensure torch tensors, float32 and on CPU for consistency
            if not isinstance(q, torch.Tensor):
                q = torch.as_tensor(q)
            if not isinstance(k, torch.Tensor):
                k = torch.as_tensor(k)
            q_t = q.to(torch.float32).cpu()
            k_t = k.to(torch.float32).cpu()
        except Exception as e:
            print(f"Failed to convert q/k for {p}: {e}")
            continue

        out.append((p, q_t, k_t, meta))

    return out


def concat_qk_from_chunks(chunks: List[Tuple[Path, dict]], dim: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate q and k tensors from a list of loaded chunk files.

    This function uses ``extract_qk_from_chunks`` to obtain (Path, q, k, meta)
    entries, preserves their order, and concatenates all q tensors into
    ``q_all`` and all k tensors into ``k_all`` along the given ``dim``.

    Args:
        chunks: list returned by ``load_chunk_files_for_layer`` (or similar).
        dim: dimension along which to concatenate (default 0).

    Returns:
        (q_all, k_all) — concatenated tensors. If no valid q/k tensors are
        found, returns two empty tensors with shape (0,).
    """
    entries = extract_qk_from_chunks(chunks)
    if not entries:
        # return empty tensors to avoid None handling downstream
        return torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32)

    q_list = []
    k_list = []
    for p, q, k, meta in entries:
        # ensure q/k are tensors on CPU float32 (extract already does this, but be defensive)
        if not isinstance(q, torch.Tensor):
            q = torch.as_tensor(q, dtype=torch.float32)
        else:
            q = q.to(torch.float32).cpu()
        if not isinstance(k, torch.Tensor):
            k = torch.as_tensor(k, dtype=torch.float32)
        else:
            k = k.to(torch.float32).cpu()

        q_list.append(q)
        k_list.append(k)

    try:
        q_all = torch.cat(q_list, dim=dim) if len(q_list) > 1 else q_list[0]
        k_all = torch.cat(k_list, dim=dim) if len(k_list) > 1 else k_list[0]
    except Exception as e:
        # fall back to stacking along a new 0-th dimension if concat fails
        print(f"concat failed with error {e}, attempting fallback stack along dim=0")
        q_all = torch.stack(q_list, dim=0)
        k_all = torch.stack(k_list, dim=0)

    return q_all, k_all

def visualize_attention_from_tensors(output_dir, topk=None, chunks: Optional[List[Tuple[Path, dict]]] = None):
    """Visualize attention using either a list of loaded chunk files or
    directly supplied q/k tensors.

    If ``chunks`` is provided the function will call ``concat_qk_from_chunks``
    to produce ``q_all`` and ``k_all`` and will build a new ``meta`` dict
    where ``num_tokens`` reflects the total tokens across chunks. If per-chunk
    metadata contains numeric ``num_tokens`` fields, those will be summed to
    compute the total; otherwise the function will infer ``num_tokens`` from
    the concatenated `q_all` tensor shape.
    """
    try:
        data = None
        # If chunks are provided, build q_all/k_all and meta here
        if chunks is not None:
            q_all_local, k_all_local = concat_qk_from_chunks(chunks)
            if q_all_local.numel() == 0 or k_all_local.numel() == 0:
                print("No valid q/k tensors found in chunks, skipping visualization.")
                return

            q_matrix = q_all_local.to(torch.float32)
            k_matrix = k_all_local.to(torch.float32)

            # Build new_meta: copy meta from first chunk and update num_tokens
            meta0 = chunks[0][1].get("meta", {}) if chunks and isinstance(chunks[0][1], dict) else {}
            data_meta = dict(meta0) if isinstance(meta0, dict) else {}

            # Prefer summing per-chunk 'num_tokens' if available
            total_tokens = 0
            found_token_counts = False
            for _, d in chunks:
                if not isinstance(d, dict):
                    continue
                m = d.get("meta", {}) if isinstance(d.get("meta", {}), dict) else {}
                nt = m.get("num_tokens")
                try:
                    if nt is not None:
                        total_tokens += int(nt)
                        found_token_counts = True
                except Exception:
                    continue

            if found_token_counts and total_tokens > 0:
                data_meta["num_tokens"] = total_tokens
            else:
                # Fallback: infer from q_all first dimension when possible
                try:
                    data_meta["num_tokens"] = int(q_matrix.shape[0]) if q_matrix.ndim >= 1 else None
                except Exception:
                    pass

            print(f"Using concatenated q/k from {len(chunks)} chunks. q shape={q_matrix.shape}, k shape={k_matrix.shape}")

        else:
            print("No input chunks or q_all/k_all supplied to visualize_attention_from_tensors; skipping.")
            return

        # Try to use metadata-driven reshape if possible
        h_dim = data_meta.get("h_dim")
        num_tokens = data_meta.get("num_tokens")
        num_kv_heads = data_meta.get("num_kv_heads")
        num_queries_per_kv = data_meta.get("num_queries_per_kv")
        scaling = data_meta.get("scaling")
        rotary_emb = get_rope(
            h_dim,
            rotary_dim=h_dim,
            max_position=4096*32,
            base=10000,
        )

        # BUG: need to apply rotary embedding to q_matrix here before computing attention


        attention_scores = None

        if None not in (h_dim, num_kv_heads, num_queries_per_kv) and q_matrix.ndim >= 2 and k_matrix.ndim >= 2:
            # attempt the same reshaping logic as older code paths
            try:
                q_matrix = q_matrix.view(q_matrix.shape[0], num_kv_heads, num_queries_per_kv, h_dim).transpose(0, 2)
                k_matrix = k_matrix[:, :, None, :].expand(k_matrix.shape[0], num_kv_heads, num_queries_per_kv, h_dim).transpose(0, 2)
                if scaling is None:
                    # h_dim is checked above to be not None, but guard for types
                    scaling = float(h_dim) ** 0.5 if h_dim is not None else 1.0

                print(f"q_matrix shape after reshape: {q_matrix.shape}, k_matrix shape after reshape: {k_matrix.shape}")

                attn_weights = torch.matmul(q_matrix, k_matrix.transpose(2, 3)) / scaling

                query_len, key_len = attn_weights.shape[-2], attn_weights.shape[-1]
                # causal lower-triangular mask as in original code
                mask = torch.tril(torch.ones(query_len, key_len, device=attn_weights.device))
                mask = mask.masked_fill(mask == 0, float('-inf'))
                attn_weights = attn_weights + mask

                attention_scores = torch.nn.functional.softmax(attn_weights, dim=-1)
            except Exception as e:
                print(f"Metadata-based reshape failed: {e}")

        if attention_scores is None:
            print("Unable to compute attention: q/k tensors have unsupported shapes and metadata was insufficient.")
            return

        print(f"Attention scores shape: {attention_scores.shape}")
        # select topk indices if specified
        if topk is not None:
            attn_score = torch.sum(attention_scores, dim=[0, 1, 2])
            print(f"attn_score shape: {attn_score.shape}")
            top_indices = torch.topk(attn_score, k=topk).indices
            top_indices, _ = torch.sort(top_indices)
            torch.set_printoptions(profile="full")
            print(f"Top-{topk} attention indices:\n{top_indices}")
            torch.set_printoptions(profile="default")
            fname = f"attention_top{topk}_indices.pt"
            out_path = os.path.join(output_dir, fname)
            torch.save({"topk": topk, "indices": top_indices}, out_path)
        

        attn_score = attention_scores[0, 0, :, :]
        print(f"shape of attn_score: {attn_score.shape}")
        # generate the per-chunk mask: for query tokens belonging to chunk i,
        # allow attention only to key tokens from chunk i and later chunks.
        # Build chunk lengths from per-chunk metadata when available, otherwise
        # infer from per-chunk 'q' tensors when possible.
        chunk_lens: List[int] = []
        for _, d in chunks:
            nt = None
            if isinstance(d, dict):
                m = d.get("meta", {}) if isinstance(d.get("meta", {}), dict) else {}
                nt = m.get("num_tokens")
                if nt is None:
                    qv = d.get("q", None)
                    if qv is not None:
                        try:
                            if isinstance(qv, torch.Tensor):
                                nt = int(qv.shape[0])
                            else:
                                qv_t = torch.as_tensor(qv)
                                nt = int(qv_t.shape[0])
                        except Exception:
                            nt = None
            try:
                chunk_lens.append(int(nt) if nt is not None else 0)
            except Exception:
                chunk_lens.append(0)

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
            print(f"Chunk lengths: {chunk_lens}, starts: {starts}")

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
            attention_scores_copy = attn_score * mask2d

        # Aggregate attention mass per query index across all leading dims
        # attention_scores_copy shape may be (Q, K) or (T, H, Q, K) etc.; query dim is -2
        nd = attention_scores_copy.dim()
        query_dim = nd - 2
        # reduce over all dims except the query dimension
        reduce_dims = tuple(i for i in range(nd) if i != query_dim)
        per_query_sum = attention_scores_copy.sum(dim=reduce_dims)

        # Ensure per_query_sum is 1-D (length = number of queries)
        if per_query_sum.dim() != 1:
            per_query_sum = per_query_sum.view(-1)

        # find query indices where total allowed attention mass < threshold
        threshold = 0.8
        low_idx = torch.where(per_query_sum < threshold)[0]
        low_idx, _ = torch.sort(low_idx)
        torch.set_printoptions(profile="full")
        print(f"Indices with low attention sum (<{threshold}), num: {low_idx.numel()}: \n{low_idx}")
        torch.set_printoptions(profile="default")

        fname = "recomp_indices.pt"
        out_path = os.path.join(output_dir, fname)
        torch.save({"low_attention_indices": low_idx}, out_path)

        common_elements = top_indices[torch.isin(top_indices, low_idx)]
        common_elements, _ = torch.sort(common_elements)
        print(f"\nElements present in both top-{topk} and low-attention indices ({len(common_elements)} items):\n{common_elements}")

    except Exception as e:
        print(f"Error computing attention: {e}")

def main():
    parser = argparse.ArgumentParser(description="Read, process, and visualize .pt files containing attention data.")
    parser.add_argument("--data", type=str, required=True, help="Directory containing .pt files.")
    parser.add_argument("--topk", type=int, required=True, help="Number of top attention indices.")
    parser.add_argument("--layer", type=int, required=False, help="Specific layer to process.")
    parser.add_argument("--out", type=str, required=True, help="Output directory for visualizations.")
    args = parser.parse_args()

    directory = args.data
    topk = args.topk
    layer = args.layer
    output_dir = args.out

    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pt_files = [f for f in os.listdir(directory) if f.endswith(".pt")]
    if not pt_files:
        print(f"No .pt files found in directory: {directory}")
        return

    print(f"Found {len(pt_files)} .pt files in directory: {directory}\n")

    if layer is not None:
        # Use the chunk loader to collect and order chunk files for this layer.
        chunks = load_chunk_files_for_layer(directory, layer)
        print(f"Found {len(chunks)} chunk files for layer {layer} in {directory}\n")
        if not chunks:
            print(f"No chunk files found for layer {layer}, exiting.")
            return
        visualize_attention_from_tensors(output_dir, topk=topk, chunks=chunks)
    else:
        print("Processing individual .pt files without chunk and layer filtering.\n")


if __name__ == "__main__":
    main()