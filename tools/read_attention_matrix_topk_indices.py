import torch
import os
import argparse

def load_and_visualize_attention_file(file_path, output_dir, topk=None):
    """Load and visualize the contents of a .pt file."""
    try:
        data = torch.load(file_path)
        print(f"Loaded file: {file_path}")
        print("Metadata:")
        for key, value in data.get("meta", {}).items():
            print(f"  {key}: {value}")

        # Extract and visualize matrices if available
        if "q" in data and "k" in data:
            q_matrix = data["q"].to(torch.float32)
            k_matrix = data["k"].to(torch.float32)

            print(f"\nmatrices from {file_path}...\n")

            # Extract metadata for reshaping
            h_dim = data["meta"].get("h_dim")
            num_tokens = data["meta"].get("num_tokens")
            num_kv_heads = data["meta"].get("num_kv_heads")
            num_queries_per_kv = data["meta"].get("num_queries_per_kv")
            scaling = data["meta"].get("scaling")

            if None in (h_dim, num_tokens, num_kv_heads, num_queries_per_kv):
                print("Missing metadata for reshaping, skipping attention computation.")
                return

            # Reshape q and k matrices based on metadata
            q_matrix = q_matrix.view(q_matrix.shape[0], num_kv_heads, num_queries_per_kv, h_dim).transpose(0, 2)
            k_matrix = k_matrix[:, :, None, :].expand(k_matrix.shape[0], num_kv_heads, num_queries_per_kv, h_dim).transpose(0, 2)

            print(f"q_matrix shape after reshape: {q_matrix.shape}, k_matrix shape after reshape: {k_matrix.shape}")

            # Compute attention scores per query head
            # attention_scores = torch.einsum("thqd, thkd -> thqk", q_matrix, k_matrix)
            attn_weights = torch.matmul(q_matrix, k_matrix.transpose(2, 3)) / scaling

            query_len, key_len = attn_weights.shape[-2], attn_weights.shape[-1]
            mask = torch.tril(torch.ones(query_len, key_len, device=attn_weights.device))
            mask = mask.masked_fill(mask == 0, float('-inf'))
            attn_weights = attn_weights + mask

            # Normalize the attention scores
            attention_scores = torch.nn.functional.softmax(attn_weights, dim=-1)

            print(f"Attention scores shape: {attention_scores.shape}")

            # select topk indices if specified
            if topk is not None:
                attn_score = torch.sum(attention_scores, dim=[0, 1, 2])
                top_indices = torch.topk(attn_score, k=topk).indices
                top_indices, _ = torch.sort(top_indices)
                print(f"Top-{topk} attention indices:\n{top_indices}")
                fname = f"normal_imp_indices.pt"
                out_path = os.path.join(output_dir, fname)
                torch.save({
                    "topk_num": topk,
                    "imp_indices": top_indices,
                }, out_path)

    except Exception as e:
        print(f"Error loading file {file_path}: {e}")


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
        pt_files = [f for f in pt_files if f"layer{layer}_" in f]
        print(f"Filtering for layer {layer}, {len(pt_files)} files remain.\n")
        for pt_file in pt_files:
            file_path = os.path.join(directory, pt_file)
            load_and_visualize_attention_file(file_path, output_dir, topk)
    else:
        for pt_file in pt_files:
            file_path = os.path.join(directory, pt_file)
            load_and_visualize_attention_file(file_path, output_dir, topk)


if __name__ == "__main__":
    main()