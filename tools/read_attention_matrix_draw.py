import torch
import os
import argparse
import matplotlib.pyplot as plt
from xformers.ops.fmha.attn_bias import LowerTriangularFromBottomRightMask

import seaborn as sns

def load_and_visualize_attention_file(file_path, save_folder):
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

            print(f"\nVisualizing matrices from {file_path}...\n")

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

            # 确保保存文件夹存在
            os.makedirs(save_folder, exist_ok=True)

            # 遍历每个注意力头并保存对应的热力图
            for head_idx in range(attention_scores.shape[1]):
                plt.figure(figsize=(10, 8))
                sns.heatmap(attention_scores[0, head_idx].cpu().numpy(),
                            cmap="viridis",
                            vmin=0,
                            vmax=1,
                            cbar=True,
                            annot=False,
                            square=True
                        )
                plt.title(f"Attention Matrix (Head {head_idx})")
                plt.xlabel("Keys")
                plt.ylabel("Queries")
                # 拼接保存路径和文件名
                file_name = os.path.basename(file_path).replace('.pt', f'_head_{head_idx}_attention_matrix.png')
                save_path = os.path.join(save_folder, file_name)
                plt.savefig(save_path)
                plt.close()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Read, process, and visualize .pt files containing attention data.")
    parser.add_argument("--data", type=str, required=True, help="Directory containing .pt files.")
    parser.add_argument("--save-dir", type=str, required=True, help="Folder to save the generated heatmaps.")
    parser.add_argument("--layer", type=int, required=False, help="Specific layer to process.")
    parser.add_argument("--prefix", type=str, required=False, help="Prefix for filtering files.")
    args = parser.parse_args()

    directory = args.data
    save_folder = args.save_dir
    layer = args.layer
    prefix = args.prefix

    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return

    
    pt_files = [f for f in os.listdir(directory) if f.endswith(".pt")]
    if prefix:
        pt_files = [f for f in pt_files if f.startswith(prefix)]
    if not pt_files:
        print(f"No .pt files found in directory: {directory}")
        return

    print(f"Found {len(pt_files)} .pt files in directory: {directory}\n")

    if layer is not None:
        pt_files = [f for f in pt_files if f"layer{layer}_" in f]
        print(f"Filtering for layer {layer}, {len(pt_files)} files remain.\n")
        for pt_file in pt_files:
            file_path = os.path.join(directory, pt_file)
            load_and_visualize_attention_file(file_path, save_folder)
    else:
        for pt_file in pt_files:
            file_path = os.path.join(directory, pt_file)
            load_and_visualize_attention_file(file_path, save_folder)


if __name__ == "__main__":
    main()