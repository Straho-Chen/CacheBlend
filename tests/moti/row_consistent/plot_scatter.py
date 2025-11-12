#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Plot scatter plot and per_query_sum graph from log file.")
    parser.add_argument("--log_file", help="Input log file to parse (e.g., row-out-no-mask.log)")
    parser.add_argument("--output", "-o", type=str, default="scatter_plot.png", help="Output PNG file name")
    args = parser.parse_args()

    log_file = args.log_file
    output_file = args.output

    # Store pairs: (index, layer)
    pairs = []

    # Store per_query_sum values: {layer: per_query_sum}
    per_query_sums = {}

    # Parse lines 5-6 to extract document lengths and calculate prefix sum
    prefix_doc_len = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
        if len(lines) >= 6:
            # Read line 5 (index 4) and line 6 (index 5)
            line5 = lines[3]  # 0-indexed, so line 5 is index 4
            print(line5)

            # Extract all numbers that appear after commas in tuples like (..., number)
            doc_lengths = []
            matches = re.findall(r',\s*(\d+)\)', line5)
            for match in matches:
                doc_lengths.append(int(match))

            # Calculate prefix sum
            if doc_lengths:
                prefix_doc_len = [0]  # Start with 0
                for length in doc_lengths:
                    prefix_doc_len.append(prefix_doc_len[-1] + length)

    current_layer = None
    collecting_indices = False

    with open(log_file, 'r') as f:
        for line in f:
            layer_match = re.search(r'Layer (\d+):', line)
            if layer_match:
                current_layer = int(layer_match.group(1))
                collecting_indices = False

                # Check if this line also contains per_query_sum
                per_query_match = re.search(r'per_query_sum:\s*([\d.]+)', line)
                if per_query_match:
                    per_query_sum_value = float(per_query_match.group(1))
                    per_query_sums[current_layer] = per_query_sum_value
                continue

            # Check for per_query_sum in lines that don't have layer declaration
            per_query_match = re.search(r'Layer (\d+):.*per_query_sum:\s*([\d.]+)', line)
            if per_query_match:
                layer_num = int(per_query_match.group(1))
                per_query_sum_value = float(per_query_match.group(2))
                per_query_sums[layer_num] = per_query_sum_value
                continue

            # Check for "Top-0.4 Indices" line - next lines will contain indices
            if 'Top-' in line and 'Indices' in line:
                collecting_indices = True
                continue

            # If we're collecting indices for a layer
            if collecting_indices and current_layer is not None:
                if 'tensor([' in line or (re.match(r'^\s+\d+', line) and not line.strip().startswith('device')):
                    if 'device' in line.lower():
                        collecting_indices = False
                        continue

                    if 'tensor([' in line:
                        content = line.split('tensor([')[1] if 'tensor([' in line else line
                        numbers = re.findall(r'\b\d+\b', content)
                    else:
                        numbers = re.findall(r'\b\d+\b', line)

                    for num_str in numbers:
                        try:
                            index = int(num_str)
                            if index < 10000:
                                pairs.append((index, current_layer))
                        except ValueError:
                            continue

                if ']' in line and 'device' in line:
                    collecting_indices = False
                elif line.strip() and not re.match(r'^\s+\d+', line) and 'tensor' not in line.lower():
                    if 'Processing' in line or 'Layer' in line:
                        collecting_indices = False

    # Create the plots
    if pairs or per_query_sums:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # ===== First subplot: Scatter plot ("index vs layer") =====
        if pairs:
            indices, layers = zip(*pairs)
            indices = np.array(indices)
            layers = np.array(layers)
            min_index = min(indices)
            max_index = max(indices)
            min_layer = min(layers)
            max_layer = max(layers)

            # Assign a color to each layer for scatter plot
            unique_layers = sorted(set(layers))
            # Use academic-style color palette (muted, colorblind-friendly)
            if len(unique_layers) <= 8:
                cmap = plt.get_cmap('Set2')  # Muted colors, good for academic papers
            elif len(unique_layers) <= 12:
                cmap = plt.get_cmap('Dark2')  # Darker, more professional
            else:
                cmap = plt.get_cmap('Set3')  # More colors, still muted
            layer_to_color = {layer: cmap(i % cmap.N) for i, layer in enumerate(unique_layers)}

            # Lower scatter size for thousands of points to be small (e.g. s=0.5)
            point_size = 0.1 if len(indices) > 1000 else 3

            # For each layer, scatter that layer's points using its color
            for il, layer in enumerate(unique_layers):
                mask = layers == layer
                ax1.scatter(
                    indices[mask],
                    layers[mask],
                    s=point_size,
                    color=layer_to_color[layer],
                    alpha=0.8,
                    label=f'Layer {layer}' if len(unique_layers) <= 15 else None
                )

            ax1.set_xlabel('Index', fontsize=12, y=-0.02)
            ax1.set_ylabel('Layer', fontsize=12)
            # ax1.set_xlim(0, 2000)
            ax1.set_ylim(-1.5, max_layer + 0.5)
            ax1.set_title('Scatter Plot: Index vs Layer', fontsize=14)

            # Draw vertical lines at x = prefix_doc_len[i] (academic dark blue)
            if prefix_doc_len:
                for prefix_val in prefix_doc_len:
                    if 0 <= prefix_val <= max_index + 1:
                        ax1.axvline(x=prefix_val, color='#2C3E50', linestyle='--', linewidth=1, alpha=0.6)

            # Set y-ticks to integer layers if not too many
            y_tick_count = max_layer - min_layer + 1
            if y_tick_count <= 50:
                ax1.set_yticks(list(range(min_layer, max_layer + 1)))
            else:
                ax1.set_yticks(np.linspace(min_layer, max_layer, 10, dtype=int))
            
            # Set x-axis at y = -1 visually (just an extra horizontal line at y=-1)
            ax1.axhline(y=-1, color='#34495E', linewidth=1, alpha=0.8)
            
            # Add legend only if not too many layers
            if len(unique_layers) <= 15:
                ax1.legend(loc="upper right", fontsize=9, frameon=False, title='Layer')

        else:
            ax1.text(0.5, 0.5, 'No scatter plot data found', 
                    transform=ax1.transAxes, ha='center', va='center', fontsize=12)
            ax1.set_title('Scatter Plot: Index vs Layer', fontsize=14)

        # ===== Second subplot: Line chart for per_query_sum =====
        if per_query_sums:
            sorted_layers = sorted(per_query_sums.keys())
            sorted_values = [per_query_sums[layer] for layer in sorted_layers]
            ax2.plot(sorted_layers, sorted_values, marker='o', linestyle='-', linewidth=2, 
                    markersize=4, color='#2980B9', markerfacecolor='#3498DB', markeredgecolor='#2980B9')
            ax2.set_xlabel('Layer', fontsize=12)
            ax2.set_ylabel('per_query_sum', fontsize=12)
            ax2.set_title('Line Chart: per_query_sum vs Layer', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(min(sorted_layers) - 1, max(sorted_layers) + 1)
            ax2.set_xticks(sorted_layers[::2])  # Show every other layer to avoid crowding
        else:
            ax2.text(0.5, 0.5, 'No per_query_sum data found', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title('Line Chart: per_query_sum vs Layer', fontsize=14)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {output_file}")
        if pairs:
            indices, layers = zip(*pairs)
            unique_layers = sorted(set(layers))
            print(f"Total pairs: {len(pairs)}")
            print(f"Layers: {unique_layers}")
            print(f"Index range: {min(indices)} - {max(indices)}")
        else:
            print("No scatter plot data found")
        if per_query_sums:
            print(f"per_query_sum data for {len(per_query_sums)} layers")
            print(f"per_query_sum range: {min(per_query_sums.values()):.2f} - {max(per_query_sums.values()):.2f}")
        if prefix_doc_len:
            print(f"prefix_doc_len: {prefix_doc_len[:10]}..." if len(prefix_doc_len) > 10 else f"prefix_doc_len: {prefix_doc_len}")
            print(f"Total prefix_doc_len values: {len(prefix_doc_len)}")
    else:
        print("No data found in the log file")


if __name__ == "__main__":
    main()
