#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Read the log file
log_file = 'row-out.log'

# Store pairs: (index, layer)
pairs = []

# Parse lines 5-6 to extract document lengths and calculate prefix sum
prefix_doc_len = []
with open(log_file, 'r') as f:
    lines = f.readlines()
    if len(lines) >= 6:
        # Read line 5 (index 4) and line 6 (index 5)
        line5 = lines[3]  # 0-indexed, so line 5 is index 4

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
        # Check for layer declaration
        layer_match = re.search(r'Layer (\d+):', line)
        if layer_match:
            current_layer = int(layer_match.group(1))
            collecting_indices = False
            continue

        # Check for "Top-0.4 Indices" line - next lines will contain indices
        if 'Top-' in line and 'Indices' in line:
            collecting_indices = True
            continue

        # If we're collecting indices for a layer
        if collecting_indices and current_layer is not None:
            # Check if line contains tensor data (starts with "tensor([" or has indented numbers)
            if 'tensor([' in line or (re.match(r'^\s+\d+', line) and not line.strip().startswith('device')):
                # Extract all numbers from the line (excluding device='cuda:X' etc)
                if 'device' in line.lower():
                    collecting_indices = False
                    continue

                # Extract numbers
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

            # Check if we're done
            if ']' in line and 'device' in line:
                collecting_indices = False
            elif line.strip() and not re.match(r'^\s+\d+', line) and 'tensor' not in line.lower():
                if 'Processing' in line or 'Layer' in line:
                    collecting_indices = False

# Calculate frequency count for each index and plot as a line plot
if pairs:
    indices, layers = zip(*pairs)

    # Count frequency of each index
    index_counter = Counter(indices)

    # Get sorted indices and their frequencies
    sorted_indices = sorted(index_counter.keys())
    frequencies = [index_counter[idx] for idx in sorted_indices]

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot as a line plot
    ax.plot(sorted_indices, frequencies, color='blue', linewidth=1.5, marker='', alpha=0.9)

    # Set labels and title
    ax.set_xlabel('Index (k)', fontsize=12)
    ax.set_ylabel('Frequency Count', fontsize=12)
    ax.set_title('Frequency Count of Scatter Points per Index (Line Plot)', fontsize=14)

    # Set x-axis range to (0, 2000) to match the scatter plot
    ax.set_xlim(0, 2000)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')

    # Draw red vertical lines at x = prefix_doc_len[i] if available
    if prefix_doc_len:
        y_max = max(frequencies) if frequencies else 1
        for prefix_val in prefix_doc_len:
            if prefix_val <= 2000:  # Only draw if within x-axis range
                ax.axvline(x=prefix_val, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig('index_frequency_line_plot.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as index_frequency_line_plot.png")
    print(f"Total pairs: {len(pairs)}")
    print(f"Unique indices: {len(sorted_indices)}")
    print(f"Index range: {min(sorted_indices)} - {max(sorted_indices)}")
    print(f"Frequency range: {min(frequencies)} - {max(frequencies)}")
    print(f"Most frequent index: {sorted_indices[np.argmax(frequencies)]} (appears {max(frequencies)} times)")
else:
    print("No data found in the log file")

