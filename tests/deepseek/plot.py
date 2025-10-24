import pandas as pd
import matplotlib.pyplot as plt
import os

# ======= Config =======
INPUT_FILE = "performance-comparison-table"  # Input data file
OUTPUT_FORMAT = "pdf"                        # Output format

# ======= Read Data =======
df = pd.read_csv(INPUT_FILE, delim_whitespace=True)

required_cols = {"model", "dataset", "name", "ttft", "f1"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing required columns. Found: {df.columns}")

# ======= Parse name into method + size =======
df["method"] = df["name"].apply(lambda x: x.split("-")[0])
df["size"] = df["name"].apply(lambda x: x.split("-")[1] if "-" in x else "unknown")

# ======= Define colors and markers =======
colors = plt.cm.tab10.colors  # Different colors for methods
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '+']  # Different markers for sizes

methods = df["method"].unique()
sizes = df["size"].unique()

color_map = {m: colors[i % len(colors)] for i, m in enumerate(methods)}
marker_map = {s: markers[i % len(markers)] for i, s in enumerate(sizes)}

# ======= Plot for each dataset =======
datasets = df["dataset"].unique()
for dataset in datasets:
    data = df[df["dataset"] == dataset]
    plt.figure(figsize=(7, 5))

    # Draw points
    for _, row in data.iterrows():
        plt.scatter(
            row["ttft"],
            row["f1"],
            color=color_map[row["method"]],
            marker=marker_map[row["size"]],
            s=80,
            edgecolor="black",
            linewidth=0.6
        )

    # Create legends manually
    method_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=m,
                   markerfacecolor=color_map[m], markersize=8, markeredgecolor='black')
        for m in methods
    ]
    size_handles = [
        plt.Line2D([0], [0], marker=marker_map[s], color='gray', label=s,
                   markerfacecolor='white', markersize=8, markeredgecolor='black')
        for s in sizes
    ]

    plt.title(f"Dataset: {dataset}", fontsize=14, weight='bold')
    plt.xlabel("TTFT (s)", fontsize=12)
    if dataset == "samsum":
        plt.ylabel("RL Score", fontsize=12)
    else:
        plt.ylabel("F1 Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Combine legends (both method & size) and place them at bottom right
    handles = method_handles + size_handles
    plt.legend(handles=handles, title="Legend", loc='lower right', fontsize=9, title_fontsize=10)

    plt.tight_layout()

    # Save PDF
    output_file = f"{dataset}_scatter.{OUTPUT_FORMAT}"
    plt.savefig(output_file, format=OUTPUT_FORMAT)
    plt.close()

print("✅ All scatter plots have been generated and saved as PDF files in the current directory.")
