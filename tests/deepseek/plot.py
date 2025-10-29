import pandas as pd
import matplotlib.pyplot as plt
import os

# ======= Config =======
INPUT_FILE = "performance-comparison-table-bak"  # Input data file
OUTPUT_FORMAT = "pdf"                            # Output format

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

# ======= Function: draw scatter =======
def draw_scatter(data, title, output_file):
    plt.figure(figsize=(7, 5))

    for _, row in data.iterrows():
        plt.scatter(
            row["ttft"],
            row["f1"],
            color=color_map.get(row["method"], "gray"),
            marker=marker_map.get(row["size"], "o"),
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

    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel("TTFT (s)", fontsize=12)
    if "samsum" in title.lower():
        plt.ylabel("RL Score", fontsize=12)
    else:
        plt.ylabel("F1 Score", fontsize=12)

    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.6)

    handles = method_handles + size_handles
    plt.legend(handles=handles, title="Legend", loc='lower right', fontsize=9, title_fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, format=OUTPUT_FORMAT)
    plt.close()
    print(f"✅ Saved: {output_file}")

# ======= 1️⃣ Original: Draw for each dataset (all models together) =======
for dataset in df["dataset"].unique():
    data = df[df["dataset"] == dataset]
    draw_scatter(data, f"Dataset: {dataset}", f"{dataset}_scatter.{OUTPUT_FORMAT}")

# ======= 2️⃣ Per-model plots =======
for model in df["model"].unique():
    model_data = df[df["model"] == model]
    for dataset in model_data["dataset"].unique():
        data = model_data[model_data["dataset"] == dataset]
        draw_scatter(data, f"{model} — {dataset}", f"{model}_{dataset}_scatter.{OUTPUT_FORMAT}")

# ======= 3️⃣ Full prefill comparison =======
full_keywords = {"full_prefill"}
full_df = df[df["method"].isin(full_keywords)]

for dataset in full_df["dataset"].unique():
    subset = full_df[full_df["dataset"] == dataset]
    for size in subset["size"].unique():
        same_size = subset[subset["size"] == size]
        if same_size["model"].nunique() < 2:
            continue  # skip if only one model
        plt.figure(figsize=(7, 5))
        models = same_size["model"].unique()
        color_map_model = {m: colors[i % len(colors)] for i, m in enumerate(models)}
        for _, row in same_size.iterrows():
            plt.scatter(
                row["ttft"],
                row["f1"],
                color=color_map_model[row["model"]],
                marker="o",
                s=90,
                edgecolor="black",
                linewidth=0.6,
                label=row["model"]
            )

        plt.title(f"{dataset} — Full Prefill (size={size})", fontsize=14, weight="bold")
        plt.xlabel("TTFT (s)", fontsize=12)
        if dataset == "samsum":
            plt.ylabel("RL Score", fontsize=12)
        else:
            plt.ylabel("F1 Score", fontsize=12)
        plt.ylim(bottom=0)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(title="Model", loc="lower right", fontsize=9, title_fontsize=10)
        plt.tight_layout()
        output_file = f"{dataset}_fullprefill_compare_size{size}.{OUTPUT_FORMAT}"
        plt.savefig(output_file, format=OUTPUT_FORMAT)
        plt.close()
        print(f"✅ Saved: {output_file}")

print("\n🎉 All plots generated successfully!")
