import pandas as pd
import matplotlib.pyplot as plt

# 读取数据文件
input_file = "performance-comparison-table"
df = pd.read_csv(input_file, delim_whitespace=True)

# 检查必要列
required_cols = {"dataset", "name", "ttft", "f1"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Input file must contain columns: {required_cols}")

# 定义颜色和marker样式
colors = {
    "blend": "tab:blue",
    "full_reuse": "tab:green",
    "full_prefill": "tab:red",
}
markers = {
    "blend": "o",
    "full_reuse": "s",
    "full_prefill": "^",
}

# 获取所有数据集
datasets = df["dataset"].unique()

# 创建子图
num_datasets = len(datasets)
fig, axes = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 5), sharey=True)

if num_datasets == 1:
    axes = [axes]  # 确保axes可迭代

for ax, dataset in zip(axes, datasets):
    subset = df[df["dataset"] == dataset]
    
    for name in subset["name"].unique():
        part = subset[subset["name"] == name]
        ax.scatter(
            part["ttft"],
            part["f1"],
            label=name,
            color=colors.get(name, "gray"),
            marker=markers.get(name, "o"),
            s=80,
            edgecolor="black",
        )

    ax.set_title(dataset, fontsize=14)
    ax.set_xlabel("TTFT", fontsize=12)
    ax.set_xlim(left=0)
    ax.grid(True)

    # ✅ 特殊处理：samsum用 RL Score
    ax.set_ylabel("F1 Score", fontsize=12)
    if dataset == "samsum":
        ax.set_ylabel("RL Score", fontsize=12)

# 图例放右下角
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()

# 输出为 PDF
output_file = "performance_scatter.pdf"
plt.savefig(output_file, format="pdf", bbox_inches="tight")
print(f"✅ Scatter plots saved to {output_file}")
