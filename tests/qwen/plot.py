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

# ✅ 不共享坐标轴，让每个子图独立
num_datasets = len(datasets)
fig, axes = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 5), sharex=False, sharey=False)

if num_datasets == 1:
    axes = [axes]  # 确保axes可迭代

for ax, dataset in zip(axes, datasets):
    subset = df[df["dataset"] == dataset]
    
    for name in subset["name"].unique():
        part = subset[subset["name"] == name]

        # 自动识别系列类型以选择颜色和marker
        if name.startswith("blend"):
            style_key = "blend"
        else:
            style_key = name  # full_reuse / full_prefill 等
        
        ax.scatter(
            part["ttft"],
            part["f1"],
            label=name,
            color=colors.get(style_key, "gray"),
            marker=markers.get(style_key, "o"),
            s=80,
            edgecolor="black",
        )

    ax.set_title(dataset, fontsize=14)
    ax.set_xlabel("TTFT", fontsize=12)
    ax.set_xlim(left=0)
    ax.grid(True)

    # ✅ 特殊处理：samsum 用 RL Score
    if dataset == "samsum":
        ax.set_ylabel("RL Score", fontsize=12)
    else:
        ax.set_ylabel("F1 Score", fontsize=12)

    # ✅ 每个子图的纵坐标单独确定范围
    ymin = 0
    ymax = subset["f1"].max() * 1.1  # 留一点空白
    ax.set_ylim(ymin, ymax)

# ✅ 每个子图独立图例（在右下角）
for ax in axes:
    ax.legend(loc="lower right", fontsize=10)

plt.tight_layout()

# 输出为 PDF
output_file = "performance_scatter.pdf"
plt.savefig(output_file, format="pdf", bbox_inches="tight")
print(f"✅ Scatter plots saved to {output_file}")
