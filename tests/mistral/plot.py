import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import re

# 表格路径（替换为你的实际路径）
table_path = "performance-comparison-table-bak"

# 读取表格
df = pd.read_csv(table_path, delim_whitespace=True)

print("Loaded data:")
print(df.head())

# 创建 PDF 文件
with PdfPages("performance_comparison.pdf") as pdf:
    datasets = df["dataset"].unique()

    for dataset in datasets:
        subset = df[df["dataset"] == dataset]

        # 拆分出 blend 系列和 full_prefill
        blend_subset = subset[subset["name"].str.startswith("blend-")].copy()
        full_prefill_subset = subset[subset["name"] == "full_prefill"]

        # 将 blend-0.0 单独处理，并改为 full_reuse
        full_reuse_subset = blend_subset[blend_subset["name"] == "blend-0.0"].copy()
        full_reuse_subset["name"] = "full_reuse"
        blend_subset = blend_subset[blend_subset["name"] != "blend-0.0"]

        # 提取 numeric ratio 方便排序
        blend_subset["ratio"] = blend_subset["name"].apply(lambda x: float(re.search(r"blend-(\d\.\d+)", x).group(1)))
        blend_subset = blend_subset.sort_values(by="ratio")

        # 绘图
        plt.figure(figsize=(7, 5))

        # 绘制 blend 趋势线（不包含 blend-0.0）
        plt.plot(blend_subset["ttft"], blend_subset["f1"],
                 marker="o", linestyle="-", color="tab:orange",
                 label="Blend Trend (varying ratio)", zorder=10)

        # 绘制 full_reuse（原 blend-0.0）点
        if not full_reuse_subset.empty:
            plt.scatter(full_reuse_subset["ttft"], full_reuse_subset["f1"],
                        color="tab:green", s=80, label="Full Reuse", zorder=15)
            for _, row in full_reuse_subset.iterrows():
                plt.text(row["ttft"], row["f1"], "full_reuse",
                         fontsize=9, ha="left", va="bottom", color="tab:green")

        # 绘制 full_prefill 点
        if not full_prefill_subset.empty:
            plt.scatter(full_prefill_subset["ttft"], full_prefill_subset["f1"],
                        color="tab:blue", s=80, label="Full Prefill", zorder=15)
            for _, row in full_prefill_subset.iterrows():
                plt.text(row["ttft"], row["f1"], "full_prefill",
                         fontsize=9, ha="left", va="bottom", color="tab:blue")

        # 添加每个 blend 点标签
        for _, row in blend_subset.iterrows():
            plt.text(row["ttft"], row["f1"], f"{row['name']}",
                     fontsize=8, ha="left", va="bottom", color="tab:orange")

        plt.title(f"Performance Comparison for {dataset}")
        plt.xlabel("TTFT (s)")
        if dataset == "samsum":
            plt.ylabel("RL Score")
        else:
            plt.ylabel("F1 Score")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        pdf.savefig()
        plt.close()

print("✅ PDF saved as: performance_comparison.pdf")
