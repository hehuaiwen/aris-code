import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, LogFormatterMathtext

with open("irs_rank_random_init_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 所有 seed 的归一化特征值
all_eigs = np.array([
    r["eigvals_norm_all"][:10] for r in data["results"]
], dtype=float)

mean_eigs = all_eigs.mean(axis=0)
std_eigs = all_eigs.std(axis=0)

idx = np.arange(1, len(mean_eigs) + 1)

# ================= 图1：普通坐标 =================
plt.figure(figsize=(7, 4.8))
plt.bar(idx, mean_eigs, yerr=std_eigs, capsize=3, width=0.6)

plt.xlabel("Eigenvalue index")
plt.ylabel("Average normalized eigenvalue")
plt.xticks(idx)
plt.ylim(-0.1, 1.05)

# y轴只保留几个主刻度
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

plt.grid(axis="y", linestyle="--", alpha=0.45)
plt.tight_layout()
plt.savefig("irs_eig_distribution_mean.png", dpi=300, bbox_inches="tight")
plt.show()


# ================= 图2：log坐标，去掉细小刻度 =================
fig, ax = plt.subplots(figsize=(7, 4.8))

ax.bar(idx, mean_eigs, yerr=std_eigs, capsize=3, width=0.6)

ax.set_yscale("log")
ax.set_xlabel("Eigenvalue index", fontsize=16)
ax.set_ylabel("Average normalized eigenvalue", fontsize=16)
ax.set_xticks(idx)

# 手动设置主刻度，避免密密麻麻
ax.set_yticks([1e0, 1e-2, 1e-4, 1e-6, 1e-8])
ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

# 关闭次刻度
ax.yaxis.set_minor_locator(NullLocator())

# 只画主网格线
ax.grid(axis="y", which="major", linestyle="--", alpha=0.45)
ax.grid(False, which="minor")

ax.set_ylim(1e-8, 2)

plt.tight_layout()
plt.savefig("irs_eig_distribution_mean_log.png", dpi=300, bbox_inches="tight")
plt.show()