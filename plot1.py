"""
plot1.py - 模型性能对比图（更新版）
========================================
基于最终结果文件（final_results_{FEAT_TYPE}.csv）绘制：
1. 热力图：数据集-特征集 vs 模型准确率
2. 柱状图：Global vs Stratified 对比（Accuracy, Balanced_Acc, MCC, F1），
   每个特征集单独成图，不同指标用不同斜线阴影

用法：
1. 确保 final_results_*.csv 和 bootstrap_ci_*.csv 已生成
2. 直接运行（无需修改）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import warnings
warnings.filterwarnings('ignore')

# ===================================================================
# 设置学术风格
# ===================================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# ===================================================================
# 配置区域
# ===================================================================
RESULTS_DIR = '../final_results'
OUTPUT_DIR = '../figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEAT_TYPES = ['descriptor', 'deep', 'combined']
FEAT_LABELS = ['F1 (Descriptor)', 'F2 (Deep)', 'F3 (Combined)']
MODEL_NAMES = ['RF', 'LGBM', 'CatBoost', 'SoftVoting', 'HardVoting']
METRICS = ['Accuracy', 'Balanced_Acc', 'MCC', 'F1']          # 需要绘制的四个指标
HATCH_PATTERNS = ['', '/', '\\', 'x']                        # 对应四种指标的阴影样式

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ===================================================================
# 加载数据
# ===================================================================
all_data = {}
bootstrap_data = {}

for ft in FEAT_TYPES:
    res_file = f'{RESULTS_DIR}/final_results_{ft}.csv'
    if os.path.exists(res_file):
        df = pd.read_csv(res_file)
        all_data[ft] = df
    else:
        print(f"警告: {res_file} 不存在")
    boot_file = f'{RESULTS_DIR}/bootstrap_ci_{ft}.csv'
    if os.path.exists(boot_file):
        bootstrap_data[ft] = pd.read_csv(boot_file)

# ===================================================================
# 图 A: 热力图（保持不变）
# ===================================================================
row_names = []
heatmap_matrix = []
for ft in FEAT_TYPES:
    df = all_data.get(ft)
    if df is None:
        continue
    for strategy in ['Global', 'C-only', 'H-only']:
        row_name = f"{FEAT_LABELS[FEAT_TYPES.index(ft)]}\n{strategy}"
        row_names.append(row_name)
        row_vals = []
        for model in MODEL_NAMES:
            val = df[(df['Strategy'] == strategy) & (df['Model'] == model)]
            if len(val) > 0:
                row_vals.append(val['Accuracy'].values[0])
            else:
                row_vals.append(np.nan)
        heatmap_matrix.append(row_vals)

heatmap_matrix = np.array(heatmap_matrix)

fig, ax = plt.subplots(figsize=(12, 8))
cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#F8F9FA', '#2E86AB'], N=100)
im = ax.imshow(heatmap_matrix, cmap=cmap, aspect='auto', vmin=0.70, vmax=0.95)

ax.set_xticks(np.arange(len(MODEL_NAMES)))
ax.set_yticks(np.arange(len(row_names)))
ax.set_xticklabels(MODEL_NAMES, fontsize=11)
ax.set_yticklabels(row_names, fontsize=9)

for i in range(len(row_names)):
    for j in range(len(MODEL_NAMES)):
        val = heatmap_matrix[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    color='black' if val < 0.85 else 'white', fontsize=8)

ax.set_xlabel('Models', fontsize=12, fontweight='bold')
ax.set_ylabel('Dataset-Feature Set (Strategy)', fontsize=12, fontweight='bold')
ax.set_title('Test Accuracy Heatmap (Strict Validation)', fontsize=13, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label('Accuracy', fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figure_heatmap_updated.pdf', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/figure_heatmap_updated.png', dpi=300)
plt.close()
print(f"热力图已保存: {OUTPUT_DIR}/figure_heatmap_updated.pdf/png")

# ===================================================================
# 图 B: Global vs Stratified 对比（四个指标，每个特征集单独一张图）
# ===================================================================
for idx, ft in enumerate(FEAT_TYPES):
    df = all_data.get(ft)
    bf = bootstrap_data.get(ft)
    if df is None:
        continue

    # 创建独立图形
    fig, ax = plt.subplots(figsize=(6, 5))

    # 提取 CatBoost 的 Global 和 Stratified 在四个指标上的值（使用 Global 测试集）
    global_vals = []
    stratified_vals = []
    ci_errors = []   # 仅用于 Accuracy 的误差棒（如果有）

    for metric in METRICS:
        # Global
        g_val = df[(df['Strategy'] == 'Global') & (df['Model'] == 'CatBoost') & (df['Subset'] == 'Global')][metric].values[0]
        # Stratified
        s_val = df[(df['Strategy'] == 'Stratified') & (df['Model'] == 'CatBoost') & (df['Subset'] == 'Global')][metric].values[0]
        global_vals.append(g_val)
        stratified_vals.append(s_val)

        # 尝试从 bootstrap 文件中获取该指标的 CI（仅 Accuracy 有）
        if metric == 'Accuracy' and bf is not None:
            acc_row = bf[bf['Metric'] == 'Accuracy']
            if len(acc_row) > 0:
                ci_lower = acc_row['CI_2.5%'].values[0]
                ci_upper = acc_row['CI_97.5%'].values[0]
                # 转换为 Stratified 的误差范围（保持原逻辑）
                err_lower = s_val - (s_val + ci_lower)
                err_upper = (s_val + ci_upper) - s_val
                ci_errors.append((err_lower, err_upper))
            else:
                ci_errors.append((0, 0))
        else:
            ci_errors.append((0, 0))

    # 绘制分组柱状图
    x_pos = np.arange(len(METRICS))
    width = 0.35

    # Global 柱子（蓝色）
    bars_global = ax.bar(x_pos - width/2, global_vals, width,
                         label='Global', color='#1f77b4', alpha=0.8, edgecolor='black')
    # Stratified 柱子（红色）
    bars_strat = ax.bar(x_pos + width/2, stratified_vals, width,
                        label='Stratified', color='#d62728', alpha=0.8, edgecolor='black')

    # 为每个指标的柱子添加不同阴影（区分指标）
    # 由于每个指标有两根柱子，我们对同一指标的两根柱子设置相同的阴影
    for i, hatch in enumerate(HATCH_PATTERNS):
        # Global 柱子
        bars_global[i].set_hatch(hatch)
        # Stratified 柱子
        bars_strat[i].set_hatch(hatch)

    # 添加误差棒（仅 Accuracy）
    if ci_errors[0] != (0, 0):
        err_lower, err_upper = ci_errors[0]
        ax.errorbar(x_pos[0] + width/2, stratified_vals[0],
                    yerr=[[err_lower], [err_upper]],
                    fmt='none', ecolor='black', capsize=4, capthick=1)

    # 添加数值标签
    for i in range(len(METRICS)):
        # Global
        ax.text(x_pos[i] - width/2, global_vals[i] + 0.02,
                f'{global_vals[i]:.3f}', ha='center', fontsize=9)
        # Stratified
        ax.text(x_pos[i] + width/2, stratified_vals[i] + 0.02,
                f'{stratified_vals[i]:.3f}', ha='center', fontsize=9)

    # 设置轴标签和刻度
    ax.set_xticks(x_pos)
    ax.set_xticklabels(METRICS, fontsize=10)
    ax.set_ylim(0.50, 0.95)  # 根据实际数据调整，这里保留
    ax.set_title(FEAT_LABELS[idx], fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    # 保存
    plt.savefig(f'{OUTPUT_DIR}/figure_global_vs_stratified_{ft}_4metrics.pdf', dpi=300)
    plt.savefig(f'{OUTPUT_DIR}/figure_global_vs_stratified_{ft}_4metrics.png', dpi=300)
    plt.close()
    print(f"对比图（4指标）已保存: {OUTPUT_DIR}/figure_global_vs_stratified_{ft}_4metrics.pdf/png")