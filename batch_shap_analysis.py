"""
统一 SHAP 稳定性分析与可视化
==============================
自动生成 9 张柱状图 + 3 张 Venn 图，涵盖：
- 特征集：descriptor, deep, combined
- 模型类型：global, c_only, h_only
要求：已存在对应的 LightGBM 模型文件及训练数据。
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy.stats import kendalltau
from matplotlib_venn import venn3
from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm

warnings.filterwarnings('ignore')

# ===================================================================
# 全局绘图风格（学术论文标准）
# ===================================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# 配色方案（来自 shap_C.py）
MODEL_COLORS = {
    'global': '#66c2a5',    # 绿
    'c_only': '#fc8d62',    # 橙
    'h_only': '#8da0cb'     # 紫
}

# ===================================================================
# 配置
# ===================================================================
FEAT_TYPES = ['descriptor', 'deep', 'combined']
MODEL_TYPES = ['global', 'c_only', 'h_only']
MODEL_LABELS = {'global': 'Global', 'c_only': 'C-only', 'h_only': 'H-only'}

SPLIT_DIR = '../formal_splits/scaled'
MODEL_DIR = '../final_results'
OUTPUT_DIR = '../shap_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SEEDS = 10
SAMPLE_SIZE = 100
TOP_N = 20  # 柱状图和 Venn 图均使用 Top‑20

# ===================================================================
# 辅助函数：加载训练数据（始终使用 Global 训练集）
# ===================================================================
def load_training_data(feat_type):
    train_path = f'{SPLIT_DIR}/Global_{feat_type}/train.csv'
    df = pd.read_csv(train_path)
    feat_cols = [c for c in df.columns if c not in ['cid', 'y', 'type']]
    X = df[feat_cols].values
    feature_names = feat_cols
    return X, feature_names

# ===================================================================
# SHAP 稳定性分析（返回重要性 DataFrame 和稳定性指标）
# ===================================================================
def shap_stability_analysis(clf, X_data, feature_names, n_seeds=10, sample_size=100):
    all_importances = []
    all_rankings = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        idx = np.random.choice(len(X_data), size=sample_size, replace=False)
        X_sample = X_data[idx]
        try:
            explainer = shap.TreeExplainer(clf, feature_names=feature_names)
            shap_values = explainer.shap_values(X_sample)
        except Exception as e:
            print(f"Seed {seed} 失败: {e}")
            continue
        if isinstance(shap_values, list):
            shap_abs = np.abs(shap_values[1])  # 二分类取正类
        else:
            shap_abs = np.abs(shap_values)
        mean_abs = np.mean(shap_abs, axis=0)
        all_importances.append(mean_abs)
        rank = np.argsort(-mean_abs)
        rank_order = np.zeros(len(feature_names))
        for pos, idx_feat in enumerate(rank):
            rank_order[idx_feat] = pos + 1
        all_rankings.append(rank_order)
        if (seed+1) % 5 == 0:
            print(f"  已完成 {seed+1}/{n_seeds} 个种子")
    all_importances = np.array(all_importances)
    all_rankings = np.array(all_rankings)
    mean_importance = np.mean(all_importances, axis=0)
    std_importance = np.std(all_importances, axis=0)

    # Kendall's W
    m = len(all_rankings)
    n = len(feature_names)
    rank_sums = np.sum(all_rankings, axis=0)
    sum_rank_sq = np.sum(rank_sums ** 2)
    W = (12 * sum_rank_sq - 3 * m**2 * n * (n + 1)**2) / (m**2 * n * (n**2 - 1))

    # 平均 Kendall's tau
    tau_values = []
    for i in range(len(all_rankings)):
        for j in range(i+1, len(all_rankings)):
            tau, _ = kendalltau(all_rankings[i], all_rankings[j])
            tau_values.append(tau)
    mean_tau = np.mean(tau_values) if tau_values else 0

    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Mean_SHAP': mean_importance,
        'Std_SHAP': std_importance,
        'Mean_Rank': np.mean(all_rankings, axis=0)
    }).sort_values('Mean_SHAP', ascending=False)
    return df_imp, W, mean_tau

# ===================================================================
# 绘图函数：绘制带误差棒的水平柱状图（Top‑20）
# ===================================================================
def plot_shap_bar(df_top, model_type, feat_type, color, output_dir):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    y_pos = np.arange(len(df_top))
    bars = ax.barh(y_pos, df_top['Mean_SHAP'].values,
                   xerr=df_top['Std_SHAP'].values,
                   capsize=2, color=color, edgecolor='black', linewidth=0.6,
                   height=0.7)

    ax.set_yticks(y_pos)
    # 特征名若太长则截断
    labels = [f[:20] + '…' if len(f) > 20 else f for f in df_top['Feature'].values]
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP Value|', fontsize=9, fontweight='bold')
    ax.set_title(f"{MODEL_LABELS[model_type]} on {feat_type.upper()}\nTop‑20 Features", 
                 fontsize=10, fontweight='bold', pad=8)
    ax.grid(axis='x', alpha=0.2, linestyle=':')

    # 在柱内添加数值标签
    x_max = df_top['Mean_SHAP'].max()
    for bar, val in zip(bars, df_top['Mean_SHAP'].values):
        width = bar.get_width()
        label_format = f'{val:.3f}' if val > 0.01 else f'{val:.4f}'
        ax.text(width + 0.02 * x_max, bar.get_y() + bar.get_height()/2,
                label_format, va='center', ha='left', fontsize=6, color='black')

    ax.set_xlim(0, x_max * 1.20)
    plt.tight_layout(pad=0.5)
    out_path = f'{output_dir}/shap_stability_{model_type}_{feat_type}'
    plt.savefig(f'{out_path}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out_path}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  保存柱状图: {out_path}.pdf/png")

# ===================================================================
# 主循环
# ===================================================================
# 存储每个组合的 Top‑20 特征集合（用于 Venn 图）
top20_sets = {ft: {} for ft in FEAT_TYPES}

for feat_type in FEAT_TYPES:
    print(f"\n========== 处理特征集: {feat_type} ==========")
    X_train, feature_names = load_training_data(feat_type)
    print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")

    for model_type in MODEL_TYPES:
        print(f"\n--- 模型: {model_type} ---")
        model_file = f'{MODEL_DIR}/clf_{model_type}_lightgbm_{feat_type}.pkl'
        if not os.path.exists(model_file):
            print(f"警告: 模型文件 {model_file} 不存在，跳过")
            continue

        with open(model_file, 'rb') as f:
            clf = pickle.load(f)
        print(f"模型类型: {type(clf).__name__}")

        # 执行稳定性分析
        df_imp, W, mean_tau = shap_stability_analysis(
            clf, X_train, feature_names, n_seeds=N_SEEDS, sample_size=SAMPLE_SIZE
        )
        print(f"  Kendall's W = {W:.4f}, 平均 τ = {mean_tau:.4f}")

        # 保存 CSV 结果
        csv_path = f'{OUTPUT_DIR}/shap_importance_{model_type}_{feat_type}.csv'
        df_imp.to_csv(csv_path, index=False)
        with open(f'{OUTPUT_DIR}/shap_W_{model_type}_{feat_type}.txt', 'w') as f:
            f.write(f"Kendall's W: {W:.4f}\nMean tau: {mean_tau:.4f}\nSeeds: {N_SEEDS}\nSample size: {SAMPLE_SIZE}")

        # 取 Top‑20 并绘图
        df_top = df_imp.head(TOP_N)
        top20_sets[feat_type][model_type] = set(df_top['Feature'].values)

        color = MODEL_COLORS[model_type]
        plot_shap_bar(df_top, model_type, feat_type, color, OUTPUT_DIR)

# ===================================================================
# 生成 Venn 图（每个特征集一张）
# ===================================================================
print("\n========== 生成 Venn 图 ==========")
for feat_type in FEAT_TYPES:
    sets = top20_sets[feat_type]
    if len(sets) < 3:
        print(f"特征集 {feat_type} 缺少模型数据，跳过 Venn")
        continue

    # 计算七个区域的元素个数 (顺序: 100, 010, 110, 001, 101, 011, 111)
    s_global = sets['global']
    s_c = sets['c_only']
    s_h = sets['h_only']

    areas = (
        len(s_global - s_c - s_h),                     # 仅 Global
        len(s_c - s_global - s_h),                     # 仅 C‑only
        len((s_global & s_c) - s_h),                   # Global ∩ C‑only
        len(s_h - s_global - s_c),                     # 仅 H‑only
        len((s_global & s_h) - s_c),                   # Global ∩ H‑only
        len((s_c & s_h) - s_global),                   # C‑only ∩ H‑only
        len(s_global & s_c & s_h)                      # 三者交集
    )

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    venn = venn3(
        subsets=areas,
        set_labels=('Global', 'C‑only', 'H‑only'),
        set_colors=(MODEL_COLORS['global'], MODEL_COLORS['c_only'], MODEL_COLORS['h_only']),
        alpha=0.7,
        layout_algorithm=DefaultLayoutAlgorithm(1.0),
        ax=ax
    )

    # 设置标签字体
    for label in venn.set_labels:
        if label:
            label.set_fontsize(9)
            label.set_fontweight('bold')
            label.set_fontname('Times New Roman')
    for label in venn.subset_labels:
        if label:
            label.set_fontsize(8)
            label.set_fontname('Times New Roman')

    ax.set_title(f'Overlap of Top‑{TOP_N} Features\n{feat_type.upper()}', 
                 fontsize=11, fontweight='bold', pad=12)
    ax.set_axis_off()
    plt.tight_layout()
    out_venn = f'{OUTPUT_DIR}/venn_{feat_type}'
    plt.savefig(f'{out_venn}.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out_venn}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存 Venn 图: {out_venn}.pdf/png")

print("\n✅ 所有图形生成完毕！")