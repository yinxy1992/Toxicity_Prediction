"""
shap_stability.py - SHAP 特征重要性稳定性分析
================================================
功能：
- 加载训练好的分类器（无 Pipeline，直接分类器）
- 对训练集进行多次随机抽样，计算 SHAP 值
- 输出特征重要性均值、标准差，并计算 Kendall's W 系数
- 生成带误差棒的特征重要性图

用法：
1. 修改 FEAT_TYPE 和 MODEL_TYPE（'global', 'c_only', 'h_only'）
2. 运行: python shap_stability.py
"""

import pandas as pd
import numpy as np
import shap
import pickle
import os
import matplotlib.pyplot as plt
import warnings
from scipy.stats import kendalltau

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

# ==================== 配置区域 ====================
FEAT_TYPE = 'descriptor'      # 'descriptor', 'deep', 'combined'
MODEL_TYPE = 'h_only'         # 'global', 'c_only', 'h_only'
# ==================================================

SPLIT_DIR = '../formal_splits/scaled'
MODEL_DIR = '../final_results'
OUTPUT_DIR = '../shap_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SEEDS = 10
SAMPLE_SIZE = 100
RANDOM_STATE = 42

# 模型文件（根据 MODEL_TYPE 选择）
model_map = {
    'global': f'clf_global_lightgbm_{FEAT_TYPE}.pkl',
    'c_only': f'clf_c_only_lightgbm_{FEAT_TYPE}.pkl',
    'h_only': f'clf_h_only_lightgbm_{FEAT_TYPE}.pkl'
}
model_path = os.path.join(MODEL_DIR, model_map[MODEL_TYPE])
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件不存在: {model_path}，请先运行 vote.py")

# 加载训练数据（Global 训练集，用于 SHAP 解释）
def load_training_data():
    train_path = f'{SPLIT_DIR}/Global_{FEAT_TYPE}/train.csv'
    df = pd.read_csv(train_path)
    feat_cols = [c for c in df.columns if c not in ['cid', 'y', 'type']]
    X = df[feat_cols].values
    feature_names = feat_cols
    return X, feature_names

X_train, feature_names = load_training_data()
print(f"训练集样本: {X_train.shape}, 特征数: {X_train.shape[1]}")

# 加载模型
with open(model_path, 'rb') as f:
    clf = pickle.load(f)
print(f"模型类型: {type(clf).__name__}")

# 稳定性分析
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
            print(f"已完成 {seed+1}/{n_seeds} 个种子")

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

    print(f"Kendall's W: {W:.4f}, 平均 tau: {mean_tau:.4f}")

    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Mean_SHAP': mean_importance,
        'Std_SHAP': std_importance,
        'Mean_Rank': np.mean(all_rankings, axis=0)
    }).sort_values('Mean_SHAP', ascending=False)
    return df_imp, W, mean_tau

df_imp, W, mean_tau = shap_stability_analysis(
    clf, X_train, feature_names, n_seeds=N_SEEDS, sample_size=SAMPLE_SIZE
)

# 保存结果
df_imp.to_csv(f'{OUTPUT_DIR}/shap_importance_{MODEL_TYPE}_{FEAT_TYPE}.csv', index=False)
with open(f'{OUTPUT_DIR}/shap_W_{MODEL_TYPE}_{FEAT_TYPE}.txt', 'w') as f:
    f.write(f"Kendall's W: {W:.4f}\nMean tau: {mean_tau:.4f}\nSeeds: {N_SEEDS}\nSample size: {SAMPLE_SIZE}")

# 绘图（Top 20）
top_n = 20
df_top = df_imp.head(top_n)
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(df_top))
bars = ax.barh(y_pos, df_top['Mean_SHAP'].values,
               xerr=df_top['Std_SHAP'].values, capsize=3,
               color='steelblue', edgecolor='black', linewidth=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(df_top['Feature'].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
ax.set_title(f'SHAP Feature Importance (Top {top_n})\nModel: {MODEL_TYPE}, Feature Set: {FEAT_TYPE}\n'
             f"Kendall's W = {W:.3f}, Mean τ = {mean_tau:.3f}", fontsize=11, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/shap_stability_{MODEL_TYPE}_{FEAT_TYPE}.pdf', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/shap_stability_{MODEL_TYPE}_{FEAT_TYPE}.png', dpi=300)
plt.close()
print(f"图已保存: shap_stability_{MODEL_TYPE}_{FEAT_TYPE}.pdf")