"""
P-R-curves.py - Precision-Recall 曲线（含 No-skill 基线）
===========================================================
功能：
- 加载指定子集（C 或 H）的测试集数据
- 加载 Global 模型和对应的子集模型（c_only 或 h_only）
- 在子集测试集上计算 Global 和 Stratified 的预测概率并绘制 P-R 曲线
- 添加 No-skill 线（正例比例）

用法：
1. 修改 FEAT_TYPE 和 SUBSET（'C' 或 'H'）
2. 运行: python P-R-curves.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import pickle
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

# ==================== 配置 ====================
FEAT_TYPE = 'deep'   # 'descriptor', 'deep', 'combined'
SUBSET = 'C'             # 'C' 或 'H'
# ==============================================

SPLIT_DIR = '../formal_splits/scaled'
MODEL_DIR = '../final_results'
OUTPUT_DIR = '../figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载指定子集的测试集
test_path = f'{SPLIT_DIR}/{SUBSET}_{FEAT_TYPE}/test.csv'
df_test = pd.read_csv(test_path)
feat_cols = [c for c in df_test.columns if c not in ['cid', 'y', 'type']]
X_test = df_test[feat_cols].values
y_test = df_test['y'].values
cids_test = df_test['cid'].values
print(f"测试集 {SUBSET}: {len(y_test)} 样本, 正例比例: {np.mean(y_test):.3f}")

# 加载 Global 模型
global_model_path = f'{MODEL_DIR}/clf_global_lightgbm_{FEAT_TYPE}.pkl'
with open(global_model_path, 'rb') as f:
    clf_global = pickle.load(f)

# 加载子集模型（根据 SUBSET）
if SUBSET == 'C':
    subset_model_path = f'{MODEL_DIR}/clf_c_only_lightgbm_{FEAT_TYPE}.pkl'
    label_subset = 'C-Only Model'
elif SUBSET == 'H':
    subset_model_path = f'{MODEL_DIR}/clf_h_only_lightgbm_{FEAT_TYPE}.pkl'
    label_subset = 'H-Only Model'
else:
    raise ValueError("SUBSET 必须为 'C' 或 'H'")
with open(subset_model_path, 'rb') as f:
    clf_subset = pickle.load(f)

# 预测概率（直接在子集测试集上）
y_proba_global = clf_global.predict_proba(X_test)[:, 1]
y_proba_subset = clf_subset.predict_proba(X_test)[:, 1]

# 计算 P-R 曲线
prec_g, rec_g, _ = precision_recall_curve(y_test, y_proba_global)
prec_s, rec_s, _ = precision_recall_curve(y_test, y_proba_subset)
auc_g = auc(rec_g, prec_g)
auc_s = auc(rec_s, prec_s)

# 绘图
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(1-rec_g, prec_g, 'b-', linewidth=2, label=f'Global (AUC={auc_g:.3f})')
ax.plot(1-rec_s, prec_s, 'r-', linewidth=2, label=f'Stratified (AUC={auc_s:.3f})')

# No-skill 线
pos_ratio = np.mean(y_test)
ax.axhline(y=pos_ratio, color='gray', linestyle=':', linewidth=1.5,
           label=f'No-skill (pos ratio={pos_ratio:.2f})')

ax.set_xlabel('1 - Recall', fontweight='bold')
ax.set_ylabel('Precision', fontweight='bold')
ax.set_title(f'P-R Curves: {SUBSET} Subset ({FEAT_TYPE})', fontsize=11, fontweight='bold')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='lower left', frameon=True, edgecolor='black', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/pr_curve_{SUBSET}_{FEAT_TYPE}.pdf', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/pr_curve_{SUBSET}_{FEAT_TYPE}.png', dpi=300)
plt.close()
print(f"P-R 曲线已保存: pr_curve_{SUBSET}_{FEAT_TYPE}.pdf")