"""
calibration_curve.py - 概率校准曲线
=====================================
功能：
- 加载测试集和分类器
- 绘制校准曲线（可靠性图）
- 计算 Brier Score

用法：
1. 修改 FEAT_TYPE
2. 运行: python calibration_curve.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
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
FEAT_TYPE = 'combined'   # 'descriptor', 'deep', 'combined'
# ==============================================

SPLIT_DIR = '../formal_splits/scaled'
MODEL_DIR = '../final_results'
OUTPUT_DIR = '../figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载全局测试集
def load_test_data():
    test_path = f'{SPLIT_DIR}/Global_{FEAT_TYPE}/test.csv'
    df = pd.read_csv(test_path)
    feat_cols = [c for c in df.columns if c not in ['cid', 'y', 'type']]
    X_test = df[feat_cols].values
    y_test = df['y'].values
    return X_test, y_test

X_test, y_test = load_test_data()
print(f"测试集样本数: {len(y_test)}")

# 加载三个模型
models = {}
for name in ['global', 'c_only', 'h_only']:
    model_path = f'{MODEL_DIR}/clf_{name}_lightgbm_{FEAT_TYPE}.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            models[name] = pickle.load(f)
        print(f"加载模型: {name}")
    else:
        print(f"警告: {model_path} 不存在，跳过")

# 计算 Brier Score 和校准曲线
results = []
fig, ax = plt.subplots(figsize=(6, 5))
colors = {'global': '#1f77b4', 'c_only': '#d62728', 'h_only': "#027843"}
labels = {'global': 'Global Model', 'c_only': 'C-Only Model', 'h_only': 'H-Only Model'}

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_proba)
    results.append({'Model': name, 'Brier_Score': brier, 'n_samples': len(y_test)})
    print(f"{name}: Brier Score = {brier:.4f}")

    # 校准曲线
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_proba, n_bins=10, strategy='quantile'
    )
    ax.plot(mean_predicted_value, fraction_of_positives,
            'o-', color=colors[name], linewidth=2, markersize=6,
            label=f"{labels[name]} (Brier={brier:.3f})")

# 理想线
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
ax.set_title(f'Probability Calibration Curves ({FEAT_TYPE})', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/calibration_curve_{FEAT_TYPE}.pdf', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/calibration_curve_{FEAT_TYPE}.png', dpi=300)
plt.close()

# 保存 Brier Score
pd.DataFrame(results).to_csv(f'{OUTPUT_DIR}/brier_scores_{FEAT_TYPE}.csv', index=False)
print(f"校准曲线和 Brier Score 已保存至 {OUTPUT_DIR}/")