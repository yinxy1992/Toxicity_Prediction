"""
applicability_domain.py - 适用域（AD）定义和评估
==================================================
功能：
- 基于训练集（Global 或 C 或 H）的特征计算马氏距离
- 以 95% 分位数作为 AD 阈值
- 评估测试集中 In-AD vs Out-of-AD 样本的性能差异
- 保存 AD 参数供部署使用

用法：
1. 修改 FEAT_TYPE 和 MODEL_TYPE（'global', 'c_only', 'h_only'）
2. 运行: python applicability_domain.py
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, f1_score
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
FEAT_TYPE = 'combined'      # 'descriptor', 'deep', 'combined'
MODEL_TYPE = 'c_only'         # 'global', 'c_only', 'h_only'
AD_PERCENTILE = 95
# ==============================================

SPLIT_DIR = '../formal_splits/scaled'
MODEL_DIR = '../final_results'
OUTPUT_DIR = '../ad_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 根据 MODEL_TYPE 选择数据集
def load_data(dataset_name):
    train_path = f'{SPLIT_DIR}/{dataset_name}_{FEAT_TYPE}/train.csv'
    test_path = f'{SPLIT_DIR}/{dataset_name}_{FEAT_TYPE}/test.csv'
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    feat_cols = [c for c in df_train.columns if c not in ['cid', 'y', 'type']]
    X_train = df_train[feat_cols].values
    y_train = df_train['y'].values
    X_test = df_test[feat_cols].values
    y_test = df_test['y'].values
    return X_train, y_train, X_test, y_test

if MODEL_TYPE == 'global':
    dataset_name = 'Global'
elif MODEL_TYPE == 'c_only':
    dataset_name = 'C'
elif MODEL_TYPE == 'h_only':
    dataset_name = 'H'
else:
    raise ValueError("MODEL_TYPE must be 'global', 'c_only', or 'h_only'")

X_train, y_train, X_test, y_test = load_data(dataset_name)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 加载对应的分类器
model_path = f'{MODEL_DIR}/clf_{MODEL_TYPE}_lightgbm_{FEAT_TYPE}.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件不存在: {model_path}")
with open(model_path, 'rb') as f:
    clf = pickle.load(f)

# 计算马氏距离
print("计算马氏距离...")
mean_vec = np.mean(X_train, axis=0)
cov_mat = np.cov(X_train, rowvar=False)
cov_mat += np.eye(cov_mat.shape[0]) * 1e-6
cov_inv = np.linalg.pinv(cov_mat)

def mahalanobis_distances(X, mean, cov_inv):
    return np.array([mahalanobis(x, mean, cov_inv) for x in X])

train_dist = mahalanobis_distances(X_train, mean_vec, cov_inv)
test_dist = mahalanobis_distances(X_test, mean_vec, cov_inv)
threshold = np.percentile(train_dist, AD_PERCENTILE)

print(f"阈值 (95%): {threshold:.4f}")
print(f"训练集距离范围: [{train_dist.min():.4f}, {train_dist.max():.4f}]")
print(f"测试集距离范围: [{test_dist.min():.4f}, {test_dist.max():.4f}]")
print(f"训练集距离: [{train_dist}]")
print(f"测试集距离: [{test_dist}]")

# 评估 In-AD vs Out-of-AD
in_ad = test_dist <= threshold
out_ad = test_dist > threshold
print(f"In-AD: {np.sum(in_ad)} 样本 ({np.sum(in_ad)/len(test_dist)*100:.1f}%)")
print(f"Out-of-AD: {np.sum(out_ad)} 样本 ({np.sum(out_ad)/len(test_dist)*100:.1f}%)")

# 预测
y_pred = clf.predict(X_test)

def metrics(y_true, y_pred, label):
    return {
        'Group': label,
        'n': len(y_true),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced_Acc': balanced_accuracy_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }

all_metrics = metrics(y_test, y_pred, 'All')
in_metrics = metrics(y_test[in_ad], y_pred[in_ad], 'In-AD')
out_metrics = metrics(y_test[out_ad], y_pred[out_ad], 'Out-of-AD')

print("\n性能对比:")
print(f"{'Group':<10} {'Acc':<8} {'BalAcc':<8} {'MCC':<8} {'F1':<8} {'n':<6}")
print("-"*45)
for m in [all_metrics, in_metrics, out_metrics]:
    print(f"{m['Group']:<10} {m['Accuracy']:.4f}  {m['Balanced_Acc']:.4f}  {m['MCC']:.4f}  {m['F1']:.4f}  {m['n']:d}")

# 保存 AD 参数
ad_params = {
    'mean': mean_vec,
    'cov_inv': cov_inv,
    'threshold': threshold,
    'percentile': AD_PERCENTILE,
    'n_features': X_train.shape[1]
}
with open(f'{OUTPUT_DIR}/ad_params_{MODEL_TYPE}_{FEAT_TYPE}.pkl', 'wb') as f:
    pickle.dump(ad_params, f)
print(f"AD 参数已保存: ad_params_{MODEL_TYPE}_{FEAT_TYPE}.pkl")

# 保存性能结果
pd.DataFrame([all_metrics, in_metrics, out_metrics]).to_csv(
    f'{OUTPUT_DIR}/ad_performance_{MODEL_TYPE}_{FEAT_TYPE}.csv', index=False
)

# 绘制距离分布图
fig, ax = plt.subplots(figsize=(6, 4))

x_upper = 26000
# 直方图
ax.hist(train_dist, bins=100, alpha=0.6, color='steelblue', density=False,
        range=(0, x_upper), label=f'Training Set (n={len(train_dist)})')
ax.hist(test_dist, bins=100, alpha=0.5, color='orange', density=False,
        range=(0, x_upper), label=f'Test Set (n={len(test_dist)})')

# 阈值线
ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
           label=f'Threshold ({AD_PERCENTILE}%)')

# ---------- 关键修改 ----------
# 1. 设定合理的 x 轴上限（根据你的需求固定为 3，或自动取99%分位数）


# 2. 粉色区域填充到 x_upper（而非 test_dist.max()*1.05）
ax.axvspan(threshold, x_upper, alpha=0.1, color='red', label='Out-of-AD Region')

# 3. 强制 x 轴范围
ax.set_xlim(0, x_upper)
# -----------------------------

ax.set_xlabel('Mahalanobis Distance', fontsize=12, fontweight='bold')
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title(f'Applicability Domain: {MODEL_TYPE} Model ({FEAT_TYPE})', fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ad_distribution_{MODEL_TYPE}_{FEAT_TYPE}.pdf', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/ad_distribution_{MODEL_TYPE}_{FEAT_TYPE}.png', dpi=300)
plt.close()
print(f"距离分布图已保存至 {OUTPUT_DIR}/")
