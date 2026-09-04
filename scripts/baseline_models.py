"""
baseline_models.py - ECFP + LR/SVM 基线评估（全局预留测试集，公平对比分层与全局策略）
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, f1_score
)
import os
import warnings
warnings.filterwarnings('ignore')

# ===================================================================
# 配置路径
# ===================================================================
SPLIT_DIR = '../formal_splits'
RAW_DATA_DIR = '../全部化合物_原始特征'
OUTPUT_DIR = '../baseline_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# 原始特征文件（任选一个，包含 SMILES 和 dose）
ORIGINAL_FILE = '全部化合物_描述符特征.xlsx'  # 或使用其他，确保包含 cid, smiles, dose

# 基线模型
BASELINE_MODELS = {
    'Logistic_Regression': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, C=1.0),
    'SVM_RBF': SVC(kernel='rbf', probability=False, random_state=RANDOM_STATE, C=1.0, gamma='scale')
}

# ===================================================================
# 辅助函数
# ===================================================================
def load_cid_list(file_path):
    """加载 CID 列表（从 CSV）"""
    df = pd.read_csv(file_path)
    return set(df['cid'].astype(str).values)

def get_type_mapping():
    """从已分好的六个文件中获取每个 CID 的取代基类型（'C' 或 'H'）"""
    mapping = {}
    for prefix in ['碳', '杂']:
        for suffix in ['描述符', '深度', '深度加描述符']:
            fname = f'{RAW_DATA_DIR}/{prefix}_{suffix}.xlsx'
            if os.path.exists(fname):
                df = pd.read_excel(fname)
                df['cid'] = df['cid'].astype(str)
                t = 'C' if prefix == '碳' else 'H'
                for cid in df['cid'].values:
                    mapping[cid] = t
    return mapping

def smiles_to_ecfp(smiles_list, radius=2, n_bits=1024):
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(np.zeros(n_bits))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            features.append(np.array(fp))
    return np.array(features)

# ===================================================================
# 主程序
# ===================================================================
print("="*60)
print("ECFP 基线模型评估 (全局预留测试集, 公平对比)")
print("="*60)

# 1. 加载全局训练/测试 ID
train_ids = load_cid_list(f'{SPLIT_DIR}/global_train_ids.csv')
test_ids = load_cid_list(f'{SPLIT_DIR}/global_test_ids.csv')
print(f"全局训练 CID 数: {len(train_ids)}, 全局测试 CID 数: {len(test_ids)}")

# 2. 加载原始数据并划分
df_raw = pd.read_excel(f'{RAW_DATA_DIR}/{ORIGINAL_FILE}')
df_raw['cid'] = df_raw['cid'].astype(str)
df_train = df_raw[df_raw['cid'].isin(train_ids)]
df_test = df_raw[df_raw['cid'].isin(test_ids)]

# 提取 SMILES 和标签
train_smiles = df_train['smiles'].values
test_smiles = df_test['smiles'].values
y_train = np.where(df_train['dose'].values < 500, 1, 0)
y_test = np.where(df_test['dose'].values < 500, 1, 0)

print(f"训练集样本数: {len(y_train)}, 正例比例: {np.mean(y_train):.3f}")
print(f"测试集样本数: {len(y_test)}, 正例比例: {np.mean(y_test):.3f}")

# 3. 获取每个 CID 的类型
type_mapping = get_type_mapping()
train_types = df_train['cid'].map(type_mapping).values
test_types = df_test['cid'].map(type_mapping).values

# 统计训练集中 C/H 分布
train_c_mask = (train_types == 'C')
train_h_mask = (train_types == 'H')
print(f"训练集中 C 类: {np.sum(train_c_mask)}, H 类: {np.sum(train_h_mask)}")

# 测试集中 C/H 分布
test_c_mask = (test_types == 'C')
test_h_mask = (test_types == 'H')
print(f"测试集中 C 类: {np.sum(test_c_mask)}, H 类: {np.sum(test_h_mask)}")

# 4. 计算 ECFP
print("\n计算 ECFP 指纹...")
X_train_raw = smiles_to_ecfp(train_smiles)
X_test_raw = smiles_to_ecfp(test_smiles)
print(f"ECFP 维度: {X_train_raw.shape[1]}")

# 5. 标准化（仅拟合训练集）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# 6. 训练各个模型
all_results = []

for model_name, clf in BASELINE_MODELS.items():
    print(f"\n{'='*50}")
    print(f"训练基线模型: {model_name}")
    print(f"{'='*50}")

    # ---- 6.1 全局模型（用所有训练数据） ----
    print("\n  >> 训练 Global 模型...")
    global_clf = clone(clf)
    global_clf.fit(X_train, y_train)
    y_pred_global = global_clf.predict(X_test)
    res_global = {
        'Strategy': 'Global',
        'Model': model_name,
        'Subset': 'All',
        'Accuracy': accuracy_score(y_test, y_pred_global),
        'Balanced_Acc': balanced_accuracy_score(y_test, y_pred_global),
        'MCC': matthews_corrcoef(y_test, y_pred_global),
        'F1': f1_score(y_test, y_pred_global),
        'n_samples': len(y_test)
    }
    all_results.append(res_global)
    print(f"     Acc: {res_global['Accuracy']:.4f}, BalAcc: {res_global['Balanced_Acc']:.4f}, MCC: {res_global['MCC']:.4f}")

    # ---- 6.2 C-only 模型（仅用训练集中的 C 类） ----
    if np.sum(train_c_mask) > 0:
        print("\n  >> 训练 C-only 模型...")
        c_clf = clone(clf)
        c_clf.fit(X_train[train_c_mask], y_train[train_c_mask])
        # 在测试集中的 C 类上评估
        if np.sum(test_c_mask) > 0:
            y_pred_c = c_clf.predict(X_test[test_c_mask])
            y_true_c = y_test[test_c_mask]
            res_c = {
                'Strategy': 'C-Only',
                'Model': model_name,
                'Subset': 'C',
                'Accuracy': accuracy_score(y_true_c, y_pred_c),
                'Balanced_Acc': balanced_accuracy_score(y_true_c, y_pred_c),
                'MCC': matthews_corrcoef(y_true_c, y_pred_c),
                'F1': f1_score(y_true_c, y_pred_c),
                'n_samples': len(y_true_c)
            }
            all_results.append(res_c)
            print(f"     Acc: {res_c['Accuracy']:.4f}, BalAcc: {res_c['Balanced_Acc']:.4f}, MCC: {res_c['MCC']:.4f}")
        else:
            print("     测试集中无 C 类样本，跳过评估。")
    else:
        print("     训练集中无 C 类样本，跳过训练。")

    # ---- 6.3 H-only 模型（仅用训练集中的 H 类） ----
    if np.sum(train_h_mask) > 0:
        print("\n  >> 训练 H-only 模型...")
        h_clf = clone(clf)
        h_clf.fit(X_train[train_h_mask], y_train[train_h_mask])
        if np.sum(test_h_mask) > 0:
            y_pred_h = h_clf.predict(X_test[test_h_mask])
            y_true_h = y_test[test_h_mask]
            res_h = {
                'Strategy': 'H-Only',
                'Model': model_name,
                'Subset': 'H',
                'Accuracy': accuracy_score(y_true_h, y_pred_h),
                'Balanced_Acc': balanced_accuracy_score(y_true_h, y_pred_h),
                'MCC': matthews_corrcoef(y_true_h, y_pred_h),
                'F1': f1_score(y_true_h, y_pred_h),
                'n_samples': len(y_true_h)
            }
            all_results.append(res_h)
            print(f"     Acc: {res_h['Accuracy']:.4f}, BalAcc: {res_h['Balanced_Acc']:.4f}, MCC: {res_h['MCC']:.4f}")
        else:
            print("     测试集中无 H 类样本，跳过评估。")
    else:
        print("     训练集中无 H 类样本，跳过训练。")

    # ---- 6.4 Stratified 策略（组合 C-only 和 H-only 预测） ----
    print("\n  >> 训练 Stratified 策略...")
    # 确保两个子模型都存在
    if (np.sum(train_c_mask) > 0 and np.sum(train_h_mask) > 0 and
        np.sum(test_c_mask) > 0 and np.sum(test_h_mask) > 0):
        # 使用之前训练好的 c_clf 和 h_clf（如果已训练）
        # 但为了避免变量未定义，重新训练
        c_clf_strat = clone(clf)
        h_clf_strat = clone(clf)
        c_clf_strat.fit(X_train[train_c_mask], y_train[train_c_mask])
        h_clf_strat.fit(X_train[train_h_mask], y_train[train_h_mask])

        # 分别预测
        y_pred_c_strat = c_clf_strat.predict(X_test[test_c_mask])
        y_pred_h_strat = h_clf_strat.predict(X_test[test_h_mask])

        # 合并（按原始顺序）
        y_pred_strat = np.zeros_like(y_test)
        y_pred_strat[test_c_mask] = y_pred_c_strat
        y_pred_strat[test_h_mask] = y_pred_h_strat

        res_strat = {
            'Strategy': 'Stratified',
            'Model': model_name,
            'Subset': 'All',
            'Accuracy': accuracy_score(y_test, y_pred_strat),
            'Balanced_Acc': balanced_accuracy_score(y_test, y_pred_strat),
            'MCC': matthews_corrcoef(y_test, y_pred_strat),
            'F1': f1_score(y_test, y_pred_strat),
            'n_samples': len(y_test)
        }
        all_results.append(res_strat)
        print(f"     Acc: {res_strat['Accuracy']:.4f}, BalAcc: {res_strat['Balanced_Acc']:.4f}, MCC: {res_strat['MCC']:.4f}")
    else:
        print("     无法构建 Stratified 策略：训练或测试集中缺少某一类。")

# 7. 保存结果
df_results = pd.DataFrame(all_results)
df_results.to_csv(f'{OUTPUT_DIR}/baseline_performance_global_test_fixed.csv', index=False)

print("\n" + "="*60)
print("基线结果汇总 (全局测试集)")
print("="*60)
print(df_results.sort_values(['Model', 'Strategy']).to_string(index=False))

print(f"\n结果已保存至: {OUTPUT_DIR}/baseline_performance_global_test_fixed.csv")
