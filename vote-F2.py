"""
vote.py - 最终模型训练与评估（严格验证，支持 Soft/Hard Voting）
====================================================================
功能：
- 使用 GridSearch 得到的最佳超参数训练模型（参数按数据集分别加载）
- 分别对 Global、C、H 三个数据集进行独立的训练和测试集评估
- 同时，在同一个全局测试集上对比 Global 模型 vs Stratified 模型（C-only+H-only拼接）
- 使用 CID 进行精确对齐，确保 Stratified 预测正确
- 输出所有评估指标及 Bootstrap 置信区间
- 支持 Soft Voting 和 Hard Voting

用法：
1. 修改 FEAT_TYPE 为 'descriptor' / 'deep' / 'combined' 分别运行
2. 运行: python vote.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef, f1_score,
    recall_score, precision_score, confusion_matrix
)
from sklearn.utils import resample
import os
import pickle
import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# ==================== 配置区域 ====================
FEAT_TYPE = 'deep'   # 可选: 'descriptor', 'deep', 'combined'
# ==================================================

SPLIT_DIR = '../formal_splits/scaled'
BEST_PARAMS_DIR = '../best_params'
OUTPUT_DIR = '../final_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_JOBS = max(1, int(os.cpu_count() * 0.85))
RANDOM_STATE = 42
N_BOOTSTRAP = 1000

# 日志（文件名包含特征类型）
log_filename = f'{OUTPUT_DIR}/vote_{FEAT_TYPE}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
)
logging.info(f"{'='*60}")
logging.info(f"特征集: {FEAT_TYPE}")
logging.info(f"输出文件将包含特征类型: final_results_{FEAT_TYPE}.csv, bootstrap_ci_{FEAT_TYPE}.csv")
logging.info(f"{'='*60}")

# ---------- 辅助函数 ----------
def load_data_with_cid(dataset_name):
    """
    加载标准化后的数据，并返回 CID 列表
    dataset_name: 'Global', 'C', 'H'
    返回: (X_train, y_train, X_test, y_test, cids_train, cids_test, df_train, df_test)
    """
    train_path = f'{SPLIT_DIR}/{dataset_name}_{FEAT_TYPE}/train.csv'
    test_path = f'{SPLIT_DIR}/{dataset_name}_{FEAT_TYPE}/test.csv'
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    feat_cols = [c for c in df_train.columns if c not in ['cid', 'y', 'type']]
    X_train = df_train[feat_cols].values
    y_train = df_train['y'].values
    X_test = df_test[feat_cols].values
    y_test = df_test['y'].values
    cids_train = df_train['cid'].values
    cids_test = df_test['cid'].values
    return X_train, y_train, X_test, y_test, cids_train, cids_test, df_train, df_test

def load_params(model_name, dataset_name):
    """加载对应数据集和模型的最佳参数"""
    pkl_path = f'{BEST_PARAMS_DIR}/best_params_{model_name}_{dataset_name}_{FEAT_TYPE}.pkl'
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"参数文件不存在: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        params = pickle.load(f)
    clean = {k: v for k, v in params.items() if not k.startswith('pca__') and k != 'n_components'}
    return clean

def build_classifier(model_name, params):
    """根据模型名和参数实例化分类器"""
    if model_name == 'randomforest':
        return RandomForestClassifier(**params, random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=0)
    elif model_name == 'lightgbm':
        return LGBMClassifier(**params, random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1)
    elif model_name == 'catboost':
        return CatBoostClassifier(**params, random_seed=RANDOM_STATE, thread_count=N_JOBS, verbose=False)
    else:
        raise ValueError(f"未知模型: {model_name}")

def evaluate(y_true, y_pred, strategy, model_name, subset='All'):
    """计算各项指标"""
    return {
        'Strategy': strategy,
        'Model': model_name,
        'Subset': subset,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced_Acc': balanced_accuracy_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Confusion_Matrix': confusion_matrix(y_true, y_pred).tolist(),
        'n_samples': len(y_true)
    }

def bootstrap_ci(y_true, pred1, pred2, n_bootstrap=1000, alpha=0.05):
    """Bootstrap 置信区间 (差值)"""
    n = len(y_true)
    diffs = []
    for _ in range(n_bootstrap):
        idx = resample(range(n), replace=True, n_samples=n)
        acc1 = accuracy_score(y_true[idx], pred1[idx])
        acc2 = accuracy_score(y_true[idx], pred2[idx])
        diffs.append(acc1 - acc2)
    lower = np.percentile(diffs, 100 * (alpha / 2))
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    p_value = 2 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0))
    return lower, upper, p_value, np.mean(diffs)

# ---------- 主流程 ----------
def main():
    # 1. 加载所有数据（带 CID）
    logging.info("加载数据...")
    X_train_g, y_train_g, X_test_g, y_test_g, cids_train_g, cids_test_g, df_train_g, df_test_g = load_data_with_cid('Global')
    X_train_c, y_train_c, X_test_c, y_test_c, cids_train_c, cids_test_c, df_train_c, df_test_c = load_data_with_cid('C')
    X_train_h, y_train_h, X_test_h, y_test_h, cids_train_h, cids_test_h, df_train_h, df_test_h = load_data_with_cid('H')

    logging.info(f"Global 训练集: {X_train_g.shape}, 测试集: {X_test_g.shape}")
    logging.info(f"C 训练集: {X_train_c.shape}, 测试集: {X_test_c.shape}")
    logging.info(f"H 训练集: {X_train_h.shape}, 测试集: {X_test_h.shape}")

    # 2. 加载各数据集的最佳参数
    logging.info("加载最佳参数...")
    params_rf_g = load_params('randomforest', 'Global')
    params_lgbm_g = load_params('lightgbm', 'Global')
    params_cb_g = load_params('catboost', 'Global')
    params_rf_c = load_params('randomforest', 'C')
    params_lgbm_c = load_params('lightgbm', 'C')
    params_cb_c = load_params('catboost', 'C')
    params_rf_h = load_params('randomforest', 'H')
    params_lgbm_h = load_params('lightgbm', 'H')
    params_cb_h = load_params('catboost', 'H')

    # 3. 训练模型
    logging.info("训练模型...")
    # Global
    rf_g = build_classifier('randomforest', params_rf_g)
    lgbm_g = build_classifier('lightgbm', params_lgbm_g)
    cb_g = build_classifier('catboost', params_cb_g)
    rf_g.fit(X_train_g, y_train_g)
    lgbm_g.fit(X_train_g, y_train_g)
    cb_g.fit(X_train_g, y_train_g)
    vote_g_soft = VotingClassifier(estimators=[('rf', rf_g), ('lgbm', lgbm_g), ('cb', cb_g)], voting='soft')
    vote_g_hard = VotingClassifier(estimators=[('rf', rf_g), ('lgbm', lgbm_g), ('cb', cb_g)], voting='hard')
    vote_g_soft.fit(X_train_g, y_train_g)
    vote_g_hard.fit(X_train_g, y_train_g)
    models_g = {'RF': rf_g, 'LGBM': lgbm_g, 'CatBoost': cb_g, 'SoftVoting': vote_g_soft, 'HardVoting': vote_g_hard}

    # C-only
    rf_c = build_classifier('randomforest', params_rf_c)
    lgbm_c = build_classifier('lightgbm', params_lgbm_c)
    cb_c = build_classifier('catboost', params_cb_c)
    rf_c.fit(X_train_c, y_train_c)
    lgbm_c.fit(X_train_c, y_train_c)
    cb_c.fit(X_train_c, y_train_c)
    vote_c_soft = VotingClassifier(estimators=[('rf', rf_c), ('lgbm', lgbm_c), ('cb', cb_c)], voting='soft')
    vote_c_hard = VotingClassifier(estimators=[('rf', rf_c), ('lgbm', lgbm_c), ('cb', cb_c)], voting='hard')
    vote_c_soft.fit(X_train_c, y_train_c)
    vote_c_hard.fit(X_train_c, y_train_c)
    models_c = {'RF': rf_c, 'LGBM': lgbm_c, 'CatBoost': cb_c, 'SoftVoting': vote_c_soft, 'HardVoting': vote_c_hard}

    # H-only
    rf_h = build_classifier('randomforest', params_rf_h)
    lgbm_h = build_classifier('lightgbm', params_lgbm_h)
    cb_h = build_classifier('catboost', params_cb_h)
    rf_h.fit(X_train_h, y_train_h)
    lgbm_h.fit(X_train_h, y_train_h)
    cb_h.fit(X_train_h, y_train_h)
    vote_h_soft = VotingClassifier(estimators=[('rf', rf_h), ('lgbm', lgbm_h), ('cb', cb_h)], voting='soft')
    vote_h_hard = VotingClassifier(estimators=[('rf', rf_h), ('lgbm', lgbm_h), ('cb', cb_h)], voting='hard')
    vote_h_soft.fit(X_train_h, y_train_h)
    vote_h_hard.fit(X_train_h, y_train_h)
    models_h = {'RF': rf_h, 'LGBM': lgbm_h, 'CatBoost': cb_h, 'SoftVoting': vote_h_soft, 'HardVoting': vote_h_hard}

    # 4. 独立子集评估（每个数据集自己的测试集）
    logging.info("执行独立子集评估...")
    all_results = []
    model_names = ['RF', 'LGBM', 'CatBoost', 'SoftVoting', 'HardVoting']

    # Global 测试集
    for name in model_names:
        pred = models_g[name].predict(X_test_g)
        all_results.append(evaluate(y_test_g, pred, 'Global', name, 'Global'))

    # C 测试集
    for name in model_names:
        pred = models_c[name].predict(X_test_c)
        all_results.append(evaluate(y_test_c, pred, 'C-only', name, 'C'))

    # H 测试集
    for name in model_names:
        pred = models_h[name].predict(X_test_h)
        all_results.append(evaluate(y_test_h, pred, 'H-only', name, 'H'))

    # 5. 统一全局对比（Global 模型 vs Stratified 模型，在同一个全局测试集上）
    logging.info("执行统一全局对比（Global 测试集）...")

    # 强制将 CID 转换为字符串，确保字典键匹配
    cids_test_g_str = cids_test_g.astype(str)
    cids_test_c_str = cids_test_c.astype(str)
    cids_test_h_str = cids_test_h.astype(str)

    strat_preds = {}
    for name in model_names:
        pred_c = models_c[name].predict(X_test_c)
        pred_h = models_h[name].predict(X_test_h)
        # 构建 CID -> 预测的字典
        cid_to_pred = {}
        for cid, p in zip(cids_test_c_str, pred_c):
            cid_to_pred[cid] = p
        for cid, p in zip(cids_test_h_str, pred_h):
            cid_to_pred[cid] = p
        # 按 Global 测试集 CID 顺序提取预测
        try:
            strat_pred = np.array([cid_to_pred[cid] for cid in cids_test_g_str])
        except KeyError as e:
            logging.error(f"映射错误: CID {e} 不在字典中，请检查数据类型")
            raise
        strat_preds[name] = strat_pred
        all_results.append(evaluate(y_test_g, strat_pred, 'Stratified', name, 'Global'))

    # 6. Bootstrap 置信区间 (CatBoost)
    logging.info("计算 Bootstrap 置信区间 (CatBoost)...")
    pred_global_cb = models_g['CatBoost'].predict(X_test_g)
    pred_strat_cb = strat_preds['CatBoost']
    acc_lower, acc_upper, acc_p, acc_mean = bootstrap_ci(
        y_test_g, pred_global_cb, pred_strat_cb, n_bootstrap=N_BOOTSTRAP
    )
    diff_acc = accuracy_score(y_test_g, pred_global_cb) - accuracy_score(y_test_g, pred_strat_cb)

    bal_diffs = []
    for _ in range(N_BOOTSTRAP):
        idx = resample(range(len(y_test_g)), replace=True, n_samples=len(y_test_g))
        bal1 = balanced_accuracy_score(y_test_g[idx], pred_global_cb[idx])
        bal2 = balanced_accuracy_score(y_test_g[idx], pred_strat_cb[idx])
        bal_diffs.append(bal1 - bal2)
    bal_lower = np.percentile(bal_diffs, 2.5)
    bal_upper = np.percentile(bal_diffs, 97.5)
    bal_p = 2 * min(np.mean(np.array(bal_diffs) <= 0), np.mean(np.array(bal_diffs) >= 0))
    diff_bal = balanced_accuracy_score(y_test_g, pred_global_cb) - balanced_accuracy_score(y_test_g, pred_strat_cb)

    # 7. 保存模型和结果（文件名包含特征类型）
    with open(f'{OUTPUT_DIR}/clf_global_catboost_{FEAT_TYPE}.pkl', 'wb') as f:
        pickle.dump(cb_g, f)
    with open(f'{OUTPUT_DIR}/clf_c_only_catboost_{FEAT_TYPE}.pkl', 'wb') as f:
        pickle.dump(cb_c, f)
    with open(f'{OUTPUT_DIR}/clf_h_only_catboost_{FEAT_TYPE}.pkl', 'wb') as f:
        pickle.dump(cb_h, f)
    with open(f'{OUTPUT_DIR}/clf_global_lightgbm_{FEAT_TYPE}.pkl', 'wb') as f:
        pickle.dump(lgbm_g, f)
    with open(f'{OUTPUT_DIR}/clf_c_only_lightgbm_{FEAT_TYPE}.pkl', 'wb') as f:
        pickle.dump(lgbm_c, f)
    with open(f'{OUTPUT_DIR}/clf_h_only_lightgbm_{FEAT_TYPE}.pkl', 'wb') as f:
        pickle.dump(lgbm_h, f)
    with open(f'{OUTPUT_DIR}/clf_global_randomforest_{FEAT_TYPE}.pkl', 'wb') as f:
        pickle.dump(rf_g, f)
    with open(f'{OUTPUT_DIR}/clf_c_only_randomforest_{FEAT_TYPE}.pkl', 'wb') as f:
        pickle.dump(rf_c, f)
    with open(f'{OUTPUT_DIR}/clf_h_only_randomforest_{FEAT_TYPE}.pkl', 'wb') as f:
        pickle.dump(rf_h, f)
    with open(f'{OUTPUT_DIR}/clf_global_softvote_{FEAT_TYPE}.pkl', 'wb') as f:
        pickle.dump(vote_g_soft, f)

    # 保存完整结果表（文件名包含 FEAT_TYPE）
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(f'{OUTPUT_DIR}/final_results_{FEAT_TYPE}.csv', index=False)
    logging.info(f"结果表已保存: final_results_{FEAT_TYPE}.csv")

    # 保存 Bootstrap CI
    bootstrap_df = pd.DataFrame([{
        'Metric': 'Accuracy',
        'Global_Score': accuracy_score(y_test_g, pred_global_cb),
        'Stratified_Score': accuracy_score(y_test_g, pred_strat_cb),
        'Difference_(G-S)': diff_acc,
        'CI_2.5%': acc_lower,
        'CI_97.5%': acc_upper,
        'p_value': acc_p,
        'Mean_Diff': acc_mean
    }, {
        'Metric': 'Balanced_Accuracy',
        'Global_Score': balanced_accuracy_score(y_test_g, pred_global_cb),
        'Stratified_Score': balanced_accuracy_score(y_test_g, pred_strat_cb),
        'Difference_(G-S)': diff_bal,
        'CI_2.5%': bal_lower,
        'CI_97.5%': bal_upper,
        'p_value': bal_p,
        'Mean_Diff': np.mean(bal_diffs)
    }])
    bootstrap_df.to_csv(f'{OUTPUT_DIR}/bootstrap_ci_{FEAT_TYPE}.csv', index=False)
    logging.info(f"Bootstrap CI 已保存: bootstrap_ci_{FEAT_TYPE}.csv")

    # 打印关键结果
    logging.info(f"\n{'='*60}")
    logging.info("独立子集评估结果 (CatBoost):")
    logging.info(f"  Global 数据集测试集准确率: {accuracy_score(y_test_g, models_g['CatBoost'].predict(X_test_g)):.4f}")
    logging.info(f"  C 数据集测试集准确率: {accuracy_score(y_test_c, models_c['CatBoost'].predict(X_test_c)):.4f}")
    logging.info(f"  H 数据集测试集准确率: {accuracy_score(y_test_h, models_h['CatBoost'].predict(X_test_h)):.4f}")
    logging.info(f"\n统一全局对比 (Global 测试集):")
    logging.info(f"  Global 模型准确率: {accuracy_score(y_test_g, pred_global_cb):.4f}")
    logging.info(f"  Stratified 模型准确率: {accuracy_score(y_test_g, pred_strat_cb):.4f}")
    logging.info(f"  差值 (Global - Stratified): {diff_acc:.4f} [95% CI: {acc_lower:.4f}, {acc_upper:.4f}], p={acc_p:.4f}")
    logging.info(f"  Balanced Acc 差值: {diff_bal:.4f} [95% CI: {bal_lower:.4f}, {bal_upper:.4f}], p={bal_p:.4f}")

if __name__ == "__main__":
    main()