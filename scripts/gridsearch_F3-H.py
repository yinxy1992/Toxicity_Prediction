# ==================== 服务器运行变量（请在此处修改） ====================
DATASET = 'H'      # 可选: 'Global', 'C', 'H'
FEAT = 'combined'     # 可选: 'descriptor', 'deep', 'combined'
# ======================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    f1_score, recall_score, precision_score, roc_auc_score,
    make_scorer
)
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# 路径
SCALED_DIR = '../formal_splits/scaled'
OUTPUT_MODEL_DIR = '../best_params'
os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42
N_JOBS = max(1, int(os.cpu_count() * 0.85))

print(f"当前任务: {DATASET} + {FEAT}")
print(f"使用CPU核心: {N_JOBS}")

# 加载已标准化的数据
train_path = f'{SCALED_DIR}/{DATASET}_{FEAT}/train.csv'
test_path = f'{SCALED_DIR}/{DATASET}_{FEAT}/test.csv'
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# 特征列（排除cid, y, type）
feature_cols = [c for c in df_train.columns if c not in ['cid', 'y', 'type']]
X_train = df_train[feature_cols].values
y_train = df_train['y'].values
X_test = df_test[feature_cols].values
y_test = df_test['y'].values

print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
print(f"训练集正例比例: {np.mean(y_train):.3f}, 测试集正例比例: {np.mean(y_test):.3f}")

# ==================== 超参数网格（完全参照您附件的gridsearch.py） ====================
param_grid_rf = {
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 5, 7, 10, 15],
    'min_samples_split': [10, 15, 20, 25],
    'min_samples_leaf': [8, 10, 12, 15],
    'max_features': [0.7, 0.8, 0.9, 1],
    'bootstrap': [True],
    'max_samples': [0.7, 0.8, 0.9, 1],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}

param_grid_lgbm = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5],
    'min_child_samples': [10, 20, 30]
}

param_grid_cb = {
    'depth': [4, 5, 6],
    'l2_leaf_reg': [1, 3, 5, 10],
    'border_count': [32, 64, 128],
    'bagging_temperature': [0, 0.5, 1]
}

models_config = {
    'RandomForest': (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=0), param_grid_rf),
    'LightGBM': (LGBMClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS, verbose=-1), param_grid_lgbm),
    'CatBoost': (CatBoostClassifier(random_seed=RANDOM_STATE, thread_count=N_JOBS, verbose=False), param_grid_cb)
}

# ==================== 评估指标定义 ====================
scoring_metrics = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'mcc': make_scorer(matthews_corrcoef),
    'f1': 'f1',
    'recall': 'recall',
    'precision': 'precision',
    'roc_auc': 'roc_auc'
}

# ==================== 执行网格搜索 ====================
best_params = {}
cv_scores_summary = {}

for model_name, (model, param_grid) in models_config.items():
    print(f"\n{'='*60}")
    print(f"正在搜索: {model_name} for {DATASET}/{FEAT}")
    print(f"{'='*60}")
    
    # GridSearchCV（这里没有Pipeline，因为数据已标准化）
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='balanced_accuracy',
        n_jobs=N_JOBS,
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    
    best = grid_search.best_params_
    best_params[model_name] = best
    print(f"最佳参数: {best}")
    print(f"最佳CV平衡准确率: {grid_search.best_score_:.4f}")
    
    # 使用最佳模型进行5折交叉验证（输出详尽指标）
    best_estimator = grid_search.best_estimator_
    cv_results = cross_validate(
        best_estimator, X_train, y_train, cv=5,
        scoring=scoring_metrics,
        return_train_score=False,
        n_jobs=N_JOBS
    )
    
    print(f"\n{model_name} 5折CV指标 (mean ± std):")
    for metric in ['accuracy', 'balanced_accuracy', 'mcc', 'f1', 'recall', 'precision', 'roc_auc']:
        scores = cv_results[f'test_{metric}']
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
        cv_scores_summary[f'{model_name}_{metric}'] = (mean_val, std_val)
    
    # 在最终预留测试集上评估（严格一次）
    y_pred = best_estimator.predict(X_test)
    print(f"\n{model_name} 在全局预留测试集上的最终表现:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Balanced Acc: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"  MCC: {matthews_corrcoef(y_test, y_pred):.4f}")
    print(f"  F1: {f1_score(y_test, y_pred):.4f}")
    print(f"  Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    
    # 保存最佳参数
    with open(f'{OUTPUT_MODEL_DIR}/best_params_{model_name.lower()}_{DATASET}_{FEAT}.pkl', 'wb') as f:
        pickle.dump(best, f)

print(f"\n所有最佳参数已保存至 {OUTPUT_MODEL_DIR}/")
