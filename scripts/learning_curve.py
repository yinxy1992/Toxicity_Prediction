"""
learning_curve.py - 生成学习曲线（Global LightGBM）
功能：验证模型是否过拟合，展示随训练集大小增加，CV得分与训练得分的收敛趋势。
用法：直接运行（需依赖 formal_splits/scaled 目录）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from lightgbm import LGBMClassifier
import os
import pickle

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

# ==================== 路径配置 ====================
SPLIT_DIR = '../formal_splits/scaled'
OUTPUT_DIR = '../figures'  # 确保该文件夹存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 特征类型列表
FEAT_TYPES = ['descriptor', 'deep', 'combined']
FEAT_LABELS = ['F1 (RDKit)', 'F2 (KPGT)', 'F3 (Combined)']
RANDOM_STATE = 42

# ==================== 辅助函数 ====================
def load_best_params(feat_type):
    """加载之前gridsearch找到的LightGBM最佳参数（如果存在），否则使用稳健默认值"""
    pkl_path = f'../best_params/best_params_lightgbm_Global_{feat_type}.pkl'
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            params = pickle.load(f)
        clean_params = {k: v for k, v in params.items() if not k.startswith('pca__') and k != 'n_components'}
        return clean_params
    else:
        print(f"未找到最佳参数文件，使用默认参数 for {feat_type}")
        return {
            'n_estimators': 200,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
            'random_state': RANDOM_STATE,
            'verbose': -1
        }

def plot_learning_curve_single(ax, X, y, title, cv=5, n_jobs=-1):
    """绘制单个学习曲线"""
    # 获取当前参数（从外部传入，或在此处临时定义）
    # 注意：params 在外部主循环中定义，这里作为闭包使用
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=clf,
        X=X, y=y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=n_jobs,
        random_state=RANDOM_STATE
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color='orange')
    ax.plot(train_sizes, train_mean, 'o-', color='#2E86AB', linewidth=2, markersize=6, label='Training Score')
    ax.plot(train_sizes, test_mean, 'o-', color='#F18F01', linewidth=2, markersize=6, label='Cross-Validation Score')
    ax.set_xlabel('Training Examples', fontsize=11, fontweight='bold')
    ax.set_ylabel('Balanced Accuracy', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(0, X.shape[0] * 1.05)

# ==================== 主程序 ====================
print("开始生成学习曲线（独立文件）...")

for idx, feat in enumerate(FEAT_TYPES):
    print(f"\n处理特征集: {feat}")
    
    # 1. 加载全局训练集
    train_path = f'{SPLIT_DIR}/Global_{feat}/train.csv'
    df = pd.read_csv(train_path)
    feat_cols = [c for c in df.columns if c not in ['cid', 'y', 'type']]
    X = df[feat_cols].values
    y = df['y'].values
    
    # 2. 加载最佳参数并初始化分类器
    best_params = load_best_params(feat)
    global clf  # 让绘图函数能够访问
    clf = LGBMClassifier(**best_params, n_jobs=-1, verbose=-1, random_state=RANDOM_STATE)
    
    # 3. 创建独立的图 (8x6 英寸)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_learning_curve_single(ax, X, y, FEAT_LABELS[idx])
    
    # 4. 保存为独立的 PDF/PNG
    pdf_path = f'{OUTPUT_DIR}/Figure_S2_LearningCurve_{feat}.pdf'
    png_path = f'{OUTPUT_DIR}/Figure_S2_LearningCurve_{feat}.png'
    
    plt.tight_layout()
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    plt.close(fig)  # 重要：关闭图形释放内存
    
    print(f"  已保存: {pdf_path}")

print("\n所有学习曲线生成完毕！")
