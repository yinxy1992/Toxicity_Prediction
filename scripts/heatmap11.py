import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -------------------------- 1. 数据整理（与原始表格一致）--------------------------
test_acc_data = {
    'Data1': {
        'F1': {'RF': 0.750, 'LGB': 0.889, 'CatBoost': 0.889, 'Hard_voting': 0.889, 'Soft_voting': 0.880},
        'F2': {'RF': 0.815, 'LGB': 0.870, 'CatBoost': 0.861, 'Hard_voting': 0.861, 'Soft_voting': 0.861},
        'F3': {'RF': 0.833, 'LGB': 0.880, 'CatBoost': 0.870, 'Hard_voting': 0.889, 'Soft_voting': 0.880}
    },
    'Data2': {
        'F1': {'RF': 0.794, 'LGB': 0.857, 'CatBoost': 0.841, 'Hard_voting': 0.841, 'Soft_voting': 0.841},
        'F2': {'RF': 0.762, 'LGB': 0.841, 'CatBoost': 0.825, 'Hard_voting': 0.825, 'Soft_voting': 0.825},
        'F3': {'RF': 0.778, 'LGB': 0.841, 'CatBoost': 0.841, 'Hard_voting': 0.841, 'Soft_voting': 0.841}
    },
    'Data3': {
        'F1': {'RF': 0.822, 'LGB': 0.844, 'CatBoost': 0.844, 'Hard_voting': 0.889, 'Soft_voting': 0.889},
        'F2': {'RF': 0.889, 'LGB': 0.844, 'CatBoost': 0.911, 'Hard_voting': 0.911, 'Soft_voting': 0.911},
        'F3': {'RF': 0.889, 'LGB': 0.867, 'CatBoost': 0.844, 'Hard_voting': 0.867, 'Soft_voting': 0.889}
    }
}

models = ['RF', 'CatBoost', 'LGB', 'Hard_voting', 'Soft_voting']
features = ['F1', 'F2', 'F3']
datasets = ['Data1', 'Data2', 'Data3']

# -------------------------- 2. 全局绘图配置（学术论文风格）--------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0

colors = {
    'F1': '#2E86AB',
    'F2': '#A23B72',
    'F3': '#F18F01',
    'models': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'],
    'heatmap': LinearSegmentedColormap.from_list('custom_cmap', ['#F8F9FA', '#2E86AB'], N=100)
}

# -------------------------- 3. 绘制热力图（独立输出）--------------------------
fig, ax = plt.subplots(figsize=(8, 3), dpi=300, constrained_layout=True)

# 构建热力图数据（行=模型，列=数据集-特征组合）
heatmap_data = []
x_labels = [f'{d}-{f}' for d in datasets for f in features]
for m in models:
    row = [test_acc_data[d][f][m] for d in datasets for f in features]
    heatmap_data.append(row)
heatmap_data = np.array(heatmap_data)

# 绘制热力图
im = ax.imshow(heatmap_data, cmap=colors['heatmap'], aspect='auto', vmin=0.75, vmax=0.95)

# 坐标轴标签
ax.set_xticks(np.arange(len(x_labels)))
ax.set_yticks(np.arange(len(models)))
ax.set_xticklabels(x_labels, rotation=20, ha='center', fontsize=12)
ax.set_yticklabels(models, fontsize=12)

# 数值标注
for i in range(len(models)):
    for j in range(len(x_labels)):
        ax.text(j, i, f'{heatmap_data[i, j]:.4f}', ha='center', va='center',
                color='black', fontsize=10)

# 颜色条
cbar = plt.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label('Test Accuracy', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# 保存图片（可选）
fig.savefig('../figures/accuracy_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('../figures/accuracy_heatmap.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
