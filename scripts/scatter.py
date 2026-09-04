"""
scatter.py - 大鼠 vs 小鼠 LD50 相关性分析
============================================
功能：
- 读取包含大鼠和小鼠 LD50 数据的 Excel 文件（mouse_vs_rat.xlsx）
- 计算 Pearson 和 Spearman 相关系数
- 绘制散点图并显示回归线

用法：
1. 确保 mouse_vs_rat.xlsx 在正确路径（可配置）
2. 运行: python scatter.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import os

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
DATA_FILE = '../data/mouse_vs_rat_doses.xlsx' 
OUTPUT_DIR = '../figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ==============================================

# 读取数据
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"数据文件不存在: {DATA_FILE}")
df = pd.read_excel(DATA_FILE)

print(df.columns)
rat_col = 'rat'
mouse_col = 'mouse'
if rat_col not in df.columns or mouse_col not in df.columns:
    raise ValueError("请检查列名，需要包含 rat_LD50 和 mouse_LD50")

# 删除缺失值
df_clean = df[[rat_col, mouse_col]].dropna()
rat_vals = df_clean[rat_col].values
mouse_vals = df_clean[mouse_col].values

# 计算相关系数
pearson_r, pearson_p = pearsonr(rat_vals, mouse_vals)
spearman_r, spearman_p = spearmanr(rat_vals, mouse_vals)

print(f"样本数: {len(rat_vals)}")
print(f"Pearson r = {pearson_r:.4f}, p = {pearson_p:.4e}")
print(f"Spearman rho = {spearman_r:.4f}, p = {spearman_p:.4e}")

# 绘图
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(rat_vals, mouse_vals, alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
# 拟合线性回归（仅用于可视化）
z = np.polyfit(rat_vals, mouse_vals, 1)
p = np.poly1d(z)
x_line = np.linspace(min(rat_vals), max(rat_vals), 100)
ax.plot(x_line, p(x_line), 'r--', linewidth=1.5, label=f'Linear fit (R²={pearson_r**2:.3f})')

ax.set_xlabel('Rat Oral LD50 (mg/kg)', fontsize=12, fontweight='bold')
ax.set_ylabel('Mouse Oral LD50 (mg/kg)', fontsize=12, fontweight='bold')
ax.set_title(f'Interspecies LD50 Correlation\nSpearman ρ = {spearman_r:.3f} (p={spearman_p:.2e})', fontsize=11)
ax.legend(loc='best', fontsize=9)
ax.grid(alpha=0.3, linestyle='--')
# 对数刻度（如果数据跨度大）
ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/rat_mouse_ld50_correlation.pdf', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/rat_mouse_ld50_correlation.png', dpi=300)
plt.close()
print(f"散点图已保存至 {OUTPUT_DIR}/rat_mouse_ld50_correlation.pdf")
