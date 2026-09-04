import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from matplotlib import rcParams

# 设置学术论文风格 - 使用Times New Roman字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# 读取Excel数据
# 请将'your_file.xlsx'替换为你的文件名
# 假设数据在Sheet1中，列名分别为'Mmouse'和'rat'
# 请根据实际情况调整列名
df = pd.read_excel('../data/mouse_vs_rat_doses.xlsx' )

# 提取数据列
# 根据你的实际列名修改下面的'mouse'和'rat'
mouse_data = df['mouse'].values
rat_data = df['rat'].values

# 检查数据长度
if len(mouse_data) != len(rat_data):
    print("警告：两列数据长度不一致！")
    # 处理方式：删除任何一列中的NaN值对应的行
    df_clean = df.dropna(subset=['mouse', 'rat'])
    mouse_data = df_clean['mouse'].values
    rat_data = df_clean['rat'].values

# 删除可能的NaN值
mask = ~np.isnan(mouse_data) & ~np.isnan(rat_data)
mouse_data = mouse_data[mask]
rat_data = rat_data[mask]

# 计算相关性统计
slope, intercept, r_value, p_value, std_err = stats.linregress(mouse_data, rat_data)
r_squared = r_value ** 2

# 创建图形 - 4英寸×3英寸
fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

# 绘制散点图
# 使用学术论文常用的蓝色
scatter = ax.scatter(mouse_data, rat_data, 
                     color='#1f77b4',  # matplotlib默认蓝色
                     alpha=0.7,
                     edgecolor='black',
                     linewidth=0.5,
                     s=40,  # 点的大小
                     zorder=3)

# 绘制拟合线
x_fit = np.linspace(np.min(mouse_data), np.max(mouse_data), 100)
y_fit = intercept + slope * x_fit
ax.plot(x_fit, y_fit, 
        color='#d62728',  # 红色拟合线
        linewidth=1.5,
        linestyle='-',
        label=f'Fit: y = {slope:.3f}x + {intercept:.3f}',
        zorder=2)

# 添加对角线（y=x线）作为参考
min_val = min(np.min(mouse_data), np.min(rat_data))
max_val = max(np.max(mouse_data), np.max(rat_data))
ax.plot([min_val, max_val], [min_val, max_val], 
        '--', 
        color='gray', 
        linewidth=1,
        alpha=0.5,
        label='y = x',
        zorder=1)

# 设置坐标轴标签
ax.set_xlabel('Mouse LD50 (mg/kg)', fontweight='normal')
ax.set_ylabel('Rat LD50 (mg/kg)', fontweight='normal')

# 设置坐标轴范围，留出一些边距
margin_x = (max(mouse_data) - min(mouse_data)) * 0.05
margin_y = (max(rat_data) - min(rat_data)) * 0.05
ax.set_xlim([min(mouse_data) - margin_x, max(mouse_data) + margin_x])
ax.set_ylim([min(rat_data) - margin_y, max(rat_data) + margin_y])

# 添加统计信息文本
stats_text = f'R² = {r_squared:.3f}\np = {p_value:.3e}\nn = {len(mouse_data)}'
ax.text(0.74, 0.75, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))

# 添加图例
ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')

# 设置网格
ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

# 设置刻度线
ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)

# 调整布局
plt.tight_layout()

# 保存图片
output_filename = '../figures/LD50_correlation_plot.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"图片已保存为: {output_filename}")

# 显示图片
plt.show()

# 打印统计信息
print(f"样本数量: {len(mouse_data)}")
print(f"相关系数 R: {r_value:.3f}")
print(f"R²: {r_squared:.3f}")
print(f"p值: {p_value:.3e}")
print(f"拟合方程: y = {slope:.3f}x + {intercept:.3f}")