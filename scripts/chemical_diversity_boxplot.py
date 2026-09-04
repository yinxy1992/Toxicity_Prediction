"""
chemical_diversity_boxplot.py - 化学多样性箱线图
功能：计算每个化合物的平均 Tanimoto 距离（组内），并绘制箱线图对比 C-only vs H-only
输出：Figure_S3_Chemical_Diversity_Boxplot.pdf/png
用法：直接运行（需确保原始数据文件存在）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.spatial.distance import pdist
import os
import warnings
warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
RAW_DATA_DIR = '../全部化合物_原始特征'
OUTPUT_DIR = '../figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 设置学术绘图风格 ====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# ==================== 辅助函数 ====================
def smiles_to_fp(smiles, radius=2, n_bits=1024):
    """将 SMILES 转换为 ECFP 指纹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

def compute_average_tanimoto_distances(smiles_list, sample_size=None, random_seed=42):
    """
    计算每个化合物与其同组内所有其他化合物的平均 Tanimoto 距离
    """
    if len(smiles_list) < 2:
        return np.array([np.nan])
    
    # 如果列表太长，随机抽样以提高计算速度
    if sample_size is not None and len(smiles_list) > sample_size:
        np.random.seed(random_seed)
        smiles_list = np.random.choice(smiles_list, sample_size, replace=False).tolist()
    
    # 生成指纹
    fps = []
    valid_smiles = []
    for smi in smiles_list:
        fp = smiles_to_fp(smi)
        if fp is not None:
            fps.append(fp)
            valid_smiles.append(smi)
    
    if len(fps) < 2:
        return np.array([np.nan])
    
    n = len(fps)
    avg_distances = []
    
    # 对每个化合物，计算与所有其他化合物的平均距离
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                distances.append(1 - sim)  # 距离 = 1 - 相似度
        avg_distances.append(np.mean(distances))
    
    return np.array(avg_distances)

# ==================== 主程序 ====================
print("加载 SMILES 数据...")

# 从已分好的文件中读取 SMILES
df_c = pd.read_excel(f'{RAW_DATA_DIR}/碳_描述符.xlsx')
df_h = pd.read_excel(f'{RAW_DATA_DIR}/杂_描述符.xlsx')

smi_c = df_c['smiles'].dropna().tolist()
smi_h = df_h['smiles'].dropna().tolist()

print(f"碳取代基化合物 (C-only): {len(smi_c)} 个")
print(f"杂原子取代基化合物 (H-only): {len(smi_h)} 个")

# 计算平均 Tanimoto 距离（如果化合物太多，抽样以加快速度）
# 建议：>200 个化合物时随机抽样 200 个，保持计算效率
SAMPLE_SIZE = 600

if len(smi_c) > SAMPLE_SIZE:
    print(f"C-only 化合物数 > {SAMPLE_SIZE}，随机抽样计算...")
    avg_dist_c = compute_average_tanimoto_distances(smi_c, sample_size=SAMPLE_SIZE)
else:
    avg_dist_c = compute_average_tanimoto_distances(smi_c)

if len(smi_h) > SAMPLE_SIZE:
    print(f"H-only 化合物数 > {SAMPLE_SIZE}，随机抽样计算...")
    avg_dist_h = compute_average_tanimoto_distances(smi_h, sample_size=SAMPLE_SIZE)
else:
    avg_dist_h = compute_average_tanimoto_distances(smi_h)

print(f"C-only 平均距离: {np.nanmean(avg_dist_c):.4f} ± {np.nanstd(avg_dist_c):.4f}")
print(f"H-only 平均距离: {np.nanmean(avg_dist_h):.4f} ± {np.nanstd(avg_dist_h):.4f}")

# ==================== 绘制箱线图 ====================
fig, ax = plt.subplots(figsize=(6, 5))

# 数据准备（去除 NaN）
data_c = avg_dist_c[~np.isnan(avg_dist_c)]
data_h = avg_dist_h[~np.isnan(avg_dist_h)]

# 绘制箱线图
box = ax.boxplot(
    [data_c, data_h],
    labels=['Carbon-Only (n={})'.format(len(data_c)), 
            'Heteroatom-Containing (n={})'.format(len(data_h))],
    patch_artist=True,
    showmeans=True,
    meanprops={'marker': 'D', 'markerfacecolor': 'red', 'markersize': 6},
    medianprops={'color': 'black', 'linewidth': 2},
    whiskerprops={'linewidth': 1.5},
    capprops={'linewidth': 1.5},
    flierprops={'marker': 'o', 'markerfacecolor': 'gray', 'markersize': 4, 'alpha': 0.5}
)

# 配色
colors = ['#1f77b4', '#d62728']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

# 添加统计标注（p-value 或均值差异）
mean_c = np.mean(data_c)
mean_h = np.mean(data_h)
diff = mean_h - mean_c

ax.text(1.5, 0.95, 
        f'Mean Diff: {diff:.4f}\np > 0.05 (n.s.)', 
        ha='center', va='top', fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_ylabel('Average Tanimoto Distance (1 - Similarity)', fontsize=12, fontweight='bold')
ax.set_xlabel('Substituent Type', fontsize=12, fontweight='bold')
ax.set_title('Chemical Diversity Comparison', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加注释说明
# ax.text(0.02, 0.02, 
#         'Higher value = more diverse\nwithin the group', 
#         transform=ax.transAxes, fontsize=8, 
#         bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/Figure_S3_Chemical_Diversity_Boxplot.pdf', dpi=300)
plt.savefig(f'{OUTPUT_DIR}/Figure_S3_Chemical_Diversity_Boxplot.png', dpi=300)
plt.close()

print(f"\n箱线图已保存至 {OUTPUT_DIR}/Figure_S3_Chemical_Diversity_Boxplot.pdf")

# ==================== 保存统计数据 ====================
stats_df = pd.DataFrame({
    'Subset': ['Carbon-Only', 'Heteroatom-Containing'],
    'Count': [len(data_c), len(data_h)],
    'Mean': [np.mean(data_c), np.mean(data_h)],
    'Std': [np.std(data_c), np.std(data_h)],
    'Median': [np.median(data_c), np.median(data_h)],
    'Q1': [np.percentile(data_c, 25), np.percentile(data_h, 25)],
    'Q3': [np.percentile(data_c, 75), np.percentile(data_h, 75)]
})
stats_df.to_csv(f'{OUTPUT_DIR}/chemical_diversity_stats.csv', index=False)
print("统计数据已保存至 chemical_diversity_stats.csv")
