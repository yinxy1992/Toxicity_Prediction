"""
chemical_diversity.py - 化学多样性计算
========================================
功能：
- 计算 Data2 (H) 和 Data3 (C) 的平均 Tanimoto 距离
- 报告类平衡性和样本数量

用法：
1. 直接运行（需确保有 SMILES 列）
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import os

# 数据文件路径（原始数据，包含 SMILES）
RAW_DATA_DIR = '../全部化合物_原始特征'
OUTPUT_DIR = '../figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载碳和杂原子的 SMILES（从已分好的文件中）
def load_smiles_by_type():
    # 使用 '碳_描述符.xlsx' 和 '杂_描述符.xlsx'
    df_c = pd.read_excel(f'{RAW_DATA_DIR}/碳_描述符.xlsx')
    df_h = pd.read_excel(f'{RAW_DATA_DIR}/杂_描述符.xlsx')
    smi_c = df_c['smiles'].dropna().tolist()
    smi_h = df_h['smiles'].dropna().tolist()
    return smi_c, smi_h

smi_c, smi_h = load_smiles_by_type()
print(f"碳取代基化合物数: {len(smi_c)}")
print(f"杂原子取代基化合物数: {len(smi_h)}")

# 计算平均 Tanimoto 距离（两两之间）
def average_tanimoto_distance(smiles_list, sample_size=500):
    """计算平均 Tanimoto 距离（基于 ECFP4），若列表过大则随机采样"""
    if len(smiles_list) < 2:
        return np.nan
    if len(smiles_list) > sample_size:
        np.random.seed(42)
        smiles_list = np.random.choice(smiles_list, sample_size, replace=False).tolist()
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fps.append(fp)
    if len(fps) < 2:
        return np.nan
    sims = []
    for i in range(len(fps)):
        for j in range(i+1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sims.append(sim)
    return 1 - np.mean(sims)

avg_dist_c = average_tanimoto_distance(smi_c)
avg_dist_h = average_tanimoto_distance(smi_h)

print(f"碳取代基平均 Tanimoto 距离: {avg_dist_c:.4f}")
print(f"杂原子取代基平均 Tanimoto 距离: {avg_dist_h:.4f}")

# 保存结果
result = pd.DataFrame({
    'Subset': ['Carbon-only', 'Heteroatom-containing'],
    'Count': [len(smi_c), len(smi_h)],
    'Avg_Tanimoto_Distance': [avg_dist_c, avg_dist_h]
})
result.to_csv(f'{OUTPUT_DIR}/chemical_diversity.csv', index=False)
print(f"结果已保存至 {OUTPUT_DIR}/chemical_diversity.csv")