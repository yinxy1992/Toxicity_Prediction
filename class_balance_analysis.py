"""
class_balance_analysis.py - Data2 和 Data3 类平衡性分析
"""
import pandas as pd
import numpy as np

SPLIT_DIR = '../formal_splits'
FEAT_TYPE = 'combined'

# 加载数据
df_c_train = pd.read_excel(f'{SPLIT_DIR}/C_{FEAT_TYPE}_train.xlsx')
df_h_train = pd.read_excel(f'{SPLIT_DIR}/H_{FEAT_TYPE}_train.xlsx')
df_c_test = pd.read_excel(f'{SPLIT_DIR}/C_{FEAT_TYPE}_test.xlsx')
df_h_test = pd.read_excel(f'{SPLIT_DIR}/H_{FEAT_TYPE}_test.xlsx')

# 合并
df_c = pd.concat([df_c_train, df_c_test], ignore_index=True)
df_h = pd.concat([df_h_train, df_h_test], ignore_index=True)

for name, df in [('Carbon-Only (Data3)', df_c), ('Heteroatom (Data2)', df_h)]:
    y = np.where(df.iloc[:, 3] < 500, 1, 0)
    print(f"\n{name}:")
    print(f"  总样本数: {len(y)}")
    print(f"  高毒 (1): {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)")
    print(f"  低毒 (0): {len(y)-np.sum(y)} ({(len(y)-np.sum(y))/len(y)*100:.1f}%)")