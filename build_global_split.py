import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle
import sys

# 导入 SMARTS.py 中的分类函数（请确保 SMARTS.py 在同一目录或 PYTHONPATH 中）
from SMARTS import classify_compound
from rdkit import Chem

# ==================== 配置路径 ====================
RAW_DATA_DIR = '../全部化合物_原始特征'
SPLIT_DIR = '../formal_splits'
OUTPUT_SCALED_DIR = f'{SPLIT_DIR}/scaled'
os.makedirs(OUTPUT_SCALED_DIR, exist_ok=True)

RANDOM_STATE = 42

# ==================== 1. 加载 SMILES 并分类 ====================
print("1. 正在加载 SMILES 并依据分子结构分类...")
smiles_file = os.path.join(RAW_DATA_DIR, '全部化合物_SMILES.xlsx')  # 需提供该文件
if not os.path.exists(smiles_file):
    print(f"错误：未找到 SMILES 文件：{smiles_file}")
    print("请确认文件存在，或修改 smiles_file 路径。")
    sys.exit(1)

df_smiles = pd.read_excel(smiles_file)
# 假设第一列为 CID，第二列为 SMILES（与 SMARTS.py 主程序一致）
cid_col = df_smiles.columns[0]
smiles_col = df_smiles.columns[1]
df_smiles['cid'] = df_smiles[cid_col].astype(str)
df_smiles['smiles'] = df_smiles[smiles_col].astype(str)

# 对每个 SMILES 进行分类
type_mapping = {}
invalid_cids = []
for idx, row in df_smiles.iterrows():
    cid = row['cid']
    smiles = row['smiles'].strip()
    if not smiles:
        invalid_cids.append(cid)
        continue
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        invalid_cids.append(cid)
        continue
    cls = classify_compound(mol)   # 返回 'carbon_only', 'heteroatom', 'non_chlorobenzene'
    if cls in ['carbon_only', 'heteroatom']:
        # 映射为 'C' 或 'H'
        type_mapping[cid] = 'C' if cls == 'carbon_only' else 'H'
    # 忽略 non_chlorobenzene（即不纳入分层抽样）

print(f"有效化合物数（含氯苯）：{len(type_mapping)}")
print(f"无效或非氯苯化合物数：{len(invalid_cids)}")

# ==================== 2. 加载全量特征并合并 ====================
print("2. 正在加载全量特征...")
df_base = pd.read_excel(f'{RAW_DATA_DIR}/全部化合物_描述符特征.xlsx')
df_base['cid'] = df_base['cid'].astype(str)
df_f2 = pd.read_excel(f'{RAW_DATA_DIR}/全部化合物_深度学习特征.xlsx')
df_f2['cid'] = df_f2['cid'].astype(str)
df_f3 = pd.read_excel(f'{RAW_DATA_DIR}/全部化合物_深度加描述符.xlsx')
df_f3['cid'] = df_f3['cid'].astype(str)

# 仅保留在 type_mapping 中的化合物（即含氯苯的）
df_merged = df_base[['cid', 'dose']].copy()
df_merged = df_merged[df_merged['cid'].isin(type_mapping.keys())]
df_merged['type'] = df_merged['cid'].map(type_mapping)
df_merged['y_binary'] = np.where(df_merged['dose'] < 500, 1, 0)
df_merged['strata'] = df_merged['type'] + '_' + df_merged['y_binary'].astype(str)
print(f"纳入分层的样本数：{len(df_merged)}")
print(f"分层分布：\n{df_merged['strata'].value_counts()}")

# ==================== 3. 全局分层拆分 ====================
print("3. 生成全局预留测试集 ID...")
X_ids = df_merged[['cid']]
y_strata = df_merged['strata']
X_train_ids, X_test_ids, _, _ = train_test_split(
    X_ids, y_strata, test_size=0.2, random_state=RANDOM_STATE, stratify=y_strata
)
train_id_set = set(X_train_ids['cid'].values)
test_id_set = set(X_test_ids['cid'].values)
print(f"训练集 ID 数：{len(train_id_set)}，测试集 ID 数：{len(test_id_set)}")
pd.DataFrame({'cid': list(train_id_set)}).to_csv(f'{SPLIT_DIR}/global_train_ids.csv', index=False)
pd.DataFrame({'cid': list(test_id_set)}).to_csv(f'{SPLIT_DIR}/global_test_ids.csv', index=False)

# ==================== 4. 特征标准化并输出 9 组数据 ====================
print("4. 开始前置标准化并生成 9 组数据...")
feat_configs = [
    ('descriptor', '全部化合物_描述符特征.xlsx', 'F1'),
    ('deep', '全部化合物_深度学习特征.xlsx', 'F2'),
    ('combined', '全部化合物_深度加描述符.xlsx', 'F3')
]

for feat_name, file_name, feat_short in feat_configs:
    print(f"\n--- 处理特征集：{feat_short} ({feat_name}) ---")
    df_raw = pd.read_excel(f'{RAW_DATA_DIR}/{file_name}')
    df_raw['cid'] = df_raw['cid'].astype(str)
    
    # 特征列（假设从第5列开始）
    feat_cols = df_raw.columns[4:].tolist()
    
    # 拆分训练/测试（仅保留在 type_mapping 中的化合物）
    df_raw = df_raw[df_raw['cid'].isin(type_mapping.keys())]
    df_train_raw = df_raw[df_raw['cid'].isin(train_id_set)]
    df_test_raw = df_raw[df_raw['cid'].isin(test_id_set)]
    
    X_train_raw = df_train_raw[feat_cols].values
    X_test_raw = df_test_raw[feat_cols].values
    
    # 拟合缩放器
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # 保存缩放器
    with open(f'{OUTPUT_SCALED_DIR}/scaler_{feat_name}.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # 构建标准化后的 DataFrame
    df_train_scaled = pd.DataFrame(X_train_scaled, columns=feat_cols)
    df_train_scaled.insert(0, 'cid', df_train_raw['cid'].values)
    df_train_scaled['y'] = np.where(df_train_raw['dose'].values < 500, 1, 0)
    df_train_scaled['type'] = df_train_scaled['cid'].map(type_mapping)
    
    df_test_scaled = pd.DataFrame(X_test_scaled, columns=feat_cols)
    df_test_scaled.insert(0, 'cid', df_test_raw['cid'].values)
    df_test_scaled['y'] = np.where(df_test_raw['dose'].values < 500, 1, 0)
    df_test_scaled['type'] = df_test_scaled['cid'].map(type_mapping)
    
    # ---- 保存 Global ----
    global_out = f'{OUTPUT_SCALED_DIR}/Global_{feat_name}'
    os.makedirs(global_out, exist_ok=True)
    df_train_scaled.to_csv(f'{global_out}/train.csv', index=False)
    df_test_scaled.to_csv(f'{global_out}/test.csv', index=False)
    print(f"  Global 已保存至 {global_out}")
    
    # ---- 保存 C-only ----
    c_out = f'{OUTPUT_SCALED_DIR}/C_{feat_name}'
    os.makedirs(c_out, exist_ok=True)
    df_train_scaled[df_train_scaled['type'] == 'C'].to_csv(f'{c_out}/train.csv', index=False)
    df_test_scaled[df_test_scaled['type'] == 'C'].to_csv(f'{c_out}/test.csv', index=False)
    print(f"  C-only 已保存至 {c_out}")
    
    # ---- 保存 Hetero ----
    h_out = f'{OUTPUT_SCALED_DIR}/H_{feat_name}'
    os.makedirs(h_out, exist_ok=True)
    df_train_scaled[df_train_scaled['type'] == 'H'].to_csv(f'{h_out}/train.csv', index=False)
    df_test_scaled[df_test_scaled['type'] == 'H'].to_csv(f'{h_out}/test.csv', index=False)
    print(f"  Hetero 已保存至 {h_out}")

print("\n所有 9 组标准化数据已准备完毕！请检查目录：../formal_splits/scaled/")