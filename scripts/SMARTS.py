import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
import os
import sys


# ============================================================
# 1. 获取所有苯环
# ============================================================

def get_benzene_rings(mol):
    """
    获取分子中所有苯环。

    定义：
    1. 六元环
    2. 六个原子均为碳原子
    3. 六个碳原子均为芳香原子
    """

    rings = [list(ring) for ring in rdmolops.GetSymmSSSR(mol)]

    benzene_rings = []

    for ring in rings:

        # 必须是六元环
        if len(ring) != 6:
            continue

        atoms = [mol.GetAtomWithIdx(i) for i in ring]

        # 六个原子必须全部是芳香碳
        if all(
            atom.GetAtomicNum() == 6 and atom.GetIsAromatic()
            for atom in atoms
        ):
            benzene_rings.append(ring)

    return benzene_rings


# ============================================================
# 2. 判断单个苯环是否含有氯
# ============================================================

def ring_has_chlorine(ring, mol):
    """
    判断苯环上是否存在直接连接的氯原子。
    """

    ring_set = set(ring)

    for idx in ring:

        atom = mol.GetAtomWithIdx(idx)

        for neighbor in atom.GetNeighbors():

            # 忽略环内部原子
            if neighbor.GetIdx() in ring_set:
                continue

            # 如果环外直接连接Cl
            if neighbor.GetAtomicNum() == 17:
                return True

    return False


# ============================================================
# 3. 分类单个含氯苯环
# ============================================================

def classify_ring(ring, mol):
    """
    判断单个含氯苯环的类别。

    carbon_only：
        苯环所有环外直接连接的取代原子
        仅包含 C、H、Cl

    heteroatom：
        苯环存在任何直接连接的其他元素
        如 O、N、S、F、Br、P 等
    """

    ring_set = set(ring)

    for idx in ring:

        atom = mol.GetAtomWithIdx(idx)

        for neighbor in atom.GetNeighbors():

            neighbor_idx = neighbor.GetIdx()

            # 忽略苯环内部原子
            if neighbor_idx in ring_set:
                continue

            atomic_num = neighbor.GetAtomicNum()

            # 允许的原子：
            # H = 1
            # C = 6
            # Cl = 17
            if atomic_num in [1, 6, 17]:
                continue

            # 出现其他任何元素
            return 'heteroatom'

    return 'carbon_only'


# ============================================================
# 4. 化合物整体分类
# ============================================================

def classify_compound(mol):
    """
    对整个化合物进行分类。

    分类规则：

    1. 如果不存在苯环
       -> non_chlorobenzene

    2. 如果存在苯环，但没有任何苯环直接连接Cl
       -> non_chlorobenzene

    3. 对所有含Cl的苯环进行判断：

       - 只要有一个苯环存在其他杂原子
         -> heteroatom

       - 所有含Cl苯环均只包含C/H/Cl
         -> carbon_only
    """

    # 获取所有苯环
    benzene_rings = get_benzene_rings(mol)

    if not benzene_rings:
        return 'non_chlorobenzene'

    # 找出所有含Cl的苯环
    chlorinated_rings = []

    for ring in benzene_rings:

        if ring_has_chlorine(ring, mol):
            chlorinated_rings.append(ring)

    # 没有苯环发生氯取代
    if not chlorinated_rings:
        return 'non_chlorobenzene'

    # 分类所有含氯苯环
    ring_types = []

    for ring in chlorinated_rings:

        ring_class = classify_ring(ring, mol)

        ring_types.append(ring_class)

    # 只要有一个含氯苯环属于heteroatom
    if 'heteroatom' in ring_types:
        return 'heteroatom'

    # 所有含氯苯环均为carbon_only
    return 'carbon_only'


# ============================================================
# 5. 主程序
# ============================================================

def main():

    # -------------------------------
    # 文件路径
    # -------------------------------

    input_file = r"C:/Users/85328/Desktop/input.xlsx"

    output_dir = r"C:/Users/85328/Desktop"

    # -------------------------------
    # 读取Excel
    # -------------------------------

    try:
        df = pd.read_excel(input_file)

    except Exception as e:
        print(f"读取Excel失败: {e}")
        sys.exit(1)

    # -------------------------------
    # 检查列数
    # -------------------------------

    if df.shape[1] < 2:

        print("错误：Excel至少需要两列：CID和SMILES")

        sys.exit(1)

    # 默认：
    # 第一列 = CID
    # 第二列 = SMILES

    cid_col = df.columns[0]
    smiles_col = df.columns[1]

    print(f"CID列: {cid_col}")
    print(f"SMILES列: {smiles_col}")

    # -------------------------------
    # 存储分类结果
    # -------------------------------

    labels = []

    # -------------------------------
    # 逐个处理化合物
    # -------------------------------

    for idx, row in df.iterrows():

        smiles = row[smiles_col]

        # 检查SMILES是否为空
        if not isinstance(smiles, str) or not smiles.strip():

            labels.append('invalid')

            continue

        # 去除前后空格
        smiles = smiles.strip()

        # RDKit解析
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:

            print(
                f"警告：无法解析SMILES | "
                f"行号={idx + 2} | "
                f"CID={row[cid_col]} | "
                f"SMILES={repr(smiles)}"
            )

            labels.append('invalid')

            continue

        try:

            cls = classify_compound(mol)

        except Exception as e:

            print(
                f"错误处理化合物 | "
                f"行号={idx + 2} | "
                f"CID={row[cid_col]} | "
                f"SMILES={repr(smiles)} | "
                f"错误={e}"
            )

            cls = 'error'

        labels.append(cls)

    # -------------------------------
    # 添加分类结果
    # -------------------------------

    df['class'] = labels

    # ============================================================
    # 6. 输出统计结果
    # ============================================================

    print("\n" + "=" * 50)
    print("分类完成")
    print("=" * 50)

    print(df['class'].value_counts(dropna=False))

    # ============================================================
    # 7. 分别保存
    # ============================================================

    non_cl = df[df['class'] == 'non_chlorobenzene']

    carbon = df[df['class'] == 'carbon_only']

    hetero = df[df['class'] == 'heteroatom']

    invalid = df[
        df['class'].isin(['invalid', 'error'])
    ]

    # 保存所有结果
    all_output = os.path.join(
        output_dir,
        "all_compounds_classified.xlsx"
    )

    df.to_excel(
        all_output,
        index=False
    )

    # 保存各类别
    non_cl.to_csv(
        os.path.join(
            output_dir,
            "non_chlorobenzene.csv"
        ),
        index=False
    )

    carbon.to_csv(
        os.path.join(
            output_dir,
            "carbon_only.csv"
        ),
        index=False
    )

    hetero.to_csv(
        os.path.join(
            output_dir,
            "heteroatom.csv"
        ),
        index=False
    )

    invalid.to_csv(
        os.path.join(
            output_dir,
            "invalid_or_error.csv"
        ),
        index=False
    )

    # ============================================================
    # 8. 输出结果
    # ============================================================

    print("\n" + "=" * 50)
    print("最终统计")
    print("=" * 50)

    print(f"非氯苯类化合物: {len(non_cl)}")
    print(f"carbon_only: {len(carbon)}")
    print(f"heteroatom: {len(hetero)}")
    print(f"invalid/error: {len(invalid)}")

    print("\n输出文件：")

    print(all_output)

    print("\n分类完成！")


if __name__ == '__main__':
    main()
