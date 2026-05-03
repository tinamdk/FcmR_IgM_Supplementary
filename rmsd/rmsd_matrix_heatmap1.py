import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms

# 之前的
# 已比对的结构文件列表 (示例)
structures = [
    "./source_file/aligned_7YTE.pdb",
    "./source_file/aligned_7YSG.pdb",
    "./source_file/aligned_7YTC.pdb",
    "./source_file/aligned_7YTD.pdb",
    "./source_file/aligned_8BPE.pdb",
    "./source_file/aligned_8BPF.pdb",
    "./source_file/aligned_8BPG.pdb",
]

# 为每个结构指定链ID (根据您的实际结构修改)
chain_ids = ["C", "R", "R", "R", "I", "I", "C"]

# 参考结构域定义 (FcμR的Ig样结构域)
# residue_range = "resid 18-121 and name CA"
residue_range = "resid 446-551 and name CA"

n_structures = len(structures)
rmsd_matrix = np.zeros((n_structures, n_structures))

# 创建参考坐标缓存 (提升性能)
reference_coords = []
for idx, (pdb, chain_id) in enumerate(zip(structures, chain_ids)):
    u = mda.Universe(pdb)

    # 修改选择器：添加链ID
    chain_selection = f"chainID {chain_id} and {residue_range}"
    ref_atoms = u.select_atoms(chain_selection)

    # 检查选择是否有效
    if len(ref_atoms) == 0:
        raise ValueError(f"结构 {pdb} 中未找到链 {chain_id} 的残基 446-551")

    reference_coords.append(ref_atoms.positions.copy())
    print(idx, str(chain_id), structures[idx],ref_atoms.positions.shape)


    # 双重循环计算所有结构对
for i in range(n_structures):
    # 获取第i个结构的参考坐标
    coords_i = reference_coords[i]

    for j in range(i, n_structures):  # 从i开始避免重复计算
        if i == j:
            # 对角线上RMSD=0
            rmsd_matrix[i, j] = 0.0
        else:
            # 获取第j个结构的坐标
            coords_j = reference_coords[j]

            print(i, j, coords_i.shape, coords_j.shape)
            # 计算RMSD（含最优叠加）
            rmsd_val = rms.rmsd(
                coords_i,  # 结构i的Cα坐标
                coords_j,  # 结构j的Cα坐标
                superposition=False  # 自动进行最优叠加
            )

            # 填充对称矩阵
            rmsd_matrix[i, j] = rmsd_val
            rmsd_matrix[j, i] = rmsd_val  # 对称位置赋值


# 验证矩阵属性
assert np.allclose(rmsd_matrix, rmsd_matrix.T)  # 验证对称性
assert np.all(rmsd_matrix.diagonal() == 0)     # 验证对角线为零

# 可视化矩阵
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(
    rmsd_matrix,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    xticklabels=[f"{structures[i][22:26]}" for i in range(n_structures)],
    yticklabels=[f"{structures[i][22:26]}" for i in range(n_structures)]
)
plt.title("Multi-Matrix(Å) Heatmap of FcµR-IgM complexes")
plt.xlabel("Human FcμR-IgM Crystal Structures")
plt.ylabel("Human FcμR-IgM Crystal Structures")
plt.savefig("rmsd_matrix_heatmap.png", dpi=600)

