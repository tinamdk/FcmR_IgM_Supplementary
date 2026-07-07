import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 添加以下全局设置 ==========
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 7
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.5
# ====================================

# ---------- 计算 RMSD 矩阵 ----------
structures = [
    "./source_file/7YTE.pdb",
    "./source_file/7YSG.pdb",
    "./source_file/7YTC.pdb",
    "./source_file/7YTD.pdb",
    "./source_file/8BPE.pdb",
    "./source_file/8BPF.pdb",
    "./source_file/8BPG.pdb",
]

chain_ids = ["C", "R", "R", "R", "I", "I", "A"]
residue_range = "resid 18-124 and name CA"

# 第一步：收集每个结构存在的残基ID集合
resid_sets = []
atoms_list = []

for idx, (pdb, chain_id) in enumerate(zip(structures, chain_ids)):
    u = mda.Universe(pdb)
    sel = f"chainID {chain_id} and {residue_range}"
    atoms = u.select_atoms(sel)
    if len(atoms) == 0:
        raise ValueError(f"{pdb} 中未找到链 {chain_id} 的残基 18-124")
    resid_sets.append(set(atoms.resids))
    atoms_list.append(atoms)
    print(idx, chain_id, pdb, atoms.positions.shape, f"resids: {min(atoms.resids)}-{max(atoms.resids)}")

# 公共残基
common_resids = sorted(set.intersection(*resid_sets))
print(f"\nCommon residues count: {len(common_resids)}")

# 基于公共残基提取坐标
coords_list = []
for atoms in atoms_list:
    resid_to_coord = {resid: pos for resid, pos in zip(atoms.resids, atoms.positions)}
    filtered_coords = np.array([resid_to_coord[resid] for resid in common_resids])
    coords_list.append(filtered_coords)
    print(filtered_coords.shape)

# 计算成对 RMSD
n = len(structures)
rmsd_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):
        if i == j:
            rmsd_matrix[i, j] = 0.0
        else:
            rmsd_val = rms.rmsd(coords_list[i], coords_list[j], superposition=True)
            rmsd_matrix[i, j] = rmsd_val
            rmsd_matrix[j, i] = rmsd_val

print("\nRMSD matrix (Å):")
print(np.round(rmsd_matrix, 2))

# ---------- 绘制热图 ----------
labels = ["7YTE", "7YSG", "7YTC", "7YTD", "8BPE", "8BPF", "8BPG"]

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    rmsd_matrix,
    annot=True,
    fmt='.2f',
    cmap='viridis',
    xticklabels=labels,
    yticklabels=labels,
    square=True,
    linewidths=0.5,
    cbar_kws={'label': 'Cα RMSD (Å)', 'shrink': 0.8},
    annot_kws={'size': 8},
    ax=ax
)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
ax.set_yticklabels(labels, rotation=0, fontsize=7)
ax.set_xlabel('PDB ID', fontsize=7)
ax.set_ylabel('PDB ID', fontsize=7)
ax.set_title('', fontweight='normal', fontsize=7)
plt.tight_layout()
plt.savefig('fig1a_rmsd_heatmap.pdf', dpi=1200, bbox_inches='tight')
plt.savefig('fig1a_rmsd_heatmap.png', dpi=1200, bbox_inches='tight')
print("Heatmap saved.")