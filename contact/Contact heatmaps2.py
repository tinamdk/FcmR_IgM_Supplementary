import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from scipy.spatial import cKDTree
from collections import defaultdict

# ==================== CONFIGURATION ====================
PDB_LIST = [
    {"file": "source_file/7YTE.pdb", "IgM_chains": ["A", "B"], "FcµR_chains": ["C", "D"], "name": "7YTE"},
    {"file": "source_file/7YTC.pdb", "IgM_chains": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
     "FcµR_chains": ["R"], "name": "7YTC"},
    {"file": "source_file/7YTD.pdb", "IgM_chains": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
     "FcµR_chains": ["R", "S", "U", "V"], "name": "7YTD"},
    {"file": "source_file/7YSG.pdb", "IgM_chains": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
     "FcµR_chains": ["U", "R", "S", "V"], "name": "7YSG"},
    {"file": "source_file/8BPE.pdb", "IgM_chains": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
     "FcµR_chains": ["I", "M", "N", "O", "P", "Q", "R", "S"], "name": "8BPE"},
    {"file": "source_file/8BPF.pdb", "IgM_chains": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
     "FcµR_chains": ["I"], "name": "8BPF"},
    {"file": "source_file/8BPG.pdb", "IgM_chains": ["C", "D", "E", "F"],
     "FcµR_chains": ["A", "B"], "name": "8BPG"},
]
DISTANCE_CUTOFF = 3.0
FCR_RES_RANGE = range(18, 126)   # FcμR residues 18-125

# ==================== HELPER FUNCTIONS ====================
def get_chain_atom_coords_with_residue(structure, chain_id):
    """返回该链所有原子的坐标及每个原子对应的 (res_num, res_name)"""
    coords = []
    res_info = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if not is_aa(residue):
                    continue
                res_num = residue.id[1]
                res_name = residue.resname.capitalize()
                for atom in residue:
                    coords.append(atom.coord)
                    res_info.append((res_num, res_name))
    return np.array(coords), res_info

def get_residue_contacts_for_chain_pair(pdb_file, fc_chain, igm_chain, cutoff):
    """返回该链对中所有接触的残基对集合 {(fc_res, igm_res)}"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)
    igm_coords, igm_res_info = get_chain_atom_coords_with_residue(structure, igm_chain)
    if len(igm_coords) == 0:
        return set()
    tree = cKDTree(igm_coords)
    contacts = set()
    for model in structure:
        for chain in model:
            if chain.id != fc_chain:
                continue
            for residue in chain:
                if not is_aa(residue):
                    continue
                fc_res_num = residue.id[1]
                if fc_res_num not in FCR_RES_RANGE:
                    continue
                for atom in residue:
                    indices = tree.query_ball_point(atom.coord, cutoff)
                    if indices:
                        for idx in indices:
                            igm_res_num, _ = igm_res_info[idx]
                            contacts.add((fc_res_num, igm_res_num))
                        break
    return contacts

def get_residue_name_map(pdb_list, chain_key, res_range=None):
    """构建残基编号 -> 三字母+编号 的映射"""
    name_map = {}
    for entry in pdb_list:
        pdb_file = entry["file"]
        if not os.path.exists(pdb_file):
            continue
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)
        except Exception:
            continue
        target_chains = entry[chain_key]
        for model in structure:
            for chain in model:
                if chain.id not in target_chains:
                    continue
                for residue in chain:
                    if not is_aa(residue):
                        continue
                    res_id = residue.id[1]
                    if res_range is not None and res_id not in res_range:
                        continue
                    if res_id not in name_map:
                        three = residue.resname.capitalize()
                        name_map[res_id] = f"{three}{res_id}"
    if res_range is not None:
        for r in res_range:
            if r not in name_map:
                name_map[r] = str(r)
    return name_map

def compute_complex_matrix(pdb_file, fc_chains, igm_chains, cutoff, fc_res_range, fc_name_map, igm_name_map):
    """计算单个复合物内部的残基接触频率矩阵"""
    pair_count = defaultdict(int)
    total_pairs = 0
    all_igm_res = set()
    all_fc_res = set()
    for fc_chain in fc_chains:
        for igm_chain in igm_chains:
            total_pairs += 1
            contacts = get_residue_contacts_for_chain_pair(pdb_file, fc_chain, igm_chain, cutoff)
            for (fc_res, igm_res) in contacts:
                pair_count[(fc_res, igm_res)] += 1
                all_igm_res.add(igm_res)
                all_fc_res.add(fc_res)
    if total_pairs == 0:
        return None, None, None, None
    fc_res_list = sorted([r for r in fc_res_range if r in all_fc_res])
    igm_res_list = sorted(all_igm_res)
    freq = np.zeros((len(fc_res_list), len(igm_res_list)))
    for i, fc in enumerate(fc_res_list):
        for j, igm in enumerate(igm_res_list):
            freq[i, j] = pair_count.get((fc, igm), 0) / total_pairs
    row_labels = [fc_name_map.get(r, str(r)) for r in fc_res_list]
    col_labels = [igm_name_map.get(r, str(r)) for r in igm_res_list]
    return freq, row_labels, col_labels, total_pairs

# ==================== MAIN ====================
def main():
    # 获取全局残基名称映射（用于统一标签）
    fc_name_map = get_residue_name_map(PDB_LIST, 'FcµR_chains', FCR_RES_RANGE)
    igm_name_map = get_residue_name_map(PDB_LIST, 'IgM_chains', res_range=None)

    # 收集每个复合物的矩阵
    matrices = []
    for entry in PDB_LIST:
        pdb_file = entry["file"]
        if not os.path.exists(pdb_file):
            print(f"File not found: {pdb_file}, skipped.")
            continue
        print(f"Processing {entry['name']} ...")
        freq, row_lab, col_lab, n_pairs = compute_complex_matrix(
            pdb_file, entry["FcµR_chains"], entry["IgM_chains"],
            DISTANCE_CUTOFF, FCR_RES_RANGE, fc_name_map, igm_name_map
        )
        if freq is not None:
            matrices.append({
                'name': entry['name'],
                'freq': freq,
                'rows': row_lab,
                'cols': col_lab,
                'n_pairs': n_pairs
            })

    # 绘制多面板热图
    n_complex = len(matrices)
    n_cols = 3
    n_rows = (n_complex + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_complex == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    vmin, vmax = 0, 1  # 统一颜色标尺
    for idx, comp in enumerate(matrices):
        ax = axes[idx]
        df = pd.DataFrame(comp['freq'], index=comp['rows'], columns=comp['cols'])
        mask = (df == 0)
        sns.heatmap(df, mask=mask, cmap='YlOrRd', vmin=vmin, vmax=vmax,
                    cbar=False, linewidths=0.2, linecolor='lightgray',
                    square=False, xticklabels=True, yticklabels=True, ax=ax)
        ax.set_title(f"{comp['name']} (n={comp['n_pairs']} chain pairs)", fontsize=10)
        ax.set_xlabel('IgM residue', fontsize=8)
        ax.set_ylabel('FcμR residue', fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

    # 隐藏多余的子图
    for idx in range(len(matrices), len(axes)):
        axes[idx].axis('off')

    # 添加公共颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Contact frequency (proportion of chain pairs)')

    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.9, top=0.95, wspace=0.3, hspace=0.4)
    plt.savefig("Figure_complex_heatmaps.png", dpi=300)
    plt.show()
    print("Saved Figure_complex_heatmaps.png")

if __name__ == "__main__":
    main()