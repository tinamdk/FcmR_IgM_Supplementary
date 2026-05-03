import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import PDBParser, NeighborSearch, is_aa
from collections import defaultdict

# ============================================================
# 配置
# ============================================================
PDB_IDS = ['7YTE', '7YTC', '7YTD', '7YSG', '8BPE', '8BPF', '8BPG']

CHAIN_MAP = {
    "7YTE": {"file": "source_file/7YTE.pdb", "IgM": ["A", "B"], "FcμR": ["C", "D"]},
    "7YTC": {"file": "source_file/7YTC.pdb", "IgM": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
             "FcμR": ["R"]},
    "7YTD": {"file": "source_file/7YTD.pdb", "IgM": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
             "FcμR": ["R", "S", "U", "V"]},
    "7YSG": {"file": "source_file/7YSG.pdb", "IgM": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J", "P"],
             "FcμR": ["U", "R", "S", "V"]},
    "8BPE": {"file": "source_file/8BPE.pdb", "IgM": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
             "FcμR": ["I", "M", "N", "O", "P", "Q", "R", "S"]},
    "8BPF": {"file": "source_file/8BPF.pdb", "IgM": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
             "FcμR": ["I"]},
    "8BPG": {"file": "source_file/8BPG.pdb", "IgM": ["C", "D", "E", "F"], "FcμR": ["A", "B"]},
}

DIST_CUTOFF = 4.5  # 距离阈值（原子间）
OUTPUT_IMAGE = "contact_frequency6.png"


# ============================================================
# 辅助函数
# ============================================================
def format_resname(resname):
    """将三字母残基名转换为首字母大写其余小写，例如'Arg'"""
    if len(resname) == 3:
        return resname[0].upper() + resname[1:].lower()
    return resname


def residue_key(res):
    """返回带残基名的键，如'Arg45'"""
    return f"{format_resname(res.resname)}{res.id[1]}"


def get_residue_number(s):
    """从残基字符串中提取数字，例如'Arg45' -> 45"""
    return int(re.search(r'\d+', s).group())


def get_sidechain_atoms(residue):
    """获取侧链原子（排除主链）"""
    backbone = {'N', 'CA', 'C', 'O'}
    if residue.resname == 'GLY':
        return [atom for atom in residue.get_atoms() if atom.name == 'CA']
    return [atom for atom in residue.get_atoms() if atom.name not in backbone]


def get_pair_contact_fractions(structure, fcmr_chains, igm_chains, cutoff):
    """
    对于给定的结构，统计每个(FcμR残基, IgM残基)对出现在多少条FcμR链中（每条链只计一次）。
    返回字典 {(fcmr_key, igm_key): 比例（出现链数 / 总FcμR链数）}
    """
    # 收集所有IgM链的原子（用于快速搜索）
    igm_atoms = []
    for ch_id in igm_chains:
        if ch_id in structure[0]:
            for atom in structure[0][ch_id].get_atoms():
                igm_atoms.append(atom)
    if not igm_atoms:
        return {}
    ns = NeighborSearch(igm_atoms)

    total_fc_chains = 0
    # 记录每条链中每个FcμR残基接触到的IgM残基集合
    chain_pair_map = []  # 元素为 dict: {fcmr_key: set of igm_keys}
    for ch_id in fcmr_chains:
        if ch_id not in structure[0]:
            continue
        total_fc_chains += 1
        current_chain_map = defaultdict(set)
        for residue in structure[0][ch_id].get_residues():
            if not is_aa(residue):
                continue
            # 只考虑FcμR D1结构域残基（PDB编号18-124）
            res_num = residue.id[1]
            if not (18 <= res_num <= 124):
                continue
            fkey = residue_key(residue)
            # 检查侧链原子是否有接触
            sidechain = get_sidechain_atoms(residue)
            if not sidechain:
                continue
            for atom in sidechain:
                nearby = ns.search(atom.coord, cutoff)
                if nearby:
                    for igm_atom in nearby:
                        igm_res = igm_atom.get_parent()
                        ikey = residue_key(igm_res)
                        current_chain_map[fkey].add(ikey)
                    # 只要有一个侧链原子有接触即可，无需继续检查该残基的其他原子
                    break
        chain_pair_map.append(current_chain_map)

    if total_fc_chains == 0:
        return {}

    # 统计每个残基对出现在多少条链中
    pair_count = defaultdict(int)
    for cmap in chain_pair_map:
        for fkey, igm_set in cmap.items():
            for ikey in igm_set:
                pair_count[(fkey, ikey)] += 1

    # 转换为比例
    fractions = {pair: cnt / total_fc_chains for pair, cnt in pair_count.items()}
    return fractions


# ============================================================
# 主程序
# ============================================================
def main():
    # 存储每个残基对在所有PDB中出现的比例（跨PDB取平均值？或累加？）
    # 这里采用：跨PDB取平均值（因为每个PDB中的FcμR链数不同，比例已经归一化到[0,1]）
    pair_fractions = defaultdict(list)  # {(f, i): list of fractions from each PDB}

    for pdb_id in PDB_IDS:
        print(f"Processing {pdb_id}...")
        entry = CHAIN_MAP.get(pdb_id)
        if not entry:
            continue
        fpath = entry['file']
        if not os.path.exists(fpath):
            print(f"  File not found: {fpath}")
            continue
        parser = PDBParser(QUIET=True)
        try:
            struct = parser.get_structure(pdb_id, fpath)
        except Exception as e:
            print(f"  Parse error: {e}")
            continue
        fractions = get_pair_contact_fractions(struct, entry['FcμR'], entry['IgM'], DIST_CUTOFF)
        for pair, frac in fractions.items():
            pair_fractions[pair].append(frac)
        print(f"  Found {len(fractions)} residue pairs")

    # 计算每个残基对的平均比例（跨所有PDB）
    avg_fractions = {pair: np.mean(vals) for pair, vals in pair_fractions.items()}

    if not avg_fractions:
        print("No contacts found.")
        return

    # 提取所有唯一的FcμR残基和IgM残基
    fc_residues = sorted({p[0] for p in avg_fractions}, key=get_residue_number)
    igm_residues = sorted({p[1] for p in avg_fractions}, key=get_residue_number)

    # 构建矩阵
    mat = np.zeros((len(fc_residues), len(igm_residues)))
    for (f, i), frac in avg_fractions.items():
        mat[fc_residues.index(f), igm_residues.index(i)] = frac

    df = pd.DataFrame(mat, index=fc_residues, columns=igm_residues)

    # 绘制热图
    plt.figure(figsize=(20, 16), dpi=1000)
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="Reds",
                     linewidths=0.5, linecolor='gray',
                     cbar_kws={'label': 'Average contact frequency (0-1)'})
    ax.set_xlabel("Immunoglobulin heavy constant μ residues", fontsize=16)
    ax.set_ylabel("FcμR-D1 residues", fontsize=16)
    ax.set_title("FcμR‑IgM residue pair contact frequency (side-chain, distance ≤ 4.5 Å)\n"
                 "Value = proportion of FcμR chains with contact, averaged over 7 complexes", fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=1000, bbox_inches='tight')
    plt.close()
    print(f"\nHeatmap saved as: {OUTPUT_IMAGE}")

    # 可选：输出前20个高频残基对
    sorted_pairs = sorted(avg_fractions.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 20 residue pairs (average frequency):")
    for (f, i), frac in sorted_pairs[:20]:
        print(f"  {f} - {i}: {frac:.3f}")


if __name__ == "__main__":
    main()