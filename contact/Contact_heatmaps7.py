import os
import re
import pandas as pd
import numpy as np
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

DIST_CUTOFF = 4.5
OUTPUT_IMAGE = "Contact_heatmap7.png"


# ============================================================
# 强制格式化函数（残基名首字母大写，其余小写）
# ============================================================
def force_format(name):
    """将任意残基名转换为首字母大写其余小写，例如'Thr65', 'Arg45'"""
    if len(name) < 4:
        return name
    three = name[:3].capitalize()
    num = name[3:]
    return f"{three}{num}"


def get_sidechain_atoms(residue):
    backbone = {'N', 'CA', 'C', 'O'}
    if residue.resname == 'GLY':
        return [atom for atom in residue.get_atoms() if atom.name == 'CA']
    return [atom for atom in residue.get_atoms() if atom.name not in backbone]


def get_pair_participation_ratio(structure, fcmr_chains, igm_chains, cutoff):
    """
    修改自原 get_participation_ratio。
    返回字典 {(fcmr_res, igm_res): 比例}，
    其中比例 = 出现该残基对的FcμR链数 / 总FcμR链数。
    """
    # 收集所有IgM原子
    igm_atoms = []
    for ch in igm_chains:
        if ch in structure[0]:
            for atom in structure[0][ch].get_atoms():
                igm_atoms.append(atom)
    if not igm_atoms:
        return {}
    ns = NeighborSearch(igm_atoms)

    total_chains = 0
    # 记录每条链中每个FcμR残基接触的IgM残基集合
    chain_pair_sets = []  # 元素为 dict: {fcmr_res: set of igm_res}

    for ch_id in fcmr_chains:
        if ch_id not in structure[0]:
            continue
        total_chains += 1
        current_map = defaultdict(set)
        for residue in structure[0][ch_id].get_residues():
            if not is_aa(residue):
                continue
            # 可选：限制FcμR残基范围 18-124（如果您的PDB编号符合）
            # if not (18 <= residue.id[1] <= 124): continue
            sidechain = get_sidechain_atoms(residue)
            if not sidechain:
                continue
            for atom in sidechain:
                if ns.search(atom.coord, cutoff):
                    raw_f = f"{residue.resname}{residue.id[1]}"
                    fkey = force_format(raw_f)
                    # 找出具体接触的IgM残基
                    for igm_atom in ns.search(atom.coord, cutoff):
                        igm_res = igm_atom.get_parent()
                        raw_i = f"{igm_res.resname}{igm_res.id[1]}"
                        ikey = force_format(raw_i)
                        current_map[fkey].add(ikey)
                    # 只要有一个侧链原子有接触，该残基就算接触（但还需记录具体IgM）
                    # 注意：一个FcμR残基可能接触多个IgM残基，这里全部记录
                    # 不break，以收集所有接触的IgM残基
        chain_pair_sets.append(current_map)

    if total_chains == 0:
        return {}

    # 统计每个残基对出现在多少条链中
    pair_count = defaultdict(int)
    for cmap in chain_pair_sets:
        for fkey, igm_set in cmap.items():
            for ikey in igm_set:
                pair_count[(fkey, ikey)] += 1

    # 转换为比例
    fractions = {pair: cnt / total_chains for pair, cnt in pair_count.items()}
    return fractions


# ============================================================
# 主程序
# ============================================================
def main():
    # 存储每个残基对在所有PDB中的比例列表
    pair_fractions = defaultdict(list)  # {(f, i): [val1, val2, ...]}

    for idx, pdb in enumerate(PDB_IDS):
        print(f"Processing {pdb}...")
        entry = CHAIN_MAP.get(pdb)
        if not entry:
            continue
        fpath = entry['file']
        if not os.path.exists(fpath):
            print(f"  File not found: {fpath}")
            continue
        parser = PDBParser(QUIET=True)
        try:
            struct = parser.get_structure(pdb, fpath)
        except Exception as e:
            print(f"  Parse error: {e}")
            continue
        props = get_pair_participation_ratio(struct, entry['FcμR'], entry['IgM'], DIST_CUTOFF)
        for pair, val in props.items():
            pair_fractions[pair].append(val)
        print(f"  Found {len(props)} residue pairs")

    if not pair_fractions:
        print("No contacts found.")
        return

    # 计算每个残基对的平均比例（跨所有PDB）
    avg_fractions = {pair: np.mean(vals) for pair, vals in pair_fractions.items()}

    # 提取所有唯一的FcμR残基和IgM残基，并按残基编号排序
    def extract_number(res_str):
        return int(re.search(r'\d+', res_str).group())

    fc_residues = sorted({p[0] for p in avg_fractions}, key=extract_number)
    igm_residues = sorted({p[1] for p in avg_fractions}, key=extract_number)

    # 构建矩阵
    mat = np.zeros((len(fc_residues), len(igm_residues)))
    for (f, i), val in avg_fractions.items():
        row = fc_residues.index(f)
        col = igm_residues.index(i)
        mat[row, col] = val

    df = pd.DataFrame(mat, index=fc_residues, columns=igm_residues)

    # 绘图（保持原脚本风格）
    plt.figure(figsize=(20, 16), dpi=1200)
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap='Reds',
                     linewidths=0.5, linecolor='gray',
                     cbar_kws={'label': 'Proportion of FcµR chains with side-chain contact'})
    ax.set_xlabel("Immunoglobulin heavy constant μ residues", fontsize=18)
    ax.set_ylabel("FcμR-D1 residues", fontsize=18)
    ax.set_title("FcμR-IgM residue pair contact frequency (side-chain, distance ≤ 4.5 Å)", fontsize=20)
    ax.tick_params(axis='x', labelsize=16, rotation=45)
    ax.tick_params(axis='y', labelsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=1200, bbox_inches='tight')
    plt.close()
    print(f"\nHeatmap saved as: {OUTPUT_IMAGE}")

    # 输出前20个高频对
    sorted_pairs = sorted(avg_fractions.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 20 residue pairs (average proportion):")
    for (f, i), val in sorted_pairs[:20]:
        print(f"  {f} - {i}: {val:.3f}")


if __name__ == "__main__":
    main()