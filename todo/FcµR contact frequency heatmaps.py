
import os
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
    "7YTE": {"file": "source_file/7YTE.pdb", "IgM": ["A","B"], "FcμR": ["C","D"]},
    "7YTC": {"file": "source_file/7YTC.pdb", "IgM": ["A","B","C","D","E","F","G","H","K","L","J"], "FcμR": ["R"]},
    "7YTD": {"file": "source_file/7YTD.pdb", "IgM": ["A","B","C","D","E","F","G","H","K","L","J"], "FcμR": ["R","S","U","V"]},
    "7YSG": {"file": "source_file/7YSG.pdb", "IgM": ["A","B","C","D","E","F","G","H","K","L","J","P"], "FcμR": ["U","R","S","V"]},
    "8BPE": {"file": "source_file/8BPE.pdb", "IgM": ["A","B","C","D","E","F","G","H","K","L","J"], "FcμR": ["I","M","N","O","P","Q","R","S"]},
    "8BPF": {"file": "source_file/8BPF.pdb", "IgM": ["A","B","C","D","E","F","G","H","K","L","J"], "FcμR": ["I"]},
    "8BPG": {"file": "source_file/8BPG.pdb", "IgM": ["C","D","E","F"], "FcμR": ["A","B"]},
}

DIST_CUTOFF = 4.5
OUTPUT_IMAGE = "FcµR contact frequency heatmaps.png"

# ============================================================
# 强制格式化函数
# ============================================================
def force_format(name):
    """将任意残基名转换为首字母大写其余小写，例如'Thr65',  'Arg45'"""
    if len(name) < 4:
        return name
    three = name[:3].capitalize()  # 关键：首字母大写，后两个小写
    num = name[3:]
    return f"{three}{num}"

def get_sidechain_atoms(residue):
    backbone = {'N', 'CA', 'C', 'O'}
    if residue.resname == 'GLY':
        return [atom for atom in residue.get_atoms() if atom.name == 'CA']
    return [atom for atom in residue.get_atoms() if atom.name not in backbone]

def get_participation_ratio(structure, fcmr_chains, igm_chains, cutoff):
    igm_atoms = []
    for ch in igm_chains:
        if ch in structure[0]:
            for atom in structure[0][ch].get_atoms():
                igm_atoms.append(atom)
    if not igm_atoms:
        return {}
    ns = NeighborSearch(igm_atoms)
    total_chains = 0
    chain_residues = []
    for ch_id in fcmr_chains:
        if ch_id not in structure[0]:
            continue
        total_chains += 1
        residues = set()
        for residue in structure[0][ch_id].get_residues():
            if not is_aa(residue):
                continue
            sidechain = get_sidechain_atoms(residue)
            if not sidechain:
                continue
            for atom in sidechain:
                if ns.search(atom.coord, cutoff):
                    raw = f"{residue.resname}{residue.id[1]}"
                    residues.add(force_format(raw))
                    break
        chain_residues.append(residues)
    if total_chains == 0:
        return {}
    numerator = defaultdict(int)
    for rset in chain_residues:
        for r in rset:
            numerator[r] += 1
    return {r: n/total_chains for r, n in numerator.items()}

# ============================================================
# 主程序
# ============================================================
def main():
    data = defaultdict(lambda: [0.0]*len(PDB_IDS))
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
        props = get_participation_ratio(struct, entry['FcμR'], entry['IgM'], DIST_CUTOFF)
        for res, val in props.items():
            data[res][idx] = val
        print(f"  Found {len(props)} residues")

    # 构建DataFrame
    df = pd.DataFrame.from_dict(data, orient='index', columns=PDB_IDS).fillna(0)
    df['mean'] = df.mean(axis=1)
    df_sorted = df.sort_values('mean', ascending=False).drop('mean', axis=1)

    # 强制将索引中的所有残基名重新格式化一次（确保万无一失）
    new_index = [force_format(name) for name in df_sorted.index]
    df_sorted.index = new_index

    # 打印前20个标签用于验证
    print("\n纵坐标标签示例（前20个）：")
    for i, label in enumerate(df_sorted.index[:20]):
        print(f"  {label}")

    # 绘图
    plt.figure(figsize=(12, 10), dpi=1200)
    ax = sns.heatmap(df_sorted, cmap='Reds', annot=False, linewidths=0.5,
                     cbar_kws={'label': 'Proportion of FcµR chains with side-chain contact'})
    ax.set_xlabel("PDB ID", fontsize=16)
    ax.set_ylabel("FcµR-D1 residue", fontsize=16)
    ax.set_title("FcμR-D1 residue contact frequency heatmaps", fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xticklabels(PDB_IDS, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    plt.close()
    print(f"\nFigure 2 saved as: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()