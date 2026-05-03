import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import PDBParser, NeighborSearch, is_aa
from collections import defaultdict

# ==================== 配置 ====================
PDB_IDS = ['7YTE', '7YTC', '7YTD', '7YSG', '8BPE', '8BPF', '8BPG']

CHAIN_MAP = {
    "7YTE": {"file": "source_file/7YTE.pdb", "IgM": ["A","B"], "FcμR": ["C","D"]},
    "7YTC": {"file": "source_file/7YTC.pdb", "IgM": ["A","B","C","D","E","F","G","H","K","L","J"],
             "FcμR": ["R"]},
    "7YTD": {"file": "source_file/7YTD.pdb", "IgM": ["A","B","C","D","E","F","G","H","K","L","J"],
             "FcμR": ["R","S","U","V"]},
    "7YSG": {"file": "source_file/7YSG.pdb", "IgM": ["A","B","C","D","E","F","G","H","K","L","J","P"],
             "FcμR": ["U","R","S","V"]},
    "8BPE": {"file": "source_file/8BPE.pdb", "IgM": ["A","B","C","D","E","F","G","H","K","L","J"],
             "FcμR": ["I","M","N","O","P","Q","R","S"]},
    "8BPF": {"file": "source_file/8BPF.pdb", "IgM": ["A","B","C","D","E","F","G","H","K","L","J"],
             "FcμR": ["I"]},
    "8BPG": {"file": "source_file/8BPG.pdb", "IgM": ["C","D","E","F"], "FcμR": ["A","B"]},
}

DIST_CUTOFF = 4.5
OUTPUT_PNG = "Extended_Figure1_FcμR_per_PDB.png"
OUTPUT_PDF = "Extended_Figure1_FcμR_per_PDB.pdf"

def force_format(name):
    if len(name) < 4:
        return name
    return name[:3].capitalize() + name[3:]

def get_sidechain_atoms(residue):
    backbone = {'N', 'CA', 'C', 'O'}
    if residue.resname == 'GLY':
        return [atom for atom in residue.get_atoms() if atom.name == 'CA']
    return [atom for atom in residue.get_atoms() if atom.name not in backbone]

def get_residue_contact_fraction(structure, fcmr_chains, igm_chains, cutoff):
    """返回字典 {fcmr_res: 比例}，比例 = 有接触的FcμR链数 / 总链数"""
    igm_atoms = []
    for ch in igm_chains:
        if ch in structure[0]:
            igm_atoms.extend(structure[0][ch].get_atoms())
    if not igm_atoms:
        return {}
    ns = NeighborSearch(igm_atoms)
    total_chains = 0
    residue_counts = defaultdict(int)
    for ch_id in fcmr_chains:
        if ch_id not in structure[0]:
            continue
        total_chains += 1
        residues_this_chain = set()
        for residue in structure[0][ch_id].get_residues():
            if not is_aa(residue):
                continue
            sidechain = get_sidechain_atoms(residue)
            if not sidechain:
                continue
            fkey = force_format(f"{residue.resname}{residue.id[1]}")
            for atom in sidechain:
                if ns.search(atom.coord, cutoff):
                    residues_this_chain.add(fkey)
                    break
        for r in residues_this_chain:
            residue_counts[r] += 1
    if total_chains == 0:
        return {}
    return {r: cnt/total_chains for r, cnt in residue_counts.items()}

def main():
    # 存储每个残基在每个PDB中的比例
    data = defaultdict(lambda: [0.0]*len(PDB_IDS))
    for idx, pdb in enumerate(PDB_IDS):
        print(f"Processing {pdb}...")
        entry = CHAIN_MAP[pdb]
        if not entry or not os.path.exists(entry['file']):
            continue
        struct = PDBParser(QUIET=True).get_structure(pdb, entry['file'])
        fracs = get_residue_contact_fraction(struct, entry['FcμR'], entry['IgM'], DIST_CUTOFF)
        for res, val in fracs.items():
            data[res][idx] = val
    # 构建DataFrame，按残基编号排序
    def extract_num(r):
        import re
        return int(re.search(r'\d+', r).group())
    residues = sorted(data.keys(), key=extract_num)
    df = pd.DataFrame([data[r] for r in residues], index=residues, columns=PDB_IDS)
    # 绘图
    plt.figure(figsize=(10, 12), dpi=1200)
    ax = sns.heatmap(df, cmap='YlOrRd', annot=True, fmt=".2f",
                     linewidths=0.5, cbar_kws={'label': 'Proportion of FcμR chains with contact'})
    ax.set_xlabel("PDB ID", fontsize=18)
    ax.set_ylabel("FcμR-D1 residues", fontsize=18)
    ax.set_title("FcμR-D1 residue contact frequency across different IgM complexes", fontsize=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=1200, bbox_inches='tight')
    plt.savefig(OUTPUT_PDF, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved as: {OUTPUT_PNG} and {OUTPUT_PDF}")

if __name__ == "__main__":
    main()