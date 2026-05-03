import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from scipy.spatial import cKDTree
from collections import defaultdict

# ========== 配置 ==========
PDB_FILES = {
    "7YTE": {"file": "source_file/7YTE.pdb", "IgM": ["A", "B"], "FcμR": ["C", "D"]},
    "7YTC": {"file": "source_file/7YTC.pdb", "IgM": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
             "FcμR": ["R"]},
    "7YTD": {"file": "source_file/7YTD.pdb", "IgM": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
             "FcμR": ["R", "S", "U", "V"]},
    "7YSG": {"file": "source_file/7YSG.pdb", "IgM": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
             "FcμR": ["U", "R", "S", "V"]},
    "8BPE": {"file": "source_file/8BPE.pdb", "IgM": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
             "FcμR": ["I", "M", "N", "O", "P", "Q", "R", "S"]},
    "8BPF": {"file": "source_file/8BPF.pdb", "IgM": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
             "FcμR": ["I"]},
    "8BPG": {"file": "source_file/8BPG.pdb", "IgM": ["C", "D", "E", "F"], "FcμR": ["A", "B"]},
}
CUTOFF = 3.0
FCR_RANGE = range(18, 126)


# ========== 辅助函数 ==========
def get_contacts(pdb, fc_chain, igm_chain, cutoff):
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("", pdb)
    # 获取 IgM 链原子
    igm_coords = []
    igm_res = []
    for model in struct:
        for chain in model:
            if chain.id != igm_chain: continue
            for res in chain:
                if not is_aa(res): continue
                for atom in res:
                    igm_coords.append(atom.coord)
                    igm_res.append((res.id[1], res.resname.capitalize()))
    if not igm_coords:
        return set()
    tree = cKDTree(igm_coords)
    contacts = set()
    for model in struct:
        for chain in model:
            if chain.id != fc_chain: continue
            for res in chain:
                if not is_aa(res): continue
                fc_num = res.id[1]
                if fc_num not in FCR_RANGE: continue
                for atom in res:
                    idxs = tree.query_ball_point(atom.coord, cutoff)
                    if idxs:
                        for idx in idxs:
                            igm_num, _ = igm_res[idx]
                            contacts.add((fc_num, igm_num))
                        break
    return contacts


def get_resname_map(pdb, chains, res_range=None):
    name_map = {}
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("", pdb)
    for model in struct:
        for chain in model:
            if chain.id not in chains: continue
            for res in chain:
                if not is_aa(res): continue
                rnum = res.id[1]
                if res_range and rnum not in res_range: continue
                if rnum not in name_map:
                    name_map[rnum] = f"{res.resname.capitalize()}{rnum}"
    if res_range:
        for r in res_range:
            if r not in name_map:
                name_map[r] = str(r)
    return name_map


# ========== 主程序 ==========
def main():
    # 收集每个复合物的二元接触矩阵
    complexes = []
    for name, info in PDB_FILES.items():
        if not os.path.exists(info["file"]):
            print(f"Missing {info['file']}, skip {name}")
            continue
        print(f"Processing {name}...")
        all_contacts = set()
        for fc in info["FcμR"]:
            for igm in info["IgM"]:
                all_contacts.update(get_contacts(info["file"], fc, igm, CUTOFF))
        if not all_contacts:
            continue
        fc_res = sorted({c[0] for c in all_contacts if c[0] in FCR_RANGE})
        igm_res = sorted({c[1] for c in all_contacts})
        mat = np.zeros((len(fc_res), len(igm_res)), dtype=int)
        for i, fr in enumerate(fc_res):
            for j, ir in enumerate(igm_res):
                if (fr, ir) in all_contacts:
                    mat[i, j] = 1
        # 获取标签
        fc_map = get_resname_map(info["file"], info["FcμR"], FCR_RANGE)
        igm_map = get_resname_map(info["file"], info["IgM"])
        rows = [fc_map.get(r, str(r)) for r in fc_res]
        cols = [igm_map.get(r, str(r)) for r in igm_res]
        complexes.append((name, mat, rows, cols))

    # 绘制 Figure 2a (多面板)
    n = len(complexes)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    cmap_bin = ListedColormap(['white', '#2c7bb6'])
    for i, (name, mat, rows, cols) in enumerate(complexes):
        ax = axes[i]
        df = pd.DataFrame(mat, index=rows, columns=cols)
        sns.heatmap(df, cmap=cmap_bin, cbar=False, linewidths=0.2, ax=ax)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel('IgM residue', fontsize=7)
        ax.set_ylabel('FcμR residue', fontsize=7)
        ax.tick_params(axis='x', rotation=45, labelsize=5)
        ytl = ax.get_yticklabels()
        for l in ytl: l.set_rotation(0); l.set_ha('right')
        ax.set_yticklabels(ytl, fontsize=5)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    # 添加颜色条
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_bin)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    fig.colorbar(sm, cax=cbar_ax, label='Contact (1=yes, 0=no)')
    plt.suptitle('a', fontsize=20, x=0.02, y=0.98, ha='left')
    plt.subplots_adjust(left=0.08, bottom=0.12, right=0.90, top=0.92, wspace=0.4, hspace=0.5)
    plt.savefig("Figure2a.png", dpi=300)
    plt.show()

    # 绘制 Figure 2b (汇总频率)
    # 重新统计所有链对的频率
    pair_cnt = defaultdict(int)
    total_pairs = 0
    all_fc = set()
    all_igm = set()
    for name, info in PDB_FILES.items():
        if not os.path.exists(info["file"]): continue
        for fc in info["FcμR"]:
            for igm in info["IgM"]:
                total_pairs += 1
                contacts = get_contacts(info["file"], fc, igm, CUTOFF)
                for (fr, ir) in contacts:
                    pair_cnt[(fr, ir)] += 1
                    all_fc.add(fr);
                    all_igm.add(ir)
    fc_list = sorted([r for r in FCR_RANGE if r in all_fc])
    igm_list = sorted(all_igm)
    freq_mat = np.zeros((len(fc_list), len(igm_list)))
    for i, fr in enumerate(fc_list):
        for j, ir in enumerate(igm_list):
            freq_mat[i, j] = pair_cnt.get((fr, ir), 0) / total_pairs
    # 标签（从第一个文件获取名称）
    first_pdb = next(iter(PDB_FILES.values()))["file"]
    first_fc = next(iter(PDB_FILES.values()))["FcμR"]
    first_igm = next(iter(PDB_FILES.values()))["IgM"]
    fc_map = get_resname_map(first_pdb, first_fc, FCR_RANGE)
    igm_map = get_resname_map(first_pdb, first_igm)
    rows = [fc_map.get(r, str(r)) for r in fc_list]
    cols = [igm_map.get(r, str(r)) for r in igm_list]
    df_freq = pd.DataFrame(freq_mat, index=rows, columns=cols)
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_freq, cmap='YlOrRd', cbar_kws={'label': 'Contact frequency'})
    plt.xlabel('IgM residue', fontsize=12)
    plt.ylabel('FcμR residue', fontsize=12)
    plt.title('b', fontsize=20, loc='left', x=-0.02, y=1.02)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig("Figure2b.png", dpi=300)
    plt.show()
    print("Saved Figure2a.png and Figure2b.png")


if __name__ == "__main__":
    main()