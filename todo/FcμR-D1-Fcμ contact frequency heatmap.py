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

DIST_CUTOFF = 4.5                # 原子间距离阈值 (Å)
THRESHOLD = 0.0                  # 只显示平均比例 > 0.0 的残基对
OUTPUT_IMAGE_PNG = "FcμR-D1-Fcμ contact frequency heatmap.png"
OUTPUT_IMAGE_PDF = "FcμR-D1-Fcμ contact frequency heatmap.pdf"


def force_format(name):
    """将残基名转换为首字母大写其余小写，例如 'Thr65', 'Arg45'"""
    if len(name) < 4:
        return name
    three = name[:3].capitalize()
    num = name[3:]
    return f"{three}{num}"


def get_sidechain_atoms(residue):
    """获取侧链原子（排除主链原子 N, CA, C, O）；对于 GLY 使用 CA 作为代表"""
    backbone = {'N', 'CA', 'C', 'O'}
    if residue.resname == 'GLY':
        return [atom for atom in residue.get_atoms() if atom.name == 'CA']
    return [atom for atom in residue.get_atoms() if atom.name not in backbone]


def get_pair_participation_ratio(structure, fcmr_chains, igm_chains, cutoff):
    """
    对于单个 PDB 结构，计算每个 (FcμR残基, IgM残基) 对出现在多少比例的 FcμR 链中。
    返回值：字典 {(fcmr_res, igm_res): 比例}
    """
    # 收集所有 IgM 链的原子（用于快速空间搜索）
    igm_atoms = []
    for ch in igm_chains:
        if ch in structure[0]:
            igm_atoms.extend(structure[0][ch].get_atoms())
    if not igm_atoms:
        return {}
    ns = NeighborSearch(igm_atoms)

    total_fc_chains = 0
    pair_count = defaultdict(int)   # {(f, i): 出现该残基对的 FcμR 链数}

    for ch_id in fcmr_chains:
        if ch_id not in structure[0]:
            continue
        total_fc_chains += 1
        # 记录当前链中每个 FcμR 残基接触的所有 IgM 残基（去重）
        chain_map = defaultdict(set)   # {fcmr_key: set of igm_keys}
        for residue in structure[0][ch_id].get_residues():
            if not is_aa(residue):
                continue
            # 可选：限制 FcμR 残基编号范围 18-124（如果您的 PDB 编号符合）
            # if not (18 <= residue.id[1] <= 124): continue
            sidechain = get_sidechain_atoms(residue)
            if not sidechain:
                continue
            fkey = force_format(f"{residue.resname}{residue.id[1]}")
            for atom in sidechain:
                nearby = ns.search(atom.coord, cutoff)
                if nearby:
                    for igm_atom in nearby:
                        igm_res = igm_atom.get_parent()
                        ikey = force_format(f"{igm_res.resname}{igm_res.id[1]}")
                        chain_map[fkey].add(ikey)
        # 将当前链中所有出现的残基对计数 +1
        for fkey, igm_set in chain_map.items():
            for ikey in igm_set:
                pair_count[(fkey, ikey)] += 1

    if total_fc_chains == 0:
        return {}
    # 转换为比例（该残基对出现在多少比例的 FcμR 链中）
    fractions = {pair: cnt / total_fc_chains for pair, cnt in pair_count.items()}
    return fractions


def main():
    # 存储每个残基对在所有 PDB 中的比例列表
    pair_fractions = defaultdict(list)   # {(f, i): [frac1, frac2, ...]}

    for pdb in PDB_IDS:
        print(f"Processing {pdb}...")
        entry = CHAIN_MAP.get(pdb)
        if not entry:
            continue
        pdb_file = entry['file']
        if not os.path.exists(pdb_file):
            print(f"  File not found: {pdb_file}")
            continue
        parser = PDBParser(QUIET=True)
        try:
            struct = parser.get_structure(pdb, pdb_file)
        except Exception as e:
            print(f"  Parse error: {e}")
            continue
        fractions = get_pair_participation_ratio(struct, entry['FcμR'], entry['IgM'], DIST_CUTOFF)
        for pair, val in fractions.items():
            pair_fractions[pair].append(val)
        print(f"  Found {len(fractions)} residue pairs")

    if not pair_fractions:
        print("No contacts found.")
        return

    # 计算每个残基对在所有 PDB 中的平均比例
    avg_fractions = {pair: np.mean(vals) for pair, vals in pair_fractions.items()}

    # 提取所有唯一的 FcμR 残基和 IgM 残基，并按残基编号排序
    def extract_number(res_str):
        return int(re.search(r'\d+', res_str).group())

    fc_residues = sorted({p[0] for p in avg_fractions}, key=extract_number)
    igm_residues = sorted({p[1] for p in avg_fractions}, key=extract_number)

    # 构建矩阵：只保留平均比例 > THRESHOLD 的值，其余设为 NaN
    mat = np.full((len(fc_residues), len(igm_residues)), np.nan)
    for (f, i), val in avg_fractions.items():
        if val > THRESHOLD:
            row = fc_residues.index(f)
            col = igm_residues.index(i)
            mat[row, col] = val

    df = pd.DataFrame(mat, index=fc_residues, columns=igm_residues)

    # ========== 绘图美化部分 ==========
    # 设置全局字体为 sans-serif (Arial/Helvetica)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['pdf.fonttype'] = 42          # 确保文本可编辑
    plt.rcParams['ps.fonttype'] = 42

    fig, ax = plt.subplots(figsize=(20, 16), dpi=300)
    mask = df.isna()

    # 使用 'YlOrRd' 配色，从浅黄到深红，更醒目
    # 设置 vmin=0.0, vmax=1.0 可以增强高值对比，但会导致低于0.0的单元格不显示（已mask），所以不需要
    # 如果希望颜色从0.0开始渐变，可以设置 norm=plt.Normalize(0.0, 1.0)
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0.0, vmax=1.0)

    heatmap = sns.heatmap(df, annot=True, fmt=".2f", cmap='YlOrRd',
                          mask=mask, norm=norm,
                          linewidths=0.5, linecolor='white',
                          cbar_kws={'label': 'Average proportion of FcµR chains with contact',
                                    'shrink': 0.8, 'pad': 0.02},
                          square=False, ax=ax)
    # NaN 区域背景色
    ax.set_facecolor('#F0F0F0')

    # 坐标轴标签
    ax.set_xlabel("Immunoglobulin heavy constant μ residues", fontsize=18, fontweight='bold')
    ax.set_ylabel("FcμR-D1 residues", fontsize=18, fontweight='bold')
    ax.set_title(f"FcμR-D1-Fcμ contact frequency (side-chain, distance ≤ {DIST_CUTOFF} Å)\n",
                 fontsize=20, fontweight='bold', pad=20)

    # 横坐标标签：每隔2个显示一个，避免拥挤（可根据实际列数调整 step）
    step = max(1, len(igm_residues) // 20)   # 大约显示20个标签
    xticks = range(0, len(igm_residues), step)
    xticklabels = [igm_residues[i] if i % step == 0 else '' for i in range(len(igm_residues))]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=16)

    # 纵坐标全部显示（行数通常不多）
    ax.set_yticks(range(len(fc_residues)))
    ax.set_yticklabels(fc_residues, rotation=0, fontsize=16)

    # 添加外框线
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

    # 调整布局
    plt.tight_layout()

    # 保存为 PNG（高分辨率）和 PDF（矢量）
    plt.savefig(OUTPUT_IMAGE_PNG, dpi=1200, bbox_inches='tight')
    plt.savefig(OUTPUT_IMAGE_PDF, bbox_inches='tight')
    plt.close()
    print(f"\nHeatmap saved as: {OUTPUT_IMAGE_PNG} and {OUTPUT_IMAGE_PDF}")

    # 输出满足阈值的残基对统计
    above_threshold = [(f, i, v) for (f, i), v in avg_fractions.items() if v > THRESHOLD]
    above_threshold.sort(key=lambda x: x[2], reverse=True)
    print(f"\nTotal residue pairs with average proportion > {THRESHOLD}: {len(above_threshold)}")
    print("\nTop 20 such pairs:")
    for f, i, v in above_threshold[:20]:
        print(f"  {f} - {i}: {v:.3f}")


if __name__ == "__main__":
    main()