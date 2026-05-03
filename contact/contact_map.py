import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBParser
import matplotlib.patches as mpatches
from matplotlib import colormaps
import os


def analyze_protein_contacts(pdb_file, receptor_chains, ligand_chains, cutoff=4.5):
    """
    分析蛋白质接触的完整函数
    """
    print(f"开始分析 {pdb_file}...")

    try:
        # 检查文件是否存在
        if not os.path.exists(pdb_file):
            print(f"错误: 文件 {pdb_file} 不存在")
            return None

        # 解析PDB文件
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('complex', pdb_file)
        model = structure[0]

        print(f"找到的链: {[chain.id for chain in model]}")

        # 检查指定的链是否存在
        all_chains = receptor_chains + ligand_chains
        missing_chains = [chain for chain in all_chains if chain not in model]

        if missing_chains:
            print(f"错误: 以下链不存在: {missing_chains}")
            return None

        contacts = []

        # 分析每个配体链与每个受体链的接触
        for lig_chain in ligand_chains:
            for rec_chain in receptor_chains:
                print(f"分析 {rec_chain} (受体) ↔ {lig_chain} (配体) 接触...")

                # 获取原子（排除氢原子）
                rec_atoms = []
                for atom in model[rec_chain].get_atoms():
                    if atom.element != 'H':  # 排除氢原子
                        rec_atoms.append(atom)

                lig_atoms = []
                for atom in model[lig_chain].get_atoms():
                    if atom.element != 'H':  # 排除氢原子
                        lig_atoms.append(atom)

                print(f"  受体链 {rec_chain} 有 {len(rec_atoms)} 个非氢原子")
                print(f"  配体链 {lig_chain} 有 {len(lig_atoms)} 个非氢原子")

                # 计算距离
                for lig_atom in lig_atoms:
                    for rec_atom in rec_atoms:
                        distance = np.linalg.norm(lig_atom.coord - rec_atom.coord)
                        if distance < cutoff:
                            rec_res = rec_atom.get_parent()
                            lig_res = lig_atom.get_parent()

                            # 判断相互作用类型
                            interaction_type = classify_interaction(rec_atom, lig_atom, distance)

                            contacts.append({
                                'receptor_chain': rec_chain,
                                'receptor_residue': f"{rec_res.resname}{rec_res.id[1]}",
                                'receptor_resid': rec_res.id[1],
                                'ligand_chain': lig_chain,
                                'ligand_residue': f"{lig_res.resname}{lig_res.id[1]}",
                                'distance': distance,
                                'interaction_type': interaction_type,
                                'receptor_atom': rec_atom.name,
                                'ligand_atom': lig_atom.name
                            })

        print(f"分析完成，共找到 {len(contacts)} 个接触")
        return pd.DataFrame(contacts)

    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        return None


def classify_interaction(atom_rec, atom_lig, distance):
    """判断相互作用的类型"""
    elem_rec, elem_lig = atom_rec.element, atom_lig.element
    elements = {elem_rec, elem_lig}

    # 氢键
    if elements <= {'N', 'O'} and distance < 3.5:
        return 'Hydrogen Bond'
    # 盐桥（带相反电荷的原子）
    elif ({'N', 'O'} in elements and distance < 4.0):
        return 'Salt Bridge'
    # 疏水作用
    elif elements <= {'C', 'S'}:
        return 'Hydrophobic'
    # π-π堆积（芳香族原子）
    elif (distance < 5.0 and
          any(name in atom_rec.name for name in ['CG', 'CD', 'CE', 'CZ']) and
          any(name in atom_lig.name for name in ['CG', 'CD', 'CE', 'CZ'])):
        return 'π-π Stacking'
    else:
        return 'Other'


def create_comprehensive_dual_map(contacts_df, output_file='comprehensive_dual_map.png'):
    """创建综合性的双向残基接触图谱"""

    if contacts_df is None or contacts_df.empty:
        print("没有接触数据可绘制")
        return

    # 创建更大的图形
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 0.5],
                          hspace=0.4, wspace=0.3)

    # 定义子图位置
    ax_main = fig.add_subplot(gs[0, 0])  # 主接触图
    ax_heatmap = fig.add_subplot(gs[0, 1])  # 热力图
    ax_receptor = fig.add_subplot(gs[1, 0])  # 受体残基详情
    ax_ligand = fig.add_subplot(gs[1, 1])  # 配体残基详情
    ax_legend = fig.add_subplot(gs[2, :])  # 图例

    ax_legend.axis('off')  # 关闭图例轴显示

    # 颜色映射
    distance_cmap = colormaps['viridis_r']

    # 1. 主接触图：显示所有接触点
    unique_receptor_chains = contacts_df['receptor_chain'].unique()
    unique_ligand_chains = contacts_df['ligand_chain'].unique()

    # 为每个链对创建唯一的y坐标
    y_positions = {}
    y_ticks = []
    y_tick_labels = []

    current_y = 0
    for rec_chain in unique_receptor_chains:
        for lig_chain in unique_ligand_chains:
            chain_pair = f"FcµR{rec_chain}-sIgM{lig_chain}"
            y_positions[chain_pair] = current_y
            y_ticks.append(current_y)
            y_tick_labels.append(chain_pair)
            current_y += 1

    # 绘制主接触图
    all_x, all_y, all_distances, all_colors = [], [], [], []
    critical_interactions = []

    for _, row in contacts_df.iterrows():
        chain_pair = f"FcµR{row['receptor_chain']}-sIgM{row['ligand_chain']}"
        y_pos = y_positions[chain_pair]

        all_x.append(row['receptor_resid'])
        all_y.append(y_pos)
        all_distances.append(row['distance'])

        # 根据距离确定颜色
        if row['distance'] < 3.0:
            color = 'red'
            critical_interactions.append(row)
        elif row['distance'] < 3.5:
            color = 'orange'
        else:
            color = distance_cmap((row['distance'] - 2.0) / 2.5)

        all_colors.append(color)

    scatter = ax_main.scatter(all_x, all_y, c=all_colors, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

    # 设置主图标签
    ax_main.set_yticks(y_ticks)
    ax_main.set_yticklabels(y_tick_labels, fontsize=10)
    ax_main.set_xlabel('FcµR Residue Number', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('Receptor-Ligand Chain Pairs', fontsize=12, fontweight='bold')
    ax_main.set_title('FcµR - sIgM Comprehensive Interaction Map', fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.2)

    # 2. 热力图：显示残基接触频率
    pivot_data = contacts_df.pivot_table(
        values='distance',
        index='receptor_residue',
        columns='ligand_residue',
        aggfunc='count',
        fill_value=0
    )

    if not pivot_data.empty:
        im = ax_heatmap.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        ax_heatmap.set_xticks(range(len(pivot_data.columns)))
        ax_heatmap.set_xticklabels(pivot_data.columns, rotation=45, ha='right', fontsize=8)
        ax_heatmap.set_yticks(range(len(pivot_data.index)))
        ax_heatmap.set_yticklabels(pivot_data.index, fontsize=8)
        ax_heatmap.set_xlabel('sIgM Residues', fontsize=10, fontweight='bold')
        ax_heatmap.set_ylabel('FcµR Residues', fontsize=10, fontweight='bold')
        ax_heatmap.set_title('Contact Frequency Heatmap', fontsize=12, fontweight='bold')

        # 添加热力图颜色条
        cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
        cbar.set_label('Number of Contacts', rotation=270, labelpad=15)

    # 3. 受体残基详情
    receptor_stats = contacts_df.groupby('receptor_residue').agg({
        'distance': ['min', 'mean', 'count'],
        'ligand_residue': lambda x: ', '.join(sorted(set(x)))
    }).reset_index()

    receptor_stats.columns = ['residue', 'min_dist', 'mean_dist', 'contact_count', 'partners']
    receptor_stats = receptor_stats.nlargest(10, 'contact_count')

    bars_receptor = ax_receptor.barh(range(len(receptor_stats)), receptor_stats['contact_count'],
                                     color='lightcoral', edgecolor='darkred', alpha=0.7)

    ax_receptor.set_yticks(range(len(receptor_stats)))
    ax_receptor.set_yticklabels(
        [f"{r} ({d:.2f}Å)" for r, d in zip(receptor_stats['residue'], receptor_stats['min_dist'])],
        fontsize=9)
    ax_receptor.set_xlabel('Number of Contacts', fontsize=10, fontweight='bold')
    ax_receptor.set_ylabel('FcµR Residue (Min Distance)', fontsize=10, fontweight='bold')
    ax_receptor.set_title('Top 10 FcµR Residues by Contact Count', fontsize=12, fontweight='bold')
    ax_receptor.grid(True, alpha=0.2, axis='x')

    # 添加受体残基的配体伙伴信息
    for i, (_, row) in enumerate(receptor_stats.iterrows()):
        ax_receptor.text(row['contact_count'] + 0.1, i, f"→ {row['partners']}",
                         va='center', fontsize=7, fontstyle='italic')

    # 4. 配体残基详情
    ligand_stats = contacts_df.groupby('ligand_residue').agg({
        'distance': ['min', 'mean', 'count'],
        'receptor_residue': lambda x: ', '.join(sorted(set(x)))
    }).reset_index()

    ligand_stats.columns = ['residue', 'min_dist', 'mean_dist', 'contact_count', 'partners']
    ligand_stats = ligand_stats.nlargest(10, 'contact_count')

    bars_ligand = ax_ligand.barh(range(len(ligand_stats)), ligand_stats['contact_count'],
                                 color='lightblue', edgecolor='darkblue', alpha=0.7)

    ax_ligand.set_yticks(range(len(ligand_stats)))
    ax_ligand.set_yticklabels([f"{r} ({d:.2f}Å)" for r, d in zip(ligand_stats['residue'], ligand_stats['min_dist'])],
                              fontsize=9)
    ax_ligand.set_xlabel('Number of Contacts', fontsize=10, fontweight='bold')
    ax_ligand.set_title('Top 10 sIgM Residues by Contact Count', fontsize=12, fontweight='bold')
    ax_ligand.grid(True, alpha=0.2, axis='x')

    # 添加配体残基的受体伙伴信息
    for i, (_, row) in enumerate(ligand_stats.iterrows()):
        ax_ligand.text(row['contact_count'] + 0.1, i, f"→ {row['partners']}",
                       va='center', fontsize=7, fontstyle='italic')

    # 5. 创建综合图例
    legend_elements = [
        mpatches.Patch(facecolor='red', label='Very Strong (< 3.0 Å)', alpha=0.7),
        mpatches.Patch(facecolor='orange', label='Strong (3.0-3.5 Å)', alpha=0.7),
        mpatches.Patch(facecolor='lightcoral', label='FcµR Residues', alpha=0.7),
        mpatches.Patch(facecolor='lightblue', label='sIgM Residues', alpha=0.7),
        mpatches.Patch(facecolor='lightgray', label=f'Total Contacts: {len(contacts_df)}', alpha=0.7)
    ]

    ax_legend.legend(handles=legend_elements, loc='center', ncol=3,
                     frameon=True, fancybox=True, shadow=True, fontsize=11)

    # 添加统计信息到图例区域
    stats_text = (
        f"FcµR Residues: {contacts_df['receptor_residue'].nunique()} | "
        f"sIgM Residues: {contacts_df['ligand_residue'].nunique()} | "
        f"Avg Distance: {contacts_df['distance'].mean():.2f} Å | "
        f"Min Distance: {contacts_df['distance'].min():.2f} Å"
    )

    ax_legend.text(0.5, 0.2, stats_text, ha='center', va='center',
                   fontsize=12, fontweight='bold', transform=ax_legend.transAxes)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    # 打印关键相互作用对
    print_critical_interactions(critical_interactions)

    return contacts_df


def print_critical_interactions(critical_interactions):
    """打印关键相互作用对"""
    if critical_interactions:
        print("\n🔬 关键强相互作用对 (距离 < 3.0 Å):")
        print("=" * 70)
        print(f"{'FcµR Residue':<15} {'sIgM Residue':<15} {'Distance (Å)':<12} {'Chain Pair':<15}")
        print("-" * 70)

        for interaction in sorted(critical_interactions, key=lambda x: x['distance']):
            chain_pair = f"FcµR{interaction['receptor_chain']}-sIgM{interaction['ligand_chain']}"
            print(f"{interaction['receptor_residue']:<15} {interaction['ligand_residue']:<15} "
                  f"{interaction['distance']:<12.2f} {chain_pair:<15}")
    else:
        print("未找到距离 < 3.0 Å 的关键相互作用")


# 主程序
def main():
    """主函数"""

    print("FcµR - sIgM 综合性双向残基相互作用分析")
    print("=" * 60)

    # 配置参数 - 请根据您的实际结构修改
    PDB_FILE = "source_file/7YSG.pdb"  # 替换为您的PDB文件
    RECEPTOR_CHAINS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L','J','P']  # sIgM配体链
    LIGAND_CHAINS = ['R', 'S', 'U', 'V']  # FcµR受体链

    print(f"分析文件: {PDB_FILE}")
    print(f"FcµR受体链: {RECEPTOR_CHAINS}")
    print(f"sIgM配体链: {LIGAND_CHAINS}")

    # 执行分析
    contacts_df = analyze_protein_contacts(PDB_FILE, RECEPTOR_CHAINS, LIGAND_CHAINS, cutoff=4.5)

    if contacts_df is not None:
        # 创建综合性图谱
        create_comprehensive_dual_map(contacts_df, 'FcμR_sIgM_comprehensive_map.png')

        # 保存详细数据
        contacts_df.to_csv('FcμR_sIgM_detailed_interactions.csv', index=False)
        print("详细相互作用数据已保存")

        # 打印统计摘要
        print(f"\n📊 统计摘要:")
        print(f"   总接触数: {len(contacts_df)}")
        print(f"   涉及的FcµR残基数: {contacts_df['receptor_residue'].nunique()}")
        print(f"   涉及的sIgM残基数: {contacts_df['ligand_residue'].nunique()}")
        print(f"   平均距离: {contacts_df['distance'].mean():.2f} Å")
        print(f"   最小距离: {contacts_df['distance'].min():.2f} Å")

    else:
        print("分析失败，请检查文件路径和链ID")


# 运行程序
if __name__ == "__main__":
    main()