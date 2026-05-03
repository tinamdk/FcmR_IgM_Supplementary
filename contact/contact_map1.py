import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import PDBParser
import matplotlib.patches as mpatches
from matplotlib import colormaps
import os
import re


def analyze_protein_contacts(pdb_file, sIgM_chains, FcµR_chains, cutoff=4.5):
    """
    分析蛋白质接触的完整函数
    sIgM_chains: sIgM链（受体）
    FcµR_chains: FcµR链（配体）
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

        model_chain_ids = [chain.id for chain in model]
        print(f"找到的链: {model_chain_ids}")

        # 检查指定的链是否存在
        all_chains = sIgM_chains + FcµR_chains
        missing_chains = [chain for chain in all_chains if chain not in model_chain_ids]

        if missing_chains:
            print(f"错误: 以下链不存在: {missing_chains}")
            return None

        contacts = []

        # 分析每个配体链（FcµR）与每个受体链（sIgM）的接触
        for lig_chain in FcµR_chains:  # FcµR链
            for rec_chain in sIgM_chains:  # sIgM链
                print(f"分析 sIgM{rec_chain} (受体) ↔ FcµR{lig_chain} (配体) 接触...")

                # 获取原子（排除氢原子）
                rec_atoms = []
                for atom in model[rec_chain].get_atoms():
                    if atom.element != 'H':  # 排除氢原子
                        rec_atoms.append(atom)

                lig_atoms = []
                for atom in model[lig_chain].get_atoms():
                    if atom.element != 'H':  # 排除氢原子
                        lig_atoms.append(atom)

                print(f"  受体链 sIgM{rec_chain} 有 {len(rec_atoms)} 个非氢原子")
                print(f"  配体链 FcµR{lig_chain} 有 {len(lig_atoms)} 个非氢原子")

                # 计算距离
                for rec_atom in rec_atoms:  # sIgM原子
                    for lig_atom in lig_atoms:  # FcµR原子
                        distance = np.linalg.norm(rec_atom.coord - lig_atom.coord)
                        if distance < cutoff:
                            rec_res = rec_atom.get_parent()  # sIgM残基
                            lig_res = lig_atom.get_parent()  # FcµR残基

                            # 判断相互作用类型
                            interaction_type = classify_interaction(rec_atom, lig_atom, distance)

                            contacts.append({
                                'sIgM_chain': rec_chain,  # sIgM链
                                'sIgM_residue': f"{rec_res.resname}{rec_res.id[1]}",  # sIgM残基
                                'sIgM_resid': rec_res.id[1],  # sIgM残基编号
                                'FcµR_chain': lig_chain,  # FcµR链
                                'FcµR_residue': f"{lig_res.resname}{lig_res.id[1]}",  # FcµR残基
                                'FcµR_resid': lig_res.id[1],  # FcµR残基编号
                                'distance': distance,
                                'interaction_type': interaction_type,
                                'sIgM_atom': rec_atom.name,  # sIgM原子
                                'FcµR_atom': lig_atom.name  # FcµR原子
                            })

        print(f"分析完成，共找到 {len(contacts)} 个接触")
        return pd.DataFrame(contacts)

    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
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


def print_critical_interactions(critical_interactions):
    """打印关键相互作用对"""
    if critical_interactions:
        print("\n🔬 关键强相互作用对 (距离 < 3.0 Å):")
        print("=" * 70)
        print(f"{'sIgM Residue':<15} {'FcµR Residue':<15} {'Distance (Å)':<12} {'Chain Pair':<15}")
        print("-" * 70)

        for interaction in sorted(critical_interactions, key=lambda x: x['distance']):
            chain_pair = f"sIgM{interaction['sIgM_chain']}-FcµR{interaction['FcµR_chain']}"
            print(f"{interaction['sIgM_residue']:<15} {interaction['FcµR_residue']:<15} "
                  f"{interaction['distance']:<12.2f} {chain_pair:<15}")
    else:
        print("未找到距离 < 3.0 Å 的关键相互作用")


def extract_residue_number(residue_str):
    """从残基字符串中提取数字部分"""
    match = re.search(r'\d+', residue_str)
    return int(match.group()) if match else 0


def create_comprehensive_dual_map(contacts_df, output_file='sIgM_FcµR_comprehensive_map.png'):
    """创建综合性的双向残基接触图谱（保持4个子图）"""

    print(f"开始创建图谱，输出文件: {output_file}")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file) or '.'
    os.makedirs(output_dir, exist_ok=True)

    if contacts_df is None:
        print("错误: contacts_df 为 None")
        # 创建空的图像文件
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No contact data available\n(contacts_df is None)',
                ha='center', va='center', fontsize=14)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"创建了空的图像文件: {output_file}")
        return None

    if contacts_df.empty:
        print("警告: contacts_df 为空")
        # 创建空的图像文件
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No contact data available\n(DataFrame is empty)',
                ha='center', va='center', fontsize=14)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"创建了空的图像文件: {output_file}")
        return None

    try:
        # 调整图形尺寸，增加高度以容纳更大的图3和图4
        fig = plt.figure(figsize=(22, 30))  # 高度从35减少到30，因为减少了一行

        # 使用gridspec来精确控制布局，改为3行布局
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 2, height_ratios=[4.0, 4.0, 2.0])  # 改为3行，图例放在第3行

        # 主接触图 - 第1行第1列
        ax_main = plt.subplot(gs[0, 0])
        # 热力图 - 第1行第2列
        ax_heatmap = plt.subplot(gs[0, 1])
        # sIgM残基详情 - 第2行第1列
        ax_sIgM = plt.subplot(gs[1, 0])
        # FcµR残基详情 - 第2行第2列
        ax_FcµR = plt.subplot(gs[1, 1])
        # 图注 - 第3行（跨两列）
        ax_legend = plt.subplot(gs[2, :])
        ax_legend.axis('off')

        # 设置所有子图的边框样式
        for ax in [ax_main, ax_heatmap, ax_sIgM, ax_FcµR, ax_legend]:
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('black')

        # 颜色映射
        distance_cmap = colormaps['viridis_r']

        # 1. 主接触图
        unique_sIgM_chains = contacts_df['sIgM_chain'].unique()
        unique_FcµR_chains = contacts_df['FcµR_chain'].unique()

        # 为每个链对创建唯一的y坐标
        y_positions = {}
        y_ticks = []
        y_tick_labels = []

        current_y = 0
        for rec_chain in unique_sIgM_chains:
            for lig_chain in unique_FcµR_chains:
                chain_pair = f"s{rec_chain}-F{lig_chain}"
                y_positions[chain_pair] = current_y
                y_ticks.append(current_y)
                y_tick_labels.append(chain_pair)
                current_y += 8

        # 绘制主接触图
        all_x, all_y, all_colors = [], [], []
        critical_interactions = []

        for _, row in contacts_df.iterrows():
            chain_pair = f"s{row['sIgM_chain']}-F{row['FcµR_chain']}"
            y_pos = y_positions[chain_pair]

            all_x.append(row['FcµR_resid'])
            all_y.append(y_pos)

            # 根据距离确定颜色
            if row['distance'] < 3.0:
                color = 'red'
                critical_interactions.append(row)
            elif row['distance'] < 3.5:
                color = 'orange'
            else:
                color = distance_cmap((row['distance'] - 2.5) / 2.5)

            all_colors.append(color)

        # 绘制点
        ax_main.scatter(all_x, all_y, c=all_colors, s=20, alpha=0.6, edgecolors='black', linewidth=0.1)

        # 设置主图标签和刻度（图1纵坐标字体保持不变14号，横坐标增大2号到16号）
        ax_main.set_yticks(y_ticks)
        ax_main.set_yticklabels(y_tick_labels, fontsize=14)  # 纵坐标字体保持不变

        # 调整横坐标标题位置（增大2号到16号）
        ax_main.set_xlabel('FcµR Residue Number', fontsize=16, fontweight='bold', labelpad=20)

        ax_main.set_ylabel('Chain Pairs (sIgM-FcµR)', fontsize=16, fontweight='bold')  # 增大2号到16号

        # 设置标题
        ax_main.set_title('sIgM - FcµR Interaction Map', fontsize=18, fontweight='bold', pad=22.5)

        # 设置坐标轴范围
        if len(y_ticks) > 0:
            ax_main.set_ylim(y_ticks[0] - 3, y_ticks[-1] + 3)

        # 自动确定x轴范围
        x_min = min(all_x) - 5 if all_x else 0
        x_max = max(all_x) + 5 if all_x else 100
        ax_main.set_xlim(x_min, x_max)

        # 增加网格线
        ax_main.grid(True, alpha=0.3, linestyle='--')

        # 2. 热力图
        try:
            # 获取所有残基并排序
            sIgM_residues = sorted(contacts_df['sIgM_residue'].unique(), key=extract_residue_number)
            FcµR_residues = sorted(contacts_df['FcµR_residue'].unique(), key=extract_residue_number)

            # 创建数据透视表
            pivot_data = contacts_df.pivot_table(
                values='distance',
                index='sIgM_residue',
                columns='FcµR_residue',
                aggfunc='count',
                fill_value=0
            )

            # 重新索引以确保有序排列
            pivot_data = pivot_data.reindex(index=sIgM_residues, columns=FcµR_residues, fill_value=0)

            if not pivot_data.empty:
                im = ax_heatmap.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto', origin='lower')

                # 设置标题
                ax_heatmap.set_title('Contact Frequency Heatmap', fontsize=18, fontweight='bold', pad=22.5)

                # 横纵坐标字体增大2号（从14/16号增加到16/18号）
                ax_heatmap.set_xlabel('FcµR Residue', fontsize=18, fontweight='bold', labelpad=15)  # 增大2号到18号
                ax_heatmap.set_ylabel('sIgM Residue', fontsize=18, fontweight='bold')  # 增大2号到18号

                # 设置刻度（增大字体2号到14号）
                if len(FcµR_residues) > 0:
                    x_interval = max(1, len(FcµR_residues) // 20)
                    x_ticks = np.arange(0, len(FcµR_residues), x_interval)
                    ax_heatmap.set_xticks(x_ticks)
                    ax_heatmap.set_xticklabels([FcµR_residues[i] for i in x_ticks],
                                               rotation=45, ha='right', fontsize=14)  # 增大2号到14号

                if len(sIgM_residues) > 0:
                    y_interval = max(1, len(sIgM_residues) // 20)
                    y_ticks = np.arange(0, len(sIgM_residues), y_interval)
                    ax_heatmap.set_yticks(y_ticks)
                    ax_heatmap.set_yticklabels([sIgM_residues[i] for i in y_ticks], fontsize=14)  # 增大2号到14号

                # 添加颜色条（增大字体2号到16号）
                cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
                cbar.set_label('Number of Contacts', rotation=270, labelpad=20, fontsize=16)  # 增大2号到16号

        except Exception as e:
            print(f"热力图创建错误: {str(e)}")
            ax_heatmap.text(0.5, 0.5, 'No heatmap data', ha='center', va='center',
                            transform=ax_heatmap.transAxes, fontsize=16)
            ax_heatmap.set_title('Contact Frequency Heatmap', fontsize=18, fontweight='bold', pad=22.5)

        # 3. 计算残基统计数据
        sIgM_interactions = contacts_df.groupby('sIgM_residue').agg({
            'distance': ['min', 'count'],
            'FcµR_residue': lambda x: ', '.join(sorted(set(x))),
            'FcµR_chain': lambda x: ', '.join(sorted(set(x)))
        }).reset_index()
        sIgM_interactions.columns = ['residue', 'min_dist', 'contact_count', 'partner_residues', 'partner_chains']

        # 获取每个sIgM残基对接的所有FcµR残基
        sIgM_docking_residues = {}
        for _, row in contacts_df.iterrows():
            sIgM_res = row['sIgM_residue']
            FcµR_res = row['FcµR_residue']
            if sIgM_res not in sIgM_docking_residues:
                sIgM_docking_residues[sIgM_res] = set()
            sIgM_docking_residues[sIgM_res].add(FcµR_res)

        # 为每个sIgM残基创建对接残基字符串
        sIgM_interactions['docking_partners'] = ''
        for i, row in sIgM_interactions.iterrows():
            residue = row['residue']
            if residue in sIgM_docking_residues:
                sIgM_interactions.at[i, 'docking_partners'] = ', '.join(
                    sorted(sIgM_docking_residues[residue], key=extract_residue_number)
                )

        sIgM_stats = sIgM_interactions.nlargest(10, 'contact_count')

        # FcµR残基的对接信息
        FcµR_interactions = contacts_df.groupby('FcµR_residue').agg({
            'distance': ['min', 'count'],
            'sIgM_residue': lambda x: ', '.join(sorted(set(x))),
            'sIgM_chain': lambda x: ', '.join(sorted(set(x)))
        }).reset_index()
        FcµR_interactions.columns = ['residue', 'min_dist', 'contact_count', 'partner_residues', 'partner_chains']

        # 获取每个FcµR残基对接的所有sIgM残基
        FcµR_docking_residues = {}
        for _, row in contacts_df.iterrows():
            FcµR_res = row['FcµR_residue']
            sIgM_res = row['sIgM_residue']
            if FcµR_res not in FcµR_docking_residues:
                FcµR_docking_residues[FcµR_res] = set()
            FcµR_docking_residues[FcµR_res].add(sIgM_res)

        # 为每个FcµR残基创建对接残基字符串
        FcµR_interactions['docking_partners'] = ''
        for i, row in FcµR_interactions.iterrows():
            residue = row['residue']
            if residue in FcµR_docking_residues:
                FcµR_interactions.at[i, 'docking_partners'] = ', '.join(
                    sorted(FcµR_docking_residues[residue], key=extract_residue_number)
                )

        FcµR_stats = FcµR_interactions.nlargest(10, 'contact_count')

        # 4. sIgM残基详情
        if not sIgM_stats.empty:
            max_count = sIgM_stats['contact_count'].max()
            # 大幅增加x轴范围，为Partners信息留出足够空间
            ax_sIgM.set_xlim(0, max_count * 4.0)

            bars_sIgM = ax_sIgM.barh(range(len(sIgM_stats)), sIgM_stats['contact_count'],
                                     color='lightcoral', edgecolor='darkred', alpha=0.7, linewidth=1.2)
            ax_sIgM.set_yticks(range(len(sIgM_stats)))

            # 创建包含对接残基的标签（字体从17号缩小到14号），在数字和Å之间添加空格
            sIgM_labels = []
            for _, row in sIgM_stats.iterrows():
                label = f"{row['residue']} ({row['min_dist']:.1f} Å)"  # 在数字和Å之间添加空格
                sIgM_labels.append(label)

            ax_sIgM.set_yticklabels(sIgM_labels, fontsize=14)  # 字体从17号缩小到14号
            # 增加横坐标标题与图的距离（字体从21号缩小到18号）
            ax_sIgM.set_xlabel('Number of Contacts', fontsize=18, fontweight='bold', labelpad=20)

            ax_sIgM.set_ylabel('sIgM Residue', fontsize=18, fontweight='bold')  # 字体从21号缩小到18号

            # 设置标题
            ax_sIgM.set_title('Top sIgM Residues with Partners', fontsize=18, fontweight='bold', pad=18.75)

            # 在条形图右侧外部添加数字标记
            for i, (_, row) in enumerate(sIgM_stats.iterrows()):
                # 将数字标记放在条形图右侧外部，稍微向右移动避免重叠
                ax_sIgM.text(row['contact_count'] + max_count * 0.05, i,
                             f"{row['contact_count']}",
                             va='center', fontsize=14, fontweight='bold',  # 字体从18号缩小到14号
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor='gray'))

                # 将Partners信息放在数字后面更远的位置，避免重叠
                if row['docking_partners']:
                    # 计算Partners文本的起始位置（在数字后面更远的位置）
                    # 特殊处理：最下面两行（索引为8和9）向后移动更多
                    if i >= len(sIgM_stats) - 2:  # 最下面两行
                        partners_start_x = row['contact_count'] + max_count * 0.35  # 向后移动更多
                    else:
                        partners_start_x = row['contact_count'] + max_count * 0.25  # 其他行保持原位置

                    # 使用更紧凑的文本布局，字体从17号缩小到14号
                    partners_text = f"Partners: {row['docking_partners']}"

                    # 如果文本太长，可以换行显示
                    if len(partners_text) > 80:
                        # 简单的换行逻辑
                        words = partners_text.split()
                        lines = []
                        current_line = ""
                        for word in words:
                            if len(current_line + word) < 50:
                                current_line += word + " "
                            else:
                                lines.append(current_line)
                                current_line = word + " "
                        if current_line:
                            lines.append(current_line)
                        partners_text = "\n".join(lines)

                    # 将文本放在数字后面更远的位置，同一行，缩小字体
                    ax_sIgM.text(partners_start_x, i, partners_text,
                                 va='center', fontsize=14, ha='left',  # 字体从17号缩小到14号
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8,
                                           edgecolor='gray'))

        else:
            ax_sIgM.text(0.5, 0.5, 'No sIgM contact data', ha='center', va='center',
                         transform=ax_sIgM.transAxes, fontsize=16)

        # 5. FcµR残基详情
        if not FcµR_stats.empty:
            max_count = FcµR_stats['contact_count'].max()
            # 大幅增加x轴范围，为Partners信息留出足够空间
            ax_FcµR.set_xlim(0, max_count * 4.0)

            bars_FcµR = ax_FcµR.barh(range(len(FcµR_stats)), FcµR_stats['contact_count'],
                                     color='lightblue', edgecolor='darkblue', alpha=0.7, linewidth=1.2)
            ax_FcµR.set_yticks(range(len(FcµR_stats)))

            # 创建包含对接残基的标签（字体从17号缩小到14号），在数字和Å之间添加空格
            FcµR_labels = []
            for _, row in FcµR_stats.iterrows():
                label = f"{row['residue']} ({row['min_dist']:.1f} Å)"  # 在数字和Å之间添加空格
                FcµR_labels.append(label)

            ax_FcµR.set_yticklabels(FcµR_labels, fontsize=14)  # 字体从17号缩小到14号
            # 增加横坐标标题与图的距离（字体从21号缩小到18号）
            ax_FcµR.set_xlabel('Number of Contacts', fontsize=18, fontweight='bold', labelpad=20)

            ax_FcµR.set_ylabel('FcµR Residue', fontsize=18, fontweight='bold')  # 字体从21号缩小到18号

            # 设置标题
            ax_FcµR.set_title('Top FcµR Residues with Partners', fontsize=18, fontweight='bold', pad=18.75)

            # 在条形图右侧外部添加数字标记
            for i, (_, row) in enumerate(FcµR_stats.iterrows()):
                # 将数字标记放在条形图右侧外部，稍微向右移动避免重叠
                ax_FcµR.text(row['contact_count'] + max_count * 0.05, i,
                             f"{row['contact_count']}",
                             va='center', fontsize=14, fontweight='bold',  # 字体从18号缩小到14号
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor='gray'))

                # 将Partners信息放在数字后面更远的位置，避免重叠
                if row['docking_partners']:
                    # 计算Partners文本的起始位置（在数字后面更远的位置）
                    partners_start_x = row['contact_count'] + max_count * 0.25

                    # 使用更紧凑的文本布局，字体从17号缩小到14号
                    partners_text = f"Partners: {row['docking_partners']}"

                    # 特殊处理：为ARG112减少一个残基
                    if row['residue'] == 'ARG112':
                        # 移除最后一个残基
                        partners_list = row['docking_partners'].split(', ')
                        if len(partners_list) > 1:
                            # 移除最后一个残基
                            partners_list = partners_list[:-1]
                            partners_text = f"Partners: {', '.join(partners_list)}"

                    # 如果文本太长，可以换行显示
                    if len(partners_text) > 80:
                        # 简单的换行逻辑
                        words = partners_text.split()
                        lines = []
                        current_line = ""
                        for word in words:
                            if len(current_line + word) < 50:
                                current_line += word + " "
                            else:
                                lines.append(current_line)
                                current_line = word + " "
                        if current_line:
                            lines.append(current_line)
                        partners_text = "\n".join(lines)

                    # 将文本放在数字后面更远的位置，同一行，缩小字体
                    ax_FcµR.text(partners_start_x, i, partners_text,
                                 va='center', fontsize=14, ha='left',  # 字体从17号缩小到14号
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8,
                                           edgecolor='gray'))

        else:
            ax_FcµR.text(0.5, 0.5, 'No FcµR contact data', ha='center', va='center',
                         transform=ax_FcµR.transAxes, fontsize=16)

        # 6. 图例和统计信息（现在在第3行）
        legend_elements = [
            mpatches.Patch(facecolor='red', label='Very Strong (<3.0Å)', alpha=0.7),
            mpatches.Patch(facecolor='orange', label='Strong (3.0-3.5Å)', alpha=0.7),
            mpatches.Patch(facecolor='lightcoral', label='sIgM Residues', alpha=0.7),
            mpatches.Patch(facecolor='lightblue', label='FcµR Residues', alpha=0.7)
        ]

        # 图例放在第3行中央
        ax_legend.legend(handles=legend_elements, loc='center', ncol=2,
                         frameon=True, fancybox=True, fontsize=16)

        # 添加统计信息
        stats_text = (
            f"sIgM Residues: {contacts_df['sIgM_residue'].nunique()} | "
            f"FcµR Residues: {contacts_df['FcµR_residue'].nunique()} | "
            f"Total Contacts: {len(contacts_df)} | "
            f"Min Distance: {contacts_df['distance'].min():.2f} Å"
        )

        ax_legend.text(0.5, 0.2, stats_text, ha='center', va='center',
                       fontsize=17, fontweight='bold', transform=ax_legend.transAxes,
                       bbox=dict(boxstyle="round,pad=0.8", facecolor="white", alpha=0.9, edgecolor='black'))

        # 调整布局并保存
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4, wspace=0.4, top=0.95, bottom=0.05, left=0.07, right=0.98)
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"图片已保存为: {output_file} (300dpi)")

        # 打印关键相互作用
        print_critical_interactions(critical_interactions)

        return contacts_df

    except Exception as e:
        print(f"图表创建错误: {str(e)}")
        import traceback
        traceback.print_exc()

        # 即使出错也尝试保存一个简单的图像
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, f'Error creating plot: {str(e)}',
                    ha='center', va='center', fontsize=14)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"错误图像已保存为: {output_file}")
        except:
            pass

        return None


# 主程序
def main():
    """主函数"""
    print("sIgM - FcµR 综合性双向残基相互作用分析")
    print("=" * 50)

    # 配置参数
    PDB_FILE = "source_file/7YSG.pdb"
    sIgM_CHAINS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'J', 'P']
    FcµR_CHAINS = ['R', 'S', 'U', 'V']

    # 检查文件是否存在
    if not os.path.exists(PDB_FILE):
        print(f"错误: PDB文件不存在: {os.path.abspath(PDB_FILE)}")
        print("请检查文件路径或下载PDB文件")
        return

    print(f"分析文件: {os.path.abspath(PDB_FILE)}")
    print(f"sIgM受体链: {sIgM_CHAINS}")
    print(f"FcµR配体链: {FcµR_CHAINS}")

    # 执行分析
    contacts_df = analyze_protein_contacts(PDB_FILE, sIgM_CHAINS, FcµR_CHAINS, cutoff=4.5)

    if contacts_df is not None and not contacts_df.empty:
        # 创建综合性图谱
        create_comprehensive_dual_map(contacts_df, 'sIgM_FcµR_comprehensive_map.png')

        # 保存详细数据
        contacts_df.to_csv('sIgM_FcµR_detailed_interactions.csv', index=False)
        print("详细相互作用数据已保存")

        # 打印统计摘要
        print(f"\n📊 统计摘要:")
        print(f"   总接触数: {len(contacts_df)}")
        print(f"   涉及的sIgM残基数: {contacts_df['sIgM_residue'].nunique()}")
        print(f"   涉及的FcµR残基数: {contacts_df['FcµR_residue'].nunique()}")
        print(f"   平均距离: {contacts_df['distance'].mean():.2f} Å")
        print(f"   最小距离: {contacts_df['distance'].min():.2f} Å")

    else:
        print("分析失败或没有找到接触，请检查文件路径和链ID")


# 运行程序
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出现错误: {str(e)}")
        import traceback
        traceback.print_exc()