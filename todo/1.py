#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
顶刊流程图 - 使用matplotlib绘制，输出矢量PDF
完全匹配用户手稿数据，六层布局，色彩区分
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def rounded_box(ax, x, y, w, h, text, fontsize=8, facecolor='white', edgecolor='black', lw=0.8):
    rect = patches.FancyBboxPatch((x, y), w, h,
                                  boxstyle=patches.BoxStyle("Round", pad=0.05),
                                  facecolor=facecolor, edgecolor=edgecolor, linewidth=lw)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, color='black', linespacing=1.2)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color='black', lw=0.8))

# 画布尺寸 (双栏宽17.6 cm)
w_cm = 17.6
w_inch = w_cm / 2.54
h_inch = 12.5
fig, ax = plt.subplots(figsize=(w_inch, h_inch), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# 坐标映射 (x, y 均归一化 0-1，实际绘制时方便调整)
# 第一层: 输入
rounded_box(ax, 0.05, 0.86, 0.90, 0.10,
            "INPUT: 7 FcμR-Fcμ complexes\nPDB: 7YTE,7YTC,7YTD,7YSG,8BPE,8BPF,8BPG\n"
            "mIgM -> 2:1      pIgM -> 1:1/4:1 (same side) & 8:1 (both sides)      sIgM -> 4:1 (opposite SC)",
            fontsize=7.5, facecolor='#E8F0FE')
arrow(ax, 0.5, 0.86, 0.5, 0.80)

# 第二层: 三列并行
# 列1
rounded_box(ax, 0.05, 0.65, 0.28, 0.13,
            "STRUCTURAL ALIGNMENT\n· Pairwise Cα RMSD (MDAnalysis)\n· Superposition (ChimeraX)",
            fontsize=7, facecolor='white')
# 列2
rounded_box(ax, 0.36, 0.65, 0.28, 0.13,
            "CONTACT ANALYSIS\n· Residue contacts (Biopython)\n· Contact proportion heatmaps\n· Interaction types (PLIP/PDBePISA)",
            fontsize=7, facecolor='white')
# 列3
rounded_box(ax, 0.67, 0.65, 0.28, 0.13,
            "EVOLUTION & ALANINE SCANNING\n· ConSurf conservation (MAFFT)\n· ΔΔG predictions (FoldX+PremPS)\n· In silico mutation (PyMOL) & H-bond (ChimeraX)",
            fontsize=7, facecolor='white')

arrow(ax, 0.19, 0.65, 0.19, 0.59)
arrow(ax, 0.50, 0.65, 0.50, 0.59)
arrow(ax, 0.81, 0.65, 0.81, 0.59)

# 第三层: 过滤
rounded_box(ax, 0.20, 0.50, 0.60, 0.08,
            "MULTI-CRITERIA FILTERING\n(structural, evolutionary, energetic, functional occurrence)",
            fontsize=7.5, facecolor='#FFF4E0')
arrow(ax, 0.50, 0.50, 0.50, 0.44)

# 第四层: 热点分类 (两行)
# 第一行三个
rounded_box(ax, 0.03, 0.32, 0.30, 0.11,
            "PRIMARY ENERGETIC HOTSPOT\nArg45\n· ΔΔG >1 (both platforms)\n· ConSurf 9, full salt bridges",
            fontsize=6.5, facecolor='#FFEEEE')
rounded_box(ax, 0.35, 0.32, 0.30, 0.11,
            "STRUCTURAL ANCHOR\nPhe67\n· Hydrophobic packing (5/5)\n· Moderate conservation (grade 6)\n· ΔΔG method-dependent >1",
            fontsize=6.5, facecolor='#EEFFEE')
rounded_box(ax, 0.67, 0.32, 0.30, 0.11,
            "STRUCTURAL SUPPORTING HOTSPOTS\nThr60, Ser63, Thr110, Asp111\n· Highly conserved (grades 8-9)\n· Form H-bonds/salt bridges (5/5)\n· ΔΔG inconsistent",
            fontsize=6.5, facecolor='#EEF4FF')
# 第二行两个
rounded_box(ax, 0.15, 0.18, 0.32, 0.11,
            "COOPERATIVE HOTSPOT\nLys69\n· Near-zero ΔΔG alone\n· Synergistic with Phe67 (double mutant abolishes binding)",
            fontsize=6.5, facecolor='#FFF2E6')
rounded_box(ax, 0.55, 0.18, 0.32, 0.11,
            "NON-ESSENTIAL RESIDUES\nThr65, Asn66, Asn109, Arg112\n· Low conservation (grades 3-4)\n· Infrequent contacts, weak ΔΔG",
            fontsize=6.5, facecolor='#F2F2F2')

# 从过滤到第一行中心的箭头
arrow(ax, 0.50, 0.44, 0.50, 0.38)
# 从第一行到第二行的箭头 (从每个框底中心向下指向第二行对应框)
# 简化: 从第一行中间框底中心到第二行两个框之间的中心
arrow(ax, 0.35, 0.32, 0.31, 0.24)  # 左
arrow(ax, 0.65, 0.32, 0.69, 0.24)  # 右

# 第五层: 统一模型
rounded_box(ax, 0.05, 0.04, 0.90, 0.11,
            "UNIFIED HIERARCHICAL MODEL\n· Rigid lock-and-key interface (Cα RMSD 1.13-2.07 Å)\n· Stoichiometric plasticity without binding-mode change\n· IgM-exclusive Cμ4 network explains isotype specificity",
            fontsize=7.5, facecolor='#EAF5F0')
arrow(ax, 0.50, 0.04, 0.50, -0.03)  # 注意y负值，需调整ylim

# 第六层: 治疗蓝图
rounded_box(ax, 0.10, -0.10, 0.80, 0.09,
            "THERAPEUTIC BLUEPRINT\nStructure-guided targeting of primary energetic and cooperative hotspots for CLL, SLE and B-cell malignancies",
            fontsize=8, facecolor='#D9EAD3')

ax.set_ylim(-0.20, 1.0)
plt.tight_layout(pad=0.2)
plt.savefig('Figure8_top_workflow.pdf', format='pdf', dpi=1200)
plt.savefig('Figure8_top_workflow.png', dpi=300)
print("Figure 8 saved as PDF and PNG.")