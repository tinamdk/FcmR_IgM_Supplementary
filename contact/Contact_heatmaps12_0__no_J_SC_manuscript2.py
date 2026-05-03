#!/usr/bin/env python3
"""
回答三个核心问题的脚本：
1. 7个复合物的共有残基对（序列对齐后）
2. 不同聚集体类型的特异性接触
3. 不同FcμR结合比例的接触差异
"""

import re
from collections import defaultdict


# ============================================================
# 第一步：建立跨PDB的残基对齐映射
# ============================================================

# 基于您提供的编号信息
# 7YTE/7YTC/7YTD/7YSG: Cμ4 范围 446-557 (112残基)
# 8BPE/8BPF/8BPG:      Cμ4 范围 464-576 (113残基，但实际也是112左右)

# 对齐规则：8系列残基编号 - 18 = 7系列残基编号
# 验证：470 - 18 = 452 (应该在446-557内)

def normalize_igm_residue(residue_id, pdb_id):
    """
    将IgM残基标准化为统一编号（基于7YTE系统）
    """
    # 提取残基名和编号
    match = re.search(r'([A-Za-z]+)(\d+)', residue_id)
    if not match:
        return residue_id
    name, num = match.groups()
    num = int(num)

    # 映射
    if pdb_id in ["8BPE", "8BPF", "8BPG"]:
        # 8系列比7系列大约大18
        unified_num = num - 18
    else:
        unified_num = num

    # 检查是否在Cμ4范围内
    is_cmu4 = 446 <= unified_num <= 557

    return f"{name}{unified_num}", is_cmu4


def normalize_fcmur_residue(residue_id, pdb_id):
    """
    将FcμR残基标准化（需要您提供FcμR的序列比对信息）
    """
    # 提取残基名和编号
    match = re.search(r'([A-Za-z]+)(\d+)', residue_id)
    if not match:
        return residue_id
    name, num = match.groups()

    # FcμR在不同PDB中编号可能也不同
    # 暂时只移除链ID
    residue_clean = residue_id.split('_')[-1] if '_' in residue_id else residue_id

    return residue_clean


# ============================================================
# 第二步：找出7个复合物的共有残基对（对齐后）
# ============================================================

def find_common_contacts_after_alignment(all_pairs_by_pdb):
    """
    序列对齐后，找出出现在所有7个复合物中的残基对
    """
    # 标准化
    normalized_sets = {}

    for pdb_id, pairs in all_pairs_by_pdb.items():
        norm_set = set()
        for fc_res, igm_res in pairs:
            fc_norm = normalize_fcmur_residue(fc_res, pdb_id)
            igm_norm, is_cmu4 = normalize_igm_residue(igm_res, pdb_id)
            norm_set.add((fc_norm, igm_norm, is_cmu4))
        normalized_sets[pdb_id] = norm_set
        print(f"{pdb_id}: {len(pairs)} → {len(norm_set)} (after normalization)")

    # 取所有7个的交集
    all_sets = list(normalized_sets.values())
    common = set.intersection(*all_sets)

    print(f"\n=== After sequence alignment ===")
    print(f"Contacts present in ALL 7 complexes: {len(common)}")

    for fc, igm, is_cmu4 in sorted(common):
        cmu4_note = " (Cμ4)" if is_cmu4 else ""
        print(f"  {fc} ←→ {igm}{cmu4_note}")

    return common, normalized_sets


# ============================================================
# 第三步：不同聚集体类型的特异性接触
# ============================================================

def find_morphology_specific_contacts(normalized_sets):
    """
    找出二聚体、五聚体、分泌型特有的接触
    """
    morphology_groups = {
        "Dimeric": ["7YTE", "8BPG"],
        "Pentameric": ["7YTC", "8BPF", "7YTD", "8BPE"],
        "Secretory": ["7YSG"]
    }

    # 计算每组所有复合物的并集
    group_unions = {}
    for morph, pdb_ids in morphology_groups.items():
        union = set()
        for pdb_id in pdb_ids:
            if pdb_id in normalized_sets:
                # 只保留残基对，去掉Cμ4标记
                union.update({(fc, igm) for fc, igm, _ in normalized_sets[pdb_id]})
        group_unions[morph] = union
        print(f"{morph}: {len(union)} unique contacts in union")

    # 找出特异性接触
    specific = {}
    for morph in morphology_groups.keys():
        other_union = set()
        for other_morph in morphology_groups.keys():
            if other_morph != morph:
                other_union.update(group_unions[other_morph])
        specific[morph] = group_unions[morph] - other_union
        print(f"\n{morph} specific: {len(specific[morph])} contacts")

        # 显示前10个
        for fc, igm in list(specific[morph])[:10]:
            print(f"    {fc} ←→ {igm}")

    return specific


# ============================================================
# 第四步：不同FcμR结合比例的接触差异（五聚体内部）
# ============================================================

def find_valency_specific_contacts(normalized_sets):
    """
    在五聚体内部，比较1个、4个、8个FcμR的接触差异
    """
    valency_groups = {
        "1_FcμR": ["7YTC", "8BPF"],
        "4_FcμR": ["7YTD"],
        "8_FcμR": ["8BPE"]
    }

    # 计算每组所有复合物的并集
    group_unions = {}
    for valency, pdb_ids in valency_groups.items():
        union = set()
        for pdb_id in pdb_ids:
            if pdb_id in normalized_sets:
                union.update({(fc, igm) for fc, igm, _ in normalized_sets[pdb_id]})
        group_unions[valency] = union
        print(f"{valency}: {len(union)} unique contacts")

    # 找出特异性接触
    print("\n=== Valency-specific contacts ===")

    # 1 FcμR 特有的
    one_specific = group_unions["1_FcμR"] - group_unions["4_FcμR"] - group_unions["8_FcμR"]
    print(f"\nUnique to 1 FcμR: {len(one_specific)} contacts")
    for fc, igm in list(one_specific)[:10]:
        print(f"    {fc} ←→ {igm}")

    # 4 FcμR 特有的
    four_specific = group_unions["4_FcμR"] - group_unions["1_FcμR"] - group_unions["8_FcμR"]
    print(f"\nUnique to 4 FcμR: {len(four_specific)} contacts")
    for fc, igm in list(four_specific)[:10]:
        print(f"    {fc} ←→ {igm}")

    # 8 FcμR 特有的
    eight_specific = group_unions["8_FcμR"] - group_unions["1_FcμR"] - group_unions["4_FcμR"]
    print(f"\nUnique to 8 FcμR: {len(eight_specific)} contacts")
    for fc, igm in list(eight_specific)[:10]:
        print(f"    {fc} ←→ {igm}")

    # 所有五聚体共有的（1,4,8都出现）
    common_in_pentamer = group_unions["1_FcμR"] & group_unions["4_FcμR"] & group_unions["8_FcμR"]
    print(f"\nCommon to ALL pentameric forms (1,4,8 FcμR): {len(common_in_pentamer)} contacts")
    for fc, igm in list(common_in_pentamer)[:10]:
        print(f"    {fc} ←→ {igm}")

    return {
        "1_specific": one_specific,
        "4_specific": four_specific,
        "8_specific": eight_specific,
        "common_to_pentamer": common_in_pentamer
    }


# ============================================================
# 第五步：生成论文表格
# ============================================================

def generate_summary_tables(common, morphology_specific, valency_specific):
    """
    生成可以直接放入论文的表格
    """

    # Table: 7个复合物共有的保守接触
    if common:
        print("\n" + "=" * 70)
        print("TABLE: Conserved contacts across all 7 complexes")
        print("=" * 70)
        print("| FcμR | IgM | Domain |")
        print("|------|-----|--------|")
        for fc, igm, is_cmu4 in sorted(common):
            domain = "Cμ4" if is_cmu4 else "Other"
            print(f"| {fc} | {igm} | {domain} |")

    # Table: 聚集体类型特异性接触
    print("\n" + "=" * 70)
    print("TABLE: Morphology-specific contacts")
    print("=" * 70)
    for morph, pairs in morphology_specific.items():
        if pairs:
            print(f"\n### {morph}-specific ({len(pairs)} pairs):")
            print("| FcμR | IgM |")
            print("|------|-----|")
            for fc, igm in sorted(pairs)[:20]:
                print(f"| {fc} | {igm} |")
            if len(pairs) > 20:
                print(f"| ... | {len(pairs) - 20} more pairs |")

    # Table: 不同FcμR结合比例的特异性接触
    print("\n" + "=" * 70)
    print("TABLE: FcμR-valency specific contacts (within pentameric IgM)")
    print("=" * 70)

    for valency, pairs in valency_specific.items():
        if pairs:
            print(f"\n### {valency} ({len(pairs)} pairs):")
            print("| FcμR | IgM |")
            print("|------|-----|")
            for fc, igm in sorted(pairs)[:15]:
                print(f"| {fc} | {igm} |")
            if len(pairs) > 15:
                print(f"| ... | {len(pairs) - 15} more pairs |")


# ============================================================
# 主程序
# ============================================================

def run_full_analysis(all_pairs_by_pdb):
    """
    运行完整分析
    """
    print("=" * 70)
    print("COMPLETE ANALYSIS: Conserved & Specific Contacts")
    print("=" * 70)

    # 1. 序列对齐后的7个复合物共有残基对
    print("\n[1] Finding common contacts after sequence alignment...")
    common, normalized_sets = find_common_contacts_after_alignment(all_pairs_by_pdb)

    # 2. 不同聚集体类型的特异性接触
    print("\n[2] Finding morphology-specific contacts...")
    morphology_specific = find_morphology_specific_contacts(normalized_sets)

    # 3. 不同FcμR结合比例的特异性接触
    print("\n[3] Finding valency-specific contacts (within pentamer)...")
    valency_specific = find_valency_specific_contacts(normalized_sets)

    # 4. 生成表格
    generate_summary_tables(common, morphology_specific, valency_specific)

    return common, morphology_specific, valency_specific