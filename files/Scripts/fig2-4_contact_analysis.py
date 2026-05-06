#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FcμR-IgM contact analysis with J chain and SC specificity
- Original figures (2a, 2b, 3, 4) preserved
- J chain analysis for pentamers (7YTC, 7YTD, 8BPE) and sIgM (7YSG)
- New: FcμR–SC (secretory component) contact analysis for sIgM (7YSG)
- Outputs: full contact CSVs, filtered heatmaps for pentamer J chain
- All comments in English
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.PDB import PDBParser, NeighborSearch, is_aa
from collections import defaultdict

# ============================================================
# Global plot settings
# ============================================================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 7
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['patch.linewidth'] = 0.5

# ============================================================
# Configuration
# ============================================================
PDB_IDS = ['7YTE', '7YTC', '7YTD', '7YSG', '8BPE', '8BPF', '8BPG']

CHAIN_MAP = {
    "7YTE": {"file": "source_file/7YTE.pdb", "IgM": ["A", "B"], "FcμR": ["C", "D"]},
    "7YTC": {"file": "source_file/7YTC.pdb", "IgM": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L"],
             "FcμR": ["R"], "J_chain": ["J"]},
    "7YTD": {"file": "source_file/7YTD.pdb", "IgM": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L"],
             "FcμR": ["R", "S", "U", "V"], "J_chain": ["J"]},
    "7YSG": {"file": "source_file/7YSG.pdb", "IgM": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L"],
             "FcμR": ["U", "R", "S", "V"], "J_chain": ["J"], "SC": ["P"]},
    "8BPE": {"file": "source_file/8BPE.pdb", "IgM": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L"],
             "FcμR": ["I", "M", "N", "O", "P", "Q", "R", "S"], "J_chain": ["J"]},
    "8BPF": {"file": "source_file/8BPF.pdb", "IgM": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L"],
             "FcμR": ["I"]},
    "8BPG": {"file": "source_file/8BPG.pdb", "IgM": ["C", "D", "E", "F"], "FcμR": ["A", "B"]},
}

DIST_CUTOFF = 4.5

OUTPUT_DIR = "contact_analysis_manuscript2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output paths for original figures and data
COMMON_HEATMAP_PNG = os.path.join(OUTPUT_DIR, "fig2a_contact_heatmap.png")
COMMON_HEATMAP_PDF = os.path.join(OUTPUT_DIR, "fig2a_contact_heatmap.pdf")
PERSISTENCE_STACKED_PNG = os.path.join(OUTPUT_DIR, "fig2b_stacked_bar.png")
PERSISTENCE_STACKED_PDF = os.path.join(OUTPUT_DIR, "fig2b_stacked_bar.pdf")
GROUP_COMPARE_HEATMAP_PNG = os.path.join(OUTPUT_DIR, "fig4a_stoichiometry_heatmap.png")
GROUP_COMPARE_HEATMAP_PDF = os.path.join(OUTPUT_DIR, "fig4a_stoichiometry_heatmap.pdf")
FULL_HEATMAP_CSV = os.path.join(OUTPUT_DIR, "source_data_fig4a_stoichiometry_full.csv")
THREE_GROUPS_BAR_PNG = os.path.join(OUTPUT_DIR, "fig3_three_groups_barplot.png")
THREE_GROUPS_BAR_PDF = os.path.join(OUTPUT_DIR, "fig3_three_groups_barplot.pdf")

PER_PDB_CSV = os.path.join(OUTPUT_DIR, "source_data_fig2-4_all_contacts.csv")
DIFF_DIMER_VS_PENTAMER_CSV = os.path.join(OUTPUT_DIR, "source_data_dimer_vs_pentamer_diff.csv")
DIFF_SIGM_VS_PENTAMER_CSV = os.path.join(OUTPUT_DIR, "source_data_sIgM_vs_pentamer_diff.csv")
THREE_GROUPS_CSV = os.path.join(OUTPUT_DIR, "source_data_fig3_three_groups.csv")

# J chain analysis outputs
J_PENTAMER_FULL_CSV = os.path.join(OUTPUT_DIR, "fig4b_pentamer_j_chain_all_contacts.csv")
J_PENTAMER_HEATMAP_PNG = os.path.join(OUTPUT_DIR, "fig4b_pentamer_j_chain_heatmap.png")
J_PENTAMER_HEATMAP_PDF = os.path.join(OUTPUT_DIR, "fig4b_pentamer_j_chain_heatmap.pdf")
J_SIGM_CSV = os.path.join(OUTPUT_DIR, "fig4b_j_sIgM_contacts.csv")

# New: SC chain analysis for sIgM
J_SIGM_SC_CSV = os.path.join(OUTPUT_DIR, "sIgM_FcmuR_SC_contacts.csv")

# Grouping
DIMER_PDBS = ['7YTE', '8BPG']
PENTAMER_PDBS = ['7YTC', '7YTD', '8BPE', '8BPF']
SIGM_PDBS = ['7YSG']
FCMUR1_PDBS = ['7YTC', '8BPF']
FCMUR4_PDBS = ['7YTD']
FCMUR8_PDBS = ['8BPE']

# ============================================================
# Helper functions
# ============================================================
def force_format(name):
    """Format residue name + number, e.g., ARG123 -> Arg123"""
    if len(name) < 4:
        return name
    three = name[:3].capitalize()
    num = name[3:]
    return f"{three}{num}"

def get_sidechain_atoms(residue):
    """Return sidechain atoms (for GLY, use CA as pseudo-sidechain)"""
    backbone = {'N', 'CA', 'C', 'O'}
    if residue.resname == 'GLY':
        return [atom for atom in residue.get_atoms() if atom.name == 'CA']
    return [atom for atom in residue.get_atoms() if atom.name not in backbone]

def get_pair_participation_ratio(structure, fcmr_chains, igm_chains, cutoff):
    """
    Compute per-residue-pair contact proportion between FcμR D1 (18-124)
    and any IgM chain (full length). Proportion normalized by number of FcμR chains.
    """
    igm_atoms = []
    for ch in igm_chains:
        if ch in structure[0]:
            igm_atoms.extend(structure[0][ch].get_atoms())
    if not igm_atoms:
        return {}
    ns = NeighborSearch(igm_atoms)
    total_fc_chains = 0
    pair_count = defaultdict(int)
    for ch_id in fcmr_chains:
        if ch_id not in structure[0]:
            continue
        total_fc_chains += 1
        chain_map = defaultdict(set)
        for residue in structure[0][ch_id].get_residues():
            if not is_aa(residue):
                continue
            if not (18 <= residue.id[1] <= 124):
                continue
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
        for fkey, igm_set in chain_map.items():
            for ikey in igm_set:
                pair_count[(fkey, ikey)] += 1
    if total_fc_chains == 0:
        return {}
    fractions = {pair: cnt / total_fc_chains for pair, cnt in pair_count.items()}
    return fractions

def extract_number(res_str):
    """Extract residue number from formatted string, e.g., 'Arg123' -> 123"""
    return int(re.search(r'\d+', res_str).group())

def compute_group_average(group_pdbs, per_pdb_fracs):
    """Average contact proportions across a group of PDBs"""
    group_data = defaultdict(list)
    for pdb in group_pdbs:
        if pdb not in per_pdb_fracs:
            continue
        for pair, prop in per_pdb_fracs[pdb].items():
            group_data[pair].append(prop)
    avg = {pair: np.mean(vals) for pair, vals in group_data.items()}
    return avg

def describe_group_diff(avg1, avg2):
    """Compute per-pair difference between two group averages"""
    common = set(avg1.keys()) & set(avg2.keys())
    results = []
    for pair in common:
        diff = avg1[pair] - avg2[pair]
        results.append((pair, avg1[pair], avg2[pair], diff))
    results.sort(key=lambda x: abs(x[3]), reverse=True)
    return results

def get_jchain_contacts(structure, fcmr_chains, jchain_chains, cutoff):
    """
    Compute contact proportion between FcμR D1 and J chain residues (full length).
    Returns dict: {(FcμR_res, J_res): proportion}
    """
    jchain_atoms = []
    for ch in jchain_chains:
        if ch in structure[0]:
            jchain_atoms.extend(structure[0][ch].get_atoms())
    if not jchain_atoms:
        return {}
    ns = NeighborSearch(jchain_atoms)
    total_fc_chains = 0
    pair_count = defaultdict(int)
    for ch_id in fcmr_chains:
        if ch_id not in structure[0]:
            continue
        total_fc_chains += 1
        chain_map = defaultdict(set)
        for residue in structure[0][ch_id].get_residues():
            if not is_aa(residue):
                continue
            if not (18 <= residue.id[1] <= 124):
                continue
            sidechain = get_sidechain_atoms(residue)
            if not sidechain:
                continue
            fkey = force_format(f"{residue.resname}{residue.id[1]}")
            for atom in sidechain:
                nearby = ns.search(atom.coord, cutoff)
                if nearby:
                    for j_atom in nearby:
                        j_res = j_atom.get_parent()
                        if not is_aa(j_res):
                            continue
                        jkey = force_format(f"{j_res.resname}{j_res.id[1]}")
                        chain_map[fkey].add(jkey)
        for fkey, j_set in chain_map.items():
            for jkey in j_set:
                pair_count[(fkey, jkey)] += 1
    if total_fc_chains == 0:
        return {}
    fractions = {pair: cnt / total_fc_chains for pair, cnt in pair_count.items()}
    return fractions

def get_fcmur_sc_contacts(structure, fcmr_chains, sc_chains, cutoff):
    """
    NEW: Compute contact proportion between FcμR D1 and secretory component (SC) chain P.
    SC residues are taken full-length. Returns dict: {(FcμR_res, SC_res): proportion}
    """
    sc_atoms = []
    for ch in sc_chains:
        if ch in structure[0]:
            sc_atoms.extend(structure[0][ch].get_atoms())
    if not sc_atoms:
        return {}
    ns = NeighborSearch(sc_atoms)
    total_fc_chains = 0
    pair_count = defaultdict(int)
    for ch_id in fcmr_chains:
        if ch_id not in structure[0]:
            continue
        total_fc_chains += 1
        chain_map = defaultdict(set)
        for residue in structure[0][ch_id].get_residues():
            if not is_aa(residue):
                continue
            if not (18 <= residue.id[1] <= 124):
                continue
            sidechain = get_sidechain_atoms(residue)
            if not sidechain:
                continue
            fkey = force_format(f"{residue.resname}{residue.id[1]}")
            for atom in sidechain:
                nearby = ns.search(atom.coord, cutoff)
                if nearby:
                    for sc_atom in nearby:
                        sc_res = sc_atom.get_parent()
                        if not is_aa(sc_res):
                            continue
                        sckey = force_format(f"{sc_res.resname}{sc_res.id[1]}")
                        chain_map[fkey].add(sckey)
        for fkey, sc_set in chain_map.items():
            for sckey in sc_set:
                pair_count[(fkey, sckey)] += 1
    if total_fc_chains == 0:
        return {}
    fractions = {pair: cnt / total_fc_chains for pair, cnt in pair_count.items()}
    return fractions

# ============================================================
# Main analysis
# ============================================================
def main():
    print("=" * 60)
    print("Processing PDB files...")
    print("=" * 60)

    pair_fractions = defaultdict(list)
    per_pdb_fractions = {}

    for pdb in PDB_IDS:
        print(f"Processing {pdb}...")
        entry = CHAIN_MAP.get(pdb)
        if not entry:
            continue
        if not os.path.exists(entry['file']):
            print(f"  File not found: {entry['file']}")
            continue
        parser = PDBParser(QUIET=True)
        try:
            struct = parser.get_structure(pdb, entry['file'])
        except Exception as e:
            print(f"  Error: {e}")
            continue
        fractions = get_pair_participation_ratio(struct, entry['FcμR'], entry['IgM'], DIST_CUTOFF)
        for pair, val in fractions.items():
            pair_fractions[pair].append(val)
        per_pdb_fractions[pdb] = fractions
        print(f"  -> {len(fractions)} contacts")

    if not pair_fractions:
        print("No contacts found. Exiting.")
        return

    # Save per-PDB contact details
    all_rows = []
    for pdb, frac in per_pdb_fractions.items():
        for (f, i), prop in frac.items():
            all_rows.append({'PDB': pdb, 'FcμR-D1_residue': f, 'IgM_residue': i, 'proportion': prop})
    pd.DataFrame(all_rows).to_csv(PER_PDB_CSV, index=False)
    print(f"Saved per-PDB CSV: {PER_PDB_CSV}")

    # -------------------------
    # Fig. 2A High-frequency contacts heatmap
    # -------------------------
    print("\n--- Heatmap of average contact proportion of high-frequency FcµR-IgM residue pairs ---")
    common_pairs = [p for p, v in pair_fractions.items() if len(v) == len(PDB_IDS)]
    if common_pairs:
        common_avg = {p: np.mean(pair_fractions[p]) for p in common_pairs}
        fc_res = sorted({p[0] for p in common_avg}, key=extract_number)
        igm_res = sorted({p[1] for p in common_avg}, key=extract_number)
        mat = np.full((len(fc_res), len(igm_res)), np.nan)
        for (f, i), val in common_avg.items():
            mat[fc_res.index(f), igm_res.index(i)] = val
        df_common = pd.DataFrame(mat, index=fc_res, columns=igm_res)

        fig, ax = plt.subplots(figsize=(9, 8), dpi=1200)
        heatmap = sns.heatmap(df_common, annot=True, fmt=".2f", cmap='Reds', mask=df_common.isna(),
                              linewidths=0.5,
                              cbar_kws={'label': 'Average contact proportion',
                                        'ticks': [0.5, 0.6, 0.7, 0.8, 0.9, 1.00]},
                              vmin=0.8, vmax=1.0, ax=ax, square=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
        rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        ax.set_xlabel("IgM heavy constant μ residues", fontsize=12)
        ax.set_ylabel("FcμR-D1 residues", fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        cbar = heatmap.collections[0].colorbar
        cbar.ax.set_ylabel('Average contact proportion', fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        plt.tight_layout()
        plt.savefig(COMMON_HEATMAP_PNG, dpi=1200)
        plt.savefig(COMMON_HEATMAP_PDF, dpi=1200)
        plt.close()
        print(f"Saved: {COMMON_HEATMAP_PNG} / {COMMON_HEATMAP_PDF}")
    else:
        print("No common pairs found.")

    # -------------------------
    # Fig. 2B Stacked bar
    # -------------------------
    print("\n--- Fig. 2B (Oligomer selectivity stacked bar)")
    type_groups = [DIMER_PDBS, PENTAMER_PDBS, SIGM_PDBS]
    type_names = ['Dimer', 'Pentamer', 'sIgM']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    count_by_type = defaultdict(lambda: [0,0,0])
    for pair in pair_fractions:
        for i, grp in enumerate(type_groups):
            cnt = sum(1 for pdb in grp if pair in per_pdb_fractions.get(pdb, {}))
            count_by_type[pair][i] = cnt
    max_total = len(PDB_IDS)
    stacks = {t: [0,0,0] for t in range(1, max_total+1)}
    for cnts in count_by_type.values():
        total = sum(cnts)
        if 1 <= total <= max_total:
            for i in range(3):
                stacks[total][i] += cnts[i]

    fig, ax = plt.subplots(figsize=(8,6), dpi=1200)
    bottom = np.zeros(max_total)
    bar_width = 0.6
    for i, name in enumerate(type_names):
        heights = [stacks[t][i] for t in range(1, max_total+1)]
        ax.bar(range(1, max_total+1), heights, bottom=bottom, label=name,
               color=colors[i], edgecolor='black', linewidth=0.5, width=bar_width)
        for j, (h,b) in enumerate(zip(heights, bottom)):
            if h>0:
                ax.text(j+1, b+h/2, f"{name[:1]}:{int(h)}", ha='center', va='center',
                        fontsize=10, color='white', fontweight='bold')
        bottom += heights
    for x in range(1, max_total+1):
        total_h = bottom[x-1]
        if total_h > 0:
            ax.text(x,total_h+0.1, f"Total: {int(total_h)}", ha='center', va='bottom', fontsize=12)
    ax.set_xlabel('Number of PDB complexes containing the residue pair (out of 7)', fontsize=12)
    ax.set_ylabel('Number of residue pairs', fontsize=12)
    ax.set_xticks(range(1, max_total+1))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_xticklabels([f'{i} complex(es)' for i in range(1, max_total+1)])
    ax.set_xlim(0.5, max_total+0.5)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.subplots_adjust(left=0.3)
    plt.tight_layout()
    plt.savefig(PERSISTENCE_STACKED_PNG, dpi=1200)
    plt.savefig(PERSISTENCE_STACKED_PDF, dpi=1200)
    plt.close()
    print(f"Saved: {PERSISTENCE_STACKED_PNG} / {PERSISTENCE_STACKED_PDF}")

    # -------------------------
    # Differential analysis
    # -------------------------
    dimer_avg = compute_group_average(DIMER_PDBS, per_pdb_fractions)
    pentamer_avg = compute_group_average(PENTAMER_PDBS, per_pdb_fractions)
    sigm_avg = compute_group_average(SIGM_PDBS, per_pdb_fractions)

    diff_dp = describe_group_diff(dimer_avg, pentamer_avg)
    pd.DataFrame(diff_dp, columns=['pair','dimer_avg','pentamer_avg','diff']).to_csv(DIFF_DIMER_VS_PENTAMER_CSV, index=False)
    diff_sp = describe_group_diff(sigm_avg, pentamer_avg)
    pd.DataFrame(diff_sp, columns=['pair','sigm_avg','pentamer_avg','diff']).to_csv(DIFF_SIGM_VS_PENTAMER_CSV, index=False)
    print("Saved differential CSVs.")

    # -------------------------
    # Fig. 3 Bar chart
    # -------------------------
    print("\n--- Fig. 3. Contact proportions of shared FcµR-Fcμ interfacial residue pairs")
    common_pairs_15 = [p for p,v in pair_fractions.items() if len(v)==len(PDB_IDS)]
    if common_pairs_15:
        pairs_list, dimer_vals, pentamer_vals, sigm_vals = [], [], [], []
        for p in common_pairs_15:
            pairs_list.append(f"{p[0]}-{p[1]}")
            dimer_vals.append(dimer_avg.get(p,0))
            pentamer_vals.append(pentamer_avg.get(p,0))
            sigm_vals.append(sigm_avg.get(p,0))
        sorted_idx = np.argsort(pentamer_vals)[::-1]
        pairs_sorted = [pairs_list[i] for i in sorted_idx]
        dimer_sorted = [dimer_vals[i] for i in sorted_idx]
        pentamer_sorted = [pentamer_vals[i] for i in sorted_idx]
        sigm_sorted = [sigm_vals[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(9,6), dpi=1200)
        x = np.arange(len(pairs_sorted))
        width = 0.25
        ax.bar(x-width, dimer_sorted, width, label='Dimer', color='#1f77b4', edgecolor='black')
        ax.bar(x, pentamer_sorted, width, label='Pentamer', color='#ff7f0e', edgecolor='black')
        ax.bar(x+width, sigm_sorted, width, label='sIgM', color='#2ca02c', edgecolor='black')
        ax.set_ylabel('Contact proportion', fontsize=12)
        ax.set_xlabel('Residue pair (FcμR-D1-Fcμ)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(pairs_sorted, rotation=45, ha='right', fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=12)
        ax.set_ylim(0,1.05)
        ax.grid(axis='y', alpha=0.3)
        plt.subplots_adjust(left=0.08, bottom=0.25, right=0.80)
        plt.savefig(THREE_GROUPS_BAR_PNG, dpi=1200)
        plt.savefig(THREE_GROUPS_BAR_PDF, dpi=1200)
        plt.close()
        print(f"Saved: {THREE_GROUPS_BAR_PNG} / {THREE_GROUPS_BAR_PDF}")
        pd.DataFrame({'pair':pairs_sorted,'dimer_avg':dimer_sorted,
                      'pentamer_avg':pentamer_sorted,'sIgM_avg':sigm_sorted}).to_csv(THREE_GROUPS_CSV, index=False)
    else:
        print("No common pairs for three-group bar plot.")

    # -------------------------
    # Fig. 4A Stoichiometry heatmap
    # -------------------------
    print("\n" + "=" * 60)
    print("Fig. 4A. Hierarchical contact patterns under increasing FcµR stoichiometry")
    print("=" * 60)

    fcmur1_avg = compute_group_average(FCMUR1_PDBS, per_pdb_fractions)
    fcmur4_avg = compute_group_average(FCMUR4_PDBS, per_pdb_fractions)
    fcmur8_avg = compute_group_average(FCMUR8_PDBS, per_pdb_fractions)

    all_pairs = set(fcmur1_avg.keys()) | set(fcmur4_avg.keys()) | set(fcmur8_avg.keys())
    MIN_THRESHOLD = 0.0
    keep_pairs = []
    for p in all_pairs:
        v1 = fcmur1_avg.get(p, 0)
        v4 = fcmur4_avg.get(p, 0)
        v8 = fcmur8_avg.get(p, 0)
        if v1 > MIN_THRESHOLD and v4 > MIN_THRESHOLD and v8 > MIN_THRESHOLD:
            keep_pairs.append(p)
    if not keep_pairs:
        print(f"Warning: No pairs with all proportions > {MIN_THRESHOLD}. Using pairs with max > 0.9 instead.")
        for p in all_pairs:
            if max(fcmur1_avg.get(p, 0), fcmur4_avg.get(p, 0), fcmur8_avg.get(p, 0)) > 0.9:
                keep_pairs.append(p)
    keep_pairs.sort(key=lambda p: fcmur8_avg.get(p, 0), reverse=True)
    data_matrix = []
    index_labels = []
    for p in keep_pairs:
        data_matrix.append([fcmur1_avg.get(p, 0), fcmur4_avg.get(p, 0), fcmur8_avg.get(p, 0)])
        index_labels.append(f"{p[0]}-{p[1]}")
    if not data_matrix:
        print("No residue pairs meet the criteria. Skipping Figure 4.")
    else:
        df_compare = pd.DataFrame(data_matrix, index=index_labels, columns=['1 FcμR', '4 FcμR', '8 FcμR'])
        full_pairs = sorted(all_pairs,
                            key=lambda p: max(fcmur1_avg.get(p, 0), fcmur4_avg.get(p, 0), fcmur8_avg.get(p, 0)),
                            reverse=True)
        full_data = []
        for p in full_pairs:
            full_data.append([fcmur1_avg.get(p, 0), fcmur4_avg.get(p, 0), fcmur8_avg.get(p, 0)])
        df_full = pd.DataFrame(full_data, index=[f"{p[0]}-{p[1]}" for p in full_pairs],
                               columns=['1 FcμR', '4 FcμR', '8 FcμR'])
        df_full.to_csv(FULL_HEATMAP_CSV)
        print(f"Saved full data (supplementary) to: {FULL_HEATMAP_CSV}")

        fig_width_cm = 17.6
        fig_width_inch = fig_width_cm / 2.54
        n_rows = len(df_compare)
        row_height_inch = 0.3
        fig_height_inch = max(3.0, n_rows * row_height_inch)
        fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch), dpi=1200)
        heatmap = sns.heatmap(df_compare, annot=True, fmt=".2f", cmap='YlOrRd',
                              annot_kws={'size': 10}, vmin=0.5, vmax=1.0,
                              cbar_kws={'label': 'Contact proportion', 'shrink': 1.0, 'aspect': 30,
                                        'ticks': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
                              linewidths=0.5, linecolor='lightgray', ax=ax)
        cbar = heatmap.collections[0].colorbar
        cbar.ax.set_ylabel('Contact proportion', fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        ax.set_ylabel("Residue pair (FcμR-Fcμ)", fontsize=10)
        ax.set_xlabel("Number of FcμR-D1 chains", fontsize=10)
        ax.tick_params(axis='x', labelsize=10)
        plt.setp(ax.get_yticklabels(), rotation=0, ha='right', fontsize=10)
        plt.subplots_adjust(left=0.2)
        plt.tight_layout()
        plt.savefig(GROUP_COMPARE_HEATMAP_PNG, dpi=1200)
        plt.savefig(GROUP_COMPARE_HEATMAP_PDF, dpi=1200)
        plt.close()
        print(f"Saved heatmap with {n_rows} pairs (all proportions > {MIN_THRESHOLD}).")

    # ============================================================
    # Fig. 4B J chain specific analysis (pentamers vs sIgM)
    # ============================================================
    print("\n" + "=" * 60)
    print("J chain specific analysis: J chain - FcμR contacts")
    print("Pentamer (7YTC,7YTD,8BPE) vs sIgM (7YSG)")
    print("=" * 60)

    fig_width_cm = 17.6
    fig_width_inch = fig_width_cm / 2.54

    # Pentamer PDBs and their FcμR counts
    J_PENTAMER_PDBS = {
        '7YTC': 1,
        '7YTD': 4,
        '8BPE': 8,
    }
    J_SIGM_PDB = '7YSG'

    # Compute J chain contacts for pentamers
    j_pentamer_contacts = {}
    for pdb, n_fcmur in J_PENTAMER_PDBS.items():
        entry = CHAIN_MAP.get(pdb)
        if not entry or 'J_chain' not in entry:
            print(f"  Skip {pdb}: no J_chain defined")
            continue
        if not os.path.exists(entry['file']):
            print(f"  File not found: {entry['file']}")
            continue
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure(pdb, entry['file'])
        fractions = get_jchain_contacts(struct, entry['FcμR'], entry['J_chain'], DIST_CUTOFF)
        j_pentamer_contacts[pdb] = fractions
        print(f"  {pdb} (FcμR={n_fcmur}) -> {len(fractions)} contacts")

    # Compute J chain contacts for sIgM
    j_sigm_contacts = {}
    entry = CHAIN_MAP.get(J_SIGM_PDB)
    if entry and 'J_chain' in entry and os.path.exists(entry['file']):
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure(J_SIGM_PDB, entry['file'])
        fractions = get_jchain_contacts(struct, entry['FcμR'], entry['J_chain'], DIST_CUTOFF)
        j_sigm_contacts[J_SIGM_PDB] = fractions
        print(f"  {J_SIGM_PDB} (sIgM) -> {len(fractions)} contacts")
    else:
        print(f"  Warning: {J_SIGM_PDB} not found or missing J_chain")

    # Save pentamer full J chain contacts (no filtering)
    pentamer_all_rows = []
    for pdb, frac in j_pentamer_contacts.items():
        for (fcmur_res, j_res), prop in frac.items():
            pentamer_all_rows.append({'PDB': pdb, 'FcμR_residue': fcmur_res, 'J_chain_residue': j_res, 'proportion': prop})
    if pentamer_all_rows:
        pd.DataFrame(pentamer_all_rows).to_csv(J_PENTAMER_FULL_CSV, index=False)
        print(f"Saved pentamer full contacts: {J_PENTAMER_FULL_CSV}")

    # Average by FcμR stoichiometry for pentamers
    j_group1 = [pdb for pdb, n in J_PENTAMER_PDBS.items() if n == 1]   # ['7YTC']
    j_group4 = [pdb for pdb, n in J_PENTAMER_PDBS.items() if n == 4]   # ['7YTD']
    j_group8 = [pdb for pdb, n in J_PENTAMER_PDBS.items() if n == 8]   # ['8BPE']

    j_avg1 = compute_group_average(j_group1, j_pentamer_contacts)
    j_avg4 = compute_group_average(j_group4, j_pentamer_contacts)
    j_avg8 = compute_group_average(j_group8, j_pentamer_contacts)

    j_all_pairs = set(j_avg1.keys()) | set(j_avg4.keys()) | set(j_avg8.keys())

    # Filter for heatmap: keep pairs with max proportion > 0.3
    J_HEATMAP_THRESHOLD = 0.3
    j_keep_pairs = []
    for p in j_all_pairs:
        v1 = j_avg1.get(p, 0)
        v4 = j_avg4.get(p, 0)
        v8 = j_avg8.get(p, 0)
        if max(v1, v4, v8) > J_HEATMAP_THRESHOLD:
            j_keep_pairs.append(p)
    j_keep_pairs.sort(key=lambda p: j_avg8.get(p, 0), reverse=True)

    if j_keep_pairs:
        j_data_matrix = []
        j_index_labels = []
        for p in j_keep_pairs:
            j_data_matrix.append([j_avg1.get(p, 0), j_avg4.get(p, 0), j_avg8.get(p, 0)])
            j_index_labels.append(f"{p[0]}-{p[1]}")
        df_j_compare = pd.DataFrame(j_data_matrix, index=j_index_labels,
                                    columns=['1 FcμR', '4 FcμR', '8 FcμR'])

        # Save filtered data for heatmap
        j_heatmap_csv = os.path.join(OUTPUT_DIR, "source_data_fig4b_pentamer_j_chain_stoichiometry.csv")
        df_j_compare.to_csv(j_heatmap_csv)
        print(f"Saved filtered data for heatmap (max > {J_HEATMAP_THRESHOLD}): {j_heatmap_csv}")

        # Plot heatmap
        fig_width_cm = 17.6
        fig_width_inch = fig_width_cm / 2.54
        n_rows = len(df_j_compare)
        row_height_inch = 0.3
        fig_height_inch = max(3.0, n_rows * row_height_inch)

        fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch), dpi=1200)
        heatmap = sns.heatmap(df_j_compare, annot=True, fmt=".2f", cmap='YlOrRd',
                              annot_kws={'size': 10},
                              vmin=0.5, vmax=1.0,
                              cbar_kws={'label': 'Contact proportion', 'shrink': 1.0, 'aspect': 30,
                                        'ticks': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
                              linewidths=0.5, linecolor='lightgray', ax=ax)
        cbar = heatmap.collections[0].colorbar
        cbar.ax.set_ylabel('Contact proportion', fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        ax.set_ylabel("Residue pair (FcμR - J)", fontsize=10)
        ax.set_xlabel("Number of FcμR-D1 chains", fontsize=10)
        ax.tick_params(axis='x', labelsize=10)
        plt.setp(ax.get_yticklabels(), rotation=0, ha='right', fontsize=10)
        plt.subplots_adjust(left=0.2)
        plt.tight_layout()
        plt.savefig(J_PENTAMER_HEATMAP_PNG, dpi=1200)
        plt.savefig(J_PENTAMER_HEATMAP_PDF, dpi=1200)
        plt.close()
        print(f"Saved pentamer stoichiometry heatmap ({n_rows} pairs, max > {J_HEATMAP_THRESHOLD}).")
    else:
        print("No pentamer J-chain pairs exceed threshold for heatmap.")

    # Save sIgM J chain contacts (all)
    if j_sigm_contacts:
        sigm_rows = []
        for pdb, frac in j_sigm_contacts.items():
            for (fcmur_res, j_res), prop in frac.items():
                sigm_rows.append({'PDB': pdb, 'FcμR_residue': fcmur_res, 'J_chain_residue': j_res, 'proportion': prop})
        pd.DataFrame(sigm_rows).to_csv(J_SIGM_CSV, index=False)
        print(f"Saved sIgM J-chain contacts (all): {J_SIGM_CSV}")
    else:
        print("No sIgM J-chain contacts found.")

    # ============================================================
    # NEW: FcμR – SC (secretory component) analysis for sIgM only
    # ============================================================
    print("\n" + "=" * 60)
    print("FcμR – SC (secretory component) contact analysis for sIgM (7YSG)")
    print("=" * 60)

    sc_contacts = {}
    entry = CHAIN_MAP.get(J_SIGM_PDB)
    if entry and 'SC' in entry and os.path.exists(entry['file']):
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure(J_SIGM_PDB, entry['file'])
        fractions = get_fcmur_sc_contacts(struct, entry['FcμR'], entry['SC'], DIST_CUTOFF)
        sc_contacts[J_SIGM_PDB] = fractions
        print(f"  {J_SIGM_PDB} (sIgM) FcμR–SC -> {len(fractions)} contacts")
    else:
        print(f"  Warning: {J_SIGM_PDB} SC chain not defined or file missing.")

    # Save all FcμR–SC contacts for sIgM
    if sc_contacts:
        sc_rows = []
        for pdb, frac in sc_contacts.items():
            for (fcmur_res, sc_res), prop in frac.items():
                sc_rows.append({'PDB': pdb, 'FcμR_residue': fcmur_res, 'SC_residue': sc_res, 'proportion': prop})
        pd.DataFrame(sc_rows).to_csv(J_SIGM_SC_CSV, index=False)
        print(f"Saved sIgM FcμR–SC contacts (all): {J_SIGM_SC_CSV}")
    else:
        print("No FcμR–SC contacts found for sIgM.")

if __name__ == "__main__":
    main()