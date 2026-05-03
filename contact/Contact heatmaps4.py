#!/usr/bin/env python3
"""
Generate Figure 2: Contact heatmap + per-complex participation ratio matrix.
Core residues are expected to appear in all 7 complexes (n >= 1).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from Bio.PDB import PDBParser, NeighborSearch, is_aa
from collections import defaultdict

# ============================================================
# CONFIGURATION
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

# Core residues as they appear in the manuscript (three-letter + number)
CORE_RESIDUES = ['Arg45', 'Thr60', 'Ser63', 'Phe67', 'Lys69', 'Thr110', 'Asp111']
DIST_CUTOFF = 3.0
OUTPUT_HEATMAP = "Figure2A_heatmap.png"
OUTPUT_MATRIX = "Figure2B_ratio_matrix.png"
OUTPUT_COMBINED = "Figure2_combined.png"

# ============================================================
# Helper functions
# ============================================================
def standardize_label(label):
    """Convert 'ARG45' -> 'Arg45' (capitalized three-letter + number)"""
    if len(label) < 6:
        return label
    three = label[:3].capitalize()
    num = label[3:]
    return f"{three}{num}"

def residue_matches(target, candidate, max_diff=1):
    """
    Check if candidate residue (e.g., 'ARG45') matches target (e.g., 'Arg45')
    Allows number offset up to max_diff (due to PDB numbering variations).
    """
    target_aa = target[:3].lower()
    target_num = int(target[3:])
    cand_aa = candidate[:3].lower()
    if cand_aa != target_aa:
        return False
    cand_num = int(candidate[3:])
    return abs(cand_num - target_num) <= max_diff

def get_residue_contact_per_chain(structure, fcmr_chains, igm_chains, cutoff=3.0):
    """
    Returns:
        chain_residues: dict {chain_id: set of residue labels (e.g., 'ARG45') that have contact}
        total_chains: number of FcμR chains present
        total_contacts_per_residue: dict {residue_label: total atom contacts across all chains}
    """
    # Collect IgM atoms
    igm_atoms = []
    for ch in igm_chains:
        if ch in structure[0]:
            for atom in structure[0][ch].get_atoms():
                igm_atoms.append(atom)
    if not igm_atoms:
        return {}, 0, {}
    ns = NeighborSearch(igm_atoms)
    chain_residues = {}
    total_contacts = defaultdict(int)
    present_chains = 0
    for ch_id in fcmr_chains:
        if ch_id not in structure[0]:
            continue
        present_chains += 1
        residues_with_contact = set()
        for residue in structure[0][ch_id].get_residues():
            if not is_aa(residue):
                continue
            res_label = f"{residue.resname}{residue.id[1]}"
            atom_contacts = 0
            for atom in residue.get_atoms():
                if ns.search(atom.coord, cutoff):
                    atom_contacts += 1
            if atom_contacts > 0:
                residues_with_contact.add(res_label)
                total_contacts[res_label] += atom_contacts
        chain_residues[ch_id] = residues_with_contact
    return chain_residues, present_chains, total_contacts

# ============================================================
# Main analysis
# ============================================================
def main():
    # For heatmap: average contacts per chain per residue per PDB
    residue_contact_avg = defaultdict(lambda: [0] * len(PDB_IDS))
    # For ratio matrix: core residues participation n/m
    residue_ratio = {res: {} for res in CORE_RESIDUES}

    for pdb_idx, pdb_id in enumerate(PDB_IDS):
        print(f"\nProcessing {pdb_id}...")
        entry = CHAIN_MAP.get(pdb_id)
        if not entry:
            print(f"  No entry for {pdb_id}, skip")
            continue
        pdb_file = entry.get('file')
        if not pdb_file or not os.path.exists(pdb_file):
            print(f"  File {pdb_file} not found, skip")
            continue
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure(pdb_id, pdb_file)
        except Exception as e:
            print(f"  Error parsing: {e}, skip")
            continue

        fcmr_chains = entry.get('FcμR', [])
        igm_chains = entry.get('IgM', [])
        if not fcmr_chains or not igm_chains:
            print(f"  Missing chain assignment, skip")
            continue

        chain_contacts, total_chains, total_contacts = get_residue_contact_per_chain(
            structure, fcmr_chains, igm_chains, DIST_CUTOFF)
        if total_chains == 0:
            print(f"  No FcμR chains found, skip")
            continue

        # Store average contacts per chain for heatmap
        for res_label, tot_contacts in total_contacts.items():
            avg = tot_contacts / total_chains
            residue_contact_avg[res_label][pdb_idx] = avg

        # For each core residue, compute n/m using fuzzy matching
        print(f"  Total FcμR chains: {total_chains}")
        for core_res in CORE_RESIDUES:
            n = 0
            # Check each chain's contact set
            for ch, res_set in chain_contacts.items():
                matched = False
                for cand_label in res_set:
                    if residue_matches(core_res, cand_label, max_diff=1):
                        matched = True
                        break
                if matched:
                    n += 1
            ratio_str = f"{n}/{total_chains}"
            residue_ratio[core_res][pdb_id] = ratio_str
            print(f"    {core_res}: {ratio_str}")

    # ========== Figure 2A: Contact heatmap ==========
    df_heat = pd.DataFrame.from_dict(residue_contact_avg, orient='index', columns=PDB_IDS)
    df_heat.fillna(0, inplace=True)
    df_heat['mean'] = df_heat.mean(axis=1)
    df_heat_sorted = df_heat.sort_values('mean', ascending=False).drop('mean', axis=1)

    # Create display labels with asterisk for core residues (fuzzy match)
    core_set_lower = {res.lower() for res in CORE_RESIDUES}
    display_labels = []
    for res in df_heat_sorted.index:
        std_label = standardize_label(res)
        # Check if this residue matches any core residue (by aa and number within 1)
        is_core = False
        for core in CORE_RESIDUES:
            if residue_matches(core, res, max_diff=1):
                is_core = True
                break
        if is_core:
            std_label = f"{std_label}*"
        display_labels.append(std_label)
    df_plot = df_heat_sorted.copy()
    df_plot.index = display_labels

    plt.figure(figsize=(10, 8), dpi=300)
    ax = sns.heatmap(df_plot, cmap='Reds', annot=True, fmt='.1f', linewidths=0.5,
                     cbar_kws={'label': 'Average contacts per FcµR chain'})
    ax.set_xlabel("PDB complex", fontsize=12)
    ax.set_ylabel("FcµR residue", fontsize=12)
    ax.set_title("A", loc='left', fontweight='bold')
    ax.tick_params(axis='y', labelsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_HEATMAP)
    plt.close()
    print(f"\nSaved {OUTPUT_HEATMAP}")

    # ========== Figure 2B: Ratio matrix for core residues ==========
    ratio_df = pd.DataFrame(index=CORE_RESIDUES, columns=PDB_IDS)
    for res in CORE_RESIDUES:
        for pdb in PDB_IDS:
            ratio_df.loc[res, pdb] = residue_ratio[res].get(pdb, "0/0")
    # Numeric values for color mapping
    def fraction(s):
        try:
            n, d = s.split('/')
            return int(n) / int(d) if int(d) > 0 else 0
        except:
            return 0
    ratio_numeric = ratio_df.applymap(fraction)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    cmap = LinearSegmentedColormap.from_list('WhiteRed', ['white', 'red'], N=100)
    im = ax.imshow(ratio_numeric, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    for i in range(len(CORE_RESIDUES)):
        for j in range(len(PDB_IDS)):
            text = ratio_df.iloc[i, j]
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color='black')
    ax.set_xticks(range(len(PDB_IDS)))
    ax.set_yticks(range(len(CORE_RESIDUES)))
    ax.set_xticklabels(PDB_IDS, rotation=45, ha='right')
    ax.set_yticklabels(CORE_RESIDUES)
    ax.set_xlabel("PDB complex", fontsize=12)
    ax.set_ylabel("Core binding residue", fontsize=12)
    ax.set_title("B", loc='left', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Proportion of FcµR chains with contact')
    plt.tight_layout()
    plt.savefig(OUTPUT_MATRIX)
    plt.close()
    print(f"Saved {OUTPUT_MATRIX}")

    # Combine both subplots
    from matplotlib.image import imread
    img1 = imread(OUTPUT_HEATMAP)
    img2 = imread(OUTPUT_MATRIX)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.set_title('A', fontsize=14, fontweight='bold', loc='left')
    ax2.imshow(img2)
    ax2.axis('off')
    ax2.set_title('B', fontsize=14, fontweight='bold', loc='left')
    plt.tight_layout()
    plt.savefig(OUTPUT_COMBINED, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined figure {OUTPUT_COMBINED}")
    print("\nFigure 2 generation complete.")

if __name__ == "__main__":
    main()