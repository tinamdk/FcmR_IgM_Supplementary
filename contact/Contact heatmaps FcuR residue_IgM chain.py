import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, is_aa
from scipy.spatial import cKDTree

# ==================== CONFIGURATION ====================
PDB_LIST = [
    {"file": "source_file/7YTE.pdb", "IgM_chains": ["A", "B"], "FcµR_chains": ["C", "D"]},
    {"file": "source_file/7YTC.pdb", "IgM_chains": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
     "FcµR_chains": ["R"]},
    {"file": "source_file/7YTD.pdb", "IgM_chains": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
     "FcµR_chains": ["R", "S", "U", "V"]},
    {"file": "source_file/7YSG.pdb", "IgM_chains": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
     "FcµR_chains": ["U", "R", "S", "V"]},
    {"file": "source_file/8BPE.pdb", "IgM_chains": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
     "FcµR_chains": ["I", "M", "N", "O", "P", "Q", "R", "S"]},
    {"file": "source_file/8BPF.pdb", "IgM_chains": ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "J"],
     "FcµR_chains": ["I"]},
    {"file": "source_file/8BPG.pdb", "IgM_chains": ["C", "D", "E", "F"],
     "FcµR_chains": ["A", "B"]},
]
DISTANCE_CUTOFF = 3.0  # Angstroms
RESIDUE_RANGE = range(18, 125)  # FcµR D1 domain residues


# ==================== UTILITY FUNCTIONS ====================
def build_global_residue_name_map(pdb_list, receptor_key='FcµR_chains'):
    """
    Scan all PDB files and all receptor chains to collect residue names.
    Returns dict: {residue_number: "ThreeLetterNumber"} (e.g., 18 -> "Leu18").
    """
    name_map = {}
    for entry in pdb_list:
        pdb_file = entry["file"]
        if not os.path.exists(pdb_file):
            continue
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)
        except Exception:
            continue
        receptor_chains = entry[receptor_key]
        for model in structure:
            for chain in model:
                if chain.id not in receptor_chains:
                    continue
                for residue in chain:
                    if is_aa(residue):
                        res_id = residue.id[1]
                        if res_id not in name_map:
                            three = residue.resname.capitalize()
                            name_map[res_id] = f"{three}{res_id}"
    # Fallback for residues never found
    for r in RESIDUE_RANGE:
        if r not in name_map:
            name_map[r] = str(r)
    return name_map


def get_residue_contacts(pdb_file, receptor_chains, ligand_chains, cutoff):
    """
    For each receptor residue in RESIDUE_RANGE, determine if any atom is within cutoff of any ligand atom.
    Returns dict: (pdb_name, chain_id, res_id) -> 1 (contact) or 0 (no contact)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)

    ligand_coords = []
    for model in structure:
        for chain in model:
            if chain.id in ligand_chains:
                for residue in chain:
                    if is_aa(residue):
                        ligand_coords.extend([atom.coord for atom in residue])
    if not ligand_coords:
        print(f"Warning: No ligand atoms in {pdb_file} for chains {ligand_chains}")
        return {}

    tree = cKDTree(ligand_coords)
    pdb_name = os.path.basename(pdb_file).replace('.pdb', '')
    contacts = {}

    for model in structure:
        for chain in model:
            if chain.id not in receptor_chains:
                continue
            for residue in chain:
                if not is_aa(residue):
                    continue
                res_id = residue.id[1]
                if res_id not in RESIDUE_RANGE:
                    continue
                has_contact = False
                for atom in residue:
                    if tree.query_ball_point(atom.coord, cutoff):
                        has_contact = True
                        break
                key = (pdb_name, chain.id, res_id)
                contacts[key] = 1 if has_contact else 0
    return contacts


# ==================== MAIN ====================
def main():
    # Build global residue name map
    residue_name_map = build_global_residue_name_map(PDB_LIST)

    all_contacts = {}
    for entry in PDB_LIST:
        pdb_file = entry["file"]
        if not os.path.exists(pdb_file):
            print(f"File not found: {pdb_file}, skipped.")
            continue
        print(f"Processing {pdb_file} ...")
        contacts = get_residue_contacts(
            pdb_file, entry["FcµR_chains"], entry["IgM_chains"], DISTANCE_CUTOFF
        )
        all_contacts.update(contacts)

    if not all_contacts:
        print("No contacts found. Check PDB files.")
        return

    residues = sorted(set([k[2] for k in all_contacts.keys()]))
    residues = [r for r in residues if r in RESIDUE_RANGE]
    chain_labels = sorted(set([(k[0], k[1]) for k in all_contacts.keys()]))
    row_index = [f"{pdb}_{chain}" for pdb, chain in chain_labels]

    data = np.zeros((len(row_index), len(residues)), dtype=int)
    for (pdb, chain, res), flag in all_contacts.items():
        if res not in residues:
            continue
        row = f"{pdb}_{chain}"
        if row in row_index:
            data[row_index.index(row), residues.index(res)] = flag

    df = pd.DataFrame(data, index=row_index, columns=residues)
    df.to_csv("contact_matrix.csv")
    print(f"Contact matrix saved: {df.shape[0]} chains x {df.shape[1]} residues")

    total_chains = len(df)
    frequency = df.sum(axis=0) / total_chains
    key_residues = frequency[frequency >= 0.5].index.tolist()

    print("\n=== Key residues (frequency >= 50%) ===")
    for res in key_residues:
        occ = int(frequency[res] * total_chains)
        print(f"Residue {res}: {frequency[res]:.2f} ({occ}/{total_chains} chains)")

    # Generate italic labels for all residues
    all_labels = []
    for r in residues:
        name = residue_name_map.get(r, str(r))
        all_labels.append(rf"$\it{{{name}}}$")

    # Show only every 5th residue label to increase spacing
    STEP = 5
    display_labels = [label if i % STEP == 0 else '' for i, label in enumerate(all_labels)]

    # Plot heatmap
    plt.figure(figsize=(24, 10))
    ax = sns.heatmap(df, cmap='YlOrRd', cbar_kws={'label': 'Contact (1=yes, 0=no)'},
                     linewidths=0.5, linecolor='lightgray', square=False,
                     xticklabels=display_labels, yticklabels=True)

    ax.set_xlabel('FcμR residue', fontsize=12)
    ax.set_ylabel('FcμR chain (PDB_chain)', fontsize=12)
    ax.set_title('Contact frequency heatmap (3Å cutoff)', fontsize=14)

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)

    # 关键修改：使用 rect 参数保留左侧空间
    plt.tight_layout(rect=[0.01, 0.01, 0.01, 0.01])  # left=0.08 足够显示 ylabel
    plt.savefig("Figure2A_heatmap.png", dpi=1200)
    plt.show()
    print("Figure2A_heatmap.png saved (1200 dpi).")

    with open("key_residues.txt", "w") as f:
        f.write("Residue\tFrequency\tOccurrence/Total\n")
        for res in key_residues:
            occ = int(frequency[res] * total_chains)
            f.write(f"{res}\t{frequency[res]:.3f}\t{occ}/{total_chains}\n")
    print("Key residues saved to key_residues.txt")


if __name__ == "__main__":
    main()