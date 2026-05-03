from Bio.PDB import PDBParser, NeighborSearch
import numpy as np

def get_interface_contacts(pdb_file, chain1, chain2, cutoff=5.0):
    # Parse PDB file
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]  # Use first model

    # Extract chains
    chainA = model[chain1]
    chainB = model[chain2]

    # Get residues from each chain
    residuesA = list(chainA.get_residues())
    residuesB = list(chainB.get_residues())

    # Collect non-hydrogen atoms from ChainB
    atomsB = []
    for res in residuesB:
        atomsB.extend(atom for atom in res.get_atoms() if atom.element != 'H')

    # Build neighbor search tree for ChainB atoms
    ns = NeighborSearch(atomsB)

    # Identify interfacial residue pairs
    contact_pairs = set()
    for resA in residuesA:
        for atom in resA.get_atoms():
            if atom.element == 'H':
                continue  # Skip hydrogens
            # Find atoms in ChainB within cutoff
            neighbors = ns.search(atom.coord, cutoff)
            for nb_atom in neighbors:
                resB = nb_atom.get_parent()
                # Record residue identifiers (chain, res_seq, insertion)
                keyA = (chain1, resA.id[1], resA.id[2])  # (chain, res_seq, icode)
                keyB = (chain2, resB.id[1], resB.id[2])
                contact_pairs.add((keyA, keyB))

    # Generate contact matrix
    sorted_resA = sorted(residuesA, key=lambda r: r.id[1])
    sorted_resB = sorted(residuesB, key=lambda r: r.id[1])
    contact_matrix = np.zeros((len(sorted_resA), len(sorted_resB)), dtype=int)

    # Map residue keys to matrix indices
    resA_to_idx = {res.id[1]: idx for idx, res in enumerate(sorted_resA)}
    resB_to_idx = {res.id[1]: idx for idx, res in enumerate(sorted_resB)}

    # Populate contact matrix
    for (keyA, keyB) in contact_pairs:
        i = resA_to_idx.get(keyA[1], -1)
        j = resB_to_idx.get(keyB[1], -1)
        if i != -1 and j != -1:
            contact_matrix[i, j] = 1

    return contact_pairs, contact_matrix, sorted_resA, sorted_resB

# Example usage
pdb_file = "example.pdb"
chain1, chain2 = "A", "B"
contacts, matrix, resA_list, resB_list = get_interface_contacts(pdb_file, chain1, chain2)

# Output results
print("Residue Pairs in Contact:", contacts)
print("Contact Matrix:\n", matrix)
print("Chain A Residues:", [res.id[1] for res in resA_list])
print("Chain B Residues:", [res.id[1] for res in resB_list])