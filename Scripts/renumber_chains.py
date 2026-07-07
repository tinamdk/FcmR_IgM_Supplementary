from Bio.PDB import PDBParser, PDBIO
import sys


def modify_chains_and_residues(input_pdb, output_pdb, chain_mapping, residue_offsets):
    """
    Modify the chain ID and residue numbers of the PDB file

    PDB file
Parameters:
input_pdb (str): Path of the input PDB file
output_pdb (str): Path of the scripts PDB file
chain_mapping (dict): Dictionary for chain ID mapping, such as {'A':'X', 'B':'Y'}
residue_offsets (dict): Dictionary for chain residue offsets, such as {'A':0, 'B':1000}
    """
    # Parse the PDB file
    parser = PDBParser()
    structure = parser.get_structure("input_structure", input_pdb)

    # Traverse all models and chains
    for model in structure:
        chains = list(model.child_list)  # Copy the linked list to avoid the problem of modifying it during iteration.
        for chain in chains:
            old_chain_id = chain.id

            # Modify the chain ID
            if old_chain_id in chain_mapping:
                new_chain_id = chain_mapping[old_chain_id]
                chain.id = new_chain_id  # 更新链ID

            # Modify residue numbers (only for the specified chain)
            if old_chain_id in residue_offsets:
                offset = residue_offsets[old_chain_id]
                for residue in chain:
                    new_residue_id = (residue.id[0], residue.id[1] + offset, residue.id[2])
                    residue.id = new_residue_id

    # Save the modified structure
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)


# Example call
if __name__ == "__main__":
    input_pdb = "./source_file/7YTE.pdb"  # Input file path
    output_pdb = "./output_file/7yte_renumber.pdb"  # Output file path
    # Define the chain ID mapping and residue offset (modify as needed)
    chain_mapping = {
        # 'A': 'X',  # Change link A to X
        # 'B': 'Y',  # Change link B to Y
        # Other links can be added as well...
    }
    residue_offsets = {
        'A': 0,  # The residue numbers of chain A remain unchanged, 7YTE
        'B': 1000,  # Residue number of chain B+1000
        # 'C': 2000,  # The 7YTC and other structures can be adjusted as needed.
        # Other links can be added as well...
        # ...
    }

    # Implement the modification
    modify_chains_and_residues(input_pdb, output_pdb, chain_mapping, residue_offsets)

    # Verification
    parser = PDBParser()
    structure = parser.get_structure("scripts", output_pdb)
    for model in structure:
        for chain in model:
            print(f"Chain {chain.id}:")
            residues = list(chain)
            print(f"First residue ID: {residues[0].id[1]}")
            print(f"Last residue ID: {residues[-1].id[1]}\n")