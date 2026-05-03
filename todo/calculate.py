from Bio.PDB import PDBParser
import numpy as np


def auto_generate_constraints(pdb_file, output_tbl):
    parser = PDBParser()
    structure = parser.get_structure("sIgM", pdb_file)

    # 计算SC/J链质心
    sc_j_coords= []
    for chain in structure.get_chains():
        if chain.id in ["S", "J"]:
            for residue in chain.get_residues():
                sc_j_coords.extend([atom.coord for atom in residue.get_atoms()])
    sc_j_center = np.mean(sc_j_coords, axis=0)

    # 筛选对侧Cµ4链
    max_dist = 0
    opposite_chain = None
    for chain in structure.get_chains():
        if chain.id not in ["S", "J"]:
            cu4_coords = [atom.coord for res in chain.get_residues() if 446 <= res.id[1] <= 558 for atom in
                          res.get_atoms()]
            if cu4_coords:
                cu4_center = np.mean(cu4_coords, axis=0)
                distance = np.linalg.norm(sc_j_center - cu4_center)
                if distance > max_dist:
                    max_dist = distance
                    opposite_chain = chain.id

    # 生成HADDOCK约束文件
    with open(output_tbl, "w") as f:
        f.write(f"! 对侧链：{opposite_chain}\n")
        f.write(f"assign (resid 465 and segid {opposite_chain}) (resid 45 and segid R) 0.0 3.0 0.5\n")
        f.write(f"assign (resid 468 and segid {opposite_chain}) (resid 67 and segid R) 0.0 3.0 0.5\n")
        f.write(f"assign (resid 526 and segid {opposite_chain}) (resid 69 and segid R) 0.0 3.0 0.5\n")

    print(f"约束文件已生成：{output_tbl}")


auto_generate_constraints("6KXS_clean.pdb","restraints.tbl")