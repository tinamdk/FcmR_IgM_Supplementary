from Bio.PDB import PDBParser, PDBIO
import sys


def modify_chains_and_residues(input_pdb, output_pdb, chain_mapping, residue_offsets):
    """
    修改PDB文件的链ID和残基编号

    参数:
        input_pdb (str): 输入PDB文件路径
        output_pdb (str): 输出PDB文件路径
        chain_mapping (dict): 链ID映射字典，如 {'A':'X', 'B':'Y'}
        residue_offsets (dict): 链残基偏移量字典，如 {'A':0, 'B':1000}
    """
    # 解析PDB文件
    parser = PDBParser()
    structure = parser.get_structure("input_structure", input_pdb)

    # 遍历所有模型和链
    for model in structure:
        chains = list(model.child_list)  # 复制链列表以避免迭代时修改的问题
        for chain in chains:
            old_chain_id = chain.id

            # 修改链ID
            if old_chain_id in chain_mapping:
                new_chain_id = chain_mapping[old_chain_id]
                chain.id = new_chain_id  # 更新链ID

            # 修改残基编号（仅处理指定链）
            if old_chain_id in residue_offsets:
                offset = residue_offsets[old_chain_id]
                for residue in chain:
                    new_residue_id = (residue.id[0], residue.id[1] + offset, residue.id[2])
                    residue.id = new_residue_id

    # 保存修改后的结构
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)


# 示例调用
if __name__ == "__main__":
    input_pdb = "./source_file/8BPF.pdb"  # 输入文件路径
    output_pdb = "./output_file/8bpf_renumber.pdb"  # 输出文件路径
    # 定义链ID映射和残基偏移量（按需修改）
    chain_mapping = {
        # 'A': 'X',  # 将链A改为X
        # 'B': 'Y',  # 将链B改为Y
        # 其他链可继续添加...
    }
    residue_offsets = {
        'A': 0,  # 链A残基编号不变
        'B': 1000,  # 链B残基编号+1000
        'C': 2000,  # 链B残基编号+2000
        'D': 3000,  # 链B残基编号+3000
        'E': 4000,  # 链B残基编号+4000
        'F': 5000,  # 链B残基编号+5000
        'G': 6000,  # 链B残基编号+6000
        'H': 7000,  # 链B残基编号+7000
        'K': 8000,  # 链B残基编号+8000
        'L': 9000,  # 链B残基编号+9000
        # 其他链可继续添加...
    }

    # 执行修改
    modify_chains_and_residues(input_pdb, output_pdb, chain_mapping, residue_offsets)

    # 验证
    parser = PDBParser()
    structure = parser.get_structure("output", output_pdb)
    for model in structure:
        for chain in model:
            print(f"Chain {chain.id}:")
            residues = list(chain)
            print(f"First residue ID: {residues[0].id[1]}")
            print(f"Last residue ID: {residues[-1].id[1]}\n")