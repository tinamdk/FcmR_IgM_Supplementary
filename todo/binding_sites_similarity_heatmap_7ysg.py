from Bio.PDB import PDBParser, NeighborSearch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 配置分析参数
PDB_FILE = 'source_file/7YSG.pdb'  # 替换为你的PDB文件
RECEPTOR_CHAINS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'J', 'P']  # 受体的所有链
LIGAND_CHAINS = ['R', 'S', 'U', 'V']  # 四个配体的链ID


# 2. 定义分析函数
def analyze_complex_binding_sites(pdb_file, receptor_chains, ligand_chains, cutoff=5.0):
    """
    分析多链受体与多个配体的结合界面
    返回一个字典，键是配体链ID，值是该配体与受体所有接触的DataFrame
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)
    model = structure[0]

    # 获取受体所有原子
    receptor_atoms = []
    for chain_id in receptor_chains:
        if chain_id in model:
            receptor_atoms.extend(list(model[chain_id].get_atoms()))

    # 初始化邻居搜索
    ns = NeighborSearch(receptor_atoms)

    all_contacts = {}

    # 分析每个配体
    for lig_chain in ligand_chains:
        if lig_chain not in model:
            print(f"Warning: Chain {lig_chain} not found in structure")
            continue

        lig_atoms = list(model[lig_chain].get_atoms())
        contacts = []

        for atom in lig_atoms:
            # 在受体中找邻居
            neighbors = ns.search(atom.coord, cutoff, level='A')
            for neighbor in neighbors:
                res_rec = neighbor.get_parent()
                res_lig = atom.get_parent()

                # 记录详细信息
                contact_info = {
                    'Ligand_Chain': lig_chain,
                    'Ligand_Res': f"{res_lig.resname}{res_lig.id[1]}",
                    'Ligand_ResID': res_lig.id[1],
                    'Receptor_Chain': neighbor.get_parent().get_parent().id,  # 获取链ID
                    'Receptor_Res': f"{res_rec.resname}{res_rec.id[1]}",
                    'Receptor_ResID': res_rec.id[1],
                    'Distance': np.linalg.norm(atom.coord - neighbor.coord)
                }
                contacts.append(contact_info)

        all_contacts[lig_chain] = pd.DataFrame(contacts)
        print(f"Found {len(contacts)} contacts for ligand chain {lig_chain}")

    return all_contacts


# 3. 执行分析
print("Analyzing multi-chain complex...")
all_sites_contacts = analyze_complex_binding_sites(PDB_FILE, RECEPTOR_CHAINS, LIGAND_CHAINS)

# 4. 保存每个结合位点的详细数据
for lig_chain, df in all_sites_contacts.items():
    df.to_csv(f'contacts_site_{lig_chain}.csv', index=False)

# 5. 高级分析：比较不同结合位点的相似性
# 提取每个位点与受体作用的残基集合（用于比较位点间差异）
binding_sites = {}
for lig_chain, df in all_sites_contacts.items():
    # 获取该位点涉及的所有受体残基（去重）
    unique_residues = set(zip(df['Receptor_Chain'], df['Receptor_ResID']))
    binding_sites[lig_chain] = unique_residues
    print(f"Site {lig_chain}: {len(unique_residues)} unique receptor residues involved")


# 6. 计算位点相似性（Jaccard相似系数）
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


# 创建相似性矩阵
chains = list(binding_sites.keys())
similarity_matrix = np.zeros((len(chains), len(chains)))

for i, chain1 in enumerate(chains):
    for j, chain2 in enumerate(chains):
        similarity_matrix[i, j] = jaccard_similarity(binding_sites[chain1], binding_sites[chain2])

# 7. 绘制结合位点相似性热图
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(similarity_matrix, cmap='YlOrRd', vmin=0, vmax=1)

# 添加标签
ax.set_xticks(np.arange(len(chains)))
ax.set_yticks(np.arange(len(chains)))
ax.set_xticklabels(chains)
ax.set_yticklabels(chains)
ax.set_xlabel('Ligand Chain (Binding Site)')
ax.set_ylabel('Ligand Chain (Binding Site)')
ax.set_title('Similarity Between Different Binding Sites\n(Jaccard Index of Receptor Residues)')

# 添加数值标注
for i in range(len(chains)):
    for j in range(len(chains)):
        text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black" if similarity_matrix[i, j] < 0.6 else "white")

# 添加颜色条
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Jaccard Similarity Index', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig('binding_sites_similarity_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("Analysis complete! Check the output CSV files and heatmap.")

# 例如，深入分析位点E
site_e_df = pd.read_csv('contacts_site_E.csv')
# 找出哪些受体残基接触最频繁
print(site_e_df['Receptor_Res'].value_counts().head(10))
# 找出距离最近（可能最关键）的相互作用
print(site_e_df.sort_values('Distance').head(10))