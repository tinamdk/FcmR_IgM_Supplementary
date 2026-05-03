import os
import subprocess
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, PDBIO
import time



class FoldXAutomation:
    def __init__(self, foldx_path, pdb_file):
        self.foldx_path = foldx_path
        self.pdb_file = pdb_file
        self.base_name = os.path.splitext(pdb_file)[0]
        self.wildtype_energy = None

    def run_foldx_command(self, command, args, output_dir="."):
        """运行FoldX命令"""
        cmd = [
            self.foldx_path,
            f"--command={command}",
            f"--pdb={self.pdb_file}",
            *args
        ]

        # 添加输出目录
        if output_dir:
            cmd.append(f"--output-dir={output_dir}")

        print(f"运行命令: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("命令执行成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败: {e}")
            print(f"错误输出: {e.stderr}")
            return False

    def create_mutation_list(self, mutations, filename="individual_list.txt"):
        """创建突变列表文件"""
        with open(filename, 'w') as f:
            for mutation in mutations:
                # 格式: 突变, 链ID;
                f.write(f"{mutation[0]},{mutation[1]};\n")
        print(f"已创建突变列表文件: {filename}")
        return filename

    def calculate_wildtype_energy(self):
        """计算野生型能量"""
        print("计算野生型能量...")
        success = self.run_foldx_command(
            "Stability",
            ["--numberOfRuns=5", "--pH=7.0", "--temperature=298"],
            output_dir="./wildtype"
        )

        if success:
            # 读取野生型能量
            wt_file = "./wildtype/ST_0_complex.fxout"
            if os.path.exists(wt_file):
                df = pd.read_csv(wt_file, sep='\t')
                self.wildtype_energy = df['Total Energy'].iloc[0]
                print(f"野生型总能量: {self.wildtype_energy:.2f} kcal/mol")
                return True
        return False

    def run_mutations(self, mutations):
        """运行所有突变"""
        results = []

        # 创建突变列表文件
        mut_file = self.create_mutation_list(mutations)

        # 运行BuildModel
        print("运行BuildModel生成突变体...")
        success = self.run_foldx_command(
            "BuildModel",
            [
                f"--mutant-file={mut_file}",
                "--numberOfRuns=5",
                "--pH=7.0",
                "--temperature=298",
                "--ionStrength=0.05"
            ],
            output_dir="./mutants"
        )

        if success:
            # 分析每个突变体
            for i, mutation in enumerate(mutations, 1):
                mut_energy = self.analyze_mutant(i, mutation)
                if mut_energy is not None:
                    ddg = mut_energy - self.wildtype_energy
                    results.append({
                        'Mutation': f"{mutation[0]}_{mutation[1]}",
                        'Wildtype_Energy': self.wildtype_energy,
                        'Mutant_Energy': mut_energy,
                        'ΔΔG': ddg,
                        'Stability_Change': 'Destabilizing' if ddg > 0 else 'Stabilizing'
                    })

        return results

    def analyze_mutant(self, mutant_number, mutation):
        """分析单个突变体"""
        mut_pdb = f"./mutants/{self.base_name}_{mutant_number}.pdb"

        if not os.path.exists(mut_pdb):
            print(f"突变体文件不存在: {mut_pdb}")
            return None

        # 运行稳定性分析
        success = self.run_foldx_command(
            "Stability",
            ["--numberOfRuns=3", "--pH=7.0", "--temperature=298"],
            output_dir=f"./mutants/analysis_{mutant_number}"
        )

        if success:
            # 读取突变体能量
            mut_file = f"./mutants/analysis_{mutant_number}/ST_0_{self.base_name}_{mutant_number}.fxout"
            if os.path.exists(mut_file):
                df = pd.read_csv(mut_file, sep='\t')
                energy = df['Total Energy'].iloc[0]
                print(f"突变体 {mutation} 能量: {energy:.2f} kcal/mol")
                return energy

        return None

    def save_results(self, results, filename="mutation_results.csv"):
        """保存结果到CSV文件"""
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"结果已保存到: {filename}")
        return df


def main():
    # 配置参数
    FOLDX_PATH = r"C:\Users\12555\Desktop\wangyi\windows_Fold_2\foldx5Windows64_0\foldx5Windows64_0\foldx_20251231.exe"
    PDB_FILE = "7ytc_renumber.pdb"

    # 定义要测试的突变列表
    MUTATIONS = [
        ("M42A", "A"),  # 链A的M42突变为丙氨酸
        ("M42A", "B"),  # 链B的M42突变为丙氨酸
        ("M42A", "C"),  # 链C的M42突变为丙氨酸
        ("M42A", "D"),  # 链D的M42突变为丙氨酸
        ("R85A", "A"),  # 链A的R85突变为丙氨酸
        ("K102A", "B")  # 链B的K102突变为丙氨酸
    ]

    # 初始化自动化对象
    automator = FoldXAutomation(FOLDX_PATH, PDB_FILE)

    # 1. 计算野生型能量
    if not automator.calculate_wildtype_energy():
        print("野生型能量计算失败，退出程序")
        return

    # 2. 运行所有突变
    print("\n开始运行突变分析...")
    results = automator.run_mutations(MUTATIONS)

    # 3. 保存结果
    if results:
        result_df = automator.save_results(results)
        print("\n最终结果:")
        print(result_df.to_string(index=False))
    else:
        print("没有获得有效结果")


if __name__ == "__main__":
    main()