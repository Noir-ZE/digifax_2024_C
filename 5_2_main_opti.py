import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import MixedVariableSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
import time

# --- 1. 导入模块一的核心评估函数 ---
# 确保 q5_module_problem_definition.py 在同一目录下
try:
    from q5_module_problem_definition import evaluate_objectives, MATERIAL_MAP, WAVEFORM_MAP
except ImportError:
    print("错误: 无法导入 'q5_module_problem_definition'。")
    print("请确保该文件与本脚本在同一目录下，并且没有语法错误。")
    exit()


# --- 2. 定义多目标优化问题 ---
class MagneticComponentOptimization(Problem):
    """
    将磁性元件优化问题定义为 pymoo 的 Problem 类。
    决策变量 x = [Temperature, Frequency, Material_Code, Waveform_Code, Bm]
    """

    def __init__(self):
        # 定义决策变量的边界
        # T: [25, 90], f: [50k, 500k], M: [0, 3], W: [0, 2], Bm: [0.05, 0.35]
        xl = np.array([25, 50000, 0, 0, 0.05])
        xu = np.array([90, 500000, 3, 2, 0.35])

        super().__init__(n_var=5, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        评估函数，pymoo会批量调用此函数。
        x 是一个 (n_individuals, n_vars) 的二维数组。
        """
        # 为这批个体计算目标函数值
        # 我们使用列表推导式来为x中的每个个体（每一行）调用评估函数
        results = [evaluate_objectives(individual) for individual in x]

        # 将结果列表转换为NumPy数组
        # out["F"] 需要一个 (n_individuals, n_objectives) 的数组
        out["F"] = np.array(results)


# --- 3. 主优化流程 ---
if __name__ == '__main__':
    print("--- 模块二：主优化器 ---")

    # 实例化我们定义的问题
    problem = MagneticComponentOptimization()

    # 定义变量类型掩码：'real'代表连续变量，'int'代表整数变量
    mask = ['real', 'real', 'int', 'int', 'real']

    # 配置NSGA-II算法
    algorithm = NSGA2(
        pop_size=100,  # 种群大小
        sampling=MixedVariableSampling(mask, {
            "real": np.random.random,
            "int": (lambda low, high: np.random.randint(low, high + 1))
        }),
        crossover=SBX(prob=0.9, eta=15, vtype=float),  # 模拟二进制交叉用于连续变量
        mutation=PM(eta=20, vtype=float),  # 多项式变异用于连续变量
        eliminate_duplicates=True
    )

    # 设置终止条件
    from pymoo.termination import get_termination

    termination = get_termination("n_gen", 200)  # 迭代200代

    print("\n--- 开始执行NSGA-II优化 ---")
    print(f"种群大小: {algorithm.pop_size}, 迭代代数: {termination.n_gen}")
    start_time = time.time()

    # 执行优化
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,  # 设置种子以保证结果可复现
                   save_history=True,
                   verbose=True)  # 打印每一代的进度

    end_time = time.time()
    print(f"\n--- 优化完成 ---")
    print(f"耗时: {end_time - start_time:.2f} 秒")
    print(f"找到 {len(res.F)} 个帕累托最优解。")

    # --- 4. 处理并保存结果 ---
    print("\n--- 正在处理并保存结果 ---")

    # 提取决策变量和目标函数值
    pareto_solutions = res.X
    pareto_front = res.F

    # 创建结果DataFrame
    df_results = pd.DataFrame(pareto_solutions,
                              columns=['Temperature', 'Frequency', 'Material_Code', 'Waveform_Code', 'Bm'])

    # 添加目标函数值列
    df_results['Core_Loss'] = pareto_front[:, 0]
    df_results['Neg_Energy_Transfer'] = pareto_front[:, 1]
    df_results['Energy_Transfer'] = -df_results['Neg_Energy_Transfer']  # 添加一列正的传输磁能，方便分析

    # 将整数编码的分类变量解码为字符串，方便阅读
    df_results['Material'] = df_results['Material_Code'].round().astype(int).map(MATERIAL_MAP)
    df_results['Waveform'] = df_results['Waveform_Code'].round().astype(int).map(WAVEFORM_MAP)

    # 保存到CSV文件
    output_path = 'pareto_results_raw.csv'
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"帕累托最优解集的原始数据已保存至: '{output_path}'")
    print("\n结果预览 (前5行):")
    print(df_results.head())