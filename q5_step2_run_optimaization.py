import numpy as np
import pandas as pd
import json
import joblib
import time
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# --- 1. 导入问题四的特征工程函数 ---
# 为保证特征提取逻辑的绝对一致性，我们直接从问题四的脚本中导入核心函数
try:
    from q4_step1_feature_engineering import calculate_bt_features, get_bt_cols
except ImportError:
    print("错误: 无法从'q4_step1_feature_engineering'导入所需函数。")
    print("请确保该脚本与本脚本在同一目录下。")
    exit()

# --- 2. 全局加载与设置 ---
print("--- 步骤2/3：执行高保真多目标优化 ---")
try:
    MODEL = joblib.load('q4_lgbm_model_optimized.pkl')
    TEMPLATE_LIBRARY = json.load(open('waveform_template_library.json'))
except FileNotFoundError as e:
    print(f"错误: 依赖文件未找到 -> {e}")
    print("请先运行 q4_all_steps.py 和 q5_step1_build_template_library.py")
    exit()

MODEL_FEATURES = MODEL.feature_name_
MATERIAL_MAP = {0: '1', 1: '2', 2: '3', 3: '4'}
WAVEFORM_MAP = {0: 'sin', 1: 'tri', 2: 'tra'}


# --- 3. 定义高保真优化问题 ---
class HighFidelityOptimization(Problem):
    def __init__(self):
        # 定义决策变量边界 [T, f, M_code, W_code, Bm]
        # Bm 范围根据附件一真实计算的Bm值设定
        xl = np.array([25, 50000, 0, 0, 0.005])

        # --- 核心修正：调整分类变量的上界 ---
        # 将上界从 X.99 调整为小于 X.5 的值，以防止 round() 函数越界。
        # Material (0,1,2,3) -> 上界 < 3.5
        # Waveform (0,1,2) -> 上界 < 2.5
        xu = np.array([90, 500000, 3.49, 2.49, 0.38])

        super().__init__(n_var=5, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        objectives = []
        for individual in x:
            T, f, M_code, W_code, Bm = individual

            # --- 高保真实践：使用真实波形模板 ---
            M_key = MATERIAL_MAP[int(round(M_code))]
            W_key = WAVEFORM_MAP[int(round(W_code))]
            template_key = f"M{M_key}_W{W_key}"

            template_info = TEMPLATE_LIBRARY[template_key]
            template_bt = np.array(template_info['template_bt'])
            template_bm = template_info['template_bm']

            # 线性缩放模板以匹配当前的Bm
            scaled_bt = template_bt * (Bm / template_bm) if template_bm > 0 else template_bt

            # --- 高保真实践：复用问题四的特征工程 ---
            # 构造一个临时的单行DataFrame来调用特征函数
            temp_df_row = pd.Series({
                'Temperature': T, 'Frequency': f,
                **dict(zip([f'B_t_{i}' for i in range(len(scaled_bt))], scaled_bt))
            })
            b_t_cols = get_bt_cols(pd.DataFrame([temp_df_row]))

            # 计算B(t)派生特征
            bt_features = calculate_bt_features(temp_df_row, b_t_cols)

            # 组合所有特征
            feature_dict = {
                'Temperature': T, 'Frequency': f,
                **bt_features.to_dict()
            }
            for w_map_key, w_map_val in WAVEFORM_MAP.items():
                feature_dict[f'Waveform_Type_{w_map_val}'] = 1.0 if int(round(W_code)) == w_map_key else 0.0
            for m_map_key, m_map_val in MATERIAL_MAP.items():
                feature_dict[f'Material_{m_map_val}'] = 1.0 if int(round(M_code)) == m_map_key else 0.0

            features_df = pd.DataFrame([feature_dict])[MODEL_FEATURES]  # 确保顺序一致

            # --- 目标函数计算 ---
            log_pred = MODEL.predict(features_df)[0]
            core_loss = np.expm1(log_pred)
            neg_energy_transfer = -(f * Bm)

            objectives.append([core_loss, neg_energy_transfer])

        out["F"] = np.array(objectives)


# --- 4. 主优化流程 ---
if __name__ == '__main__':
    problem = HighFidelityOptimization()
    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination("n_gen", 200)

    print("\n开始执行NSGA-II优化 (预计需要较长时间)...")
    start_time = time.time()
    res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=True)
    end_time = time.time()

    print(f"\n--- 优化完成, 耗时: {end_time - start_time:.2f} 秒 ---")
    print(f"找到 {len(res.F)} 个帕累托最优解。")

    # --- 5. 处理并保存结果 ---
    pareto_solutions = res.X
    pareto_front = res.F

    df_results = pd.DataFrame(pareto_solutions,
                              columns=['Temperature', 'Frequency', 'Material_Code', 'Waveform_Code', 'Bm'])
    df_results['Core_Loss'] = pareto_front[:, 0]
    df_results['Energy_Transfer'] = -pareto_front[:, 1]

    df_results['Material'] = df_results['Material_Code'].round().astype(int).map(MATERIAL_MAP)
    df_results['Waveform'] = df_results['Waveform_Code'].round().astype(int).map(WAVEFORM_MAP)

    output_path = 'q5_pareto_results.csv'
    df_results.to_csv(output_path, index=False)

    print(f"帕累托最优解集已保存至: '{output_path}'")
    print("--- 步骤2/3 完成 ---")