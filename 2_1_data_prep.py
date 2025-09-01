import pandas as pd
import numpy as np
from tqdm import tqdm


def prepare_data_for_q2(input_csv_path: str) -> pd.DataFrame:
    """
    加载并筛选数据，为问题二的斯坦麦茨方程拟合做准备。

    Args:
        input_csv_path (str): 整合后的训练数据CSV文件路径。

    Returns:
        pd.DataFrame: 包含材料1、正弦波工况下，建模所需列的数据框。
    """
    print(f"正在从 '{input_csv_path}' 加载数据...")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"错误: 文件 '{input_csv_path}' 未找到。")
        print("请确保已运行问题一的脚本并生成了 'train_data_combined.csv'。")
        return pd.DataFrame()

    # 1. 筛选数据
    print("正在筛选材料1在正弦波激励下的数据...")
    df_filtered = df[(df['Material'] == 1) & (df['Waveform_Type'] == 'sin')].copy()

    if df_filtered.empty:
        print("警告: 未找到符合条件（材料1，正弦波）的数据。")
        return pd.DataFrame()

    print(f"筛选出 {len(df_filtered)} 条符合条件的样本。")

    # 2. 计算峰值磁通密度 (Bm)
    print("正在计算峰值磁通密度 (Bm)...")
    b_field_cols = [col for col in df.columns if col.startswith('B_t_')]

    # 提取B(t)数据
    b_series_data = df_filtered[b_field_cols]

    # 计算 Bm
    b_max = b_series_data.max(axis=1)
    b_min = b_series_data.min(axis=1)
    df_filtered['Bm'] = (b_max - b_min) / 2

    # 3. 整理最终数据
    # 只保留后续建模需要的列
    df_prepared = df_filtered[['Temperature', 'Frequency', 'Core_Loss', 'Bm']].copy()

    print("数据准备完成。")
    return df_prepared


def main():
    """主函数，执行数据准备流程并保存结果"""

    # 定义输入和输出文件路径
    input_file = 'train_data_combined.csv'
    output_file = 'q2_material1_sine_wave_data.csv'

    # 执行数据准备
    prepared_df = prepare_data_for_q2(input_file)

    # 检查是否成功并保存
    if not prepared_df.empty:
        print("\n--- 问题二数据准备概览 ---")
        print(f"总样本数: {len(prepared_df)}")
        print("\n数据预览 (前5行):")
        print(prepared_df.head())
        print("\n数据描述性统计:")
        print(prepared_df.describe())

        # 保存处理好的数据
        prepared_df.to_csv(output_file, index=False)
        print(f"\n已将准备好的数据保存至 '{output_file}'，可用于后续建模。")


if __name__ == '__main__':
    main()