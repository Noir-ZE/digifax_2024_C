import pandas as pd
import numpy as np


def prepare_data_for_q2(input_csv_path: str) -> pd.DataFrame:
    """
    加载并筛选数据，为问题二的斯坦麦茨方程拟合做准备。

    Args:
        input_csv_path (str): 整合后的训练数据CSV文件路径。

    Returns:
        pd.DataFrame: 包含材料1、正弦波工况下，建模所需列的数据框。
    """
    print("--- 步骤 1/2：开始执行数据准备 ---")
    print(f"正在从 '{input_csv_path}' 加载数据...")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"错误: 文件 '{input_csv_path}' 未找到。请先运行问题一的脚本。")

    # 1. 筛选数据
    print("正在筛选材料1在正弦波激励下的数据...")
    df_filtered = df[(df['Material'] == 1) & (df['Waveform_Type'] == 'sin')].copy()

    if df_filtered.empty:
        raise ValueError("错误: 未在数据中找到符合条件（材料1，正弦波）的样本。")

    print(f"筛选出 {len(df_filtered)} 条符合条件的样本。")

    # 2. 计算峰值磁通密度 (Bm)
    print("正在计算峰值磁通密度 (Bm)...")
    b_field_cols = [col for col in df.columns if col.startswith('B_t_')]
    b_series_data = df_filtered[b_field_cols]

    # Bm的定义是峰峰值的一半
    df_filtered['Bm'] = (b_series_data.max(axis=1) - b_series_data.min(axis=1)) / 2

    # 3. 整理最终数据
    df_prepared = df_filtered[['Temperature', 'Frequency', 'Core_Loss', 'Bm']].copy()

    print("数据准备完成。")
    return df_prepared


def main():
    """主函数，执行数据准备流程并保存结果"""
    input_file = 'train_data_combined.csv'
    output_file = 'q2_material1_sine_wave_data.csv'

    try:
        prepared_df = prepare_data_for_q2(input_file)

        print("\n--- 问题二数据准备概览 ---")
        print(f"总样本数: {len(prepared_df)}")
        print("\n数据预览 (前5行):")
        print(prepared_df.head())

        prepared_df.to_csv(output_file, index=False)
        print(f"\n已将准备好的数据保存至 '{output_file}'，可用于后续建模。")
        print("--- 步骤 1/2 完成 ---")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n处理失败: {e}")


if __name__ == '__main__':
    main()