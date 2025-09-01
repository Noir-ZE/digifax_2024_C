import pandas as pd
import numpy as np


def prepare_data_for_q3(input_csv_path: str) -> pd.DataFrame:
    """
    加载并预处理数据，为问题三的因素分析做准备。

    Args:
        input_csv_path (str): 整合后的训练数据CSV文件路径。

    Returns:
        pd.DataFrame: 包含对数变换和Bm列，并已设置好数据类型的数据框。
    """
    print(f"正在从 '{input_csv_path}' 加载数据...")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"错误: 文件 '{input_csv_path}' 未找到。")
        print("请确保已运行问题一的脚本并生成了 'train_data_combined.csv'。")
        return pd.DataFrame()

    # 1. 数据清洗：确保 Core_Loss 为正值
    initial_rows = len(df)
    df = df[df['Core_Loss'] > 0].copy()
    cleaned_rows = len(df)
    if initial_rows > cleaned_rows:
        print(f"警告: 已移除 {initial_rows - cleaned_rows} 行 'Core_Loss' 为非正值的数据。")

    # --- 新增步骤：计算峰值磁通密度 (Bm) ---
    print("正在计算峰值磁通密度 (Bm)...")
    # 假设B(t)的列名是从'B_t_0'到'B_t_1023'，或者使用位置来定位
    # 一个更稳健的方法是找到所有可以转换为数值且非我们已知列的列
    known_cols = ['Temperature', 'Frequency', 'Core_Loss', 'Waveform_Type', 'Material', 'Sample_ID']  # 假设有Sample_ID
    b_field_cols = [col for col in df.columns if col not in known_cols and pd.api.types.is_numeric_dtype(df[col])]

    # 如果列名不规范，则按位置选择
    if len(b_field_cols) != 1024:
        print("警告：通过列名未能精确定位1024个B(t)列，将尝试按位置选择。")
        # 假设B(t)列在第4列之后（从0开始计数）
        b_field_cols = df.columns[4:1028]  # 这是一个基于原始数据结构的假设
        # 更安全的假设是，除了我们知道的几个列，其他都是B(t)
        # 在问题一的脚本中，列名应为 'Temperature', 'Frequency', 'Core_Loss', 'Waveform_Type', 'Material'
        # 以及 B_t_0 到 B_t_1023
        b_field_cols = [col for col in df.columns if col.startswith('B_t_')]

    if len(b_field_cols) == 1024:
        b_series_data = df[b_field_cols]
        b_max = b_series_data.max(axis=1)
        b_min = b_series_data.min(axis=1)
        df['Bm'] = (b_max - b_min) / 2
        print("'Bm' 列计算完成。")
    else:
        print(f"错误：无法找到1024个B(t)数据列来计算'Bm'。找到了 {len(b_field_cols)} 列。")
        return pd.DataFrame()
    # ----------------------------------------

    # 2. 对数变换
    print("正在对 'Core_Loss' 列进行对数变换，生成 'log_Core_Loss'...")
    df['log_Core_Loss'] = np.log(df['Core_Loss'])

    # 3. 数据类型转换
    print("正在转换因子列的数据类型为 'category'...")
    try:
        df['Temperature'] = df['Temperature'].astype('category')
        df['Waveform_Type'] = df['Waveform_Type'].astype('category')
        df['Material'] = df['Material'].astype('category')
    except KeyError as e:
        print(f"错误: 数据中缺少关键列: {e}")
        return pd.DataFrame()

    print("数据准备完成。")
    return df


def main():
    """主函数，执行数据准备流程并保存结果"""

    input_file = 'train_data_combined.csv'
    output_file = 'q3_analysis_data.csv'

    prepared_df = prepare_data_for_q3(input_file)

    if not prepared_df.empty:
        print("\n--- 问题三数据准备概览 ---")
        print(f"总样本数: {len(prepared_df)}")
        print("\n数据信息:")
        # 只显示关键列的信息，避免刷屏
        prepared_df[['Temperature', 'Waveform_Type', 'Material', 'Core_Loss', 'Bm', 'log_Core_Loss']].info()

        print("\n数据预览 (前5行):")
        print(prepared_df[['Temperature', 'Waveform_Type', 'Material', 'Core_Loss', 'Bm', 'log_Core_Loss']].head())

        # 保存处理好的数据
        columns_to_save = ['Temperature', 'Frequency', 'Core_Loss', 'Waveform_Type', 'Material', 'Bm', 'log_Core_Loss']
        prepared_df[columns_to_save].to_csv(output_file, index=False)
        print(f"\n已将准备好的数据保存至 '{output_file}'，可用于后续因素分析。")


if __name__ == '__main__':
    main()