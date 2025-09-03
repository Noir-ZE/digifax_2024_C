import pandas as pd
import numpy as np


def prepare_data_for_q3(input_csv_path: str) -> pd.DataFrame:
    """
    加载并预处理数据，为问题三的ANCOVA因素分析做准备。
    核心任务：计算Bm，对因变量和协变量进行对数变换，并转换数据类型。
    """
    print("--- 步骤 1/3：开始执行数据准备 ---")
    print(f"正在从 '{input_csv_path}' 加载数据...")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"错误: 文件 '{input_csv_path}' 未找到。请先运行问题一的脚本。")

    # 1. 数据清洗
    initial_rows = len(df)
    df = df[(df['Core_Loss'] > 0) & (df['Frequency'] > 0)].copy()
    if len(df) < initial_rows:
        print(f"警告: 已移除 {initial_rows - len(df)} 行 'Core_Loss' 或 'Frequency' 为非正值的数据。")

    # 2. 计算并清洗 Bm
    print("正在计算峰值磁通密度 (Bm)...")
    b_field_cols = [col for col in df.columns if col.startswith('B_t_')]
    if len(b_field_cols) == 1024:
        b_series_data = df[b_field_cols]
        df['Bm'] = (b_series_data.max(axis=1) - b_series_data.min(axis=1)) / 2
        df = df[df['Bm'] > 0].copy()
        print("'Bm' 列计算并清洗完成。")
    else:
        raise ValueError(f"错误：无法找到1024个B(t)数据列来计算'Bm'。实际找到 {len(b_field_cols)} 列。")

    # 3. 对数变换
    print("正在对因变量(Core_Loss)和协变量(Frequency, Bm)进行对数变换...")
    df['log_Core_Loss'] = np.log(df['Core_Loss'])
    df['log_Frequency'] = np.log(df['Frequency'])
    df['log_Bm'] = np.log(df['Bm'])

    # 4. 数据类型转换 (核心修正点)
    print("正在转换因子列的数据类型为 'category'，并指定顺序...")
    df['Temperature'] = df['Temperature'].astype('category')

    # --- 核心修正：显式定义波形顺序 ---
    waveform_order = ['sin', 'tri', 'tra']
    df['Waveform_Type'] = pd.Categorical(df['Waveform_Type'], categories=waveform_order, ordered=True)
    # ---------------------------------

    df['Material'] = df['Material'].astype('category')

    print("数据准备完成。")
    return df


def main():
    """主函数，执行数据准备流程并保存结果"""
    input_file = 'train_data_combined.csv'
    output_file = 'q3_analysis_data_prepared.csv'

    try:
        prepared_df = prepare_data_for_q3(input_file)

        print("\n--- 问题三数据准备概览 ---")
        print(f"总样本数: {len(prepared_df)}")

        key_cols = ['Temperature', 'Waveform_Type', 'Material', 'log_Core_Loss', 'log_Frequency', 'log_Bm']
        print("\n关键列数据信息:")
        prepared_df[key_cols].info()

        prepared_df.to_csv(output_file, index=False)
        print(f"\n已将准备好的数据保存至 '{output_file}'，可用于后续因素分析。")
        print("--- 步骤 1/3 完成 ---")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n处理失败: {e}")


if __name__ == '__main__':
    main()