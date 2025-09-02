import pandas as pd
import numpy as np
import os
import glob
import re


def combine_training_data(data_path: str) -> pd.DataFrame:
    """
    加载、整合并清理所有附件一的训练数据文件。

    Args:
        data_path (str): 存放训练数据xlsx文件的目录路径。

    Returns:
        pd.DataFrame: 一个包含所有训练数据的合并后的DataFrame。
    """
    print("--- 步骤 1/3：开始执行数据整合与预处理 ---")

    # 使用glob查找所有符合命名规则的训练数据文件
    search_pattern = os.path.join(data_path, "Training_Data_[!_]*.xlsx")
    training_files = glob.glob(search_pattern)

    if not training_files:
        raise FileNotFoundError(f"在路径 '{data_path}' 中未找到任何训练数据文件。请确保数据文件存在。")

    print(f"找到以下训练文件: {', '.join(os.path.basename(f) for f in sorted(training_files))}")

    all_data_frames = []
    for file_path in sorted(training_files):
        match = re.search(r'Training_Data_(\d+)\.xlsx', os.path.basename(file_path))
        if not match:
            continue
        material_id = int(match.group(1))

        print(f"正在加载文件: {os.path.basename(file_path)} (材料 {material_id})...")
        df = pd.read_excel(file_path, header=0)
        df['Material'] = material_id
        all_data_frames.append(df)

    combined_df = pd.concat(all_data_frames, ignore_index=True)
    print("所有数据文件已成功合并。")

    # 清理和重命名列名
    b_field_cols_original = [col for col in combined_df.columns if isinstance(col, int) or col.isdigit()]
    b_field_cols_new = [f'B_t_{i}' for i in range(len(b_field_cols_original))]

    rename_dict = {
        '温度，oC': 'Temperature',
        '频率，Hz': 'Frequency',
        '磁芯损耗，w/m3': 'Core_Loss',
        '励磁波形': 'Waveform_Type',
    }
    rename_dict.update(dict(zip(b_field_cols_original, b_field_cols_new)))

    combined_df.rename(columns=rename_dict, inplace=True)

    # 重新排列列的顺序
    final_cols = ['Material', 'Temperature', 'Frequency', 'Core_Loss', 'Waveform_Type'] + b_field_cols_new
    combined_df = combined_df[final_cols]
    print("列名已清理并重排。")

    return combined_df


if __name__ == '__main__':
    DATA_DIRECTORY = '.'
    try:
        train_df = combine_training_data(DATA_DIRECTORY)

        output_filename = 'train_data_combined.csv'
        train_df.to_csv(output_filename, index=False)
        print(f"\n整合后的数据已成功保存到 '{output_filename}'")
        print("--- 步骤 1/3 完成 ---")

    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
