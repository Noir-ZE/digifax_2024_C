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
    # 使用glob查找所有符合命名规则的训练数据文件
    # [!_]用于排除像'Training_Data_4_test.xlsx'这样的测试样本文件
    search_pattern = os.path.join(data_path, "Training_Data_[!_]*.xlsx")
    training_files = glob.glob(search_pattern)

    if not training_files:
        print(f"在路径 '{data_path}' 中未找到任何训练数据文件。")
        print("请确保 'Training_Data_1.xlsx', 'Training_Data_2.xlsx' 等文件位于该目录下。")
        return pd.DataFrame()

    print(f"找到以下训练文件: {', '.join(os.path.basename(f) for f in training_files)}")

    all_data_frames = []
    for file_path in sorted(training_files):
        # 从文件名中提取材料编号
        match = re.search(r'Training_Data_(\d+)\.xlsx', os.path.basename(file_path))
        if not match:
            continue
        material_id = int(match.group(1))

        print(f"正在加载文件: {os.path.basename(file_path)} (材料 {material_id})...")
        df = pd.read_excel(file_path, header=0)  # 第一行是表头
        df['Material'] = material_id
        all_data_frames.append(df)

    # 合并所有DataFrame
    combined_df = pd.concat(all_data_frames, ignore_index=True)
    print("所有数据文件已成功合并。")

    # 清理和重命名列名
    # 获取磁通密度列（从第5列开始）
    b_field_cols_original = combined_df.columns[4:-1]  # 排除最后添加的'Material'列
    b_field_cols_new = [f'B_t_{i}' for i in range(len(b_field_cols_original))]

    rename_dict = {
        '温度，oC': 'Temperature',
        '频率，Hz': 'Frequency',
        '磁芯损耗，w/m3': 'Core_Loss',
        '励磁波形': 'Waveform_Type',
    }

    # 将原始的磁通密度列名（如 0, 1, 2...）也加入到重命名词典中
    rename_dict.update(dict(zip(b_field_cols_original, b_field_cols_new)))

    combined_df.rename(columns=rename_dict, inplace=True)

    # 重新排列列的顺序，将'Material'列提前
    cols = ['Material', 'Temperature', 'Frequency', 'Core_Loss', 'Waveform_Type'] + b_field_cols_new
    combined_df = combined_df[cols]

    print("列名已清理并重排。")

    return combined_df


if __name__ == '__main__':
    # 假设数据文件存放在名为 'data' 的子目录中
    # 如果您的文件在其他地方，请修改此路径
    DATA_DIRECTORY = '.'

    # 执行数据整合
    train_df = combine_training_data(DATA_DIRECTORY)

    if not train_df.empty:
        # 显示整合后数据的信息
        print("\n--- 整合后的数据概览 ---")
        print(f"总行数: {train_df.shape[0]}")
        print(f"总列数: {train_df.shape[1]}")
        print("\n前5行数据:")
        print(train_df.head())

        print("\n各材料样本数量:")
        print(train_df['Material'].value_counts().sort_index())

        # 保存整合后的数据到CSV文件，方便后续步骤直接调用
        output_filename = 'train_data_combined.csv'
        train_df.to_csv(output_filename, index=False)
        print(f"\n整合后的数据已保存到 '{output_filename}'")
