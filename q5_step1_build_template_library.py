import pandas as pd
import numpy as np
import json
import os


def get_bt_cols(df):
    """从DataFrame中识别出B(t)数据列"""
    df.columns = df.columns.map(str)
    known_feature_cols = ['Sample_ID', 'Temperature', 'Frequency', 'Waveform_Type', 'Material', 'Core_Loss']
    return [col for col in df.columns if col not in known_feature_cols]


def build_waveform_template_library(train_data_path: str, output_path: str):
    """
    构建并保存一个基于真实实验数据的波形模板库。
    """
    print("--- 步骤1/3：构建真实波形模板库 ---")

    # --- 1. 加载数据并计算Bm ---
    try:
        df = pd.read_csv(train_data_path)
    except FileNotFoundError:
        print(f"错误: 训练数据 '{train_data_path}' 未找到。")
        return

    bt_cols = get_bt_cols(df)
    b_series = df[bt_cols].astype(float)
    df['Bm'] = (b_series.max(axis=1) - b_series.min(axis=1)) / 2
    print("数据加载完成，Bm已计算。")

    # --- 2. 筛选代表性波形 ---
    template_library = {}
    grouped = df.groupby(['Material', 'Waveform_Type'])

    for (material, waveform), group in grouped:
        if group.empty:
            continue

        # 找到Bm最接近中位数的样本作为模板
        median_bm = group['Bm'].median()
        representative_sample = group.iloc[(group['Bm'] - median_bm).abs().argsort()[:1]]

        if representative_sample.empty:
            continue

        # 提取模板信息
        template_bt_series = representative_sample[bt_cols].iloc[0].values.tolist()
        template_bm = representative_sample['Bm'].iloc[0]

        # 存入库中
        key = f"M{material}_W{waveform}"
        template_library[key] = {
            'template_bt': template_bt_series,
            'template_bm': template_bm
        }
        print(f"为 '{key}' 找到代表性波形模板 (Bm={template_bm:.4f})")

    # --- 3. 保存模板库 ---
    with open(output_path, 'w') as f:
        json.dump(template_library, f, indent=4)

    print(f"\n波形模板库构建完成，共 {len(template_library)} 个模板。")
    print(f"已保存至: '{output_path}'")
    print("--- 步骤1/3 完成 ---")


if __name__ == '__main__':
    # 使用问题一中合并好的全量训练数据
    TRAIN_DATA_FILE = 'train_data_combined.csv'
    OUTPUT_LIBRARY_FILE = 'waveform_template_library.json'

    build_waveform_template_library(TRAIN_DATA_FILE, OUTPUT_LIBRARY_FILE)