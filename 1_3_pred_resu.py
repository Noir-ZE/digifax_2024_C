import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import fft
import joblib
from tqdm import tqdm
import os

# 初始化tqdm，使其能与pandas的apply方法良好配合
tqdm.pandas(desc="为测试数据提取特征")


def extract_features(b_series: pd.Series) -> pd.Series:
    """
    从单个磁通密度时间序列中提取所有定义的特征。
    (此函数与训练脚本中的函数完全相同，以确保特征提取的一致性)

    Args:
        b_series (pd.Series): 包含1024个采样点的时间序列。

    Returns:
        pd.Series: 包含所有计算出的特征的Series。
    """
    # 1. 标准化 (Normalization)
    std_dev = b_series.std()
    if std_dev < 1e-9:
        return pd.Series({
            'kurtosis': 0, 'skewness': 0, 'crest_factor': 0,
            'thd': 0, 'h3_h1_ratio': 0, 'h5_h1_ratio': 0,
            'derivative_kurtosis': 0
        })

    b_normalized = (b_series - b_series.mean()) / std_dev

    # 2. 时域特征 (Time-Domain Features)
    kurt = kurtosis(b_normalized, fisher=False)
    sk = skew(b_normalized)
    crest_factor = np.max(np.abs(b_normalized))

    # 3. 频域特征 (Frequency-Domain Features)
    N = len(b_normalized)
    yf = fft(b_normalized.to_numpy())
    amplitudes = 2.0 / N * np.abs(yf[0:N // 2])

    A1 = amplitudes[1] if len(amplitudes) > 1 else 0
    if A1 < 1e-9:
        thd, h3_h1_ratio, h5_h1_ratio = 1.0, 1.0, 1.0
    else:
        harmonics_sq_sum = np.sum(amplitudes[2:] ** 2)
        thd = np.sqrt(harmonics_sq_sum) / A1
        A3 = amplitudes[3] if len(amplitudes) > 3 else 0
        A5 = amplitudes[5] if len(amplitudes) > 5 else 0
        h3_h1_ratio = (A3 ** 2) / (A1 ** 2)
        h5_h1_ratio = (A5 ** 2) / (A1 ** 2)

    # 4. 导数形态特征 (Derivative & Shape Features)
    derivative = np.diff(b_normalized)
    derivative_kurtosis = kurtosis(derivative, fisher=False) if len(derivative) > 0 and np.std(derivative) > 1e-9 else 0

    return pd.Series({
        'kurtosis': kurt, 'skewness': sk, 'crest_factor': crest_factor,
        'thd': thd, 'h3_h1_ratio': h3_h1_ratio, 'h5_h1_ratio': h5_h1_ratio,
        'derivative_kurtosis': derivative_kurtosis
    })


def main():
    """主函数：加载模型和数据、预测、生成结果文件和报告"""

    # --- 1. 加载模型和编码器 ---
    try:
        model = joblib.load('waveform_classifier.joblib')
        le = joblib.load('label_encoder.joblib')
    except FileNotFoundError:
        print("错误: 未找到 'waveform_classifier.joblib' 或 'label_encoder.joblib'。")
        print("请先运行 'step2_train_classifier.py' 来训练并保存模型。")
        return
    print("已成功加载分类模型和标签编码器。")

    # --- 2. 加载并准备测试数据 ---
    test_data_path = 'Test_Data_2.xlsx'
    try:
        df_test = pd.read_excel(test_data_path)
    except FileNotFoundError:
        print(f"错误: 测试数据文件 '{test_data_path}' 未找到。")
        return

    # 清理列名以匹配训练时的格式
    b_field_cols_original = df_test.columns[4:]
    b_field_cols_new = [f'B_t_{i}' for i in range(len(b_field_cols_original))]
    rename_dict = dict(zip(b_field_cols_original, b_field_cols_new))
    df_test.rename(columns=rename_dict, inplace=True)

    print(f"已加载测试数据 '{test_data_path}'。")

    # --- 3. 为测试数据提取特征 ---
    b_field_cols = [col for col in df_test.columns if col.startswith('B_t_')]
    X_test = df_test[b_field_cols].progress_apply(extract_features, axis=1)

    # --- 4. 执行预测 ---
    predictions_encoded = model.predict(X_test)
    predictions_str = le.inverse_transform(predictions_encoded)
    print("已完成对测试数据的预测。")

    # --- 5. 生成结果 ---
    # 定义题目要求的波形类别映射
    final_mapping = {'sin': 1, 'tri': 2, 'tra': 3}
    final_predictions = pd.Series(predictions_str).map(final_mapping)

    # a. 生成附件四
    df_results = pd.DataFrame({
        '样本序号': np.arange(1, len(final_predictions) + 1),
        '波形类别': final_predictions
    })

    output_filename = '附件四.xlsx'
    df_results.to_excel(output_filename, index=False)
    print(f"\n结果文件已生成: '{output_filename}'")

    # b. 统计波形数量
    waveform_counts = df_results['波形类别'].map({v: k for k, v in final_mapping.items()}).value_counts()
    print("\n--- 附件二波形数量统计 ---")
    print(waveform_counts)
    print("--------------------------")

    # c. 展示特定样本结果
    specific_samples = [1, 5, 15, 25, 35, 45, 55, 65, 75, 80]
    df_specific = df_results[df_results['样本序号'].isin(specific_samples)].copy()
    df_specific['波形类别名称'] = df_specific['波形类别'].map({v: k for k, v in final_mapping.items()})

    print("\n--- 指定样本的分类结果 ---")
    print(df_specific[['样本序号', '波形类别', '波形类别名称']].to_string(index=False))
    print("--------------------------")


if __name__ == '__main__':
    main()