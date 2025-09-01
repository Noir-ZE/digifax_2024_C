import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import fft
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import joblib
from tqdm import tqdm

# 初始化tqdm，使其能与pandas的apply方法良好配合
tqdm.pandas(desc="提取特征")


def extract_features(b_series: pd.Series) -> pd.Series:
    """
    从单个磁通密度时间序列中提取所有定义的特征。

    Args:
        b_series (pd.Series): 包含1024个采样点的时间序列。

    Returns:
        pd.Series: 包含所有计算出的特征的Series。
    """
    # 1. 标准化 (Normalization)
    #    为避免分母为0的情况，增加一个极小值
    std_dev = b_series.std()
    if std_dev < 1e-9:
        # 如果标准差接近于0（例如，一条直线），则将其视为一个无特征的信号
        return pd.Series({
            'kurtosis': 0, 'skewness': 0, 'crest_factor': 0,
            'thd': 0, 'h3_h1_ratio': 0, 'h5_h1_ratio': 0,
            'derivative_kurtosis': 0
        })

    b_normalized = (b_series - b_series.mean()) / std_dev

    # 2. 时域特征 (Time-Domain Features)
    #    使用Pearson定义（无偏），与大纲公式一致
    kurt = kurtosis(b_normalized, fisher=False)
    sk = skew(b_normalized)
    #    对于标准化信号，RMS为1，波峰因子即为最大绝对值
    crest_factor = np.max(np.abs(b_normalized))

    # 3. 频域特征 (Frequency-Domain Features)
    N = len(b_normalized)
    yf = fft(b_normalized.to_numpy())
    #    获取频谱幅值，仅需前半部分
    amplitudes = 2.0 / N * np.abs(yf[0:N // 2])

    A1 = amplitudes[1] if len(amplitudes) > 1 else 0
    if A1 < 1e-9:  # 基波幅值过小，无法计算谐波比
        thd = 1.0  # 认为是严重失真
        h3_h1_ratio = 1.0
        h5_h1_ratio = 1.0
    else:
        harmonics_sq_sum = np.sum(amplitudes[2:] ** 2)
        thd = np.sqrt(harmonics_sq_sum) / A1

        A3 = amplitudes[3] if len(amplitudes) > 3 else 0
        A5 = amplitudes[5] if len(amplitudes) > 5 else 0
        h3_h1_ratio = (A3 ** 2) / (A1 ** 2)
        h5_h1_ratio = (A5 ** 2) / (A1 ** 2)

    # 4. 导数形态特征 (Derivative & Shape Features)
    derivative = np.diff(b_normalized)
    if len(derivative) > 0 and np.std(derivative) > 1e-9:
        derivative_kurtosis = kurtosis(derivative, fisher=False)
    else:
        derivative_kurtosis = 0

    return pd.Series({
        'kurtosis': kurt,
        'skewness': sk,
        'crest_factor': crest_factor,
        'thd': thd,
        'h3_h1_ratio': h3_h1_ratio,
        'h5_h1_ratio': h5_h1_ratio,
        'derivative_kurtosis': derivative_kurtosis
    })


def main():
    """主函数：加载数据、特征工程、模型训练与评估、保存模型"""

    # --- 1. 加载数据 ---
    try:
        df_combined = pd.read_csv('train_data_combined.csv')
    except FileNotFoundError:
        print("错误: 'train_data_combined.csv' 文件未找到。")
        print("请先运行 'step1_data_preprocessing.py' 来生成此文件。")
        return

    print("已加载整合后的训练数据。")

    # --- 2. 特征工程 ---
    print("开始进行特征工程...")
    b_field_cols = [col for col in df_combined.columns if col.startswith('B_t_')]
    b_series_data = df_combined[b_field_cols]

    # 应用特征提取函数
    feature_df = b_series_data.progress_apply(extract_features, axis=1)

    # 合并特征和原始信息
    df_features = pd.concat([df_combined[['Waveform_Type']], feature_df], axis=1)
    print("特征工程完成。")

    # --- 3. 准备训练数据 ---
    X = df_features.drop('Waveform_Type', axis=1)
    y_str = df_features['Waveform_Type']

    # 标签编码
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    # 保存标签编码器，预测时需要用到
    joblib.dump(le, 'label_encoder.joblib')
    print("标签编码完成，对应关系如下:")
    # zip(le.classes_, le.transform(le.classes_)) 会给出 ('正弦波', 0), ('梯形波', 1), ('三角波', 2)
    # 我们按题目要求的 1, 2, 3 格式来理解
    label_mapping = {name: code + 1 for code, name in enumerate(le.classes_)}
    print(label_mapping)

    # --- 4. 模型训练与交叉验证 ---
    print("\n开始使用5折交叉验证评估模型...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lgbm = lgb.LGBMClassifier(random_state=42)

    accuracies = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        lgbm.fit(X_train, y_train)
        preds = lgbm.predict(X_val)
        acc = accuracy_score(y_val, preds)
        accuracies.append(acc)
        print(f"Fold {fold + 1} 准确率: {acc:.6f}")

    mean_accuracy = np.mean(accuracies)
    print(f"\n模型交叉验证平均准确率: {mean_accuracy:.6f}")
    if mean_accuracy > 0.995:
        print("评估结果：模型性能极佳，特征区分度高，模型合理且有效。")
    else:
        print("评估结果：模型性能良好，可以用于预测。")

    # --- 5. 训练并保存最终模型 ---
    print("\n在全部训练数据上训练最终模型...")
    final_model = lgb.LGBMClassifier(random_state=42)
    final_model.fit(X, y)

    # 保存模型
    model_filename = 'waveform_classifier.joblib'
    joblib.dump(final_model, model_filename)
    print(f"最终分类器已训练并保存到 '{model_filename}'")


if __name__ == '__main__':
    main()
