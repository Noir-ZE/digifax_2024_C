import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import rfft
import warnings
from tqdm import tqdm

# --- 全局设置 ---
warnings.filterwarnings('ignore', category=FutureWarning)
tqdm.pandas(desc="特征计算中")


# --- 特征计算函数 (封装以提高复用性) ---

def get_bt_cols(df):
    """从DataFrame中识别出B(t)数据列"""
    df.columns = df.columns.map(str)
    known_feature_cols = ['Sample_ID', 'Temperature', 'Frequency', 'Waveform_Type', 'Material', 'Core_Loss']
    return [col for col in df.columns if col not in known_feature_cols]


def calculate_bt_features(row, b_t_cols):
    """计算单个样本的所有B(t)派生特征"""
    signal = row[b_t_cols].values.astype(float)
    frequency = row['Frequency']
    N = len(signal)

    if N == 0:
        return pd.Series([np.nan] * 6, index=['Bm', 'max_dB_dt', 'Kurtosis', 'Skewness', 'Crest_Factor', 'THD'])

    # 物理派生特征
    b_max, b_min = np.max(signal), np.min(signal)
    bm = (b_max - b_min) / 2

    delta_t = 1 / (frequency * N) if frequency > 0 else np.inf
    derivatives = np.diff(signal) / delta_t if delta_t != np.inf else np.zeros(N - 1)
    max_db_dt = np.max(np.abs(derivatives))

    # 形态学派生特征
    kurt = kurtosis(signal, fisher=False)
    skewness = skew(signal)

    rms = np.sqrt(np.mean(signal ** 2))
    crest_factor = b_max / rms if rms > 0 else np.nan

    # THD 计算
    thd = np.nan
    if N > 1 and frequency > 0:
        yf = rfft(signal)
        amplitudes = np.abs(yf)
        if len(amplitudes) > 1 and amplitudes[1] > 0:
            harmonics_sq_sum = np.sum(amplitudes[2:] ** 2)
            thd = np.sqrt(harmonics_sq_sum) / amplitudes[1]

    return pd.Series([bm, max_db_dt, kurt, skewness, crest_factor, thd],
                     index=['Bm', 'max_dB_dt', 'Kurtosis', 'Skewness', 'Crest_Factor', 'THD'])


# --- 主特征工程流程 ---

def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """对输入的数据框执行完整的物理信息增强特征工程流程"""
    print("开始执行物理信息增强的特征工程...")

    b_t_cols = get_bt_cols(df)
    print(f"识别出 {len(b_t_cols)} 个B(t)数据列。")

    # 并行计算所有B(t)派生特征
    bt_features = df.progress_apply(lambda row: calculate_bt_features(row, b_t_cols), axis=1)

    # 合并基础工况特征和派生特征
    features_df = pd.concat([df[['Temperature', 'Frequency', 'Waveform_Type', 'Material']], bt_features], axis=1)

    print("特征工程完成。")
    return features_df


# --- 主执行模块 ---

def main():
    """主函数，执行特征工程并保存结果"""
    print("--- 步骤1/3：特征工程 ---")

    # --- 加载数据 ---
    try:
        df_train_raw = pd.read_csv('train_data_combined.csv')
        df_test_raw = pd.read_excel('Test_Data_3.xlsx', header=0)
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 -> {e}")
        return

    # --- 标准化测试集列名 ---
    df_test_raw.rename(columns={
        '序号': 'Sample_ID', '温度，oC': 'Temperature', '频率，Hz': 'Frequency',
        '磁芯材料': 'Material', '励磁波形': 'Waveform_Type'
    }, inplace=True)

    # --- 应用特征工程 ---
    print("\n处理训练集...")
    train_features = feature_engineering_pipeline(df_train_raw)
    y_train = np.log1p(df_train_raw['Core_Loss'])  # 对目标进行log1p变换

    print("\n处理测试集...")
    test_features = feature_engineering_pipeline(df_test_raw)

    # --- One-Hot 编码 ---
    print("\n执行One-Hot编码...")
    combined_features = pd.concat([train_features, test_features], ignore_index=True)
    combined_encoded = pd.get_dummies(combined_features, columns=['Waveform_Type', 'Material'], dtype=float)

    # --- 分离并保存最终数据集 ---
    X_train_final = combined_encoded.iloc[:len(df_train_raw)]
    X_test_final = combined_encoded.iloc[len(df_train_raw):]

    train_final_df = pd.concat([X_train_final, y_train.rename('log1p_Core_Loss')], axis=1)
    test_final_df = pd.concat([df_test_raw[['Sample_ID']].reset_index(drop=True), X_test_final.reset_index(drop=True)],
                              axis=1)

    train_final_df.to_csv('q4_train_featured.csv', index=False)
    test_final_df.to_csv('q4_test_featured.csv', index=False)

    print("\n特征化后的训练/测试数据已保存。")
    print(f"训练集特征维度: {X_train_final.shape}")
    print(f"测试集特征维度: {X_test_final.shape}")
    print("--- 步骤1/3 完成 ---")


if __name__ == '__main__':
    main()