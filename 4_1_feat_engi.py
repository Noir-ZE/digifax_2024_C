import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.fft import rfft, rfftfreq
import warnings
from tqdm import tqdm

# --- Setup ---
warnings.filterwarnings('ignore', category=FutureWarning)
tqdm.pandas(desc="Processing rows")


# --- Feature Calculation Functions ---

def calculate_thd(row, b_t_cols):
    """计算单个样本的总谐波失真 (THD)"""
    signal = row[b_t_cols].values
    frequency = row['Frequency']

    N = len(signal)
    if N == 0 or pd.isna(frequency): return np.nan

    # 执行快速傅里叶变换
    yf = rfft(signal)
    amplitudes = np.abs(yf)

    if len(amplitudes) < 2: return np.nan
    A1 = amplitudes[1]  # 基波 (Fundamental) 的幅值

    if A1 == 0: return np.nan

    harmonics_amplitudes = amplitudes[2:]  # 谐波 (Harmonics)
    thd = np.sqrt(np.sum(harmonics_amplitudes ** 2)) / A1
    return thd


# --- Main Feature Engineering Pipeline ---

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    对输入的数据框进行特征工程，提取用于预测磁芯损耗的关键特征。
    """
    print("开始特征工程...")

    # 1. 识别B(t)列和特征列
    # 确保所有列名都是字符串，以处理来自Excel的整数列名
    df.columns = df.columns.map(str)

    # 定义已知的元数据/特征列
    known_feature_cols = ['Sample_ID', 'Temperature', 'Frequency', 'Waveform_Type', 'Material', 'Core_Loss']
    # B(t)列是除了已知特征列之外的所有列
    b_t_cols = [col for col in df.columns if col not in known_feature_cols]

    print(f"识别出 {len(b_t_cols)} 个B(t)数据列。")
    if len(b_t_cols) != 1024:
        print(f"警告: B(t)列的数量 ({len(b_t_cols)}) 不是预期的1024。请检查数据格式。")

    # 2. 计算 Bm (峰值磁通密度)
    print("计算 Bm...")
    b_series = df[b_t_cols].astype(float)
    df['Bm'] = (b_series.max(axis=1) - b_series.min(axis=1)) / 2

    # 3. 计算 max(dB/dt)
    print("计算 max(dB/dt)...")
    N = len(b_t_cols)
    delta_t = 1 / (df['Frequency'] * N)
    # diff()计算差分，iloc[:, 1:]去除第一个NaN值，然后除以时间间隔
    derivatives = b_series.diff(axis=1).iloc[:, 1:].div(delta_t, axis=0)
    df['max_dB_dt'] = derivatives.max(axis=1)

    # 4. 计算 Kurtosis (峭度)
    print("计算 Kurtosis...")
    mean = b_series.mean(axis=1)
    std = b_series.std(axis=1)
    std[std == 0] = 1  # 避免除以零
    z_scores = b_series.sub(mean, axis=0).div(std, axis=0)
    df['Kurtosis'] = kurtosis(z_scores, axis=1, fisher=False)  # fisher=False 得到原始四阶矩/二阶矩平方

    # 5. 计算 THD (总谐波失真)
    print("计算 THD (此步骤可能较慢)...")
    df['THD'] = df.progress_apply(lambda row: calculate_thd(row, b_t_cols), axis=1)

    print("特征工程完成。")
    return df


# --- Main Execution Block ---

def main():
    """主函数，执行完整的特征工程流程并保存结果。"""

    train_file = 'train_data_combined.csv'
    test_file = 'Test_Data_3.xlsx'  # 使用您提供的文件名

    # --- 1. 加载并预处理数据 ---
    print("--- 加载数据 ---")
    try:
        df_train_raw = pd.read_csv(train_file)
        # 直接读取Excel文件，第一行为表头
        df_test_raw = pd.read_excel(test_file, header=0)
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 -> {e}")
        print(f"请确保 '{train_file}' 和 '{test_file}' 文件在当前目录下。")
        return

    # --- 2. 标准化测试集的列名 ---
    # 根据您提供的准确信息重命名测试集列
    df_test_raw.rename(columns={
        '序号': 'Sample_ID',
        '温度，oC': 'Temperature',
        '频率，Hz': 'Frequency',
        '磁芯材料': 'Material',
        '励磁波形': 'Waveform_Type'
    }, inplace=True)

    # --- 3. 应用特征工程 ---
    print("\n--- 处理训练集 ---")
    df_train_featured = feature_engineering(df_train_raw.copy())

    print("\n--- 处理测试集 ---")
    df_test_featured = feature_engineering(df_test_raw.copy())

    # --- 4. One-Hot 编码 ---
    print("\n--- 执行One-Hot编码 ---")

    categorical_features = ['Waveform_Type', 'Material']
    numerical_features = ['Frequency', 'Temperature', 'Bm', 'max_dB_dt', 'Kurtosis', 'THD']

    y_train = df_train_featured['Core_Loss']

    # 合并以确保编码列一致
    combined_df = pd.concat([
        df_train_featured[numerical_features + categorical_features],
        df_test_featured[numerical_features + categorical_features]
    ], ignore_index=True)

    combined_df_encoded = pd.get_dummies(combined_df, columns=categorical_features, prefix=categorical_features,
                                         dtype=float)

    # 分离回训练集和测试集
    X_train_final = combined_df_encoded.iloc[:len(df_train_raw)]
    X_test_final = combined_df_encoded.iloc[len(df_train_raw):]

    # 组合最终数据框
    train_final_df = pd.concat([X_train_final.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test_final_df = pd.concat(
        [df_test_featured[['Sample_ID']].reset_index(drop=True), X_test_final.reset_index(drop=True)], axis=1)

    # --- 5. 保存结果 ---
    print("\n--- 保存处理后的数据 ---")
    train_output_file = 'q4_train_featured.csv'
    test_output_file = 'q4_test_featured.csv'

    train_final_df.to_csv(train_output_file, index=False)
    test_final_df.to_csv(test_output_file, index=False)

    print(f"特征化后的训练数据已保存至: '{train_output_file}'")
    print(f"特征化后的测试数据已保存至: '{test_output_file}'")

    print("\n训练数据预览 (前5行):")
    print(train_final_df.head())
    print("\n测试数据预览 (前5行):")
    print(test_final_df.head())


if __name__ == '__main__':
    main()