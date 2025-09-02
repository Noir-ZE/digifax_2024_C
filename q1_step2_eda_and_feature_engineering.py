import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import fft
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# --- 全局绘图设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号


# --- 特征提取函数 (与您原版一致) ---
def extract_features(b_series: pd.Series) -> pd.Series:
    std_dev = b_series.std()
    if std_dev < 1e-9:
        return pd.Series({'kurtosis': 0, 'skewness': 0, 'crest_factor': 0, 'thd': 0, 'h3_h1_ratio': 0, 'h5_h1_ratio': 0,
                          'derivative_kurtosis': 0})
    b_normalized = (b_series - b_series.mean()) / std_dev
    kurt = kurtosis(b_normalized, fisher=False)
    sk = skew(b_normalized)
    crest_factor = np.max(np.abs(b_normalized))
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
    derivative = np.diff(b_normalized)
    derivative_kurtosis = kurtosis(derivative, fisher=False) if len(derivative) > 0 and np.std(derivative) > 1e-9 else 0
    return pd.Series(
        {'kurtosis': kurt, 'skewness': sk, 'crest_factor': crest_factor, 'thd': thd, 'h3_h1_ratio': h3_h1_ratio,
         'h5_h1_ratio': h5_h1_ratio, 'derivative_kurtosis': derivative_kurtosis})


# --- 可视化函数 (已按要求优化) ---
def perform_eda_visualizations(df_full, b_t_cols):
    """
    接收一个包含所有列的完整DataFrame来进行可视化。
    """
    print("--- 2a. 执行探索性数据分析 (EDA) ---")

    # 1. 绘制典型波形图 (优化为并排三图)
    waveform_types = sorted(df_full['Waveform_Type'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('典型励磁波形对比 (标准化后)', fontsize=18, y=1.02)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(waveform_types)))

    for ax, wf_type, color in zip(axes, waveform_types, colors):
        sample = df_full[df_full['Waveform_Type'] == wf_type].iloc[0]
        b_series = sample[b_t_cols].values
        b_normalized = (b_series - b_series.mean()) / b_series.std()

        ax.plot(b_normalized, color=color, linewidth=2)
        ax.set_title(f'波形: {wf_type}', fontsize=14)
        ax.set_xlabel('采样点')
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[0].set_ylabel('标准化磁通密度 B(t)')

    plt.tight_layout()
    plt.savefig('q1_typical_waveforms.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("典型波形对比图已保存至 'q1_typical_waveforms.png'")

    # 2. 绘制核心特征分布图
    key_features = ['kurtosis', 'thd', 'derivative_kurtosis']
    feature_names_cn = ['峭度 (Kurtosis)', '总谐波失真 (THD)', '导数的峭度']

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('核心特征在不同波形下的分布对比', fontsize=18)

    for i, (feat, name_cn) in enumerate(zip(key_features, feature_names_cn)):
        sns.violinplot(ax=axes[i], x='Waveform_Type', y=feat, data=df_full, palette='viridis', order=waveform_types)
        axes[i].set_title(f'{name_cn} 分布', fontsize=14)
        axes[i].set_xlabel('波形类型', fontsize=12)
        axes[i].set_ylabel(name_cn, fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('q1_feature_distributions.png', dpi=300)
    plt.show()
    print("核心特征分布对比图已保存至 'q1_feature_distributions.png'")


def main():
    print("\n--- 步骤 2/3：开始执行EDA与特征工程 ---")

    try:
        df_combined = pd.read_csv('train_data_combined.csv')
    except FileNotFoundError:
        print("错误: 'train_data_combined.csv' 未找到。请先运行步骤1的脚本。")
        return

    b_t_cols = [col for col in df_combined.columns if col.startswith('B_t_')]

    # --- 特征工程 ---
    print("\n--- 2b. 构建物理驱动的特征工程体系 ---")
    tqdm.pandas(desc="提取特征")
    feature_df = df_combined[b_t_cols].progress_apply(extract_features, axis=1)
    print("特征工程完成。")

    # --- EDA 可视化 ---
    df_for_eda = pd.concat([df_combined, feature_df], axis=1)
    perform_eda_visualizations(df_for_eda, b_t_cols)

    # --- 保存特征化后的数据 ---
    df_to_save = pd.concat([df_combined[['Waveform_Type']], feature_df], axis=1)
    output_filename = 'q1_train_featured.csv'
    df_to_save.to_csv(output_filename, index=False)
    print(f"\n特征化后的训练数据已保存至 '{output_filename}'，用于下一步建模。")
    print("--- 步骤 2/3 完成 ---")


if __name__ == '__main__':
    main()