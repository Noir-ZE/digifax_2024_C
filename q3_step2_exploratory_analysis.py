import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.factorplots import interaction_plot


def analyze_main_effects(df: pd.DataFrame):
    """步骤2a：独立效应的探索性分析"""
    print("\n--- 步骤 2a：生成独立效应探索图 (小提琴图) ---")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    fig.suptitle('独立效应探索性分析: 各因素对对数磁芯损耗的初步影响', fontsize=18)

    # 修复Seaborn的FutureWarning
    sns.violinplot(x='Temperature', y='log_Core_Loss', data=df, ax=axes[0], hue='Temperature', palette='coolwarm',
                   legend=False)
    axes[0].set_title('温度 (Temperature) 的影响', fontsize=14)
    axes[0].set_xlabel('温度 (°C)', fontsize=12)
    axes[0].set_ylabel('对数磁芯损耗 (log(W/m3))', fontsize=12)

    sns.violinplot(x='Waveform_Type', y='log_Core_Loss', data=df, ax=axes[1], hue='Waveform_Type', palette='viridis',
                   legend=False)
    axes[1].set_title('励磁波形 (Waveform_Type) 的影响', fontsize=14)
    axes[1].set_xlabel('波形类型', fontsize=12)

    sns.violinplot(x='Material', y='log_Core_Loss', data=df, ax=axes[2], hue='Material', palette='plasma', legend=False)
    axes[2].set_title('磁芯材料 (Material) 的影响', fontsize=14)
    axes[2].set_xlabel('材料编号', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('q3_exploratory_main_effects.png', dpi=300)
    plt.show()


def analyze_interaction_effects(df: pd.DataFrame):
    """步骤2b：协同效应的探索性分析"""
    print("\n--- 步骤 2b：生成协同效应探索图 (交互图) ---")

    # --- 核心修正：为满足interaction_plot的API要求，将因子列强制转换为字符串 ---
    df_plot = df.copy()
    df_plot['Temperature'] = df_plot['Temperature'].astype(str)
    df_plot['Waveform_Type'] = df_plot['Waveform_Type'].astype(str)
    df_plot['Material'] = df_plot['Material'].astype(str)
    # ---------------------------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('协同效应探索性分析: 因素间的交互影响', fontsize=18)

    interaction_plot(df_plot['Temperature'], df_plot['Waveform_Type'], df_plot['log_Core_Loss'], ax=axes[0],
                     markers=['o', 's', '^'])
    axes[0].set_title('温度与波形的交互作用', fontsize=14)
    axes[0].set_xlabel('温度 (°C)', fontsize=12)
    axes[0].set_ylabel('对数磁芯损耗均值', fontsize=12)

    interaction_plot(df_plot['Temperature'], df_plot['Material'], df_plot['log_Core_Loss'], ax=axes[1],
                     markers=['o', 's', '^', 'D'])
    axes[1].set_title('温度与材料的交互作用', fontsize=14)
    axes[1].set_xlabel('温度 (°C)', fontsize=12)

    interaction_plot(df_plot['Waveform_Type'], df_plot['Material'], df_plot['log_Core_Loss'], ax=axes[2],
                     markers=['o', 's', '^', 'D'])
    axes[2].set_title('波形与材料的交互作用', fontsize=14)
    axes[2].set_xlabel('波形类型', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('q3_exploratory_interaction_effects.png', dpi=300)
    plt.show()


def main():
    print("\n--- 步骤 2/3：开始执行探索性数据分析 ---")
    input_file = 'q3_analysis_data_prepared.csv'
    try:
        df = pd.read_csv(input_file)
        # 从文件中读取时，category类型会丢失，需要重新设置，这对于后续的ANCOVA很重要
        df['Temperature'] = df['Temperature'].astype('category')
        df['Waveform_Type'] = pd.Categorical(df['Waveform_Type'], categories=['sin', 'tri', 'tra'], ordered=True)
        df['Material'] = df['Material'].astype('category')
    except FileNotFoundError:
        raise FileNotFoundError(f"错误: 文件 '{input_file}' 未找到。请先运行步骤1的脚本。")

    analyze_main_effects(df)
    analyze_interaction_effects(df)

    print("\n探索性分析完成。这些图表为后续的ANCOVA分析提供了初步假设。")
    print("--- 步骤 2/3 完成 ---")


if __name__ == '__main__':
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 中文字体'SimHei'未找到，图形显示可能不正常。")
    main()