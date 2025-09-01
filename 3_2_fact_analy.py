import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from statsmodels.graphics.factorplots import interaction_plot


def analyze_main_effects(df: pd.DataFrame):
    """
    步骤一：独立效应分析 - 绘制小提琴图
    """
    print("\n--- 步骤一：正在生成独立效应分析图 (小提琴图) ---")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle('独立效应分析: 各因素对对数磁芯损耗的影响', fontsize=18, y=0.98)

    # 1. 温度的影响
    sns.violinplot(x='Temperature', y='log_Core_Loss', data=df, ax=axes[0], palette='coolwarm')
    axes[0].set_title('温度 (Temperature) 的影响', fontsize=14)
    axes[0].set_xlabel('温度 (°C)', fontsize=12)
    axes[0].set_ylabel('对数磁芯损耗 (log(W/m3))', fontsize=12)

    # 2. 波形的影响
    sns.violinplot(x='Waveform_Type', y='log_Core_Loss', data=df, ax=axes[1], palette='viridis')
    axes[1].set_title('励磁波形 (Waveform_Type) 的影响', fontsize=14)
    axes[1].set_xlabel('波形类型', fontsize=12)
    axes[1].set_ylabel('')  # 共享Y轴

    # 3. 材料的影响
    sns.violinplot(x='Material', y='log_Core_Loss', data=df, ax=axes[2], palette='plasma')
    axes[2].set_title('磁芯材料 (Material) 的影响', fontsize=14)
    axes[2].set_xlabel('材料编号', fontsize=12)
    axes[2].set_ylabel('')  # 共享Y轴

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('q3_main_effects_violin_analysis.png')
    plt.show()
    print("独立效应分析(小提琴图)已保存至 'q3_main_effects_violin_analysis.png'")


def analyze_interaction_effects(df: pd.DataFrame):
    """
    步骤二：协同效应分析 - 绘制交互效应图
    """
    print("\n--- 步骤二：正在生成协同效应分析图 (交互图) ---")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('协同效应分析: 因素间的交互影响', fontsize=18, y=0.98)

    # 1. 温度 & 波形
    interaction_plot(df['Temperature'], df['Waveform_Type'], df['log_Core_Loss'], ax=axes[0], markers=['o', 's', '^'])
    axes[0].set_title('温度与波形的交互作用', fontsize=14)
    axes[0].set_xlabel('温度 (°C)', fontsize=12)
    axes[0].set_ylabel('对数磁芯损耗均值', fontsize=12)

    # 2. 温度 & 材料
    interaction_plot(df['Temperature'], df['Material'], df['log_Core_Loss'], ax=axes[1], markers=['o', 's', '^', 'D'])
    axes[1].set_title('温度与材料的交互作用', fontsize=14)
    axes[1].set_xlabel('温度 (°C)', fontsize=12)
    axes[1].set_ylabel('')

    # 3. 波形 & 材料
    interaction_plot(df['Waveform_Type'], df['Material'], df['log_Core_Loss'], ax=axes[2], markers=['o', 's', '^', 'D'])
    axes[2].set_title('波形与材料的交互作用', fontsize=14)
    axes[2].set_xlabel('波形类型', fontsize=12)
    axes[2].set_ylabel('')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('q3_interaction_effects_analysis.png')
    plt.show()
    print("协同效应分析图已保存至 'q3_interaction_effects_analysis.png'")


def perform_anova_and_quantify(df: pd.DataFrame):
    """
    步骤三：影响程度量化 - 执行三因素ANOVA分析并可视化结果
    """
    print("\n--- 步骤三：正在执行三因素ANOVA并量化影响程度 ---")

    anova_results = pg.anova(
        data=df,
        dv='log_Core_Loss',
        between=['Temperature', 'Waveform_Type', 'Material'],
        detailed=True
    )

    print("三因素方差分析 (Three-Way ANOVA) 结果:")
    # 仅显示我们关心的效应
    effects_to_show = anova_results.iloc[:-1]  # 排除最后的'Residual'行
    print(effects_to_show.to_string())

    # --- 新增：ANOVA结果可视化 ---
    print("\n正在生成ANOVA影响程度可视化图 (条形图)...")

    # 准备绘图数据，并按np2排序
    plot_data = effects_to_show[['Source', 'np2']].sort_values(by='np2', ascending=False)

    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(x='np2', y='Source', data=plot_data, palette='mako')

    # 在条形上添加数值标签
    for index, row in plot_data.iterrows():
        barplot.text(row.np2 + 0.005, index, f"{row.np2:.3f}", color='black', ha="left", va='center')

    plt.title('各因素对磁芯损耗影响程度量化 (ANOVA 偏Eta平方)', fontsize=16)
    plt.xlabel('影响程度 (Partial Eta-Squared, np2)', fontsize=12)
    plt.ylabel('效应来源 (Source)', fontsize=12)
    plt.xlim(0, max(plot_data['np2']) * 1.1)  # 调整X轴范围以容纳标签
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('q3_anova_effect_sizes.png')
    plt.show()
    print("ANOVA影响程度可视化图已保存至 'q3_anova_effect_sizes.png'")


def main():
    input_file = 'q3_analysis_data.csv'
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误: 文件 '{input_file}' 未找到。")
        print("请先运行 'q3_step1_data_preparation.py' 来生成此文件。")
        return
    print(f"已加载准备好的数据 '{input_file}'。")

    analyze_main_effects(df)
    analyze_interaction_effects(df)
    perform_anova_and_quantify(df)

    print("\n\n问题三核心分析完成。请根据输出的图表和ANOVA表格撰写分析报告。")


if __name__ == '__main__':
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 未找到'SimHei'字体，图形中的中文可能无法正常显示。")

    main()