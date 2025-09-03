import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def set_chinese_font():
    """设置中文字体"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("中文字体 'SimHei' 设置成功。")
    except:
        print("警告: 未找到 'SimHei' 字体，中文可能无法正常显示。")


def analyze_pareto_front(df):
    """对帕累托前沿进行全景式分析与可视化"""
    print("--- 步骤3/3：帕累托前沿全景式分析 ---")

    # --- 1. 定位三个关键点 ---
    low_loss_point = df.loc[df['Core_Loss'].idxmin()]
    high_energy_point = df.loc[df['Energy_Transfer'].idxmax()]

    norm_loss = (df['Core_Loss'] - df['Core_Loss'].min()) / (df['Core_Loss'].max() - df['Core_Loss'].min())
    norm_energy = (df['Energy_Transfer'] - df['Energy_Transfer'].min()) / (
                df['Energy_Transfer'].max() - df['Energy_Transfer'].min())
    distances = np.sqrt(norm_loss ** 2 + (1 - norm_energy) ** 2)
    knee_point = df.loc[distances.idxmin()]
    print("已定位三个关键最优解。")

    # --- 2. 帕累托前沿主图与着色分析 ---
    print("正在生成帕累托前沿可视化图表...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    fig.suptitle('帕累托前沿全景式分析', fontsize=20, weight='bold')

    # 图一：按材料着色
    sns.scatterplot(ax=axes[0], data=df, x='Core_Loss', y='Energy_Transfer', hue='Material', palette='viridis', s=60,
                    alpha=0.8)
    axes[0].set_title('按“材料”着色分析', fontsize=16)
    axes[0].set_xlabel('磁芯损耗 (Core Loss, W/m3)', fontsize=12)
    axes[0].set_ylabel('传输磁能 (Energy Transfer, Hz·T)', fontsize=12)
    axes[0].grid(True, linestyle='--')
    axes[0].legend(title='材料')

    # 图二：按波形着色
    sns.scatterplot(ax=axes[1], data=df, x='Core_Loss', y='Energy_Transfer', hue='Waveform', palette='plasma', s=60,
                    alpha=0.8)
    axes[1].set_title('按“波形”着色分析', fontsize=16)
    axes[1].set_xlabel('磁芯损耗 (Core Loss, W/m3)', fontsize=12)
    axes[1].set_ylabel('')
    axes[1].grid(True, linestyle='--')
    axes[1].legend(title='波形')

    # 统一高亮关键点
    for ax in axes:
        ax.scatter(low_loss_point['Core_Loss'], low_loss_point['Energy_Transfer'], c='lime', s=200, ec='black',
                   marker='D', zorder=5, label='最低损耗')
        ax.scatter(high_energy_point['Core_Loss'], high_energy_point['Energy_Transfer'], c='red', s=200, ec='black',
                   marker='*', zorder=5, label='最高能传')
        ax.scatter(knee_point['Core_Loss'], knee_point['Energy_Transfer'], c='blue', s=200, ec='black', marker='P',
                   zorder=5, label='最佳均衡')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('q5_pareto_analysis_colored.png', dpi=300)
    plt.show()

    # --- 3. 决策变量分布分析 ---
    print("正在生成决策变量分布图...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('帕累托最优解的决策变量分布', fontsize=20, weight='bold')

    sns.violinplot(ax=axes[0], y=df['Temperature'], color='skyblue')
    axes[0].set_title('温度 (°C) 分布', fontsize=16)

    sns.violinplot(ax=axes[1], y=df['Frequency'], color='salmon')
    axes[1].set_title('频率 (Hz) 分布', fontsize=16)

    sns.violinplot(ax=axes[2], y=df['Bm'], color='lightgreen')
    axes[2].set_title('磁通密度峰值 (T) 分布', fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('q5_decision_variables_distribution.png', dpi=300)
    plt.show()

    # --- 4. 生成最终结论表格 ---
    summary_df = pd.DataFrame([low_loss_point, knee_point, high_energy_point])
    summary_df['Solution_Type'] = ['最低损耗 (效率优先)', '最佳均衡 (性价比最高)', '最高能传 (性能优先)']
    cols_to_show = ['Solution_Type', 'Core_Loss', 'Energy_Transfer', 'Temperature', 'Frequency', 'Bm', 'Material',
                    'Waveform']
    summary_df = summary_df[cols_to_show]

    # 格式化
    for col in ['Core_Loss', 'Energy_Transfer', 'Temperature', 'Frequency', 'Bm']:
        summary_df[col] = summary_df[col].map('{:,.2f}'.format)

    print("\n--- 最终结论：最优工况条件总结 ---")
    print(summary_df.to_string())
    summary_df.to_csv('q5_optimal_conditions_summary.csv', index=False, encoding='utf-8-sig')
    print("\n最优工况总结已保存至 'q5_optimal_conditions_summary.csv'")
    print("--- 步骤3/3 完成 ---")


if __name__ == '__main__':
    set_chinese_font()
    try:
        results_df = pd.read_csv('q5_pareto_results.csv')
    except FileNotFoundError:
        print("错误: 'q5_pareto_results.csv' 未找到。")
        print("请先成功运行步骤1和2的脚本。")
        exit()
    analyze_pareto_front(results_df)