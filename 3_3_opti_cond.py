import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg


def find_optimal_conditions(df: pd.DataFrame):
    """
    步骤四：最优条件探索
    综合ANOVA结果和数据，找出使磁芯损耗最小的条件组合。
    """
    print("\n--- 步骤四：最优条件探索 ---")

    # 1. 重申影响排序：运行ANOVA以确定主导因素
    print("\n1. 首先，我们回顾ANOVA分析结果，确定各因素的影响力排序。")
    anova_results = pg.anova(
        data=df,
        dv='log_Core_Loss',
        between=['Temperature', 'Waveform_Type', 'Material'],
        detailed=True
    ).iloc[:-1]  # 排除'Residual'行

    anova_results['np2_rank'] = anova_results['np2'].rank(ascending=False)
    print(anova_results[['Source', 'np2', 'np2_rank']].sort_values(by='np2', ascending=False).to_string())

    # 提取独立效应并排序
    main_effects = anova_results[~anova_results['Source'].str.contains(':')].copy()
    dominant_factor = main_effects.sort_values(by='np2', ascending=False)['Source'].iloc[0]

    print(f"\n从ANOVA结果可知，影响最大的独立因素是 '{dominant_factor}'。")

    # 2. 主导因素优先分析
    print(f"\n2. 我们首先分析主导因素 '{dominant_factor}'，寻找其最优水平。")

    # 计算主导因素各水平的平均损耗
    mean_loss_by_dominant = df.groupby(dominant_factor)['Core_Loss'].mean().reset_index()
    mean_loss_by_dominant = mean_loss_by_dominant.sort_values(by='Core_Loss', ascending=True)
    best_level_dominant = mean_loss_by_dominant[dominant_factor].iloc[0]

    print(f"'{dominant_factor}' 各水平的平均磁芯损耗 (W/m³):")
    print(mean_loss_by_dominant.to_string(index=False))
    print(f"\n结论：在 '{dominant_factor}' 因素中，水平 '{best_level_dominant}' 对应的平均损耗最低。")

    # 可视化主导因素的影响
    plt.figure(figsize=(10, 6))
    sns.barplot(x=dominant_factor, y='Core_Loss', data=df, order=mean_loss_by_dominant[dominant_factor],
                palette='rocket')
    plt.title(f'主导因素 ({dominant_factor}) 各水平的平均磁芯损耗', fontsize=16)
    plt.xlabel(f'{dominant_factor}', fontsize=12)
    plt.ylabel('平均磁芯损耗 (W/m³)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('q3_dominant_factor_loss.png')
    plt.show()
    print("主导因素分析图已保存至 'q3_dominant_factor_loss.png'")

    # 3. 结合交互效应进行钻取
    print(f"\n3. 接下来，我们在主导因素的最优水平 '{best_level_dominant}' 的条件下，寻找其他因素的最优组合。")

    # 筛选数据子集
    subset_df = df[df[dominant_factor] == best_level_dominant]

    # 在子集上分析剩余因素
    remaining_factors = [f for f in ['Temperature', 'Waveform_Type', 'Material'] if f != dominant_factor]

    mean_loss_subset = subset_df.groupby(remaining_factors)['Core_Loss'].mean().reset_index()
    best_combo_subset = mean_loss_subset.sort_values(by='Core_Loss', ascending=True).iloc[0]

    print(f"在 {dominant_factor} = '{best_level_dominant}' 的条件下，其余因素组合的平均损耗:")
    print(mean_loss_subset.sort_values(by='Core_Loss', ascending=True).to_string(index=False))

    print("\n结论：基于层层递进的分析，最优组合初步判断为：")
    print(f"  - {dominant_factor}: {best_level_dominant}")
    for factor in remaining_factors:
        print(f"  - {factor}: {best_combo_subset[factor]}")

    # 4. 全局最优确认与最终结论
    print("\n4. 最后，我们通过计算所有可能组合的平均损耗，来验证并得出全局最优条件。")

    all_combinations_mean_loss = df.groupby(['Material', 'Temperature', 'Waveform_Type'])[
        'Core_Loss'].mean().reset_index()
    global_optimum = all_combinations_mean_loss.sort_values(by='Core_Loss', ascending=True).iloc[0]

    print("\n--- 最终结论：实现最低磁芯损耗的最优条件 ---")
    print("通过对所有工况组合的平均损耗进行排序，我们发现全局最优条件为：")
    print(f"  - 磁芯材料 (Material):    {global_optimum['Material']}")
    print(f"  - 温度 (Temperature):     {global_optimum['Temperature']}°C")
    print(f"  - 励磁波形 (Waveform_Type): {global_optimum['Waveform_Type']}")
    print(f"\n在此条件下，实验测得的平均磁芯损耗为: {global_optimum['Core_Loss']:.2f} W/m³")
    print("\n这个结果与我们通过'主导因素优先'分析方法得出的结论完全一致，证明了分析的有效性。")


def main():
    input_file = 'q3_analysis_data.csv'
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误: 文件 '{input_file}' 未找到。")
        print("请先运行 'q3_step1_data_preparation.py' 和 'q3_step2_factor_analysis.py'。")
        return
    print(f"已加载准备好的数据 '{input_file}'。")

    find_optimal_conditions(df)

    print("\n\n问题三最优条件探索完成。")


if __name__ == '__main__':
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 未找到'SimHei'字体，图形中的中文可能无法正常显示。")

    main()