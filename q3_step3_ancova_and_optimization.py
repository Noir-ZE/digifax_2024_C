import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols


def perform_ancova_and_quantify(df: pd.DataFrame):
    """
    步骤3a：使用 statsmodels 执行三因素ANCOVA并精确量化影响程度
    """
    print("\n--- 步骤 3a：执行ANCOVA并精确量化影响 ---")
    print("核心库升级：使用 statsmodels 进行高级建模")

    # 1. 定义模型公式
    formula = "log_Core_Loss ~ C(Temperature) * C(Waveform_Type) * C(Material) + log_Frequency + log_Bm"
    print(f"模型公式: {formula}")

    # 2. 构建并拟合OLS模型
    model = ols(formula, data=df).fit()

    # 3. 执行方差分析 (Type II ANOVA)
    ancova_results = sm.stats.anova_lm(model, typ=2)

    # 4. 计算偏Eta平方 (Partial Eta-Squared, ηp²)
    ancova_results['np2'] = ancova_results['sum_sq'] / (
                ancova_results['sum_sq'] + ancova_results.loc['Residual', 'sum_sq'])

    print("\n三因素协方差分析 (Three-Way ANCOVA) 结果 (使用 statsmodels):")
    print(ancova_results.round(6).to_string())

    # 5. 可视化效应量
    plot_data = ancova_results.drop('Residual').reset_index().rename(columns={'index': 'Source'})
    plot_data = plot_data.sort_values('np2', ascending=False)

    def get_color(source):
        if source in ['log_Frequency', 'log_Bm']: return 'firebrick'
        if ':' in source: return 'darkorange'
        return 'steelblue'

    plot_data['color'] = plot_data['Source'].apply(get_color)

    # --- 核心修正：创建颜色映射字典，并正确使用hue和palette参数 ---
    color_map = dict(zip(plot_data['Source'], plot_data['color']))

    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='np2',
        y='Source',
        data=plot_data,
        hue='Source',  # 指定hue，让seaborn知道按哪个分类来上色
        palette=color_map,  # 提供Source -> Color的映射字典
        legend=False,  # 关闭自动生成的图例，我们手动创建
        orient='h'
    )
    # -----------------------------------------------------------

    # 添加数值标签
    for i, row in plot_data.iterrows():
        # 使用 .iloc[i] 来定位正确的y坐标
        plt.text(row.np2 + 0.01, i, f"{row.np2:.4f}", color='black', ha="left", va='center')

    plt.title('各因素对磁芯损耗影响程度量化 (ANCOVA 偏Eta平方)', fontsize=16)
    plt.xlabel('影响程度 (Partial Eta-Squared, ηp2)', fontsize=12)
    plt.ylabel('效应来源 (Source)', fontsize=12)
    plt.xlim(0, max(plot_data['np2']) * 1.15)
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='firebrick', label='协变量 (Covariate)'),
                       Patch(facecolor='steelblue', label='主效应 (Main Effect)'),
                       Patch(facecolor='darkorange', label='交互效应 (Interaction)')]
    plt.legend(handles=legend_elements, title='效应类型', loc='lower right')

    plt.tight_layout()
    plt.savefig('q3_ancova_effect_sizes.png', dpi=300)
    plt.show()

    return ancova_results


def find_optimal_conditions(df: pd.DataFrame, ancova_results: pd.DataFrame):
    """
    步骤3b：基于ANCOVA结果进行最优条件探索
    """
    print("\n--- 步骤 3b：最优条件探索 ---")

    # 1. 确定工程可控因素的影响力排序
    main_effect_sources = ['C(Temperature)', 'C(Waveform_Type)', 'C(Material)']
    main_effects = ancova_results[ancova_results.index.isin(main_effect_sources)]
    dominant_factor_source = main_effects['np2'].idxmax()
    dominant_factor = dominant_factor_source.replace('C(', '').replace(')', '')

    print(
        f"1. 根据ANCOVA结果，在工程可控因素中，影响力最大的是 '{dominant_factor}' (ηp²={main_effects.loc[dominant_factor_source, 'np2']:.4f})。")

    # 2. 主导因素优先分析
    mean_loss_by_dominant = df.groupby(dominant_factor)['Core_Loss'].mean().sort_values()
    best_level_dominant = mean_loss_by_dominant.index[0]
    print(f"\n2. 分析主导因素'{dominant_factor}'，发现水平'{best_level_dominant}'对应的平均损耗最低。")
    print(mean_loss_by_dominant)

    # 3. 结合交互效应进行钻取
    print(f"\n3. 在'{dominant_factor}'为'{best_level_dominant}'的条件下，寻找其他因素的最优组合。")
    subset_df = df[df[dominant_factor] == best_level_dominant]
    remaining_factors = [f for f in ['Temperature', 'Waveform_Type', 'Material'] if f != dominant_factor]
    # 使用 observed=False 来避免未来版本的 pandas 警告
    mean_loss_subset = subset_df.groupby(remaining_factors, observed=False)['Core_Loss'].mean().sort_values()
    print(f"   在该子集下，其余因素组合的平均损耗排序如下：")
    print(mean_loss_subset)

    # 4. 全局最优确认与最终结论
    print("\n4. 最终，我们通过计算所有工况组合的平均损耗，确认全局最优条件。")
    all_combinations_mean_loss = df.groupby(['Material', 'Temperature', 'Waveform_Type'], observed=False)[
        'Core_Loss'].mean().sort_values()
    global_optimum_combo = all_combinations_mean_loss.index[0]
    min_loss_value = all_combinations_mean_loss.iloc[0]

    print("\n--- 最终结论：实现最低磁芯损耗的最优条件 ---")
    print(f"  - 磁芯材料 (Material):    {global_optimum_combo[0]}")
    print(f"  - 温度 (Temperature):     {global_optimum_combo[1]}")
    print(f"  - 励磁波形 (Waveform_Type): {global_optimum_combo[2]}")
    print(f"在此条件下，实验测得的平均磁芯损耗为: {min_loss_value:.2f} W/m3")


def main():
    print("\n--- 步骤 3/3：开始执行ANCOVA量化分析与最优条件探索 ---")
    input_file = 'q3_analysis_data_prepared.csv'
    try:
        df = pd.read_csv(input_file)
        # 重新设置category类型
        df['Temperature'] = df['Temperature'].astype('category')
        df['Waveform_Type'] = pd.Categorical(df['Waveform_Type'], categories=['sin', 'tri', 'tra'], ordered=True)
        df['Material'] = df['Material'].astype('category')
    except FileNotFoundError:
        raise FileNotFoundError(f"错误: 文件 '{input_file}' 未找到。请先运行步骤1和2的脚本。")

    ancova_results_df = perform_ancova_and_quantify(df)
    find_optimal_conditions(df, ancova_results_df)

    print("\n\n问题三分析全部完成。")
    print("--- 步骤 3/3 完成 ---")


if __name__ == '__main__':
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告: 中文字体'SimHei'未找到，图形显示可能不正常。")
    main()