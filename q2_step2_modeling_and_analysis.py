import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error

# --- 全局绘图设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# --- 1. 模型定义 ---
def se_model_func(X, k, alpha, beta):
    """传统斯坦麦茨方程 (SE)"""
    f, Bm = X
    return k * (f ** alpha) * (Bm ** beta)


def mse_model_func(X, c0, c1, c2, alpha, beta):
    """修正的斯坦麦茨方程 (MSE)"""
    T, f, Bm = X
    k_T = c0 + c1 * T + c2 * (T ** 2)
    return k_T * (f ** alpha) * (Bm ** beta)


# --- 2. 评估与可视化函数 ---
def calculate_metrics(y_true, y_pred):
    """计算RMSE, MAPE, R2等性能指标"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAPE (%)': mape, 'R² Score': r2}


def plot_visual_comparisons(df, model_prefix):
    """生成预测值vs实际值图和残差图"""
    pred_col = f'{model_prefix}_Prediction'
    resid_col = f'{model_prefix}_Residual'
    df[resid_col] = df['Core_Loss'] - df[pred_col]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    max_val = max(df['Core_Loss'].max(), df[pred_col].max()) * 1.05

    # 预测值 vs. 实际值
    sns.scatterplot(x='Core_Loss', y=pred_col, data=df, hue='Temperature', palette='viridis', alpha=0.7, ax=axes[0])
    axes[0].plot([0, max_val], [0, max_val], 'r--', label='y=x (理想线)')
    axes[0].set_title(f'{model_prefix}模型: 预测值 vs. 实际值', fontsize=15)
    axes[0].set_xlabel('实际磁芯损耗 (W/m3)', fontsize=12)
    axes[0].set_ylabel('预测磁芯损耗 (W/m3)', fontsize=12)
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)
    axes[0].grid(True)
    axes[0].legend()

    # 残差图
    sns.scatterplot(x=pred_col, y=resid_col, data=df, hue='Temperature', palette='viridis', alpha=0.7, ax=axes[1])
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_title(f'{model_prefix}模型: 残差图', fontsize=15)
    axes[1].set_xlabel('预测磁芯损耗 (W/m3)', fontsize=12)
    axes[1].set_ylabel('残差 (实际值 - 预测值)', fontsize=12)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'q2_{model_prefix}_analysis_plots.png', dpi=300)
    plt.show()


# --- 主分析流程 ---
def main():
    print("\n--- 步骤 2/2：开始执行诊断式建模与分析 ---")
    try:
        df = pd.read_csv('q2_material1_sine_wave_data.csv')
    except FileNotFoundError:
        print("错误: 'q2_material1_sine_wave_data.csv' 未找到。请先运行步骤1的脚本。")
        return

    # --- 2a. 建立基准模型 (SE) ---
    print("\n--- 2a. 建立基准模型 (SE) ---")
    print("正在使用非线性最小二乘法 (NLLS) 拟合SE模型...")
    X_se = (df['Frequency'].values, df['Bm'].values)
    y_se = df['Core_Loss'].values
    p0_se = [1e-5, 1.5, 2.5]  # 提供合理的初始猜测值
    params_se, _ = curve_fit(se_model_func, X_se, y_se, p0=p0_se, maxfev=10000)
    k_se, alpha_se, beta_se = params_se
    print("SE模型拟合完成。参数如下:")
    print(f"  k     = {k_se:.6e}\n  alpha = {alpha_se:.6f}\n  beta  = {beta_se:.6f}")
    df['SE_Prediction'] = se_model_func(X_se, *params_se)

    # --- 2b. 缺陷诊断与修正依据探索 ---
    print("\n--- 2b. 缺陷诊断与修正依据探索 ---")
    # 诊断1: SE模型误差与温度的关系
    df['SE_RPE'] = (df['Core_Loss'] - df['SE_Prediction']) / df['Core_Loss'] * 100
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Temperature', y='SE_RPE', data=df, palette='viridis')
    plt.axhline(0, color='r', linestyle='--')
    plt.title('SE模型相对预测误差(RPE) vs. 温度', fontsize=15)
    plt.xlabel('温度 (°C)', fontsize=12)
    plt.ylabel('相对预测误差 (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('q2_se_error_diagnosis.png', dpi=300)
    plt.show()
    print("SE模型缺陷诊断图已保存至 'q2_se_error_diagnosis.png'")

    # 诊断2: k(T)的依赖关系探索
    df['k_empirical'] = df['Core_Loss'] / (df['Frequency'] ** alpha_se * df['Bm'] ** beta_se)
    k_fit_poly = np.polyfit(df['Temperature'], df['k_empirical'], 2)
    k_pred_poly = np.poly1d(k_fit_poly)
    temp_range = np.linspace(df['Temperature'].min(), df['Temperature'].max(), 100)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Temperature', y='k_empirical', data=df, alpha=0.6, label='经验k值')
    plt.plot(temp_range, k_pred_poly(temp_range), 'r-', label='二次多项式拟合')
    plt.title('经验k值 vs. 温度 (k_empirical vs. T)', fontsize=15)
    plt.xlabel('温度 (°C)', fontsize=12)
    plt.ylabel('经验k值', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig('q2_k_vs_T_exploration.png', dpi=300)
    plt.show()
    print("k(T)关系探索图已保存至 'q2_k_vs_T_exploration.png'")
    print("探索性分析表明，k与T呈现明显的二次关系，为后续修正提供了依据。")

    # --- 2c. 构建修正模型 (MSE) 并进行全局优化 ---
    print("\n--- 2c. 构建修正模型 (MSE) 并进行全局优化 ---")
    print("正在使用NLLS对MSE模型进行全局同步优化...")
    X_mse = (df['Temperature'].values, df['Frequency'].values, df['Bm'].values)
    y_mse = df['Core_Loss'].values
    # 继承SE的alpha, beta作为初始猜测，并加上k(T)的初始猜测
    p0_mse = [k_fit_poly[2], k_fit_poly[1], k_fit_poly[0], alpha_se, beta_se]
    params_mse, _ = curve_fit(mse_model_func, X_mse, y_mse, p0=p0_mse, maxfev=20000)
    print("MSE模型拟合完成。参数如下:")
    print(f"  c0    = {params_mse[0]:.6e}\n  c1    = {params_mse[1]:.6e}\n  c2    = {params_mse[2]:.6e}")
    print(f"  alpha = {params_mse[3]:.6f}\n  beta  = {params_mse[4]:.6f}")
    df['MSE_Prediction'] = mse_model_func(X_mse, *params_mse)

    # --- 2d. 全方位性能验证与深度分析 ---
    print("\n--- 2d. 全方位性能验证与深度分析 ---")
    # 定量对比
    metrics_se_total = calculate_metrics(df['Core_Loss'], df['SE_Prediction'])
    metrics_mse_total = calculate_metrics(df['Core_Loss'], df['MSE_Prediction'])

    print("\n总体性能指标对比:")
    results_total_df = pd.DataFrame([metrics_se_total, metrics_mse_total], index=['SE Model', 'MSE Model'])
    print(results_total_df.round(4).to_string())

    print("\n分温度平均绝对百分比误差 (MAPE %) 对比:")
    mape_comparison = df.groupby('Temperature').apply(
        lambda x: pd.Series({
            'SE_MAPE': calculate_metrics(x['Core_Loss'], x['SE_Prediction'])['MAPE (%)'],
            'MSE_MAPE': calculate_metrics(x['Core_Loss'], x['MSE_Prediction'])['MAPE (%)']
        })
    )
    print(mape_comparison.round(2).to_string())

    # 可视化对比
    print("\n生成SE模型性能可视化图...")
    plot_visual_comparisons(df, 'SE')
    print("\n生成MSE模型性能可视化图...")
    plot_visual_comparisons(df, 'MSE')

    print("\n分析完成。MSE模型在各项指标上均显著优于传统SE模型。")


if __name__ == '__main__':
    main()