import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# --- 1. 定义模型和评估函数 ---

def predict_se(X, params):
    """SE模型预测函数"""
    f, Bm = X
    k, alpha, beta = params
    return k * (f ** alpha) * (Bm ** beta)


def predict_mse(X, params):
    """MSE模型预测函数"""
    T, f, Bm = X
    c0, c1, c2, alpha, beta = params
    k_T = c0 + c1 * T + c2 * (T ** 2)
    return k_T * (f ** alpha) * (Bm ** beta)


def calculate_metrics(y_true, y_pred):
    """计算RMSE, MAPE, R2等性能指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0)
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]

    if len(y_true_clean) == 0:
        return {'RMSE': np.nan, 'MAPE (%)': np.nan, 'R2 Score': np.nan}

    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    r2 = r2_score(y_true_clean, y_pred_clean)
    return {'RMSE': rmse, 'MAPE (%)': mape, 'R2 Score': r2}


# --- 新增：可视化函数 ---
def plot_visual_comparisons(df):
    """生成预测值vs实际值图和残差图"""

    # 1. 预测值 vs. 实际值 对比图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    max_val = df['Core_Loss'].max()

    # SE Model
    sns.scatterplot(x='Core_Loss', y='SE_Prediction', data=df, ax=axes[0], alpha=0.6, hue='Temperature',
                    palette='viridis')
    axes[0].plot([0, max_val], [0, max_val], 'r--', label='y=x (理想线)')
    axes[0].set_title('SE模型: 预测值 vs. 实际值')
    axes[0].set_xlabel('实际磁芯损耗 (W/m3)')
    axes[0].set_ylabel('预测磁芯损耗 (W/m3)')
    axes[0].grid(True)
    axes[0].legend()

    # MSE Model
    sns.scatterplot(x='Core_Loss', y='MSE_Prediction', data=df, ax=axes[1], alpha=0.6, hue='Temperature',
                    palette='viridis')
    axes[1].plot([0, max_val], [0, max_val], 'r--', label='y=x (理想线)')
    axes[1].set_title('MSE模型: 预测值 vs. 实际值')
    axes[1].set_xlabel('实际磁芯损耗 (W/m3)')
    axes[1].set_ylabel('预测磁芯损耗 (W/m3)')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('q2_predictions_vs_actual.png')
    plt.show()
    print("预测值vs实际值对比图已保存至 'q2_predictions_vs_actual.png'")

    # 2. 残差图对比
    df['SE_Residual'] = df['Core_Loss'] - df['SE_Prediction']
    df['MSE_Residual'] = df['Core_Loss'] - df['MSE_Prediction']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # SE Model Residuals
    sns.scatterplot(x='SE_Prediction', y='SE_Residual', data=df, ax=axes[0], alpha=0.6, hue='Temperature',
                    palette='viridis')
    axes[0].axhline(0, color='r', linestyle='--')
    axes[0].set_title('SE模型: 残差图')
    axes[0].set_xlabel('预测磁芯损耗 (W/m³)')
    axes[0].set_ylabel('残差 (实际值 - 预测值)')
    axes[0].grid(True)

    # MSE Model Residuals
    sns.scatterplot(x='MSE_Prediction', y='MSE_Residual', data=df, ax=axes[1], alpha=0.6, hue='Temperature',
                    palette='viridis')
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_title('MSE模型: 残差图')
    axes[1].set_xlabel('预测磁芯损耗 (W/m³)')
    axes[1].set_ylabel('')  # Y-axis label is shared
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('q2_residuals_comparison.png')
    plt.show()
    print("残差对比图已保存至 'q2_residuals_comparison.png'")


def main():
    # --- 数据加载和准备 ---
    input_file = 'q2_material1_sine_wave_data.csv'
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误: 文件 '{input_file}' 未找到。")
        return

    # --- 拟合SE模型 ---
    print("\n--- 正在使用对数线性回归拟合传统斯坦麦茨方程 (SE Model) ---")
    df['log_Pv'] = np.log(df['Core_Loss'])
    df['log_f'] = np.log(df['Frequency'])
    df['log_Bm'] = np.log(df['Bm'])

    X_log_se = df[['log_f', 'log_Bm']]
    y_log_se = df['log_Pv']

    lr_se = LinearRegression()
    lr_se.fit(X_log_se, y_log_se)

    alpha_fit_se, beta_fit_se = lr_se.coef_
    k_fit = np.exp(lr_se.intercept_)
    params_se = (k_fit, alpha_fit_se, beta_fit_se)

    print("SE模型拟合完成。参数如下:")
    print(f"  k     = {k_fit:.6e}")
    print(f"  alpha = {alpha_fit_se:.6f}")
    print(f"  beta  = {beta_fit_se:.6f}")

    # --- 分析SE模型局限性 ---
    print("\n--- 正在分析SE模型的局限性 ---")
    df['SE_Prediction'] = predict_se((df['Frequency'], df['Bm']), params_se)
    df['SE_RPE'] = (df['Core_Loss'] - df['SE_Prediction']) / df['Core_Loss'] * 100

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Temperature', y='SE_RPE', data=df)
    plt.title('SE模型相对预测误差(RPE) vs. 温度 (材料1, 正弦波)')
    plt.xlabel('温度 (°C)')
    plt.ylabel('相对预测误差 (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('q2_se_error_analysis.png')
    plt.show()
    print(f"误差分析图已保存至 'q2_se_error_analysis.png'。")

    # --- 拟合MSE模型 ---
    print("\n--- 正在使用两步回归法拟合修正斯坦麦茨方程 (MSE Model) ---")
    alpha_fit_mse, beta_fit_mse = alpha_fit_se, beta_fit_se  # 继承自SE模型
    df['k_empirical'] = df['Core_Loss'] / (df['Frequency'] ** alpha_fit_mse * df['Bm'] ** beta_fit_mse)

    c2_fit, c1_fit, c0_fit = np.polyfit(df['Temperature'], df['k_empirical'], 2)
    params_mse = (c0_fit, c1_fit, c2_fit, alpha_fit_mse, beta_fit_mse)

    print("MSE模型拟合完成。参数如下:")
    print(f"  c0    = {c0_fit:.6e}")
    print(f"  c1    = {c1_fit:.6e}")
    print(f"  c2    = {c2_fit:.6e}")
    print(f"  alpha = {alpha_fit_mse:.6f} (继承)")
    print(f"  beta  = {beta_fit_mse:.6f} (继承)")

    # --- 效果比较与验证 ---
    print("\n--- 模型性能对比评估 ---")
    df['MSE_Prediction'] = predict_mse((df['Temperature'], df['Frequency'], df['Bm']), params_mse)

    # 定量对比
    metrics_se = calculate_metrics(df['Core_Loss'], df['SE_Prediction'])
    metrics_mse = calculate_metrics(df['Core_Loss'], df['MSE_Prediction'])

    print("\n总体性能指标:")
    results_df = pd.DataFrame([metrics_se, metrics_mse], index=['SE Model', 'MSE Model'])
    print(results_df.round(4).to_string())

    print("\n分温度平均绝对百分比误差 (MAPE %):")
    mape_comparison = df.groupby('Temperature').apply(
        lambda x: pd.Series({
            'SE_MAPE': calculate_metrics(x['Core_Loss'], x['SE_Prediction'])['MAPE (%)'],
            'MSE_MAPE': calculate_metrics(x['Core_Loss'], x['MSE_Prediction'])['MAPE (%)']
        })
    )
    print(mape_comparison.round(4).to_string())

    # --- 新增：可视化验证 ---
    print("\n--- 生成模型性能的可视化对比图 ---")
    plot_visual_comparisons(df)

    print("\n分析完成。")


if __name__ == '__main__':
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"警告: 设置中文字体失败 ({e})。")

    main()
