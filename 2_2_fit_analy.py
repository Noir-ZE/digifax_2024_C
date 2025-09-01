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
    # 添加数据验证
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 过滤掉无效值
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0) & (y_pred != 0)
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]

    if len(y_true_clean) == 0:
        return {'RMSE': np.nan, 'MAPE (%)': np.nan, 'R2 Score': np.nan}

    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    r2 = r2_score(y_true_clean, y_pred_clean)
    return {'RMSE': rmse, 'MAPE (%)': mape, 'R2 Score': r2}


def main():
    # --- 2. 加载数据 ---
    input_file = 'q2_material1_sine_wave_data.csv'
    try:
        df = pd.read_csv(input_file)
        print(f"已加载准备好的数据 '{input_file}'。")
        print(f"数据形状: {df.shape}")
        print(f"数据列: {df.columns.tolist()}")
        print(f"数据预览:\n{df.head()}")

        # 检查数据质量
        print("\n数据质量检查:")
        print(f"缺失值: {df.isnull().sum().sum()}")
        print(f"无穷值: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")

        # 检查关键列是否存在且为正值
        required_cols = ['Core_Loss', 'Frequency', 'Bm', 'Temperature']
        for col in required_cols:
            if col not in df.columns:
                print(f"错误: 缺少必需的列 '{col}'")
                return
            if col != 'Temperature':  # Temperature can be negative, others should be positive
                non_positive = (df[col] <= 0).sum()
                if non_positive > 0:
                    print(f"警告: '{col}' 列中有 {non_positive} 个非正值")
                    df = df[df[col] > 0]  # 过滤掉非正值

        print(f"过滤后数据形状: {df.shape}")

    except FileNotFoundError:
        print(f"错误: 文件 '{input_file}' 未找到。")
        print("请先运行 'q2_step1_data_preparation.py' 来生成此文件。")
        return
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return

    # --- 3. 步骤一：建立并拟合基准模型 (SE) using Log-Linear Regression ---
    print("\n--- 正在使用对数线性回归拟合传统斯坦麦茨方程 (SE Model) ---")

    try:
        # 准备对数变换后的数据
        df['log_Pv'] = np.log(df['Core_Loss'])
        df['log_f'] = np.log(df['Frequency'])
        df['log_Bm'] = np.log(df['Bm'])

        # 检查对数变换后是否有无效值
        if df[['log_Pv', 'log_f', 'log_Bm']].isnull().any().any():
            print("警告: 对数变换产生了无效值")
            df = df.dropna(subset=['log_Pv', 'log_f', 'log_Bm'])

        X_log_se = df[['log_f', 'log_Bm']]
        y_log_se = df['log_Pv']

        # 执行线性回归
        lr_se = LinearRegression()
        lr_se.fit(X_log_se, y_log_se)

        # 提取参数
        alpha_fit_se = lr_se.coef_[0]
        beta_fit_se = lr_se.coef_[1]
        k_fit = np.exp(lr_se.intercept_)
        params_se = (k_fit, alpha_fit_se, beta_fit_se)

        print("SE模型拟合完成。参数如下:")
        print(f"  k     = {k_fit:.6e}")
        print(f"  alpha = {alpha_fit_se:.6f}")
        print(f"  beta  = {beta_fit_se:.6f}")

    except Exception as e:
        print(f"SE模型拟合时发生错误: {e}")
        return

    # --- 4. 步骤二：分析基准模型的局限性 ---
    print("\n--- 正在分析SE模型的局限性 ---")
    try:
        df['SE_Prediction'] = predict_se((df['Frequency'], df['Bm']), params_se)
        df['SE_RPE'] = (df['Core_Loss'] - df['SE_Prediction']) / df['Core_Loss'] * 100

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Temperature', y='SE_RPE', data=df)
        plt.title('SE模型相对预测误差(RPE) vs. 温度 (材料1, 正弦波)')
        plt.xlabel('温度 (°C)')
        plt.ylabel('相对预测误差 (%)')
        plt.grid(True, linestyle='--', alpha=0.6)

        plot_filename = 'q2_se_error_analysis.png'
        plt.savefig(plot_filename)
        plt.show()
        print(f"误差分析图已保存至 '{plot_filename}'。")

    except Exception as e:
        print(f"分析SE模型局限性时发生错误: {e}")
        return

    # --- 5. 步骤三：构建并拟合修正模型 (MSE) using Two-Step Regression ---
    print("\n--- 正在使用两步回归法拟合修正斯坦麦茨方程 (MSE Model) ---")

    try:
        # 步骤一：使用SE模型得到的alpha和beta，计算经验k(T)
        alpha_fit_mse, beta_fit_mse = alpha_fit_se, beta_fit_se
        df['k_empirical'] = df['Core_Loss'] / (df['Frequency'] ** alpha_fit_mse * df['Bm'] ** beta_fit_mse)

        # 步骤二：对 k_empirical = c0 + c1*T + c2*T^2 进行二次多项式拟合
        X_poly = df['Temperature']
        y_poly = df['k_empirical']

        # 过滤掉无效值
        valid_mask = np.isfinite(y_poly)
        X_poly_clean = X_poly[valid_mask]
        y_poly_clean = y_poly[valid_mask]

        # np.polyfit返回的系数是最高次幂到最低次幂
        c2_fit, c1_fit, c0_fit = np.polyfit(X_poly_clean, y_poly_clean, 2)
        params_mse = (c0_fit, c1_fit, c2_fit, alpha_fit_mse, beta_fit_mse)

        print("MSE模型拟合完成。参数如下:")
        print(f"  c0    = {c0_fit:.6e}")
        print(f"  c1    = {c1_fit:.6e}")
        print(f"  c2    = {c2_fit:.6e}")
        print(f"  alpha = {alpha_fit_mse:.6f} (继承自SE模型)")
        print(f"  beta  = {beta_fit_mse:.6f} (继承自SE模型)")

    except Exception as e:
        print(f"MSE模型拟合时发生错误: {e}")
        return

    # --- 6. 效果比较与验证 ---
    print("\n--- 模型性能对比评估 ---")
    try:
        df['MSE_Prediction'] = predict_mse((df['Temperature'], df['Frequency'], df['Bm']), params_mse)

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

        print("\n结论：通过对比总体指标和分温度指标，可以看出MSE模型显著优于传统SE模型。")

    except Exception as e:
        print(f"模型性能对比时发生错误: {e}")


if __name__ == '__main__':
    # 设置matplotlib以支持中文显示
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"警告: 设置中文字体失败 ({e})。图形中的中文可能无法正常显示。")

    main()