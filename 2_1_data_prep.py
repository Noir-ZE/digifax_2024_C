import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


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
    # 转换为numpy数组并确保数据类型正确
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # 过滤掉无效值
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true != 0) & (y_pred != 0)
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]

    if len(y_true_clean) == 0:
        return {'RMSE': np.nan, 'MAPE (%)': np.nan, 'R2 Score': np.nan}

    try:
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
        r2 = r2_score(y_true_clean, y_pred_clean)
        return {'RMSE': rmse, 'MAPE (%)': mape, 'R2 Score': r2}
    except Exception as e:
        print(f"计算指标时出错: {e}")
        return {'RMSE': np.nan, 'MAPE (%)': np.nan, 'R2 Score': np.nan}


def safe_log_transform(data, column_name):
    """安全的对数变换函数"""
    print(f"正在对{column_name}进行对数变换...")
    print(f"  最小值: {data.min():.6e}")
    print(f"  最大值: {data.max():.6e}")
    print(f"  平均值: {data.mean():.6e}")

    # 检查是否有非正值
    non_positive = (data <= 0).sum()
    if non_positive > 0:
        print(f"  警告: 发现 {non_positive} 个非正值")
        data = data[data > 0]

    # 执行对数变换
    try:
        log_data = np.log(data.astype(np.float64))
        print(f"  对数变换成功，范围: [{log_data.min():.6f}, {log_data.max():.6f}]")

        # 检查是否有无效值
        invalid_count = (~np.isfinite(log_data)).sum()
        if invalid_count > 0:
            print(f"  警告: 对数变换后有 {invalid_count} 个无效值")

        return log_data
    except Exception as e:
        print(f"  对数变换失败: {e}")
        return None


def main():
    # --- 2. 加载数据 ---
    input_file = 'q2_material1_sine_wave_data.csv'
    try:
        df = pd.read_csv(input_file)
        print(f"已加载准备好的数据 '{input_file}'。")
        print(f"数据形状: {df.shape}")
        print(f"数据列: {df.columns.tolist()}")

        # 详细的数据统计
        print("\n详细数据统计:")
        print(df.describe())

        # 确保数据类型正确
        numeric_columns = ['Temperature', 'Frequency', 'Core_Loss', 'Bm']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 移除任何可能的NaN值
        original_size = len(df)
        df = df.dropna()
        if len(df) < original_size:
            print(f"移除了 {original_size - len(df)} 行包含NaN的数据")

        # 检查关键列的值范围
        for col in ['Core_Loss', 'Frequency', 'Bm']:
            if col in df.columns:
                min_val = df[col].min()
                if min_val <= 0:
                    print(f"警告: {col} 的最小值是 {min_val}，将过滤非正值")
                    df = df[df[col] > 0]

        print(f"最终数据形状: {df.shape}")

    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return

    # --- 3. 步骤一：建立并拟合基准模型 (SE) using Log-Linear Regression ---
    print("\n=== 开始SE模型拟合过程 ===")

    # 分步进行对数变换
    print("\n步骤1: 对数变换")

    log_Pv = safe_log_transform(df['Core_Loss'], 'Core_Loss')
    if log_Pv is None:
        print("Core_Loss对数变换失败，程序终止")
        return

    log_f = safe_log_transform(df['Frequency'], 'Frequency')
    if log_f is None:
        print("Frequency对数变换失败，程序终止")
        return

    log_Bm = safe_log_transform(df['Bm'], 'Bm')
    if log_Bm is None:
        print("Bm对数变换失败，程序终止")
        return

    # 确保所有数据长度一致
    min_length = min(len(log_Pv), len(log_f), len(log_Bm))
    print(f"\n数据长度检查: log_Pv={len(log_Pv)}, log_f={len(log_f)}, log_Bm={len(log_Bm)}")
    print(f"使用最短长度: {min_length}")

    # 重新创建DataFrame以确保数据一致性
    try:
        valid_indices = df.index[:min_length]
        df_clean = df.loc[valid_indices].copy()
        df_clean['log_Pv'] = log_Pv[:min_length]
        df_clean['log_f'] = log_f[:min_length]
        df_clean['log_Bm'] = log_Bm[:min_length]

        print(f"清理后的数据形状: {df_clean.shape}")

    except Exception as e:
        print(f"创建清理数据时出错: {e}")
        return

    print("\n步骤2: 准备回归数据")
    try:
        X_log_se = df_clean[['log_f', 'log_Bm']].values.astype(np.float64)
        y_log_se = df_clean['log_Pv'].values.astype(np.float64)

        print(f"X_log_se形状: {X_log_se.shape}, 类型: {X_log_se.dtype}")
        print(f"y_log_se形状: {y_log_se.shape}, 类型: {y_log_se.dtype}")
        print(f"X_log_se范围: [{X_log_se.min():.6f}, {X_log_se.max():.6f}]")
        print(f"y_log_se范围: [{y_log_se.min():.6f}, {y_log_se.max():.6f}]")

        # 检查是否有无效值
        if not np.all(np.isfinite(X_log_se)):
            print("警告: X_log_se包含无效值")
            return
        if not np.all(np.isfinite(y_log_se)):
            print("警告: y_log_se包含无效值")
            return

    except Exception as e:
        print(f"准备回归数据时出错: {e}")
        return

    print("\n步骤3: 执行线性回归")
    try:
        # 使用更稳定的求解器
        lr_se = LinearRegression(fit_intercept=True)
        print("开始拟合...")
        lr_se.fit(X_log_se, y_log_se)
        print("拟合完成！")

        # 提取参数
        alpha_fit_se = float(lr_se.coef_[0])
        beta_fit_se = float(lr_se.coef_[1])
        intercept = float(lr_se.intercept_)
        k_fit = np.exp(intercept)

        print("SE模型拟合完成。参数如下:")
        print(f"  intercept = {intercept:.6f}")
        print(f"  k     = {k_fit:.6e}")
        print(f"  alpha = {alpha_fit_se:.6f}")
        print(f"  beta  = {beta_fit_se:.6f}")

        params_se = (k_fit, alpha_fit_se, beta_fit_se)

    except Exception as e:
        print(f"线性回归时发生严重错误: {e}")
        print("尝试使用替代方法...")
        try:
            # 手动计算最小二乘解
            X_with_intercept = np.column_stack([np.ones(len(X_log_se)), X_log_se])
            coeffs = np.linalg.lstsq(X_with_intercept, y_log_se, rcond=None)[0]

            intercept = coeffs[0]
            alpha_fit_se = coeffs[1]
            beta_fit_se = coeffs[2]
            k_fit = np.exp(intercept)

            print("使用手动最小二乘法成功拟合:")
            print(f"  k     = {k_fit:.6e}")
            print(f"  alpha = {alpha_fit_se:.6f}")
            print(f"  beta  = {beta_fit_se:.6f}")

            params_se = (k_fit, alpha_fit_se, beta_fit_se)

        except Exception as e2:
            print(f"替代方法也失败: {e2}")
            return

    # 继续后续步骤...
    print("\n步骤4: 计算预测值")
    try:
        df_clean['SE_Prediction'] = predict_se((df_clean['Frequency'], df_clean['Bm']), params_se)
        df_clean['SE_RPE'] = (df_clean['Core_Loss'] - df_clean['SE_Prediction']) / df_clean['Core_Loss'] * 100

        print("SE模型预测完成")

        # 计算性能指标
        metrics_se = calculate_metrics(df_clean['Core_Loss'], df_clean['SE_Prediction'])
        print(f"SE模型性能: {metrics_se}")

    except Exception as e:
        print(f"计算预测值时出错: {e}")
        return

    print("\n=== SE模型完成，程序正常结束 ===")


if __name__ == '__main__':
    # 设置matplotlib以支持中文显示
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"警告: 设置中文字体失败 ({e})。")

    main()