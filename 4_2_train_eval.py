import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

# --- 全局设置 ---
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号


def train_and_evaluate():
    """
    执行模型训练、10折交叉验证评估和特征重要性分析。
    """
    # --- 1. 加载数据 ---
    print("--- 步骤一：加载特征化后的训练数据 ---")
    try:
        train_df = pd.read_csv('q4_train_featured.csv')
    except FileNotFoundError:
        print("错误: 'q4_train_featured.csv' 未找到。")
        print("请先运行 q4_step1_feature_engineering.py 脚本。")
        return

    # 分离特征 (X) 和目标 (y)
    X = train_df.drop('Core_Loss', axis=1)
    y = train_df['Core_Loss']

    # 对目标变量进行对数变换，缓解偏态分布问题
    y_log = np.log1p(y)
    print("数据加载完成，已对目标变量 Core_Loss 进行 log1p 变换。")

    # --- 2. 模型与交叉验证设置 ---
    print("\n--- 步骤二：执行10折交叉验证评估模型性能 ---")

    # 初始化LightGBM回归模型
    model = lgb.LGBMRegressor(random_state=42)

    # 设置10折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # 用于存储每次验证的结果
    rmse_scores, r2_scores, mape_scores = [], [], []

    # --- 3. 执行交叉验证循环 ---
    for fold, (train_index, val_index) in enumerate(kf.split(X, y_log)):
        print(f"正在进行第 {fold + 1}/10 折交叉验证...")

        # 分割数据
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_log_train, y_log_val = y_log.iloc[train_index], y_log.iloc[val_index]

        # 训练模型
        model.fit(X_train, y_log_train)

        # 进行预测 (预测结果为对数变换后的值)
        y_log_pred = model.predict(X_val)

        # 将预测值和真实值逆变换回原始尺度
        y_pred = np.expm1(y_log_pred)
        y_val = np.expm1(y_log_val)

        # 计算评估指标
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        mape = mean_absolute_percentage_error(y_val, y_pred)

        # 存储分数
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mape_scores.append(mape)

    # --- 4. 打印交叉验证结果 ---
    print("\n--- 交叉验证评估结果 ---")
    print(f"R² (决定系数):           平均值 = {np.mean(r2_scores):.4f}, 标准差 = {np.std(r2_scores):.4f}")
    print(f"RMSE (均方根误差):       平均值 = {np.mean(rmse_scores):.2f}, 标准差 = {np.std(rmse_scores):.2f}")
    print(
        f"MAPE (平均绝对百分比误差): 平均值 = {np.mean(mape_scores) * 100:.2f}%, 标准差 = {np.std(mape_scores) * 100:.2f}%")
    print("评估解读: R² 接近1，RMSE和MAPE较低且稳定（标准差小），表明模型精度高、泛化能力强。")

    # --- 5. 训练最终模型 ---
    print("\n--- 步骤三：在全部训练数据上训练最终模型 ---")
    final_model = lgb.LGBMRegressor(random_state=42)
    final_model.fit(X, y_log)
    print("最终模型训练完成。")

    # --- 6. 特征重要性分析 ---
    print("\n--- 步骤四：分析并可视化特征重要性 ---")
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("最重要的10个特征:")
    print(feature_importances.head(10))

    # 可视化
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importances.head(20), palette='viridis')
    plt.title('模型特征重要性 (Top 20)', fontsize=16)
    plt.xlabel('重要性', fontsize=12)
    plt.ylabel('特征', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("特征重要性图已保存为 'feature_importance.png'")

    # --- 7. 保存模型 ---
    print("\n--- 步骤五：保存训练好的最终模型 ---")
    joblib.dump(final_model, 'lgbm_model.pkl')
    print("模型已成功保存至 'lgbm_model.pkl'，可用于后续预测。")


if __name__ == '__main__':
    train_and_evaluate()