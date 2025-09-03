import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import optuna
import json
import warnings

# --- 全局设置 ---
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X, y):
    """Optuna的目标函数，用于评估一组超参数的性能"""

    # 定义超参数的搜索空间
    params = {
        'objective': 'regression_l1',  # MAE损失，对异常值更鲁棒
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 400, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # 使用5折交叉验证来评估当前参数的性能
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)


def main():
    """主函数，执行超参数优化流程"""
    print("--- 步骤2/3：贝叶斯超参数优化 ---")

    # --- 加载数据 ---
    try:
        train_df = pd.read_csv('q4_train_featured.csv')
    except FileNotFoundError:
        print("错误: 'q4_train_featured.csv' 未找到。请先运行步骤1的脚本。")
        return

    X = train_df.drop('log1p_Core_Loss', axis=1)
    y = train_df['log1p_Core_Loss']
    print("数据加载完成。")

    # --- 执行优化 ---
    print("启动Optuna进行超参数搜索 (预计需要一些时间)...")
    study = optuna.create_study(direction='minimize')
    # n_trials可以根据时间和计算资源调整，100是一个比较好的起点
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100, show_progress_bar=True)

    # --- 保存最优结果 ---
    best_params = study.best_params
    print("\n优化完成！")
    print(f"最优交叉验证RMSE: {study.best_value:.6f}")
    print("找到的最佳超参数:")
    print(best_params)

    # 将最优参数保存到JSON文件，供下一步使用
    with open('q4_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    print("\n最优超参数已保存至 'q4_best_params.json'")
    print("--- 步骤2/3 完成 ---")


if __name__ == '__main__':
    main()