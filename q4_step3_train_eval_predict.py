import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import shap
import warnings

# --- 全局设置 ---
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def train_evaluate_explain(X, y, best_params):
    """执行模型训练、10折交叉验证评估和SHAP可解释性分析"""
    print("\n--- 3a：执行10折交叉验证评估模型性能 ---")

    model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1, objective='regression_l1')
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X))
    feature_importances = pd.DataFrame(index=X.columns)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"正在进行第 {fold + 1}/10 折交叉验证...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict(X_val)
        feature_importances[f'fold_{fold + 1}'] = model.feature_importances_

    # --- 性能评估 ---
    y_orig = np.expm1(y)
    oof_preds_orig = np.expm1(oof_preds)

    r2 = r2_score(y_orig, oof_preds_orig)
    rmse = np.sqrt(mean_squared_error(y_orig, oof_preds_orig))
    mape = mean_absolute_percentage_error(y_orig, oof_preds_orig)

    print("\n--- 交叉验证评估结果 ---")
    print(f"R² (决定系数):           {r2:.6f}")
    print(f"RMSE (均方根误差):       {rmse:.2f}")
    print(f"MAPE (平均绝对百分比误差): {mape * 100:.4f}%")

    # --- 性能可视化 ---
    print("\n--- 3b：生成性能可视化图表 ---")
    # 1. 预测值 vs 真实值
    plt.figure(figsize=(8, 8))
    plt.scatter(y_orig, oof_preds_orig, alpha=0.3, s=10)
    plt.plot([y_orig.min(), y_orig.max()], [y_orig.min(), y_orig.max()], 'r--', lw=2)
    plt.xlabel("真实磁芯损耗 (W/m3)", fontsize=12)
    plt.ylabel("预测磁芯损耗 (W/m3)", fontsize=12)
    plt.title("预测值 vs. 真实值 (10折交叉验证)", fontsize=16)
    plt.grid(True)
    plt.savefig('q4_pred_vs_true.png', dpi=300)
    plt.show()

    # 2. 残差图
    residuals = y_orig - oof_preds_orig
    plt.figure(figsize=(10, 6))
    plt.scatter(oof_preds_orig, residuals, alpha=0.3, s=10)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("预测磁芯损耗 (W/m3)", fontsize=12)
    plt.ylabel("残差 (真实值 - 预测值)", fontsize=12)
    plt.title("残差图", fontsize=16)
    plt.grid(True)
    plt.savefig('q4_residuals_plot.png', dpi=300)
    plt.show()

    # --- 模型解释性分析 ---
    print("\n--- 3c：训练最终模型并进行SHAP可解释性分析 ---")
    final_model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1, objective='regression_l1')
    final_model.fit(X, y)

    # SHAP分析
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)

    # 绘制SHAP摘要图
    shap.summary_plot(shap_values, X, plot_type="dot", show=False, max_display=15)
    plt.title("SHAP摘要图：特征对模型输出的影响", fontsize=14)
    plt.tight_layout()
    plt.savefig('q4_shap_summary_plot.png', dpi=300)
    plt.show()
    print("SHAP摘要图已保存。")

    return final_model


def predict_on_test(model, X_test):
    """使用最终模型对测试集进行预测"""
    print("\n--- 3d：执行最终预测 ---")
    log_predictions = model.predict(X_test)
    predictions = np.expm1(log_predictions)
    print("预测完成。")
    return predictions


def main():
    """主函数，执行模型训练、评估、解释和预测的全流程"""
    print("--- 步骤3/3：模型训练、深度评估与预测 ---")

    # --- 加载数据和最优参数 ---
    try:
        train_df = pd.read_csv('q4_train_featured.csv')
        test_df = pd.read_csv('q4_test_featured.csv')
        with open('q4_best_params.json', 'r') as f:
            best_params = json.load(f)
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 -> {e}。请先运行步骤1和2的脚本。")
        return

    X = train_df.drop('log1p_Core_Loss', axis=1)
    y = train_df['log1p_Core_Loss']
    X_test = test_df.drop('Sample_ID', axis=1)

    # 确保测试集和训练集特征列顺序一致
    X_test = X_test[X.columns]

    # --- 执行核心流程 ---
    final_model = train_evaluate_explain(X, y, best_params)
    predictions = predict_on_test(final_model, X_test)

    # --- 保存模型和结果 ---
    joblib.dump(final_model, 'q4_lgbm_model_optimized.pkl')
    print("\n优化后的最终模型已保存至 'q4_lgbm_model_optimized.pkl'")

    results_df = pd.DataFrame({'Sample_ID': test_df['Sample_ID'], 'Predicted_Core_Loss': predictions})
    results_df['Predicted_Core_Loss'] = results_df['Predicted_Core_Loss'].round(1)

    # 为附件四生成完整预测
    results_df.to_csv('q4_predictions_for_submission.csv', index=False)
    print("完整的预测结果已保存至 'q4_predictions_for_submission.csv'")

    # 为论文生成特定样本的预测
    specific_ids = [16, 76, 98, 126, 168, 230, 271, 338, 348, 379]
    paper_table = results_df[results_df['Sample_ID'].isin(specific_ids)]
    paper_table.rename(columns={'Sample_ID': '样本序号', 'Predicted_Core_Loss': '磁芯损耗预测值 (W/m³)'}, inplace=True)
    paper_table.to_csv('q4_predictions_for_paper.csv', index=False, encoding='utf-8-sig')
    print("\n为论文指定的样本预测结果已保存至 'q4_predictions_for_paper.csv'")
    print(paper_table)

    print("\n--- 步骤3/3 完成 ---")


if __name__ == '__main__':
    main()