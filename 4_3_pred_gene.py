import pandas as pd
import numpy as np
import joblib
import warnings

# --- 全局设置 ---
warnings.filterwarnings('ignore')


def predict_and_generate_results():
    """
    加载模型和数据，进行预测，并生成最终的结果文件和论文专用CSV。
    """
    # --- 1. 加载模型和测试数据 ---
    print("--- 步骤一：加载模型和特征化后的测试数据 ---")
    try:
        model = joblib.load('lgbm_model.pkl')
        test_df = pd.read_csv('q4_test_featured.csv')
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 -> {e}")
        print("请确保 'lgbm_model.pkl' 和 'q4_test_featured.csv' 文件在当前目录下。")
        print("请先运行 q4_step1 和 q4_step2 脚本。")
        return

    print("模型和测试数据加载成功。")

    # --- 2. 准备预测数据 ---
    print("\n--- 步骤二：准备用于预测的特征矩阵 ---")
    X_test = test_df.drop('Sample_ID', axis=1)
    sample_ids = test_df['Sample_ID']

    model_features = model.feature_name_
    if list(X_test.columns) != model_features:
        print("警告: 测试数据的特征列与模型训练时的特征列不完全匹配或顺序不一致。")
        print("正在尝试按模型训练时的顺序重排特征列...")
        try:
            X_test = X_test[model_features]
        except KeyError as e:
            print(f"错误: 测试数据中缺少必要的特征列: {e}")
            return

    print(f"准备好 {X_test.shape[0]} 个样本进行预测。")

    # --- 3. 执行预测并进行逆变换 ---
    print("\n--- 步骤三：执行预测 ---")
    log_predictions = model.predict(X_test)
    predictions = np.expm1(log_predictions)
    print("预测完成，并已将结果从对数尺度还原。")

    # --- 4. 生成完整结果文件 ---
    print("\n--- 步骤四：生成并保存完整的预测结果 ---")
    results_df = pd.DataFrame({
        'Sample_ID': sample_ids,
        'Predicted_Core_Loss': predictions
    })
    results_df['Predicted_Core_Loss'] = results_df['Predicted_Core_Loss'].round(1)

    output_csv_path = 'q4_predictions.csv'
    results_df.to_csv(output_csv_path, index=False)
    print(f"完整的预测结果已保存至: '{output_csv_path}'")

    # --- 5. 生成论文专用CSV文件 ---
    print("\n--- 步骤五：为论文生成特定样本的预测结果文件 ---")

    # **核心修改：使用您指定的样本序号列表**
    specific_sample_ids = [16, 76, 98, 126, 168, 230, 271, 338, 348, 379]
    paper_table_df = results_df[results_df['Sample_ID'].isin(specific_sample_ids)].copy()

    # 重命名列以符合报告格式
    paper_table_df.rename(columns={
        'Sample_ID': '样本序号',
        'Predicted_Core_Loss': '磁芯损耗预测值 (W/m³)'
    }, inplace=True)

    # **核心修改：保存为独立的CSV文件**
    paper_csv_path = 'q4_predictions_for_paper.csv'
    paper_table_df.to_csv(paper_csv_path, index=False, encoding='utf-8-sig')

    print(f"为论文指定的样本预测结果已单独保存至: '{paper_csv_path}'")
    print("\n预览论文专用数据:")
    print(paper_table_df)


if __name__ == '__main__':
    predict_and_generate_results()