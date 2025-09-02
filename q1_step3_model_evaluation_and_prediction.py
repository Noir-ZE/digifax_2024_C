import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb


# --- 模型评估函数 ---
def evaluate_models(X, y):
    print("--- 3a. 执行多模型横向评估 ---")

    # 定义模型矩阵
    pipelines = {
        '逻辑回归': Pipeline(
            [('scaler', StandardScaler()), ('model', LogisticRegression(random_state=42, max_iter=1000))]),
        '支持向量机 (SVM)': Pipeline([('scaler', StandardScaler()), ('model', SVC(random_state=42))]),
        'LightGBM': Pipeline([('model', lgb.LGBMClassifier(random_state=42))])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print("正在使用5折交叉验证评估以下模型: ", list(pipelines.keys()))
    for name, pipeline in pipelines.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        results.append({'模型': name, '平均准确率': scores.mean(), '准确率标准差': scores.std()})
        print(f"  - {name}: 平均准确率 = {scores.mean():.6f} (± {scores.std():.6f})")

    results_df = pd.DataFrame(results).sort_values(by='平均准确率', ascending=False)

    print("\n模型横向评估结果总结:")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]['模型']
    print(f"\n结论: '{best_model_name}' 表现最优，将被选为最终模型。")
    return pipelines[best_model_name], best_model_name


# --- 最优模型深度分析函数 ---
def analyze_best_model(model_pipeline, model_name, X, y, le):
    print(f"\n--- 3b. 对最优模型 ({model_name}) 进行深度剖析 ---")

    # 1. 混淆矩阵
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.array([])
    y_true = np.array([])

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_pipeline.fit(X_train, y_train)
        preds = model_pipeline.predict(X_val)

        y_pred = np.concatenate([y_pred, preds])
        y_true = np.concatenate([y_true, y_val])

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'{model_name} 归一化混淆矩阵', fontsize=16)
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.savefig('q1_confusion_matrix.png', dpi=300)
    plt.show()
    print("混淆矩阵图已保存至 'q1_confusion_matrix.png'")

    # 2. 分类报告
    print("\n分类报告 (精确率、召回率、F1分数):")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    # 3. 特征重要性分析 (仅适用于LightGBM)
    if 'lightgbm' in str(model_pipeline.named_steps['model']).lower():
        final_model = model_pipeline.named_steps['model']
        final_model.fit(X, y)  # 在全量数据上训练以获取最终重要性

        feature_importances = pd.DataFrame({
            '特征': X.columns,
            '重要性': final_model.feature_importances_
        }).sort_values('重要性', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='重要性', y='特征', data=feature_importances, palette='viridis')
        plt.title('LightGBM 特征重要性', fontsize=16)
        plt.savefig('q1_feature_importance.png', dpi=300)
        plt.show()
        print("特征重要性图已保存至 'q1_feature_importance.png'")
        print("\n特征重要性排名:")
        print(feature_importances.to_string(index=False))


# --- 预测函数 ---
def predict_on_test_data(model_pipeline, le):
    print("\n--- 3c. 应用最优模型于测试集并生成结果 ---")

    # 1. 加载并准备测试数据
    test_data_path = 'Test_Data_2.xlsx'
    try:
        df_test_raw = pd.read_excel(test_data_path, header=0)
    except FileNotFoundError:
        print(f"错误: 测试数据文件 '{test_data_path}' 未找到。")
        return

    df_test = df_test_raw.rename(columns={
        '温度，oC': 'Temperature', '频率，Hz': 'Frequency', '磁芯材料': 'Material'
    })
    b_field_cols_orig = [col for col in df_test.columns if isinstance(col, int) or col.isdigit()]
    b_field_cols_new = [f'B_t_{i}' for i in range(len(b_field_cols_orig))]
    df_test.rename(columns=dict(zip(b_field_cols_orig, b_field_cols_new)), inplace=True)

    # 2. 特征工程
    print("正在为测试数据提取特征...")
    b_t_cols = [col for col in df_test.columns if col.startswith('B_t_')]
    X_test = df_test[b_t_cols].apply(extract_features, axis=1)  # 复用已定义的特征提取函数

    # 3. 预测
    predictions_encoded = model_pipeline.predict(X_test)
    predictions_str = le.inverse_transform(predictions_encoded)

    # 4. 生成结果文件
    final_mapping = {'sin': 1, 'tri': 2, 'tra': 3}
    final_predictions = pd.Series(predictions_str).map(final_mapping)

    df_results = pd.DataFrame({'样本序号': df_test_raw['序号'], '波形类别': final_predictions})
    output_filename = '附件四.xlsx'
    df_results.to_excel(output_filename, index=False)
    print(f"\n结果文件已生成: '{output_filename}'")

    # 5. 打印论文所需表格
    waveform_counts = df_results['波形类别'].map({v: k for k, v in final_mapping.items()}).value_counts().sort_index()
    print("\n附件二波形数量统计:")
    print(waveform_counts.to_string())

    specific_samples = [1, 5, 15, 25, 35, 45, 55, 65, 75, 80]
    df_specific = df_results[df_results['样本序号'].isin(specific_samples)].copy()
    df_specific['波形类别名称'] = df_specific['波形类别'].map({v: k for k, v in final_mapping.items()})
    print("\n指定样本的分类结果:")
    print(df_specific[['样本序号', '波形类别', '波形类别名称']].to_string(index=False))


def main():
    print("\n--- 步骤 3/3：开始执行模型评估、分析与预测 ---")
    try:
        df_featured = pd.read_csv('q1_train_featured.csv')
    except FileNotFoundError:
        print("错误: 'q1_train_featured.csv' 未找到。请先运行步骤2的脚本。")
        return

    X = df_featured.drop('Waveform_Type', axis=1)
    y_str = df_featured['Waveform_Type']

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_str), name='target')
    joblib.dump(le, 'label_encoder_final.joblib')

    # 评估模型并选出最优
    best_model_pipeline, best_model_name = evaluate_models(X, y)

    # 对最优模型进行深度分析
    analyze_best_model(best_model_pipeline, best_model_name, X, y, le)

    # 使用最优模型进行预测
    # 重新在全量数据上训练最终模型
    print("\n正在全量数据上训练最终模型用于预测...")
    best_model_pipeline.fit(X, y)
    joblib.dump(best_model_pipeline, 'waveform_classifier_final.joblib')
    print("最终模型已保存至 'waveform_classifier_final.joblib'")

    predict_on_test_data(best_model_pipeline, le)

    print("\n--- 步骤 3/3 完成 ---")


if __name__ == '__main__':
    # 导入特征提取函数，因为预测时需要
    from q1_step2_eda_and_feature_engineering import extract_features

    main()