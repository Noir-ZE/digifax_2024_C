import numpy as np
import pandas as pd
import joblib
from scipy.stats import kurtosis
from scipy.fft import rfft

# --- 1. 全局加载与设置 ---
print("正在加载问题定义模块...")

try:
    MODEL = joblib.load('lgbm_model.pkl')
    MODEL_FEATURES = MODEL.feature_name_
    # --- 关键调试步骤 ---
    print("\n--- 从 lgbm_model.pkl 中加载的特征列表 ---")
    print(MODEL_FEATURES)
    print("-------------------------------------------\n")

except FileNotFoundError:
    print("错误: 模型文件 'lgbm_model.pkl' 未找到。")
    print("请确保已成功运行问题四的脚本并生成了模型文件。")
    MODEL = None

N_SAMPLES = 1024

# --- 核心修正：使用与模型内部存储完全一致的后缀 ---
# (整数编码 -> 模型内部使用的后缀字符串)
MATERIAL_MAP = {0: '1', 1: '2', 2: '3', 3: '4'}
WAVEFORM_MAP = {0: 'sin', 1: 'tri', 2: 'tra'}

print("问题定义模块加载完成。")


# --- 2. B(t) 波形合成函数 (对齐新的波形名称) ---
def generate_bt_waveform(Bm, waveform_suffix, n_samples=N_SAMPLES):
    t = np.linspace(0, 1, n_samples, endpoint=False)

    if waveform_suffix == 'sin':
        return Bm * np.sin(2 * np.pi * t)

    elif waveform_suffix == 'tri':
        period = t[-1] - t[0]
        return (4 * Bm / period) * np.abs(((t - period / 4) % period) - period / 2) - Bm

    elif waveform_suffix == 'tra':
        signal = np.zeros(n_samples)
        t1, t2, t3 = 0.2, 0.5, 0.7
        mask1 = t <= t1
        signal[mask1] = (-Bm) + (2 * Bm * (t[mask1] / t1))
        mask2 = (t > t1) & (t <= t2)
        signal[mask2] = Bm
        mask3 = (t > t2) & (t <= t3)
        signal[mask3] = Bm - (2 * Bm * (t[mask3] - t2) / (t3 - t2))
        mask4 = (t > t3)
        signal[mask4] = -Bm
        return signal
    else:
        return Bm * np.sin(2 * np.pi * t)


# --- 3. 单个个体的特征工程函数 ---
def generate_features_for_individual(x):
    T, f, M_encoded, W_encoded, Bm = x
    material_suffix = MATERIAL_MAP[int(round(M_encoded))]
    waveform_suffix = WAVEFORM_MAP[int(round(W_encoded))]

    b_t_series = generate_bt_waveform(Bm, waveform_suffix)

    delta_t = 1 / (f * N_SAMPLES)
    derivatives = np.diff(b_t_series) / delta_t
    max_db_dt = np.max(np.abs(derivatives))

    kurt = kurtosis(b_t_series, fisher=False)

    yf = rfft(b_t_series)
    amplitudes = np.abs(yf)
    thd = np.sqrt(np.sum(amplitudes[2:] ** 2)) / amplitudes[1] if len(amplitudes) > 2 and amplitudes[1] != 0 else 0

    # --- 核心修正：严格按照错误日志提示的名称构建特征 ---
    feature_dict = {
        'Frequency': f,
        'Temperature': T,
        'Bm': Bm,
        'max_dB_dt': max_db_dt,
        'Kurtosis': kurt,
        'THD': thd,
    }

    for w_key, w_val in WAVEFORM_MAP.items():
        feature_dict[f'Waveform_Type_{w_val}'] = 1.0 if int(round(W_encoded)) == w_key else 0.0

    for m_key, m_val in MATERIAL_MAP.items():
        feature_dict[f'Material_{m_val}'] = 1.0 if int(round(M_encoded)) == m_key else 0.0

    features_df = pd.DataFrame([feature_dict])
    return features_df[MODEL_FEATURES]


# --- 4. 核心评估函数 (提供给优化器的接口) ---
def evaluate_objectives(x):
    if MODEL is None:
        raise RuntimeError("模型未能加载，无法进行评估。")

    features = generate_features_for_individual(x)

    log_pred = MODEL.predict(features)[0]
    core_loss = np.expm1(log_pred)

    f_val, Bm_val = x[1], x[4]
    neg_energy_transfer = -(f_val * Bm_val)

    return core_loss, neg_energy_transfer


# --- 5. 测试入口 ---
if __name__ == '__main__':
    print("\n--- 模块独立测试 ---")
    if MODEL is not None:
        test_individual = np.array([50, 100000, 1, 0, 0.2])  # T=50, f=100k, Material '2', Waveform 'sin', Bm=0.2

        print(
            f"测试工况: T={test_individual[0]}, f={test_individual[1]}, Material_Code={int(test_individual[2])} ('{MATERIAL_MAP[1]}'), Waveform_Code={int(test_individual[3])} ('{WAVEFORM_MAP[0]}'), Bm={test_individual[4]}")

        try:
            test_features = generate_features_for_individual(test_individual)
            print("\n成功生成的特征向量 (部分列):")
            print(test_features[['Frequency', 'Temperature', 'Bm', 'Waveform_Type_sin', 'Material_2']])

            loss, neg_energy = evaluate_objectives(test_individual)
            print(f"\n评估结果:")
            print(f"  - 预测磁芯损耗 (目标一): {loss:.2f} W/m³")
            print(f"  - 传输磁能的负值 (目标二): {neg_energy:.2f}")
            print(f"  - 传输磁能: {-neg_energy:.2f} Hz·T")

        except Exception as e:
            print(f"\n测试过程中发生错误: {e}")