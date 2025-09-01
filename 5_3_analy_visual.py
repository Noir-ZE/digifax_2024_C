import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties


# --- 1. 中文字体设置 ---
def set_chinese_font():
    """
    自动设置支持中文的字体，优先使用系统自带的黑体。
    """
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("中文字体 'SimHei' 设置成功。")
    except:
        print("警告: 未找到 'SimHei' 字体。图形中的中文可能无法正常显示。")
        print("您可以尝试安装 'SimHei' 字体或在代码中更换为其他已安装的中文字体。")


# --- 2. 主分析与可视化流程 ---
if __name__ == '__main__':
    print("--- 模块三：结果分析与可视化 ---")

    # 设置绘图风格和字体
    sns.set_theme(style="whitegrid")
    set_chinese_font()

    # 读取优化结果
    try:
        df = pd.read_csv('pareto_results_raw.csv')
    except FileNotFoundError:
        print("错误: 未找到 'pareto_results_raw.csv' 文件。")
        print("请先成功运行模块二 'q5_module_main_optimizer.py' 来生成该文件。")
        exit()

    print(f"成功加载 {len(df)} 个帕累托最优解。")

    # --- 3. 寻找三个关键点 ---

    # 3.1 最低损耗点 (Efficiency-First)
    low_loss_point = df.loc[df['Core_Loss'].idxmin()]

    # 3.2 最高能传点 (Performance-First)
    high_energy_point = df.loc[df['Energy_Transfer'].idxmax()]

    # 3.3 最佳均衡点 (Knee Point)
    # 使用理想点距离法：
    # a. 将两个目标归一化到 [0, 1] 区间
    #    对于Core_Loss, 0是最好, 1是最差
    #    对于Energy_Transfer, 1是最好, 0是最差
    norm_loss = (df['Core_Loss'] - df['Core_Loss'].min()) / (df['Core_Loss'].max() - df['Core_Loss'].min())
    norm_energy = (df['Energy_Transfer'] - df['Energy_Transfer'].min()) / (
                df['Energy_Transfer'].max() - df['Energy_Transfer'].min())

    # b. 计算每个点到理想点(0, 1)的距离
    #    理想点代表 (最小的归一化损耗, 最大的归一化能传)
    distances = np.sqrt(norm_loss ** 2 + (1 - norm_energy) ** 2)

    # c. 找到距离最近的点
    knee_point = df.loc[distances.idxmin()]

    print("\n已定位三个关键最优解。")

    # --- 4. 可视化帕累托前沿 ---

    plt.figure(figsize=(12, 8))

    # 绘制所有帕累托最优解
    sns.scatterplot(data=df, x='Core_Loss', y='Energy_Transfer',
                    label='帕累托最优解集', alpha=0.6, s=50)

    # 高亮标记关键点
    plt.scatter(low_loss_point['Core_Loss'], low_loss_point['Energy_Transfer'],
                color='green', s=200, ec='black', marker='D', zorder=5,
                label=f"最低损耗点\n(损耗: {low_loss_point['Core_Loss']:.2f} W/m3)")

    plt.scatter(high_energy_point['Core_Loss'], high_energy_point['Energy_Transfer'],
                color='red', s=200, ec='black', marker='*', zorder=5,
                label=f"最高能传点\n(能传: {high_energy_point['Energy_Transfer']:.0f} Hz·T)")

    plt.scatter(knee_point['Core_Loss'], knee_point['Energy_Transfer'],
                color='blue', s=200, ec='black', marker='P', zorder=5,
                label=f"最佳均衡点 (Knee Point)\n(损耗: {knee_point['Core_Loss']:.2f}, 能传: {knee_point['Energy_Transfer']:.0f})")

    # 设置图表属性
    plt.title('磁性元件多目标优化帕累托前沿', fontsize=18, weight='bold')
    plt.xlabel('磁芯损耗 (Core Loss, W/m3)', fontsize=14)
    plt.ylabel('传输磁能 (Energy Transfer, Hz·T)', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 保存图表
    output_image_path = 'pareto_front_analysis.png'
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')

    print(f"\n帕累托前沿分析图已保存至: '{output_image_path}'")
    # plt.show() # 如果需要，可以取消注释以在运行时显示图表

    # --- 5. 生成并输出最终结论 ---

    # 创建结论DataFrame
    summary_df = pd.DataFrame([low_loss_point, knee_point, high_energy_point])
    summary_df['Solution_Type'] = ['最低损耗 (效率优先)', '最佳均衡 (性价比最高)', '最高能传 (性能优先)']

    # 调整列顺序，使其更易读
    cols_to_show = ['Solution_Type', 'Core_Loss', 'Energy_Transfer', 'Temperature', 'Frequency', 'Bm', 'Material',
                    'Waveform']
    summary_df = summary_df[cols_to_show]

    # 格式化数值以提高可读性
    summary_df['Core_Loss'] = summary_df['Core_Loss'].map('{:,.2f}'.format)
    summary_df['Energy_Transfer'] = summary_df['Energy_Transfer'].map('{:,.0f}'.format)
    summary_df['Temperature'] = summary_df['Temperature'].map('{:.1f}'.format)
    summary_df['Frequency'] = summary_df['Frequency'].map('{:,.0f}'.format)
    summary_df['Bm'] = summary_df['Bm'].map('{:.3f}'.format)

    print("\n--- 最终结论：最优工况条件总结 ---")
    print(summary_df.to_string())

    # 保存结论到CSV
    output_summary_path = 'optimal_conditions_summary.csv'
    summary_df.to_csv(output_summary_path, index=False, encoding='utf-8-sig')
    print(f"\n最优工况总结已保存至: '{output_summary_path}'")