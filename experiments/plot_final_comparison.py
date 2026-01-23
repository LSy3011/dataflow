import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 配置 ---
# 定义三个实验数据的相对路径
PATHS = {
    "Baseline (No Fusion, 4x4)": "01_baseline_no_fusion_4x4/baseline_no_fusion_results.csv",
    "Optimized (Fusion, 4x4)":   "03_resource_constrained_4x4/dse_latency_results.csv",
    "Ideal Bound (Fusion, 8x8)": "02_resource_abundant_8x8/dse_8x8_results.csv"
}

def get_best_latency(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return 0
    df = pd.read_csv(filepath)
    # 找到 Latency 最小的那一行
    return df['latency'].min()

def main():
    data = []
    
    print("=== Extracting Final Results ===")
    for label, path in PATHS.items():
        lat = get_best_latency(path)
        print(f"{label:<30} : {lat}")
        data.append({"Experiment": label, "Latency": lat})
    
    df_plot = pd.DataFrame(data)
    
    # --- 绘图 ---
    plt.figure(figsize=(10, 6))
    
    # 定义颜色：灰色(基准)，绿色(你的成果)，蓝色(理想上限)
    colors = ['#95a5a6', '#2ecc71', '#3498db']
    
    ax = sns.barplot(x="Experiment", y="Latency", data=df_plot, palette=colors)
    
    # 在柱子上标数值
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=12, fontweight='bold')
    
    # 添加提升幅度标注 (Baseline vs Optimized)
    base_lat = df_plot.loc[0, 'Latency']
    opt_lat = df_plot.loc[1, 'Latency']
    improvement = (base_lat - opt_lat) / base_lat * 100
    
    plt.title(f"Final Performance Comparison\n(Improvement: {improvement:.1f}%)", fontsize=14)
    plt.ylabel("Total Latency (Cycles) - Lower is Better")
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    output_file = "final_comparison_summary.png"
    plt.savefig(output_file, dpi=300)
    print(f"\n[Success] Final summary chart saved to: {output_file}")

if __name__ == "__main__":
    main()
