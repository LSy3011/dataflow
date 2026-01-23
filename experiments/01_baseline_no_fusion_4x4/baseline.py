# -*- coding: utf-8 -*-
import subprocess
import re
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 确保路径正确回溯到项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
COMPILER_EXEC = os.path.join(PROJECT_ROOT, "build", "tools", "mlir-neura-opt", "mlir-neura-opt")
TEST_FILE = os.path.join(PROJECT_ROOT, "test", "samples", "bert", "bert_affine.mlir")

# 基准测试的搜索空间
ALPHA_RANGE = np.arange(0.5, 5.5, 0.5)
BETA_RANGE = np.arange(0.5, 5.5, 0.5)

# [关键] 强制关闭融合的参数
BASELINE_FUSION_THRESH = 1000000000.0  # 10亿，确保不融合
BASELINE_MAX_SIZE = 1                  # 强制每个块只有1个任务

def run_baseline(alpha, beta):
    cmd = [
        COMPILER_EXEC,
        "-affine-super-vectorize=virtual-vector-size=4",
        "--task-dependency-analysis",
        "--static-feature-extraction",
        f"--spatial-orchestration=chip-width=4 chip-height=4 alpha={alpha:.2f} beta={beta:.2f} fusion-threshold={BASELINE_FUSION_THRESH} max-fusion-size={BASELINE_MAX_SIZE}",
        TEST_FILE
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        match = re.search(r"\[RESULT\] TimeSteps=(\d+) Blocks=(\d+) Latency=(\d+)", res.stderr)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
    except Exception as e:
        print(f"Error: {e}")
    return float('inf'), 0, float('inf')

def plot_heatmap(df, filename):
    if df.empty: return
    
    # 转换数据格式用于绘图
    pivot = df.pivot(index="alpha", columns="beta", values="latency")
    
    plt.figure(figsize=(10, 8))
    # 使用 viridis_r (反转)，深色代表低数值（高性能），浅色代表高数值（差性能）
    # 对于 Baseline，你会看到上方(High Alpha)变亮/变黄，代表性能恶化
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis_r", 
                cbar_kws={'label': 'Total Latency (Cycles)'})
    
    plt.title(f"Baseline Performance (No Fusion, 4x4 Chip)\nNotice: High Alpha causes Congestion")
    plt.ylabel("Alpha (Expansion Weight)")
    plt.xlabel("Beta (Congestion Penalty)")
    plt.gca().invert_yaxis() # 让 Alpha 从小到大从下往上
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")

def main():
    if not os.path.exists(COMPILER_EXEC):
        print(f"Compiler not found at {COMPILER_EXEC}")
        return

    print("=== Running BASELINE (No Fusion) Experiment ===")
    print(f"Fusion Threshold: {BASELINE_FUSION_THRESH} (Disabled)")
    print(f"Max Fusion Size:  {BASELINE_MAX_SIZE}")
    print("-" * 60)

    results = []
    
    for a in ALPHA_RANGE:
        for b in BETA_RANGE:
            steps, blocks, lat = run_baseline(a, b)
            print(f"Alpha={a:.1f} Beta={b:.1f} -> Steps={steps} Blocks={blocks} Latency={lat}")
            if steps != float('inf'):
                results.append({"alpha": a, "beta": b, "steps": steps, "blocks": blocks, "latency": lat})

    if not results:
        print("No valid results.")
        return

    df = pd.DataFrame(results)
    df.to_csv("baseline_no_fusion_results.csv", index=False)

    # 找到基准线下的最优解
    best = df.loc[df['latency'].idxmin()]
    
    print("\n" + "="*60)
    print(f"🛑 BASELINE RESULT (NO FUSION)")
    print(f"   Best Alpha: {best['alpha']}")
    print(f"   Best Beta:  {best['beta']}")
    print(f"   ---------------------------")
    print(f"   Blocks:     {int(best['blocks'])}")
    print(f"   Time Steps: {int(best['steps'])}")
    print(f"   Latency:    {int(best['latency'])}")
    print("="*60)

    # [新增] 生成热力图
    plot_heatmap(df, "heatmap_baseline_4x4.png")

if __name__ == "__main__":
    main()
