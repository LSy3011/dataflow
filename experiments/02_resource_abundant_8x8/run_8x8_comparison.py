# -*- coding: utf-8 -*-
import subprocess
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np

# --- 配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 回溯到项目根目录 (假设目录结构是 experiments/02_xxx/this_script.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
COMPILER_EXEC = os.path.join(PROJECT_ROOT, "build", "tools", "mlir-neura-opt", "mlir-neura-opt")
TEST_FILE = os.path.join(PROJECT_ROOT, "test", "samples", "bert", "bert_affine.mlir")

# [关键设置] 保持与 4x4 实验完全一致的搜索空间，确保对比公平性
ALPHA_RANGE = np.arange(0.5, 3.5, 0.5)
BETA_RANGE = np.arange(0.5, 5.5, 0.5)
FUSION_VALUES = [500.0] # 我们只关注最优融合阈值下的表现

def run_compiler(alpha, beta, fusion):
    cmd = [
        COMPILER_EXEC,
        "-affine-super-vectorize=virtual-vector-size=4",
        "--task-dependency-analysis",
        "--static-feature-extraction",
        # [唯一区别] 这里设置为 8x8
        f"--spatial-orchestration=chip-width=8 chip-height=8 alpha={alpha:.2f} beta={beta:.2f} fusion-threshold={fusion:.1f} max-fusion-size=20",
        TEST_FILE
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        # 解析 Latency
        match = re.search(r"\[RESULT\] TimeSteps=(\d+) Blocks=(\d+) Latency=(\d+)", res.stderr)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
    except Exception as e:
        print(f"Error: {e}")
    return float('inf'), 0, float('inf')

def plot_heatmap(df, filename):
    if df.empty: return
    pivot = df.pivot(index="alpha", columns="beta", values="latency")
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis_r", 
                cbar_kws={'label': 'Total Latency (Cycles/Ops)'})
    plt.title(f"Performance Heatmap (8x8 Resource Abundant)\nComparison Group")
    plt.ylabel("Alpha (Expansion Weight)")
    plt.xlabel("Beta (Congestion Penalty)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")

def main():
    if not os.path.exists(COMPILER_EXEC):
        print(f"Compiler not found at {COMPILER_EXEC}")
        return

    print("=== Running 8x8 Comparison Experiment ===")
    results = []
    
    for f in FUSION_VALUES:
        for a in ALPHA_RANGE:
            for b in BETA_RANGE:
                steps, blocks, lat = run_compiler(a, b, f)
                print(f"A={a:.2f} B={b:.2f} -> Latency={lat} (Stp={steps})")
                if steps != float('inf'):
                    results.append({"alpha": a, "beta": b, "fusion": f, "steps": steps, "blocks": blocks, "latency": lat})

    df = pd.DataFrame(results)
    df.to_csv("dse_8x8_results.csv", index=False)
    
    best = df.loc[df['latency'].idxmin()]
    print("\n" + "="*40)
    print(f"🏆 8x8 BEST RESULT")
    print(f"   Latency: {best['latency']}")
    print(f"   (Use this to calculate efficiency drop in 4x4)")
    print("="*40)
    
    plot_heatmap(df, "heatmap_8x8_comparison.png")

if __name__ == "__main__":
    main()
