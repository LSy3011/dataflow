import subprocess
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
import numpy as np
import time

# --- 配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
COMPILER_EXEC = os.path.join(PROJECT_ROOT, "build", "tools", "mlir-neura-opt", "mlir-neura-opt")
TEST_FILE = os.path.join(PROJECT_ROOT, "test", "samples", "bert", "bert_affine.mlir")

# 范围设置 (根据之前的经验微调范围，捕捉变化)
# Alpha: 重点关注 0.5 ~ 3.0，步长 0.25 (更精细)
ALPHA_RANGE = np.arange(0.25, 3.25, 0.25)
# Beta:  重点关注 0.5 ~ 5.0，步长 0.5
BETA_RANGE = np.arange(0.5, 5.5, 0.5)
# Fusion: 锁定在性能突变点附近
FUSION_VALUES = [500.0, 20000.0, 25000.0]

def run_compiler(alpha, beta, fusion):
    cmd = [
        COMPILER_EXEC,
        "-affine-super-vectorize=virtual-vector-size=4",
        "--task-dependency-analysis",
        "--static-feature-extraction",
        f"--spatial-orchestration=chip-width=8 chip-height=8 alpha={alpha:.2f} beta={beta:.2f} fusion-threshold={fusion:.1f}",
        TEST_FILE
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        match = re.search(r"\[RESULT\] TimeSteps=(\d+) Blocks=(\d+)", res.stderr)
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception as e:
        print(f"Error: {e}")
    return float('inf'), 0

def plot_heatmap(df, fusion_val, filename):
    subset = df[df['fusion'] == fusion_val]
    if subset.empty: return
    
    pivot = subset.pivot(index="alpha", columns="beta", values="steps")
    
    plt.figure(figsize=(10, 8))
    # 使用 viridis_r 颜色：深色代表数值小（性能好），浅色代表数值大（性能差）
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis_r", 
                cbar_kws={'label': 'Time Steps (Lower is Better)'})
    plt.title(f"DSE Heatmap (Fusion={fusion_val})\nFinding Optimal Alpha/Beta for Adaptive Model")
    plt.ylabel("Alpha (Expansion Weight)")
    plt.xlabel("Beta (Congestion Penalty)")
    plt.gca().invert_yaxis() # 让 Alpha 从小到大向上
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")

def plot_pareto(df, filename):
    plt.figure(figsize=(10, 6))
    # 使用散点图展示 Blocks vs Steps 的权衡
    sns.scatterplot(data=df, x="blocks", y="steps", hue="fusion", style="fusion", palette="deep", s=100)
    plt.title("Resource (Blocks) vs Performance (Steps) Trade-off")
    plt.xlabel("Number of Super-Tasks (Blocks)")
    plt.ylabel("Execution Time Steps")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")

def main():
    if not os.path.exists(COMPILER_EXEC):
        print("Compiler not found! Build it first.")
        return

    results = []
    total = len(ALPHA_RANGE) * len(BETA_RANGE) * len(FUSION_VALUES)
    count = 0
    
    print(f"Starting Process DSE... Total iterations: {total}")
    
    for f in FUSION_VALUES:
        for a in ALPHA_RANGE:
            for b in BETA_RANGE:
                count += 1
                steps, blocks = run_compiler(a, b, f)
                print(f"[{count}/{total}] A={a:.2f} B={b:.2f} F={f:.0f} -> Steps={steps} Blks={blocks}")
                if steps != float('inf'):
                    results.append({"alpha": a, "beta": b, "fusion": f, "steps": steps, "blocks": blocks})

    df = pd.DataFrame(results)
    df.to_csv("dse_process_results.csv", index=False)
    
    # --- 生成过程图 ---
    # 1. 针对最优 Fusion (500) 的热力图
    plot_heatmap(df, 500.0, "process_heatmap_optimal.png")
    
    # 2. 针对临界点 Fusion (20000) 的热力图 - 观察参数敏感性
    plot_heatmap(df, 20000.0, "process_heatmap_critical.png")
    
    # 3. 帕累托图 - 展示整体权衡过程
    plot_pareto(df, "process_pareto_frontier.png")

if __name__ == "__main__":
    main()
