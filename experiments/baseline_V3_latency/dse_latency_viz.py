import subprocess
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np

# --- 配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
COMPILER_EXEC = os.path.join(PROJECT_ROOT, "build", "tools", "mlir-neura-opt", "mlir-neura-opt")
TEST_FILE = os.path.join(PROJECT_ROOT, "test", "samples", "bert", "bert_affine.mlir")

# 搜索空间 (保持之前的范围)
ALPHA_RANGE = np.arange(0.5, 3.5, 0.5)
BETA_RANGE = np.arange(0.5, 5.5, 0.5)
# 融合阈值：现在因为有了 max-fusion-size 限制，我们可以重点看低阈值下的表现
FUSION_VALUES = [500.0, 5000.0, 20000.0]

def run_compiler(alpha, beta, fusion):
    cmd = [
        COMPILER_EXEC,
        "-affine-super-vectorize=virtual-vector-size=4",
        "--task-dependency-analysis",
        "--static-feature-extraction",
        # 注意：这里我们使用了默认的 max-fusion-size=20，也可以在这里显式指定
        f"--spatial-orchestration=chip-width=8 chip-height=8 alpha={alpha:.2f} beta={beta:.2f} fusion-threshold={fusion:.1f} max-fusion-size=20",
        TEST_FILE
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        # [关键修改] 捕获 Latency
        match = re.search(r"\[RESULT\] TimeSteps=(\d+) Blocks=(\d+) Latency=(\d+)", res.stderr)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
    except Exception as e:
        print(f"Error: {e}")
    return float('inf'), 0, float('inf')

def plot_heatmap(df, fusion_val, filename):
    subset = df[df['fusion'] == fusion_val]
    if subset.empty: return
    
    # [关键修改] 这里画 Latency 而不是 Steps
    pivot = subset.pivot(index="alpha", columns="beta", values="latency")
    
    plt.figure(figsize=(10, 8))
    # 使用 viridis_r，深色代表低延迟（高性能）
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis_r", 
                cbar_kws={'label': 'Total Latency (Cycles/Ops)'})
    plt.title(f"Performance Heatmap (Fusion={fusion_val}, MaxSize=20)\nTarget: Minimize Latency")
    plt.ylabel("Alpha (Expansion Weight)")
    plt.xlabel("Beta (Congestion Penalty)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")

def plot_pareto(df, filename):
    plt.figure(figsize=(10, 6))
    # [关键修改] Y轴改为 Latency
    sns.scatterplot(data=df, x="blocks", y="latency", hue="fusion", style="fusion", palette="deep", s=100)
    plt.title("Resource (Blocks) vs Performance (Latency) Trade-off")
    plt.xlabel("Number of Super-Tasks (Blocks)")
    plt.ylabel("Total Execution Latency")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")

def main():
    if not os.path.exists(COMPILER_EXEC):
        print("Compiler not found!")
        return

    results = []
    total = len(ALPHA_RANGE) * len(BETA_RANGE) * len(FUSION_VALUES)
    count = 0
    
    print(f"Starting Latency-Aware DSE... Total: {total}")
    
    for f in FUSION_VALUES:
        for a in ALPHA_RANGE:
            for b in BETA_RANGE:
                count += 1
                steps, blocks, lat = run_compiler(a, b, f)
                print(f"[{count}/{total}] A={a:.2f} B={b:.2f} F={f:.0f} -> Latency={lat} (Stp={steps} Blk={blocks})")
                if steps != float('inf'):
                    results.append({"alpha": a, "beta": b, "fusion": f, "steps": steps, "blocks": blocks, "latency": lat})

    if not results:
        print("No valid results found.")
        return

    df = pd.DataFrame(results)
    df.to_csv("dse_latency_results.csv", index=False)
    
    # 找最优解（Latency 最小）
    best = df.loc[df['latency'].idxmin()]
    print("\n" + "="*40)
    print(f"🏆 BEST CONFIGURATION (Latency Optimized)")
    print(f"   Alpha: {best['alpha']}")
    print(f"   Beta:  {best['beta']}")
    print(f"   Fusion: {best['fusion']}")
    print(f"   Result: Latency={best['latency']} (Blocks={best['blocks']})")
    print("="*40)

    # 绘图
    plot_heatmap(df, 500.0, "heatmap_latency_500.png")
    plot_pareto(df, "pareto_latency.png")

if __name__ == "__main__":
    main()
