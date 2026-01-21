import subprocess
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
import numpy as np
import time

# --- 1. 配置区域 (Configuration) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
COMPILER_EXEC = os.path.join(PROJECT_ROOT, "build", "tools", "mlir-neura-opt", "mlir-neura-opt")
TEST_FILE = os.path.join(PROJECT_ROOT, "test", "samples", "bert", "bert_affine.mlir")

# 固定参数
CHIP_SIZE = 8
BASE_ARGS = [
    "-affine-super-vectorize=virtual-vector-size=4",
    "--task-dependency-analysis",
    "--static-feature-extraction"
]

# --- 关键修改：定义搜索范围 (Start, Stop, Step) ---
# Alpha (扩容权重): 范围 0.5 ~ 5.0, 步长 0.5
# 之前的结论是 Alpha 越小越好，所以我们重点搜小值区域
ALPHA_RANGE = np.arange(0.5, 5.5, 0.5) 

# Beta (扇出惩罚): 范围 0.1 ~ 5.0, 步长 0.5
# 之前的结论是 Beta 越大越好，所以覆盖大值
BETA_RANGE = np.arange(0.1, 5.5, 0.5)

# Fusion Threshold (融合阈值):
# 既然之前的 Profit 在 23000 左右，我们在附近细搜，同时也覆盖极大极小值
FUSION_VALUES = [500.0, 10000.0, 20000.0, 25000.0, 30000.0, 50000.0]

# --- 2. 执行函数 ---
def run_experiment(alpha, beta, fusion):
    # 格式化参数，保留一位小数
    pass_options = (
        f"chip-width={CHIP_SIZE} chip-height={CHIP_SIZE} "
        f"alpha={alpha:.2f} beta={beta:.2f} fusion-threshold={fusion:.1f}"
    )
    
    cmd = [COMPILER_EXEC] + BASE_ARGS + [
        f"--spatial-orchestration={pass_options}",
        TEST_FILE
    ]

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=10 # 现在运行很快了，超时设短一点
        )
        duration = time.time() - start_time
        
        output = result.stderr
        
        # 解析结果
        result_match = re.search(r"\[RESULT\] TimeSteps=(\d+) Blocks=(\d+) Failed=(\d+)", output)
        
        if result_match:
            return int(result_match.group(1)), int(result_match.group(2)), int(result_match.group(3)), duration
        else:
            # Fallback 解析
            if "TimeSteps=" in output:
                print(f"[Warn] Parsing failed but keyword found. Output tail: {output[-100:]}")
            return float('inf'), 0, 1, duration # 视为失败
            
    except subprocess.TimeoutExpired:
        return float('inf'), 0, 1, 10.0
    except Exception as e:
        print(f"Error: {e}")
        return float('inf'), 0, 1, 0.0

# --- 3. 主逻辑 ---
def main():
    if not os.path.exists(COMPILER_EXEC):
        print(f"Error: Compiler not found at {COMPILER_EXEC}")
        return

    # 计算总迭代次数
    total_iters = len(ALPHA_RANGE) * len(BETA_RANGE) * len(FUSION_VALUES)
    print(f"Starting Grid Search on {os.path.basename(TEST_FILE)}")
    print(f"Total Iterations: {total_iters}")
    print(f"Alpha Range: {ALPHA_RANGE}")
    print(f"Beta Range:  {BETA_RANGE}")
    print(f"Fusion Vals: {FUSION_VALUES}")
    print("=" * 60)
    print(f"{'Iter':<6} | {'Alpha':<6} {'Beta':<6} {'Fusion':<8} | {'Steps':<6} {'Blocks':<6} {'Time(s)':<6}")
    print("-" * 60)

    results = []
    count = 0
    
    for fusion in FUSION_VALUES:
        for alpha in ALPHA_RANGE:
            for beta in BETA_RANGE:
                count += 1
                steps, blocks, failed, duration = run_experiment(alpha, beta, fusion)
                
                # 只打印成功的或者特定的 log，避免刷屏
                # 这里打印所有结果以便观察
                print(f"{count}/{total_iters:<4} | {alpha:<6.1f} {beta:<6.1f} {fusion:<8.0f} | {steps:<6} {blocks:<6} {duration:<6.2f}")
                
                if steps != float('inf') and failed == 0:
                    results.append({
                        "alpha": round(alpha, 2),
                        "beta": round(beta, 2),
                        "fusion": fusion,
                        "steps": steps,
                        "blocks": blocks
                    })

    if not results:
        print("\n[Error] No valid results collected.")
        return

    df = pd.DataFrame(results)
    
    # --- 4. 寻找最优解 ---
    # 排序优先级: 
    # 1. Time Steps (越小越好) -> 性能
    # 2. Blocks (越小越好) -> 融合得越好，通信开销越小
    # 3. Beta (越大越好) -> 倾向于更稳健的扇出控制 (可选)
    
    sorted_df = df.sort_values(by=['steps', 'blocks', 'beta'], ascending=[True, True, False])
    best_row = sorted_df.iloc[0]

    print("\n" + "="*40)
    print(f"🏆 GLOBAL OPTIMAL PARAMETERS FOUND")
    print(f"   Alpha: {best_row['alpha']}")
    print(f"   Beta:  {best_row['beta']}")
    print(f"   Fusion Threshold: {best_row['fusion']}")
    print(f"   Performance: {int(best_row['steps'])} Steps")
    print(f"   Resource: {int(best_row['blocks'])} Blocks (Merged from 190 tasks)")
    print("="*40)

    # --- 5. 可视化 (针对最优的 Fusion Threshold) ---
    best_fusion = best_row['fusion']
    subset = df[df['fusion'] == best_fusion]
    
    if not subset.empty:
        # Pivot table for Heatmap
        # Index: Alpha (y-axis), Columns: Beta (x-axis), Values: Steps
        pivot_steps = subset.pivot(index="alpha", columns="beta", values="steps")
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_steps, annot=True, fmt=".0f", cmap="viridis_r", cbar_kws={'label': 'Time Steps'})
        plt.title(f"Performance Heatmap (Fusion Threshold = {best_fusion})\n(Lower is Better)")
        plt.ylabel("Alpha (Vectorization Weight)")
        plt.xlabel("Beta (Fan-out Penalty)")
        plt.gca().invert_yaxis() # 让 Alpha 从小到大从下往上，或者保持矩阵直觉，看个人喜好
        plt.savefig("grid_search_heatmap.png")
        print(f"\n[Visual] Heatmap saved to 'grid_search_heatmap.png'")

    # 保存所有数据到 CSV 以备后续分析
    df.to_csv("dse_results.csv", index=False)
    print(f"[Data] Full results saved to 'dse_results.csv'")

if __name__ == "__main__":
    main()
