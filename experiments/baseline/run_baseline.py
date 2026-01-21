import subprocess
import re
import pandas as pd
import os
import numpy as np

# --- 配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 确保路径正确回溯到项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
COMPILER_EXEC = os.path.join(PROJECT_ROOT, "build", "tools", "mlir-neura-opt", "mlir-neura-opt")
TEST_FILE = os.path.join(PROJECT_ROOT, "test", "samples", "bert", "bert_affine.mlir")

# 基准测试的搜索空间
# 我们依然需要搜索 Alpha/Beta，以找到“无融合”状态下的物理极限
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
        f"--spatial-orchestration=chip-width=8 chip-height=8 alpha={alpha:.2f} beta={beta:.2f} fusion-threshold={BASELINE_FUSION_THRESH} max-fusion-size={BASELINE_MAX_SIZE}",
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

    # 找到基准线下的最优解（哪怕不融合，我们也取表现最好的那一组作为对比基线）
    best = df.loc[df['latency'].idxmin()]
    
    print("\n" + "="*60)
    print(f"🛑 BASELINE RESULT (NO FUSION)")
    print(f"   Best Alpha: {best['alpha']}")
    print(f"   Best Beta:  {best['beta']}")
    print(f"   ---------------------------")
    print(f"   Blocks:     {int(best['blocks'])} (Should be ~190)")
    print(f"   Time Steps: {int(best['steps'])}")
    print(f"   Latency:    {int(best['latency'])}")
    print("="*60)

if __name__ == "__main__":
    main()
