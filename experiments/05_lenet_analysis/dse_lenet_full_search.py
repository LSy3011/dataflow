import subprocess
import re
import pandas as pd
import os
import numpy as np

# --- 配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
COMPILER_EXEC = os.path.join(PROJECT_ROOT, "build", "tools", "mlir-neura-opt", "mlir-neura-opt")
TEST_FILE = os.path.join(PROJECT_ROOT, "test", "samples", "lenet", "lenet_affine.mlir")

# --- 搜索空间 ---
ALPHA_RANGE = [0.5, 1.5, 2.5]
BETA_RANGE = [0.5, 2.5, 4.5]
# 融合阈值扫描: 0.0 代表激进融合, 1e9 代表不融合
FUSION_VALUES = [0.0, 100.0, 500.0, 2000.0, 5000.0, 10000.0]

def run_compiler(alpha, beta, fusion):
    cmd = [
        COMPILER_EXEC,
        "-affine-super-vectorize=virtual-vector-size=4",
        "--task-dependency-analysis",
        "--static-feature-extraction",
        f"--spatial-orchestration=chip-width=4 chip-height=4 alpha={alpha:.2f} beta={beta:.2f} fusion-threshold={fusion:.1f} max-fusion-size=20",
        "--negotiated-routing=width=4 height=4 max-iter=50",
        TEST_FILE
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        match = re.search(r"\[RESULT\] TimeSteps=(\d+) Blocks=(\d+) Latency=(\d+)", res.stderr)
        if match:
            return int(match.group(3)), int(match.group(2)) # Latency, Blocks
    except Exception:
        pass
    return float('inf'), 0

def main():
    if not os.path.exists(COMPILER_EXEC):
        print("Compiler not found.")
        return

    results = []
    print(f"=== Running LeNet DSE Search ===")
    
    total = len(ALPHA_RANGE) * len(BETA_RANGE) * len(FUSION_VALUES)
    count = 0

    for f in FUSION_VALUES:
        for a in ALPHA_RANGE:
            for b in BETA_RANGE:
                count += 1
                lat, blks = run_compiler(a, b, f)
                if lat != float('inf'):
                    print(f"[{count}/{total}] A={a} B={b} F={f} -> Lat={lat}")
                    results.append({"Alpha": a, "Beta": b, "Fusion": f, "Latency": lat, "Blocks": blks})

    if results:
        df = pd.DataFrame(results)
        df.to_csv("lenet_full_dse.csv", index=False)
        best = df.loc[df['Latency'].idxmin()]
        print("\n" + "="*40)
        print(f"🏆 Best Configuration Found:")
        print(f"   Alpha: {best['Alpha']}, Beta: {best['Beta']}, Fusion: {best['Fusion']}")
        print(f"   Latency: {int(best['Latency'])}")
        print("="*40)

if __name__ == "__main__":
    main()

