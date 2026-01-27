#!/bin/bash

# ================= 配置区域 (基于 DSE 结果) =================
BEST_ALPHA=0.5
BEST_BETA=0.5
BEST_FUSION=0.0
# ==========================================================

COMPILER="../../build/tools/mlir-neura-opt/mlir-neura-opt"
INPUT_FILE="../../test/samples/lenet/lenet_affine.mlir"

echo "=== Experiment 05: LeNet Baseline vs Fusion ==="

# 1. 运行 Baseline (通过设置超高阈值 1e9 和 Size=1 来强制关闭融合)
echo "[1/2] Running Baseline (No Fusion)..."
$COMPILER $INPUT_FILE \
  --static-feature-extraction \
  --task-dependency-analysis \
  --spatial-orchestration="chip-width=4 chip-height=4 alpha=$BEST_ALPHA beta=$BEST_BETA fusion-threshold=1000000000.0 max-fusion-size=1" \
  --negotiated-routing="width=4 height=4 max-iter=50" \
  -o lenet_baseline.mlir > lenet_baseline.log 2>&1

echo "Done. Log: lenet_baseline.log"

# 2. 运行 Fusion (使用最优参数开启融合)
echo "[2/2] Running Fusion (Optimized)..."
$COMPILER $INPUT_FILE \
  --static-feature-extraction \
  --task-dependency-analysis \
  --spatial-orchestration="chip-width=4 chip-height=4 alpha=$BEST_ALPHA beta=$BEST_BETA fusion-threshold=$BEST_FUSION max-fusion-size=20" \
  --negotiated-routing="width=4 height=4 max-iter=50" \
  -o lenet_fusion.mlir > lenet_fusion.log 2>&1

echo "Done. Log: lenet_fusion.log"
