#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/ADT/DenseSet.h"
#include <iostream>
#include <numeric>

#define DEBUG_TYPE "static-feature-extraction"

namespace mlir {
namespace neura {

struct StaticFeatureExtractionPass
    : public PassWrapper<StaticFeatureExtractionPass, OperationPass<func::FuncOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StaticFeatureExtractionPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, 
                    memref::MemRefDialect, 
                    arith::ArithDialect,
                    vector::VectorDialect>();
  }

  StringRef getArgument() const override { return "static-feature-extraction"; }
  StringRef getDescription() const override { return "Extract static features and annotate tasks in IR"; }

  // 辅助函数：估算操作的 Latency
  unsigned getLatencyEstimate(Operation *op) {
    if (isa<arith::DivFOp, arith::DivSIOp>(op)) return 10;
    if (isa<memref::LoadOp, affine::AffineLoadOp>(op)) return 4;
    if (isa<vector::TransferReadOp, vector::TransferWriteOp>(op)) return 4; 
    if (op->getName().getStringRef().contains("exp")) return 10;
    return 1;
  }

  // --- 核心函数：分析单个 Task 并写入属性 ---
  void analyzeAndAnnotateTask(Operation *taskOp) {
    long long n_vec = 0;
    long long n_scalar = 0;
    long long n_compute = 0;
    long long n_memory = 0;
    long long n_affine_access = 0;
    long long v_live_out_bits = 0; 
    
    unsigned loop_depth = 0;
    unsigned max_latency = 0;
    long long fan_out_sum = 0;
    long long op_count = 0; // 绝对指令数

    // 用于跟踪当前 Task 访问了哪些 Buffer
    llvm::DenseSet<Value> accessedBuffers;

    // 递归遍历 Task 内部的所有指令
    taskOp->walk([&](Operation *op) {
        if (op == taskOp) return; 
        op_count++;

        // 1. Vectorization Rate
        bool isVec = false;
        for (auto t : op->getResultTypes()) if (isa<VectorType>(t)) isVec = true;
        if (isVec) n_vec++; else n_scalar++;

        // 2. Arithmetic Intensity
        if (op->getDialect()->getNamespace() == "arith" || 
            op->getDialect()->getNamespace() == "math") {
            n_compute++;
        }
        
        // 3. 统计访存 & 收集 Buffer
        if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
            n_memory++; n_affine_access++; accessedBuffers.insert(loadOp.getMemRef());
        } 
        else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
            n_memory++; n_affine_access++; accessedBuffers.insert(storeOp.getMemRef());
            v_live_out_bits += 32; 
        } 
        else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
            n_memory++; accessedBuffers.insert(loadOp.getMemRef());
        } 
        else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
            n_memory++; accessedBuffers.insert(storeOp.getMemRef());
            v_live_out_bits += 32;
        }
        else if (auto vecRead = dyn_cast<vector::TransferReadOp>(op)) {
            n_memory++; n_affine_access++; accessedBuffers.insert(vecRead.getSource());
        }
        else if (auto vecWrite = dyn_cast<vector::TransferWriteOp>(op)) {
            n_memory++; n_affine_access++; accessedBuffers.insert(vecWrite.getSource());
            if (auto vecType = vecWrite.getVector().getType().dyn_cast<VectorType>()) {
                v_live_out_bits += vecType.getNumElements() * vecType.getElementTypeBitWidth();
            }
        }

        // 5. Loop Depth
        if (isa<affine::AffineForOp>(op)) loop_depth++;

        // 6. RecMII Latency 估算
        unsigned lat = getLatencyEstimate(op);
        if (lat > max_latency) max_latency = lat;

        // 7. Fan Out
        for (auto res : op->getResults()) {
            fan_out_sum += std::distance(res.use_begin(), res.use_end());
        }
    });

    // 4. 计算 Static Buffer Size
    long long s_buffer = 0;
    for (Value buf : accessedBuffers) {
        auto type = buf.getType().dyn_cast<MemRefType>();
        if (type && type.hasStaticShape()) {
            s_buffer += type.getNumElements() * type.getElementTypeBitWidth();
        }
    }

    // 计算指标
    double r_vec = (double)n_vec / (n_vec + n_scalar + 1e-9);
    double i_arith = (double)n_compute / (n_memory + 1e-9);
    double r_affine = (double)n_affine_access / (n_memory + 1e-9);
    double fan_avg = (double)fan_out_sum / (op_count + 1e-9);
    double rec_mii = (max_latency > 5) ? 10.0 : 1.0; 

    // --- 写入 IR 属性 ---
    OpBuilder builder(taskOp->getContext());
    taskOp->setAttr("neura.r_vec", builder.getF64FloatAttr(r_vec));
    taskOp->setAttr("neura.i_arith", builder.getF64FloatAttr(i_arith));
    taskOp->setAttr("neura.r_affine", builder.getF64FloatAttr(r_affine));
    taskOp->setAttr("neura.fan_avg", builder.getF64FloatAttr(fan_avg));
    taskOp->setAttr("neura.s_buffer", builder.getI64IntegerAttr(s_buffer));
    taskOp->setAttr("neura.v_live_out", builder.getI64IntegerAttr(v_live_out_bits));
    taskOp->setAttr("neura.rec_mii", builder.getF64FloatAttr(rec_mii));
    taskOp->setAttr("neura.l_depth", builder.getI32IntegerAttr(loop_depth));
    
    // [关键新增] 显式标注指令数，用于后续时延计算和融合限制
    taskOp->setAttr("neura.op_count", builder.getI64IntegerAttr(op_count));
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    bool hasAnnotatedTasks = false;
    func.walk([&](affine::AffineForOp op) {
        if (op->hasAttr("neura.task_id")) {
            analyzeAndAnnotateTask(op);
            hasAnnotatedTasks = true;
        }
    });

    if (!hasAnnotatedTasks) {
        for (Operation &op : func.getBody().front()) {
            if (isa<affine::AffineForOp>(op)) {
                analyzeAndAnnotateTask(&op);
            }
        }
    }

    llvm::errs() << "--------------------------------------------------\n";
    llvm::errs() << "Static Features Extracted & Annotated\n";
    llvm::errs() << "--------------------------------------------------\n";
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createStaticFeatureExtractionPass() {
  return std::make_unique<StaticFeatureExtractionPass>();
}

} // namespace neura
} // namespace mlir
