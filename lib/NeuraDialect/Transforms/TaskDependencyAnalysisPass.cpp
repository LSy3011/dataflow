#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h" // 必须包含这个
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>

namespace mlir {
namespace neura {

struct TaskInfo {
  int id;
  Operation* op;
  llvm::DenseSet<Value> readBuffers;
  llvm::DenseSet<Value> writtenBuffers;
  std::vector<int> streamingSuccs;
  std::vector<int> conflictSuccs;
};

class TaskDependencyAnalysisPass
    : public PassWrapper<TaskDependencyAnalysisPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TaskDependencyAnalysisPass)

  StringRef getArgument() const override { return "task-dependency-analysis"; }
  StringRef getDescription() const override { return "Analyze dependencies (Scalar & Vector) and annotate tasks"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, memref::MemRefDialect, scf::SCFDialect, vector::VectorDialect>();
  }

  // --- 关键修复：增加 Vector 和 Copy 的支持 ---
  void collectMemoryAccess(Operation *op, TaskInfo &info) {
    op->walk([&](Operation *innerOp) {
      // 1. Affine / Standard Load
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(innerOp)) {
        info.readBuffers.insert(loadOp.getMemRef());
      } else if (auto loadOp = dyn_cast<memref::LoadOp>(innerOp)) {
        info.readBuffers.insert(loadOp.getMemRef());
      } 
      // 2. Vector Transfer Read (你的 IR 主要用这个)
      else if (auto vecRead = dyn_cast<vector::TransferReadOp>(innerOp)) {
        info.readBuffers.insert(vecRead.getSource());
      }
      // 3. Affine / Standard Store
      else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(innerOp)) {
        info.writtenBuffers.insert(storeOp.getMemRef());
      } else if (auto storeOp = dyn_cast<memref::StoreOp>(innerOp)) {
        info.writtenBuffers.insert(storeOp.getMemRef());
      } 
      // 4. Vector Transfer Write (你的 IR 主要用这个)
      else if (auto vecWrite = dyn_cast<vector::TransferWriteOp>(innerOp)) {
        info.writtenBuffers.insert(vecWrite.getSource());
      }
      // 5. MemRef Copy (读 source, 写 target)
      else if (auto copyOp = dyn_cast<memref::CopyOp>(innerOp)) {
        info.readBuffers.insert(copyOp.getSource());
        info.writtenBuffers.insert(copyOp.getTarget());
      }
    });
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(&getContext());

    llvm::errs() << "==================================================================\n";
    llvm::errs() << "Task Analysis: Analyzing & Annotating IR for " << func.getName() << "\n";
    llvm::errs() << "==================================================================\n";

    std::vector<TaskInfo> tasks;
    int taskIdCounter = 0;

    // 1. 识别任务
    for (Operation &op : func.getBody().front()) {
      if (isa<affine::AffineForOp>(op) || isa<scf::ForOp>(op)) {
        TaskInfo info;
        info.id = taskIdCounter++;
        info.op = &op;
        collectMemoryAccess(&op, info);
        tasks.push_back(info);
      }
    }

    if (tasks.empty()) return;

    // 2. 分析依赖
    int edgeCount = 0;
    for (size_t i = 0; i < tasks.size(); ++i) {
      for (size_t j = i + 1; j < tasks.size(); ++j) {
        auto &taskA = tasks[i];
        const auto &taskB = tasks[j];

        bool isFlowDep = false;
        bool isConflict = false;

        // RAW (Flow Dependency)
        for (Value val : taskA.writtenBuffers) {
          if (taskB.readBuffers.count(val)) { isFlowDep = true; break; }
        }
        
        // WAW / WAR (Conflict)
        if (!isFlowDep) {
           for (Value val : taskA.writtenBuffers) {
               if (taskB.writtenBuffers.count(val)) { isConflict = true; break; }
           }
           if (!isConflict) {
               for (Value val : taskA.readBuffers) {
                   if (taskB.writtenBuffers.count(val)) { isConflict = true; break; }
               }
           }
        }

        if (isFlowDep) {
            taskA.streamingSuccs.push_back(taskB.id);
            edgeCount++;
        } else if (isConflict) {
            taskA.conflictSuccs.push_back(taskB.id);
            edgeCount++;
        }
      }
    }
    llvm::errs() << "Found " << tasks.size() << " tasks and " << edgeCount << " dependencies.\n";

    // 3. 写入 IR
    for (auto &task : tasks) {
        Operation* op = task.op;
        op->setAttr("neura.task_id", builder.getI32IntegerAttr(task.id));

        if (!task.streamingSuccs.empty()) {
            SmallVector<Attribute> succAttrs;
            for (int id : task.streamingSuccs) succAttrs.push_back(builder.getI32IntegerAttr(id));
            op->setAttr("neura.streaming_succs", builder.getArrayAttr(succAttrs));
        }
        if (!task.conflictSuccs.empty()) {
            SmallVector<Attribute> conflictAttrs;
            for (int id : task.conflictSuccs) conflictAttrs.push_back(builder.getI32IntegerAttr(id));
            op->setAttr("neura.waiting_succs", builder.getArrayAttr(conflictAttrs));
        }
    }

    // 4. 生成 DOT
    std::string dotOutput;
    llvm::raw_string_ostream dotStream(dotOutput);
    dotStream << "digraph TaskGraph {\n  rankdir=TB;\n  node [shape=box, style=filled, fillcolor=white];\n";
    for (const auto& t : tasks) {
        dotStream << "  Task" << t.id << " [label=\"Task " << t.id << "\"];\n";
    }
    for (const auto& t : tasks) {
        for (int succId : t.streamingSuccs)
             dotStream << "  Task" << t.id << " -> Task" << succId << " [color=\"#006400\", penwidth=2.0, label=\"Flow\"];\n";
        for (int succId : t.conflictSuccs)
             dotStream << "  Task" << t.id << " -> Task" << succId << " [color=red, style=dashed, label=\"Conflict\"];\n";
    }
    dotStream << "}\n";
    llvm::errs() << "\n[DOT Graph Visualization]\n" << dotOutput << "\n";
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createTaskDependencyAnalysisPass() {
  return std::make_unique<TaskDependencyAnalysisPass>();
}

} // namespace neura
} // namespace mlir
