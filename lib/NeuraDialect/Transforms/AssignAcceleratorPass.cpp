#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

// 引入 TableGen 生成的 Pass 定义
#define GEN_PASS_DEF_ASSIGNACCELERATOR
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
struct AssignAcceleratorPass
    : public PassWrapper<AssignAcceleratorPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignAcceleratorPass)

  StringRef getArgument() const override { return "assign-accelerator"; }
  StringRef getDescription() const override {
    return "Unconditionally tags functions as accelerator='neura' for backend codegen.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(&getContext());

    // 遍历所有 func.func 操作
    module.walk([&](func::FuncOp func) {
      // 1. 跳过外部函数声明（例如 printf, malloc 等）
      if (func.isExternal()) return;

      // 2. 强制设置属性
      // 关键修改：直接使用字符串字面量 "accelerator" 和 "neura"
      // 这必须与 GenerateCodePass.cpp 第 496 行的检查完全一致！
      func->setAttr("accelerator", builder.getStringAttr("neura"));
    });
  }
};
} // namespace

namespace mlir {
namespace neura {
std::unique_ptr<Pass> createAssignAcceleratorPass() {
  return std::make_unique<AssignAcceleratorPass>();
}
} // namespace neura
} // namespace mlir
