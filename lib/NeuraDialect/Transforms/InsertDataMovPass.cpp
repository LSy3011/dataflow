#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct InsertDataMovPass
    : public PassWrapper<InsertDataMovPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertDataMovPass)

  StringRef getArgument() const override { return "insert-data-mov"; }
  StringRef getDescription() const override {
    return "Insert neura.data_mov for data dependencies between compute operations.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(&getContext());

    // 收集需要替换的操作数位置 (Operation*, OperandIndex)
    SmallVector<std::pair<Operation*, unsigned>> operandsToReplace;

    func.walk([&](Operation *op) {
      // 1. 跳过自身就是搬运指令的操作 (防止无限递归)
      if (isa<neura::DataMovOp>(op) || isa<neura::CtrlMovOp>(op)) return;
      
      // 2. 跳过 Terminator (如 return/yield)，通常不需要搬运
      if (op->hasTrait<OpTrait::IsTerminator>()) return;

      // 3. 检查所有操作数
      for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        Value operand = op->getOperand(i);
        Operation *defOp = operand.getDefiningOp();

        // 只有当操作数由另一个操作产生时（非函数参数 BlockArgument），才插入搬运
        if (defOp) {
            // 避免重复包装：如果生产者已经是 DataMov，跳过
            if (isa<neura::DataMovOp>(defOp)) continue;

            // [关键修改]：不再检查 defOp 的 Dialect 必须是 neura
            // 只要是本地定义的值，都视为需要通过 NoC 搬运的数据依赖
            operandsToReplace.push_back({op, i});
        }
      }
    });

    // 执行插入
    for (auto pair : operandsToReplace) {
      Operation *op = pair.first;
      unsigned idx = pair.second;
      Value originalVal = op->getOperand(idx);

      builder.setInsertionPoint(op); // 在消费者之前插入
      
      // 创建 DataMovOp
      auto mov = builder.create<neura::DataMovOp>(
          op->getLoc(), 
          originalVal.getType(), 
          originalVal
      );

      // 替换操作数：让 op 使用 mov 的结果，而不是原始值
      op->setOperand(idx, mov.getResult());
    }
  }
};
} // namespace

namespace mlir {
namespace neura {
std::unique_ptr<Pass> createInsertDataMovPass() {
  return std::make_unique<InsertDataMovPass>();
}
} // namespace neura
} // namespace mlir
