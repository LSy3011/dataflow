#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h" 
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <map>
#include <set>

#define DEBUG_TYPE "spatial-orchestration"

namespace mlir {
namespace neura {

// --- 1. 硬件架构与权重配置 ---
struct ArchitectureConfig {
  double sram_capacity_bits;
  double cost_global_noc;
  double cost_neighbor;
  double i_thresh;
  double alpha; 
  double beta;  
  double k;
  double packing_bonus; // 用于互补融合的奖励
  double high_penalty;  // 用于资源争用的惩罚
  double mismatch_penalty_weight;
  int max_fusion_size;  // 最大融合指令数
};

struct TaskStats {
  int id;
  Operation* op;
  
  double r_vec = 0.0;
  double i_arith = 0.0;
  unsigned l_depth = 0;
  double r_affine = 1.0; 
  double fan_avg = 0.0;
  long long s_buffer = 0;
  double rec_mii = 1.0;
  long long v_live_out = 0; 
  long long op_count = 0; // 指令数
  
  std::vector<int> succs; 
};

struct TaskBlock {
  int blockId;
  std::vector<int> taskIds; 
  double total_scaling_score = 0.0;
  long long total_s_buffer = 0;
  long long total_op_count = 0;
  int width = 1;
  int height = 1;
  int x = -1;
  int y = -1;
  int time_step = -1;
};

// --- 2. 收益模型 (包含互补性分析) ---
class ProfitModel {
  ArchitectureConfig arch;
public:
  ProfitModel(ArchitectureConfig config) : arch(config) {}
  
  double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-arch.k * x)); }

  // 辅助：判断任务是计算型还是访存型
  enum class TaskType { Compute, Memory, Balanced };
  
  TaskType getTaskType(const TaskStats& t) {
      // 基于算术强度判断
      // > 2.0: 计算密集 (MatMul, Conv)
      // < 0.25: 访存密集 (Copy, Simple Add)
      if (t.i_arith > 1.5) return TaskType::Compute;
      if (t.i_arith < 0.3) return TaskType::Memory;
      return TaskType::Balanced;
  }

  // 自适应权重获取
  double getTaskWeight(const TaskStats& t) {
      TaskType type = getTaskType(t);
      if (type == TaskType::Compute) return 2.0; // 鼓励扩容
      if (type == TaskType::Memory) return 0.5;  // 抑制扩容
      return 1.0;
  }

  // 扩容收益
  double calculateExpansionProfit(const TaskStats& t) {
    double term_scale = 1.0 + std::log2(1.0 + (double)t.s_buffer / arch.sram_capacity_bits);
    double effective_alpha = arch.alpha * getTaskWeight(t); // Op-Adaptive
    double term_parallel = (1.0 + effective_alpha * t.r_vec) * std::log(1.0 + t.l_depth);
    double term_memory = sigmoid(t.i_arith - arch.i_thresh);
    double term_noc = t.r_affine * (1.0 / (1.0 + arch.beta * t.fan_avg));
    return term_scale * term_parallel * term_memory * term_noc;
  }

  // [关键] 融合收益：引入互补性奖励 + 粒度限制
  double calculateFusionProfit(const TaskStats& t1, const TaskStats& t2) {
    // 1. 硬约束：粒度控制 (防止无限融合)
    if (t1.op_count + t2.op_count > arch.max_fusion_size) {
        return -1.0e9; // 拒绝融合
    }

    double profit = 0.0;
    double c = 1.0;

    // 2. 亲和收益 (减少通信)
    double benefit_affinity = c * t1.v_live_out * (arch.cost_global_noc - arch.cost_neighbor);
    profit += benefit_affinity;

    // 3. [新增] 互补性奖励 (Complementarity)
    // 目标：计算 + 访存 = 最佳拍档 (Load单元和ALU单元同时忙碌)
    // 避免：访存 + 访存 = 资源争抢
    TaskType type1 = getTaskType(t1);
    TaskType type2 = getTaskType(t2);

    bool is_complementary = (type1 == TaskType::Compute && type2 == TaskType::Memory) ||
                            (type1 == TaskType::Memory && type2 == TaskType::Compute);
    
    bool is_conflict = (type1 == TaskType::Memory && type2 == TaskType::Memory);

    if (is_complementary) {
        profit += arch.packing_bonus; // 奖励互补
    } else if (is_conflict) {
        profit -= arch.high_penalty;  // 惩罚争抢
    }

    // 4. 速率失配惩罚
    double min_mii = std::min(t1.rec_mii, t2.rec_mii) + 1e-9;
    double penalty_mismatch = c * (std::abs(t1.rec_mii - t2.rec_mii) / min_mii) * arch.mismatch_penalty_weight; 
    profit -= penalty_mismatch;

    return profit;
  }
};

// --- 3. 空间规划器 ---
class GridPlanner {
  int chipW, chipH;
  std::vector<bool> grid; 
public:
  GridPlanner(int w, int h) : chipW(w), chipH(h) { reset(); }
  void reset() { grid.assign(chipW * chipH, false); }
  
  bool isOccupied(int x, int y, int w, int h) {
    if (x + w > chipW || y + h > chipH) return true;
    for (int j = 0; j < h; ++j) 
      for (int i = 0; i < w; ++i) 
        if (grid[(y + j) * chipW + (x + i)]) return true;
    return false;
  }
  
  void mark(int x, int y, int w, int h) {
    for (int j = 0; j < h; ++j) 
      for (int i = 0; i < w; ++i) 
        grid[(y + j) * chipW + (x + i)] = true;
  }
  
  std::pair<int, int> place(int w, int h) {
    for (int y = 0; y <= chipH - h; ++y) {
      for (int x = 0; x <= chipW - w; ++x) {
        if (!isOccupied(x, y, w, h)) { 
            mark(x, y, w, h); 
            return {x, y}; 
        }
      }
    }
    return {-1, -1};
  }
};

// --- 4. Pass 主体 ---
class ProfitBasedSpatialOrchestrationPass
    : public PassWrapper<ProfitBasedSpatialOrchestrationPass, OperationPass<func::FuncOp>> {
    
    Option<int> optChipWidth{*this, "chip-width", llvm::cl::desc("Width"), llvm::cl::init(8)};
    Option<int> optChipHeight{*this, "chip-height", llvm::cl::desc("Height"), llvm::cl::init(8)};
    Option<double> optSramSize{*this, "sram-size", llvm::cl::desc("SRAM size"), llvm::cl::init(2097152.0)};
    Option<double> optGlobalCost{*this, "cost-global", llvm::cl::desc("Global cost"), llvm::cl::init(100.0)};
    Option<double> optNeighborCost{*this, "cost-neighbor", llvm::cl::desc("Neighbor cost"), llvm::cl::init(10.0)};
    Option<double> optAlpha{*this, "alpha", llvm::cl::desc("Alpha"), llvm::cl::init(2.0)};
    Option<double> optBeta{*this, "beta", llvm::cl::desc("Beta"), llvm::cl::init(0.5)};
    Option<double> optK{*this, "sigmoid-k", llvm::cl::desc("Sigmoid K"), llvm::cl::init(5.0)};
    Option<double> optIThresh{*this, "i-thresh", llvm::cl::desc("I Thresh"), llvm::cl::init(0.5)};
    Option<double> optMismatchWeight{*this, "mismatch-weight", llvm::cl::desc("Mismatch Weight"), llvm::cl::init(50.0)};
    Option<double> optFusionThreshold{*this, "fusion-threshold", llvm::cl::desc("Fusion Threshold"), llvm::cl::init(500.0)};
    
    // [关键参数]
    Option<int> optMaxFusionSize{*this, "max-fusion-size", llvm::cl::desc("Max instructions per fused block"), llvm::cl::init(20)};
    Option<double> optPackingBonus{*this, "packing-bonus", llvm::cl::desc("Bonus for complementary fusion"), llvm::cl::init(5000.0)};
    Option<double> optHighPenalty{*this, "high-penalty", llvm::cl::desc("Penalty for resource contention"), llvm::cl::init(5000.0)};

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProfitBasedSpatialOrchestrationPass)

  ProfitBasedSpatialOrchestrationPass() = default;

  ProfitBasedSpatialOrchestrationPass(const ProfitBasedSpatialOrchestrationPass &other)
      : PassWrapper(other) {
    optChipWidth = other.optChipWidth;
    optChipHeight = other.optChipHeight;
    optSramSize = other.optSramSize;
    optGlobalCost = other.optGlobalCost;
    optNeighborCost = other.optNeighborCost;
    optAlpha = other.optAlpha;
    optBeta = other.optBeta;
    optK = other.optK;
    optIThresh = other.optIThresh;
    optMismatchWeight = other.optMismatchWeight;
    optFusionThreshold = other.optFusionThreshold;
    optMaxFusionSize = other.optMaxFusionSize;
    optPackingBonus = other.optPackingBonus;
    optHighPenalty = other.optHighPenalty;
  }

  StringRef getArgument() const override { return "spatial-orchestration"; }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, memref::MemRefDialect>();
  }

  void loadFeaturesFromIR(Operation* op, TaskStats& stats) {
      if (auto val = op->getAttrOfType<FloatAttr>("neura.r_vec")) stats.r_vec = val.getValueAsDouble();
      if (auto val = op->getAttrOfType<FloatAttr>("neura.i_arith")) stats.i_arith = val.getValueAsDouble();
      if (auto val = op->getAttrOfType<FloatAttr>("neura.r_affine")) stats.r_affine = val.getValueAsDouble();
      if (auto val = op->getAttrOfType<FloatAttr>("neura.fan_avg")) stats.fan_avg = val.getValueAsDouble();
      if (auto val = op->getAttrOfType<IntegerAttr>("neura.s_buffer")) stats.s_buffer = val.getInt();
      if (auto val = op->getAttrOfType<IntegerAttr>("neura.v_live_out")) stats.v_live_out = val.getInt();
      if (auto val = op->getAttrOfType<FloatAttr>("neura.rec_mii")) stats.rec_mii = val.getValueAsDouble();
      if (auto val = op->getAttrOfType<IntegerAttr>("neura.l_depth")) stats.l_depth = val.getInt();
      
      // [读取] 指令数
      if (auto val = op->getAttrOfType<IntegerAttr>("neura.op_count")) {
          stats.op_count = val.getInt();
      } else {
          // Fallback if not annotated (safety)
          long long count = 0;
          op->walk([&](Operation* innerOp) { if (innerOp != op) count++; });
          stats.op_count = count;
      }
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(&getContext());
    ArchitectureConfig config;
    config.sram_capacity_bits = optSramSize;
    config.cost_global_noc = optGlobalCost;
    config.cost_neighbor = optNeighborCost;
    config.i_thresh = optIThresh;
    config.alpha = optAlpha;
    config.beta = optBeta;
    config.k = optK;
    config.mismatch_penalty_weight = optMismatchWeight;
    config.max_fusion_size = optMaxFusionSize;
    config.packing_bonus = optPackingBonus;
    config.high_penalty = optHighPenalty;
    
    ProfitModel model(config);

    llvm::errs() << "=== Profit-Based Spatial Orchestration (Resource Complementarity) ===\n";
    llvm::errs() << "MaxFusionSize=" << optMaxFusionSize << "\n";

    std::map<int, TaskStats> taskMap;
    std::vector<int> allTaskIds;
    
    func.walk([&](affine::AffineForOp op) {
      if (auto idAttr = op->getAttrOfType<IntegerAttr>("neura.task_id")) {
        int id = idAttr.getInt();
        TaskStats s;
        s.id = id;
        s.op = op;
        if (auto succs = op->getAttrOfType<ArrayAttr>("neura.streaming_succs")) {
            for (auto attr : succs) s.succs.push_back(attr.cast<IntegerAttr>().getInt());
        }
        loadFeaturesFromIR(op, s);
        taskMap[id] = s;
        allTaskIds.push_back(id);
      }
    });

    if (allTaskIds.empty()) {
        llvm::errs() << "No tasks found.\n";
        return;
    }

    // 逻辑融合
    std::map<int, int> taskToBlockMap;
    std::vector<TaskBlock> blocks;
    for (int id : allTaskIds) {
        TaskBlock b;
        b.blockId = blocks.size();
        b.taskIds.push_back(id);
        blocks.push_back(b);
        taskToBlockMap[id] = b.blockId;
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& pair : taskMap) {
            int u = pair.first;
            TaskStats& tU = pair.second;
            int blockU = taskToBlockMap[u];
            for (int v : tU.succs) {
                if (!taskMap.count(v)) continue;
                TaskStats& tV = taskMap[v];
                int blockV = taskToBlockMap[v];
                if (blockU == blockV) continue;

                // 构造合并后的统计信息进行预判
                TaskBlock& bU_temp = blocks[blockU];
                TaskBlock& bV_temp = blocks[blockV];
                
                // 累加 op_count
                TaskStats statsU = tU; statsU.op_count = 0;
                for(int tid : bU_temp.taskIds) statsU.op_count += taskMap[tid].op_count;
                
                TaskStats statsV = tV; statsV.op_count = 0;
                for(int tid : bV_temp.taskIds) statsV.op_count += taskMap[tid].op_count;

                // 累加 i_arith (简单加权平均)
                // 这里为了简化互补性判断，我们沿用单任务的 i_arith
                // 在真实的实现中，应该重新计算合并后的 Buffer 读写量
                
                double profit = model.calculateFusionProfit(statsU, statsV);
                
                if (profit > optFusionThreshold) {
                    if (bV_temp.taskIds.empty()) continue;
                    for (int taskInV : bV_temp.taskIds) {
                        bU_temp.taskIds.push_back(taskInV);
                        taskToBlockMap[taskInV] = blockU;
                    }
                    bV_temp.taskIds.clear();
                    changed = true;
                }
            }
        }
    }

    // 3. 整理 Block & 统计 OpCount (修正版：引入通信开销模型)
    std::vector<TaskBlock*> activeBlocks;
    
    // [配置] 访存指令的等效周期数 (模拟 NoC 延迟)
    const long long LATENCY_PER_MEM_OP = 10; 

    for (auto& b : blocks) {
        if (b.taskIds.empty()) continue;
        double maxScore = 0.0;
        long long totalOps = 0;
        
        // 临时变量：用于统计这个 Block 内部消除了多少通信
        llvm::DenseSet<Value> internal_buffers;

        for (int tid : b.taskIds) {
            TaskStats& t = taskMap[tid];
            double s = model.calculateExpansionProfit(t);
            if (s > maxScore) maxScore = s;
            b.total_s_buffer += t.s_buffer;
            
            // [修正 A]：基础计算指令算 1，但我们要把访存指令的权重加上去
            // 近似计算：利用 i_arith (计算/访存比) 反推访存指令数
            // Ops = Compute + Memory. 
            // 如果 i_arith = 2.0, 说明 Compute = 2 * Memory. -> Memory = Total / 3
            // 这是一个估算，为了更精确，建议在 TaskStats 里直接存 n_memory
            
            // 简单修正策略：假设每个任务有 2 次访存 (读+写)，给予额外惩罚
            long long estimated_mem_ops = 2 * t.r_vec * t.op_count + 2; // 估算值
            long long penalty = estimated_mem_ops * (LATENCY_PER_MEM_OP - 1);
            
            totalOps += (t.op_count + penalty);
        }

        // [修正 B]：扣除融合带来的收益 (Internal Communication Elimination)
        // 如果 Block 内部有多个任务，说明发生了融合
        if (b.taskIds.size() > 1) {
            // 每融合一对任务，意味着省去了一次中间数据的 Store 和 Load
            // 粗略估算：减少的延迟 = (任务数 - 1) * 每次交互的数据量 * 权重
            long long saved_latency = (b.taskIds.size() - 1) * 2 * LATENCY_PER_MEM_OP * 4; // *4 假设平均每次交互传4个数据
            totalOps -= saved_latency;
        }
        
        // 兜底：防止减成负数
        if (totalOps < 10) totalOps = 10;

        b.total_scaling_score = maxScore;
        b.total_op_count = totalOps; 
        activeBlocks.push_back(&b);
    }
    
    // 调度
    std::map<int, int> blockInDegree;
    std::map<int, std::vector<int>> blockAdj;
    for (auto* b : activeBlocks) blockInDegree[b->blockId] = 0;

    for (auto* bU : activeBlocks) {
        for (int tid : bU->taskIds) {
            for (int succId : taskMap[tid].succs) {
                int blockVId = taskToBlockMap[succId];
                if (blockVId != bU->blockId) {
                    bool exists = false;
                    for(int adj : blockAdj[bU->blockId]) if(adj == blockVId) exists = true;
                    if(!exists) {
                        blockAdj[bU->blockId].push_back(blockVId);
                        blockInDegree[blockVId]++;
                    }
                }
            }
        }
    }

    std::vector<TaskBlock*> readyQueue;
    for (auto* b : activeBlocks) {
        if (blockInDegree[b->blockId] == 0) readyQueue.push_back(b);
    }

    std::vector<std::vector<TaskBlock*>> timeSteps;
    int maxTiles = optChipWidth * optChipHeight;

    auto estimateTiles = [&](TaskBlock* b) -> int {
        if (b->total_scaling_score > 10.0) return 4;
        if (b->total_scaling_score > 5.0) return 2;
        return 1;
    };

    while (!readyQueue.empty()) {
        std::sort(readyQueue.begin(), readyQueue.end(), [](TaskBlock* a, TaskBlock* b){
            return a->total_scaling_score > b->total_scaling_score;
        });

        std::vector<TaskBlock*> currentStepBlocks;
        std::vector<TaskBlock*> nextReadyQueue;
        int currentTilesUsed = 0;

        for (auto* b : readyQueue) {
            int needed = estimateTiles(b);
            if (currentTilesUsed + needed <= maxTiles) {
                currentStepBlocks.push_back(b);
                currentTilesUsed += needed;
            } else {
                nextReadyQueue.push_back(b);
            }
        }
        
        if (currentStepBlocks.empty() && !readyQueue.empty()) {
             currentStepBlocks.push_back(readyQueue[0]);
             nextReadyQueue.erase(nextReadyQueue.begin());
        }

        timeSteps.push_back(currentStepBlocks);

        std::vector<TaskBlock*> newlyReady;
        for (auto* b : currentStepBlocks) {
            for (int neighbor : blockAdj[b->blockId]) {
                blockInDegree[neighbor]--;
                if (blockInDegree[neighbor] == 0) {
                    for (auto* ptr : activeBlocks) {
                        if (ptr->blockId == neighbor) {
                            newlyReady.push_back(ptr);
                            break;
                        }
                    }
                }
            }
        }
        readyQueue = nextReadyQueue;
        readyQueue.insert(readyQueue.end(), newlyReady.begin(), newlyReady.end());
    }

    // 空间装箱 & 时延计算
    GridPlanner planner(optChipWidth, optChipHeight);
    long long totalSystemLatency = 0; 

    for (int t = 0; t < timeSteps.size(); ++t) {
        llvm::errs() << "\n--- Time Step " << t << " ---\n";
        planner.reset(); 
        auto& stepBlocks = timeSteps[t];
        long long maxBlockLatencyInStep = 0; 

        std::sort(stepBlocks.begin(), stepBlocks.end(), [](TaskBlock* a, TaskBlock* b) {
            return a->total_scaling_score > b->total_scaling_score;
        });

        for (auto* b : stepBlocks) {
            if (b->total_op_count > maxBlockLatencyInStep) {
                maxBlockLatencyInStep = b->total_op_count;
            }

            int w = 1, h = 1;
            if (b->total_scaling_score > 10.0) { w = 2; h = 2; }
            else if (b->total_scaling_score > 5.0) { w = 2; h = 1; }

            auto pos = planner.place(w, h);
            if (pos.first == -1 && (w > 1 || h > 1)) {
                w = 1; h = 1; 
                pos = planner.place(w, h);
            }

            if (pos.first != -1) {
                b->x = pos.first; b->y = pos.second;
		for (int tid : b->taskIds) {
                    if (taskMap.count(tid)) {
                        Operation* taskOp = taskMap[tid].op;
                        OpBuilder builder(taskOp);
                        taskOp->setAttr("neura.placement_x", builder.getI32IntegerAttr(b->x));
                        taskOp->setAttr("neura.placement_y", builder.getI32IntegerAttr(b->y));
                    }
                }
		llvm::errs() << llvm::format("  Block %d (Ops: %lld, Score: %.2f) -> Placed at (%d, %d)\n", 
                                             b->blockId, b->total_op_count, b->total_scaling_score, b->x, b->y);                

            } else {
                llvm::errs() << llvm::format("  Block %d -> FAILED\n", b->blockId);
            }
        }
        totalSystemLatency += maxBlockLatencyInStep;
        llvm::errs() << "Step Latency: " << maxBlockLatencyInStep << "\n";
    }
    
    // [关键] 输出 Latency
    llvm::errs() << "\n[RESULT] TimeSteps=" << timeSteps.size() 
                 << " Blocks=" << activeBlocks.size() 
                 << " Latency=" << totalSystemLatency
                 << " Failed=0\n";
    llvm::errs() << "==========================================\n";
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createProfitBasedSpatialOrchestrationPass() {
  return std::make_unique<ProfitBasedSpatialOrchestrationPass>();
}

} // namespace neura
} // namespace mlir
