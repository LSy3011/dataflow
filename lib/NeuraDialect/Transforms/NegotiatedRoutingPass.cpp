#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/NeuraDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include <queue>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <iostream>
#include <algorithm> // 添加这行，解决 std::reverse 找不到的问题

#define DEBUG_TYPE "negotiated-routing"

namespace mlir {
namespace neura {

// --- 1. 基础数据结构 ---

enum Direction { NORTH = 0, SOUTH, EAST, WEST, LOCAL, DIR_COUNT };

struct RoutingLink {
  int id;
  int srcX, srcY;
  int dstX, dstY;
  
  int capacity = 1;          // 物理带宽
  int current_usage = 0;     // 当前使用量
  double history_cost = 0.0; // 历史拥塞代价
  double base_cost = 1.0;    // 基础延迟

  // PathFinder 代价函数
  double getDynamicCost(double pres_fac) const {
    double congestion_penalty = 1.0;
    if (current_usage >= capacity) {
      congestion_penalty = 1.0 + pres_fac * (current_usage + 1 - capacity);
    }
    return (base_cost + history_cost) * congestion_penalty;
  }
};

struct RouterNode {
  int x, y;
  int id; // y * W + x
  RoutingLink* outLinks[DIR_COUNT]; // 邻接表
  
  RouterNode() { for(int i=0; i<DIR_COUNT; ++i) outLinks[i] = nullptr; }
};

struct Net {
  int netId;
  int srcTaskId;
  int dstTaskId;
  int srcNodeId; // Router ID
  int dstNodeId; // Router ID
  std::vector<RoutingLink*> path; // 当前路径
};

// --- 2. 路由资源图 (RRG) ---

class RoutingResourceGraph {
public:
  int width, height;
  std::vector<RouterNode> nodes;
  std::vector<RoutingLink> links;

  RoutingResourceGraph(int w, int h) : width(w), height(h) {
    nodes.resize(width * height);
    links.reserve(width * height * 5);
    
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int u = y * width + x;
        nodes[u].x = x; nodes[u].y = y; nodes[u].id = u;

        // 添加双向连接
        if (y < height - 1) addLink(x, y, x, y + 1, NORTH);
        if (y > 0)          addLink(x, y, x, y - 1, SOUTH);
        if (x < width - 1)  addLink(x, y, x + 1, y, EAST);
        if (x > 0)          addLink(x, y, x - 1, y, WEST);
        addLink(x, y, x, y, LOCAL); // Loopback/Local Port
      }
    }
  }

  void addLink(int sx, int sy, int dx, int dy, Direction dir) {
    links.push_back({(int)links.size(), sx, sy, dx, dy});
    RoutingLink* l = &links.back();
    if (dir == LOCAL) l->capacity = 999; // 本地端口假设无限带宽
    nodes[sy * width + sx].outLinks[dir] = l;
  }

  RouterNode* getNode(int id) { return &nodes[id]; }
  
  int getCoordsId(int x, int y) { return y * width + x; }

  // 协商机制：更新拥塞历史
  void updateHistoryCosts() {
    for (auto& link : links) {
      if (link.current_usage > link.capacity) {
        link.history_cost += 0.5 * (link.current_usage - link.capacity);
      }
    }
  }

  void resetUsage() {
    for (auto& link : links) link.current_usage = 0;
  }
};

// --- 3. Pass 实现 ---

class NegotiatedRoutingPass
    : public PassWrapper<NegotiatedRoutingPass, OperationPass<func::FuncOp>> {

  Option<int> optMaxIter{*this, "max-iter", llvm::cl::desc("Max negotiation iterations"), llvm::cl::init(30)};
  Option<int> optChipW{*this, "width", llvm::cl::desc("Chip Width"), llvm::cl::init(8)};
  Option<int> optChipH{*this, "height", llvm::cl::desc("Chip Height"), llvm::cl::init(8)};

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NegotiatedRoutingPass)

  // [修复 1] 添加默认构造函数
  NegotiatedRoutingPass() = default;

  // [修复 2] 添加拷贝构造函数，显式拷贝 Option 的值
  NegotiatedRoutingPass(const NegotiatedRoutingPass& other) : PassWrapper(other) {
    optMaxIter = other.optMaxIter;
    optChipW = other.optChipW;
    optChipH = other.optChipH;
  }

  StringRef getArgument() const override { return "negotiated-routing"; }
  StringRef getDescription() const override { return "Congestion-aware routing using PathFinder algorithm"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
  }

  // A* Search
  bool routeNet(Net& net, RoutingResourceGraph& rrg, double pres_fac) {
    // 1. 拆除旧路径 (Rip-up)
    for (auto* link : net.path) {
      if (link->current_usage > 0) link->current_usage--;
    }
    net.path.clear();

    // 2. 初始化 A*
    struct State {
      int u;
      double g; // Cost from start
      double f; // g + h
      RoutingLink* incomingLink; // 如何到达 u
      int parentNode;

      bool operator>(const State& other) const { return f > other.f; }
    };

    std::priority_queue<State, std::vector<State>, std::greater<State>> pq;
    std::vector<double> dist(rrg.nodes.size(), 1e9);
    std::vector<RoutingLink*> parentLink(rrg.nodes.size(), nullptr);
    std::vector<int> parentNode(rrg.nodes.size(), -1);

    auto heuristic = [&](int u, int v) {
      int ux = u % rrg.width, uy = u / rrg.width;
      int vx = v % rrg.width, vy = v / rrg.width;
      return std::abs(ux - vx) + std::abs(uy - vy);
    };

    int start = net.srcNodeId;
    int goal = net.dstNodeId;

    dist[start] = 0;
    pq.push({start, 0, (double)heuristic(start, goal), nullptr, -1});

    bool found = false;
    while (!pq.empty()) {
      State curr = pq.top(); pq.pop();

      if (curr.g > dist[curr.u]) continue;
      if (curr.u == goal) { found = true; break; }

      RouterNode* uNode = rrg.getNode(curr.u);
      
      // 遍历邻居
      for (int i = 0; i < DIR_COUNT; ++i) {
        RoutingLink* link = uNode->outLinks[i];
        if (!link) continue;

        // 如果是 LOCAL 端口，只允许在终点进入
        if (i == LOCAL && rrg.getCoordsId(link->dstX, link->dstY) != goal) continue;

        int v = rrg.getCoordsId(link->dstX, link->dstY);
        double new_g = curr.g + link->getDynamicCost(pres_fac);

        if (new_g < dist[v]) {
          dist[v] = new_g;
          parentLink[v] = link;
          parentNode[v] = curr.u;
          pq.push({v, new_g, new_g + heuristic(v, goal), link, curr.u});
        }
      }
    }

    if (!found) return false;

    // 3. 回溯路径 & 更新使用量 (Commit)
    int curr = goal;
    while (curr != start) {
      RoutingLink* link = parentLink[curr];
      net.path.push_back(link);
      link->current_usage++;
      curr = parentNode[curr];
    }
    // 路径是反向的，需要翻转
    std::reverse(net.path.begin(), net.path.end());
    return true;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    
    llvm::errs() << "=== Negotiated Routing (PathFinder) ===\n";

    // 1. 初始化 RRG
    RoutingResourceGraph rrg(optChipW, optChipH);
    
    // 2. 读取放置结果 & 构建 Nets
    std::map<int, int> taskIdToNodeId;
    func.walk([&](affine::AffineForOp op) {
      if (auto idAttr = op->getAttrOfType<IntegerAttr>("neura.task_id")) {
         int x = 0, y = 0;
         if (auto xAttr = op->getAttrOfType<IntegerAttr>("neura.placement_x")) x = xAttr.getInt();
         if (auto yAttr = op->getAttrOfType<IntegerAttr>("neura.placement_y")) y = yAttr.getInt();
         // 边界检查
         if(x >= optChipW) x = optChipW - 1;
         if(y >= optChipH) y = optChipH - 1;
         
         taskIdToNodeId[idAttr.getInt()] = rrg.getCoordsId(x, y);
      }
    });

    std::vector<Net> nets;
    int netCounter = 0;
    func.walk([&](affine::AffineForOp op) {
      if (auto idAttr = op->getAttrOfType<IntegerAttr>("neura.task_id")) {
        int u = idAttr.getInt();
        if (auto succs = op->getAttrOfType<ArrayAttr>("neura.streaming_succs")) {
          for (auto attr : succs) {
            // [修复 3] 修复 cast 警告
            int v = cast<IntegerAttr>(attr).getInt();
            if (taskIdToNodeId.count(u) && taskIdToNodeId.count(v)) {
               // 只有不同 Tile 之间才需要路由
               if (taskIdToNodeId[u] != taskIdToNodeId[v]) {
                   Net net;
                   net.netId = netCounter++;
                   net.srcTaskId = u; net.dstTaskId = v;
                   net.srcNodeId = taskIdToNodeId[u];
                   net.dstNodeId = taskIdToNodeId[v];
                   nets.push_back(net);
               }
            }
          }
        }
      }
    });

    llvm::errs() << "Identified " << nets.size() << " inter-tile nets.\n";

    // 3. 协商式路由主循环
    double pres_fac = 0.5; // 当前拥塞因子

    for (int iter = 0; iter < optMaxIter; ++iter) {
      int overflow_links = 0;
      
      for (auto& net : nets) {
        routeNet(net, rrg, pres_fac);
      }

      // 检查拥塞
      for (const auto& link : rrg.links) {
        if (link.current_usage > link.capacity) overflow_links++;
      }

      if (overflow_links == 0) {
        llvm::errs() << "Routing converged at iter " << iter << "!\n";
        break;
      }

      llvm::errs() << "Iter " << iter << ": " << overflow_links << " overflow links.\n";
      
      // 更新参数
      rrg.updateHistoryCosts();
      pres_fac *= 1.5; // 逐渐增大的惩罚，迫使解决冲突
    }

    // 4. 输出 Log
    for (const auto& net : nets) {
        std::string pathStr;
        for (auto* l : net.path) {
            pathStr += "(" + std::to_string(l->srcX) + "," + std::to_string(l->srcY) + ")->";
        }
        pathStr += "(" + std::to_string(rrg.getNode(net.dstNodeId)->x) + "," + std::to_string(rrg.getNode(net.dstNodeId)->y) + ")";
        llvm::errs() << "Net " << net.srcTaskId << "->" << net.dstTaskId << ": " << pathStr << "\n";
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createNegotiatedRoutingPass() {
  return std::make_unique<NegotiatedRoutingPass>();
}

} // namespace neura
} // namespace mlir
