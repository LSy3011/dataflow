import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- 配置文件名 ---
BASELINE_LOG = "bert_baseline_optimal.log"
FUSION_LOG = "bert_fusion_optimal.log"
OUTPUT_IMAGE = "bert_result_comparison_auto.png"

def parse_log_file(filepath):
    """
    解析 Log 文件，提取核心指标。
    返回一个包含 metrics 的字典，如果文件不存在或解析失败则返回 None。
    """
    metrics = {
        'TimeSteps': 0,
        'Blocks': 0,
        'Latency': 0,
        'Inter-tile Nets': 0
    }
    
    if not os.path.exists(filepath):
        print(f"Error: File not found - {filepath}")
        return None

    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
            # 1. 提取 [RESULT] 行的数据
            # 格式: [RESULT] TimeSteps=61 Blocks=190 Latency=1680 Failed=0
            result_match = re.search(r"\[RESULT\] TimeSteps=(\d+) Blocks=(\d+) Latency=(\d+)", content)
            if result_match:
                metrics['TimeSteps'] = int(result_match.group(1))
                metrics['Blocks'] = int(result_match.group(2))
                metrics['Latency'] = int(result_match.group(3))
            else:
                print(f"Warning: Could not find '[RESULT]' line in {filepath}")

            # 2. 提取路由网络数量
            # 格式: Identified 120 inter-tile nets.
            nets_match = re.search(r"Identified (\d+) inter-tile nets", content)
            if nets_match:
                metrics['Inter-tile Nets'] = int(nets_match.group(1))
            else:
                print(f"Warning: Could not find 'Identified ... nets' in {filepath}")
                
        return metrics

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def main():
    print("--- Starting Automated Log Analysis ---")
    
    # 1. 自动解析日志文件
    baseline_data = parse_log_file(BASELINE_LOG)
    fusion_data = parse_log_file(FUSION_LOG)

    if not baseline_data or not fusion_data:
        print("Aborting: Could not parse one or both log files.")
        # 为了演示，如果文件不存在，这里提供默认值（Fallback）防止代码直接崩溃，
        # 实际使用时请确保 Log 文件存在。
        print("Using fallback data for demonstration...")
        baseline_data = {'TimeSteps': 61, 'Blocks': 190, 'Latency': 1680, 'Inter-tile Nets': 120}
        fusion_data = {'TimeSteps': 35, 'Blocks': 129, 'Latency': 523, 'Inter-tile Nets': 82}

    print(f"Baseline Data: {baseline_data}")
    print(f"Fusion Data:   {fusion_data}")

    # 2. 构建 DataFrame
    metrics_list = ['TimeSteps', 'Blocks', 'Latency', 'Inter-tile Nets']
    data = {
        'Metric': metrics_list,
        'Baseline': [baseline_data[m] for m in metrics_list],
        'Fusion': [fusion_data[m] for m in metrics_list]
    }
    df = pd.DataFrame(data)

    # 3. 计算提升幅度 (Improvement Percentage)
    df['Improvement'] = (df['Baseline'] - df['Fusion']) / df['Baseline'] * 100

    # 4. 绘图逻辑
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Comparison: Baseline vs. Fusion (BERT Model)', fontsize=18, fontweight='bold', y=0.98)
    axes = axes.flatten()

    titles = [
        'Schedule Length (TimeSteps)\n(Lower is Better)', 
        'Number of Task Blocks\n(Lower means better granularity)', 
        'Total System Latency (Cycles)\n(Lower is Better)', 
        'Inter-tile Communication Nets\n(Lower means less congestion)'
    ]
    colors = ['#95a5a6', '#2ecc71'] # 灰色 vs 绿色

    for i, metric in enumerate(metrics_list):
        ax = axes[i]
        
        # 准备数据
        vals = [df.loc[i, 'Baseline'], df.loc[i, 'Fusion']]
        labels = ['Baseline\n(No Fusion)', 'Fusion\n(Optimized)']
        
        # 绘制柱状图
        bars = ax.bar(labels, vals, color=colors, edgecolor='black', alpha=0.9, width=0.6)
        
        # 设置标题
        ax.set_title(titles[i], fontsize=13, fontweight='bold', pad=15)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        # 标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (max(vals)*0.02),
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
            
        # 标注提升百分比
        imp = df.loc[i, 'Improvement']
        # 计算标注位置（两柱中间）
        text_y = max(vals) * 0.85
        # 防止文字溢出下边界
        if text_y < max(vals) * 0.2: text_y = max(vals) * 0.5
            
        ax.annotate(f'-{imp:.1f}%', 
                    xy=(0.5, text_y), xycoords='data',
                    xytext=(0, 0), textcoords='offset points',
                    ha='center', va='center', fontsize=16, color='#c0392b', fontweight='heavy',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c0392b", lw=2, alpha=0.9))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 5. 保存
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"\n✅ Chart successfully generated: {OUTPUT_IMAGE}")
    # plt.show() # 如果是在服务器无头模式下运行，请注释掉这行

if __name__ == "__main__":
    main()
