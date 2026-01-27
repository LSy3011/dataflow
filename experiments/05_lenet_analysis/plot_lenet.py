import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 文件名配置
BASELINE_LOG = "lenet_baseline.log"
FUSION_LOG = "lenet_fusion.log"
OUTPUT_IMAGE = "lenet_result_comparison.png"

def parse_log(filepath):
    metrics = {'TimeSteps': 0, 'Blocks': 0, 'Latency': 0, 'Nets': 0}
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None
    
    with open(filepath, 'r') as f:
        content = f.read()
        res = re.search(r"\[RESULT\] TimeSteps=(\d+) Blocks=(\d+) Latency=(\d+)", content)
        if res:
            metrics['TimeSteps'] = int(res.group(1))
            metrics['Blocks'] = int(res.group(2))
            metrics['Latency'] = int(res.group(3))
        net = re.search(r"Identified (\d+) inter-tile nets", content)
        if net:
            metrics['Nets'] = int(net.group(1))
    return metrics

def main():
    b_data = parse_log(BASELINE_LOG)
    f_data = parse_log(FUSION_LOG)
    
    if not b_data or not f_data: return

    print(f"Baseline Data: {b_data}")
    print(f"Fusion Data:   {f_data}")

    metrics = ['TimeSteps', 'Blocks', 'Latency', 'Nets']
    titles = ['Schedule Steps', 'Task Blocks', 'Total Latency', 'NoC Links']
    
    data = {
        'Metric': metrics,
        'Baseline': [b_data[m] for m in metrics],
        'Fusion': [f_data[m] for m in metrics]
    }
    df = pd.DataFrame(data)

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Performance Comparison: Baseline vs. Fusion (LeNet)', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    colors = ['#95a5a6', '#3498db']

    for i, m in enumerate(metrics):
        ax = axes[i]
        vals = [df.loc[i, 'Baseline'], df.loc[i, 'Fusion']]
        bars = ax.bar(['Baseline', 'Fusion'], vals, color=colors, edgecolor='black', width=0.6)
        ax.set_title(titles[i], fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + max(vals)*0.02, f'{int(h)}', ha='center', va='bottom', fontweight='bold')
            
        imp = (vals[0] - vals[1]) / vals[0] * 100
        text_y = max(vals) * 0.85
        if text_y < max(vals) * 0.2: text_y = max(vals) * 0.5
        ax.annotate(f'-{imp:.1f}%', xy=(0.5, text_y), ha='center', fontsize=14, color='#e74c3c', fontweight='heavy',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e74c3c", alpha=0.9))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Chart saved to: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
