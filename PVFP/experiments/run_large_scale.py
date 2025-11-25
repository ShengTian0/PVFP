# -*- coding: utf-8 -*-
"""
大规模网络实验脚本
35节点、79链路的网络配置
"""

import sys
sys.path.append('..')

from main import PVFPFramework
import time


def run_large_scale_experiment():
    """运行大规模网络实验"""
    
    print("\n" + "="*80)
    print(" "*25 + "大规模网络实验")
    print("="*80)
    print("\n配置:")
    print("  - 网络规模: 35节点, 79链路")
    print("  - 域数量: 3")
    print("  - CPU容量: 20核/节点")
    print("  - 带宽: 2 Mbps/链路")
    print("  - SFC数量: 25")
    print("  - 聚合轮数: 50")
    print("="*80 + "\n")
    
    # 创建PVFP框架
    start_time = time.time()
    pvfp = PVFPFramework(scale='large', num_domains=3)
    
    # 执行训练
    training_results = pvfp.run_federated_training(
        num_sfcs=25,
        aggregation_rounds=50
    )
    
    # 评估
    evaluation_results = pvfp.evaluate(num_sfcs=50)
    
    # 打印统计
    pvfp.aggregator.print_summary()
    
    total_time = time.time() - start_time
    
    print(f"\n" + "="*80)
    print(f"实验总用时: {total_time/60:.2f} 分钟")
    print("="*80 + "\n")
    
    return training_results, evaluation_results


if __name__ == "__main__":
    results = run_large_scale_experiment()
