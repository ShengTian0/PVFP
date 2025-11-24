# -*- coding: utf-8 -*-
"""
小规模网络实验脚本
12节点、15链路的网络配置
"""

import sys
sys.path.append('..')

from main import PVFPFramework
import time


def run_small_scale_experiment():
    """运行小规模网络实验"""
    
    print("\n" + "="*80)
    print(" "*25 + "小规模网络实验")
    print("="*80)
    print("\n配置:")
    print("  - 网络规模: 12节点, 15链路")
    print("  - 域数量: 4")
    print("  - CPU容量: 30核/节点")
    print("  - 带宽: 4 Mbps/链路")
    print("  - SFC数量: 15")
    print("  - 聚合轮数: 30")
    print("="*80 + "\n")
    
    # 创建PVFP框架
    start_time = time.time()
    pvfp = PVFPFramework(scale='small', num_domains=4)
    
    # 执行训练
    training_results = pvfp.run_federated_training(
        num_sfcs=15,
        aggregation_rounds=30
    )
    
    # 评估
    evaluation_results = pvfp.evaluate(num_sfcs=30)
    
    # 打印统计
    pvfp.aggregator.print_summary()
    
    total_time = time.time() - start_time
    
    print(f"\n" + "="*80)
    print(f"实验总用时: {total_time/60:.2f} 分钟")
    print("="*80 + "\n")
    
    return training_results, evaluation_results


if __name__ == "__main__":
    results = run_small_scale_experiment()
