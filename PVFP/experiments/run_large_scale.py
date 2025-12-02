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

    
    # 创建PVFP框架
    start_time = time.time()
    pvfp = PVFPFramework(scale='large', num_domains=3)
    
    # 执行训练（聚合轮数适当下调以匹配loss收敛速度）
    training_results = pvfp.run_federated_training(
        num_sfcs=25,
        aggregation_rounds=20
    )
    
    # 保存训练好的模型（large规模）
    model_path = pvfp.save_model()
    
    # 评估
    evaluation_results = pvfp.evaluate(num_sfcs=50)
    
    # 打印统计
    pvfp.aggregator.print_summary()
    
    total_time = time.time() - start_time
    
    print(f"\n" + "="*80)
    print(f"实验总用时: {total_time/60:.2f} 分钟")
    print("="*80 + "\n")
    
    return training_results, evaluation_results, model_path


if __name__ == "__main__":
    results = run_large_scale_experiment()
