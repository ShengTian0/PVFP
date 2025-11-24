# -*- coding: utf-8 -*-
"""
联邦聚合器实现 (Algorithm 2)
实现时滞加权的联邦学习聚合策略
"""

import numpy as np
import time
import sys
sys.path.append('../..')
from config import *


class FederatedAggregator:
    """联邦聚合器 - 实现论文Algorithm 2"""
    
    def __init__(self, num_domains=NUM_DOMAINS, lambda_staleness=LAMBDA_STALENESS, 
                 delta_base=DELTA_BASE):
        """
        初始化联邦聚合器
        
        Args:
            num_domains: 域的数量
            lambda_staleness: 时滞衰减指数 λ
            delta_base: 基础聚合权重 δ
        """
        self.num_domains = num_domains
        self.lambda_staleness = lambda_staleness
        self.delta_base = delta_base
        
        # 记录每个域的时间戳
        self.domain_timestamps = {}
        self.first_upload_time = None
        
        # 全局模型权重
        self.global_weights = None
        
        # 聚合统计
        self.aggregation_count = 0
        self.staleness_history = []
    
    def compute_staleness_factor(self, upload_time, first_time):
        """
        计算时滞因子 ς(ti - t1)
        
        公式: ς(ti - t1) = 1 / (ti - t1 + 1)^λ
        
        Args:
            upload_time: 当前域的上传时间 ti
            first_time: 第一个域的上传时间 t1
        
        Returns:
            时滞因子 ς
        """
        time_diff = upload_time - first_time
        staleness = 1.0 / ((time_diff + 1) ** self.lambda_staleness)
        return staleness
    
    def compute_aggregation_weight(self, staleness_factor):
        """
        计算聚合权重 δi
        
        公式: δi = ς(ti - t1) * δ
        
        Args:
            staleness_factor: 时滞因子 ς
        
        Returns:
            聚合权重 δi
        """
        return staleness_factor * self.delta_base
    
    def aggregate(self, domain_weights_list, domain_timestamps_list):
        """
        执行联邦聚合
        
        公式: Θ(t+1) = (1 - Σδi) * Θ(t) + Σ(δi * θi(t))
        
        Args:
            domain_weights_list: 各域上传的权重列表 [weights_1, weights_2, ...]
            domain_timestamps_list: 各域的上传时间戳列表 [t1, t2, ...]
        
        Returns:
            aggregated_weights: 聚合后的全局权重
            aggregation_info: 聚合信息字典
        """
        num_domains_uploaded = len(domain_weights_list)
        
        if num_domains_uploaded == 0:
            print("[警告] 没有域上传权重，跳过聚合")
            return self.global_weights, {}
        
        # 找到最早的上传时间作为参考
        first_time = min(domain_timestamps_list)
        if self.first_upload_time is None:
            self.first_upload_time = first_time
        
        # 计算每个域的时滞因子和聚合权重
        staleness_factors = []
        aggregation_weights = []
        
        for timestamp in domain_timestamps_list:
            staleness = self.compute_staleness_factor(timestamp, first_time)
            delta_i = self.compute_aggregation_weight(staleness)
            
            staleness_factors.append(staleness)
            aggregation_weights.append(delta_i)
        
        # 归一化聚合权重（确保权重和不超过1）
        total_delta = sum(aggregation_weights)
        if total_delta > 1.0:
            aggregation_weights = [w / total_delta for w in aggregation_weights]
            total_delta = 1.0
        
        # 如果是第一次聚合，直接使用加权平均
        if self.global_weights is None:
            self.global_weights = self._weighted_average(
                domain_weights_list, 
                aggregation_weights
            )
        else:
            # 应用聚合公式: Θ(t+1) = (1 - Σδi) * Θ(t) + Σ(δi * θi(t))
            # 保留部分全局权重
            retention_weight = 1.0 - total_delta
            
            # 初始化新的全局权重
            new_global_weights = []
            
            for i, global_layer_weight in enumerate(self.global_weights):
                # 保留的全局权重部分
                retained_part = retention_weight * global_layer_weight
                
                # 新上传权重的加权和
                aggregated_part = np.zeros_like(global_layer_weight)
                for domain_weights, delta_i in zip(domain_weights_list, aggregation_weights):
                    aggregated_part += delta_i * domain_weights[i]
                
                # 合并
                new_layer_weight = retained_part + aggregated_part
                new_global_weights.append(new_layer_weight)
            
            self.global_weights = new_global_weights
        
        # 记录统计信息
        self.aggregation_count += 1
        self.staleness_history.append({
            'staleness_factors': staleness_factors,
            'aggregation_weights': aggregation_weights,
            'total_delta': total_delta,
            'retention_weight': 1.0 - total_delta if self.aggregation_count > 1 else 0.0
        })
        
        # 返回聚合信息
        aggregation_info = {
            'aggregation_round': self.aggregation_count,
            'num_domains': num_domains_uploaded,
            'staleness_factors': staleness_factors,
            'aggregation_weights': aggregation_weights,
            'total_delta': total_delta,
            'avg_staleness': np.mean(staleness_factors),
            'min_staleness': np.min(staleness_factors),
            'max_staleness': np.max(staleness_factors)
        }
        
        print(f"\n[聚合轮次 {self.aggregation_count}]")
        print(f"  参与域数: {num_domains_uploaded}")
        print(f"  时滞因子: {[f'{s:.4f}' for s in staleness_factors]}")
        print(f"  聚合权重: {[f'{w:.4f}' for w in aggregation_weights]}")
        print(f"  总权重: {total_delta:.4f}, 保留权重: {1.0-total_delta:.4f}")
        
        return self.global_weights, aggregation_info
    
    def _weighted_average(self, weights_list, weights):
        """
        计算加权平均
        
        Args:
            weights_list: 权重列表
            weights: 权重系数
        
        Returns:
            加权平均后的权重
        """
        # 归一化权重
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 计算加权平均
        averaged_weights = []
        num_layers = len(weights_list[0])
        
        for layer_idx in range(num_layers):
            layer_weights = [w[layer_idx] for w in weights_list]
            
            # 加权求和
            weighted_sum = np.zeros_like(layer_weights[0])
            for weight, layer_w in zip(weights, layer_weights):
                weighted_sum += weight * layer_w
            
            averaged_weights.append(weighted_sum)
        
        return averaged_weights
    
    def get_global_weights(self):
        """获取全局模型权重"""
        return self.global_weights
    
    def set_global_weights(self, weights):
        """设置全局模型权重"""
        self.global_weights = weights
    
    def reset(self):
        """重置聚合器状态"""
        self.domain_timestamps = {}
        self.first_upload_time = None
        self.aggregation_count = 0
        self.staleness_history = []
    
    def get_statistics(self):
        """
        获取聚合统计信息
        
        Returns:
            统计信息字典
        """
        if not self.staleness_history:
            return {}
        
        all_staleness = []
        all_weights = []
        
        for record in self.staleness_history:
            all_staleness.extend(record['staleness_factors'])
            all_weights.extend(record['aggregation_weights'])
        
        stats = {
            'total_aggregations': self.aggregation_count,
            'avg_staleness_overall': np.mean(all_staleness),
            'std_staleness_overall': np.std(all_staleness),
            'avg_weight_overall': np.mean(all_weights),
            'std_weight_overall': np.std(all_weights),
            'staleness_history': self.staleness_history
        }
        
        return stats
    
    def print_summary(self):
        """打印聚合总结"""
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print("联邦聚合总结")
        print("="*50)
        print(f"总聚合轮次: {stats.get('total_aggregations', 0)}")
        print(f"平均时滞因子: {stats.get('avg_staleness_overall', 0):.4f}")
        print(f"时滞因子标准差: {stats.get('std_staleness_overall', 0):.4f}")
        print(f"平均聚合权重: {stats.get('avg_weight_overall', 0):.4f}")
        print(f"聚合权重标准差: {stats.get('std_weight_overall', 0):.4f}")
        print("="*50)


class DomainCoordinator:
    """域协调器 - 管理多个域的训练和上传"""
    
    def __init__(self, num_domains, aggregator):
        """
        初始化域协调器
        
        Args:
            num_domains: 域的数量
            aggregator: 联邦聚合器实例
        """
        self.num_domains = num_domains
        self.aggregator = aggregator
        
        # 域状态跟踪
        self.domain_states = {
            i: {'status': 'idle', 'last_update': None}
            for i in range(num_domains)
        }
    
    def coordinate_training_round(self, domain_agents, domain_datasets, 
                                 local_epochs=LOCAL_EPOCHS):
        """
        协调一轮训练
        
        Args:
            domain_agents: 域DQN代理列表
            domain_datasets: 每个域的数据集
            local_epochs: 本地训练轮数
        
        Returns:
            aggregated_weights: 聚合后的权重
            round_info: 本轮信息
        """
        domain_weights_list = []
        domain_timestamps_list = []
        
        print(f"\n{'='*60}")
        print(f"开始新的训练轮次")
        print(f"{'='*60}")
        
        # 各域并行训练
        for domain_id in range(self.num_domains):
            agent = domain_agents[domain_id]
            dataset = domain_datasets.get(domain_id, [])
            
            if len(dataset) == 0:
                print(f"[域 {domain_id}] 无数据，跳过训练")
                continue
            
            print(f"\n[域 {domain_id}] 开始本地训练 ({local_epochs} 轮)...")
            
            # 记录开始时间
            start_time = time.time()
            
            # 本地训练
            for epoch in range(local_epochs):
                # 这里应该根据实际环境进行训练
                # 示例：简单的训练循环
                pass
            
            # 记录结束时间
            end_time = time.time()
            training_time = end_time - start_time
            
            # 上传权重和时间戳
            weights = agent.get_weights()
            domain_weights_list.append(weights)
            domain_timestamps_list.append(end_time)
            
            # 更新域状态
            self.domain_states[domain_id]['status'] = 'completed'
            self.domain_states[domain_id]['last_update'] = end_time
            
            print(f"[域 {domain_id}] 训练完成，用时 {training_time:.2f}秒")
        
        # 云端聚合
        print(f"\n{'='*60}")
        print(f"云端开始聚合")
        print(f"{'='*60}")
        
        aggregated_weights, agg_info = self.aggregator.aggregate(
            domain_weights_list,
            domain_timestamps_list
        )
        
        # 下发全局模型到各域
        for domain_id in range(self.num_domains):
            agent = domain_agents[domain_id]
            agent.set_weights(aggregated_weights)
            print(f"[域 {domain_id}] 接收到全局模型")
        
        round_info = {
            'participating_domains': len(domain_weights_list),
            'aggregation_info': agg_info
        }
        
        return aggregated_weights, round_info
