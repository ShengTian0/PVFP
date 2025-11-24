# -*- coding: utf-8 -*-
"""
PVFP主程序
整合所有模块，执行完整的联邦学习训练流程
"""

import os
import sys
import numpy as np
import time
import json
from datetime import datetime

# 添加路径
sys.path.append('.')

# 导入配置
from config import *

# 导入模块
from pvfp.utils.topo_loader import TopologyLoader, SFCGenerator
from pvfp.domain.vnf_parallel import VNFParallelRules
from pvfp.cloud.decomposer import SFCDecomposer
from pvfp.cloud.aggregator import FederatedAggregator, DomainCoordinator
from pvfp.domain.dqn_agent import DQNAgent
from pvfp.env.network_env import VNFPlacementEnv, MultiSFCEnvironment


class PVFPFramework:
    """PVFP框架主类"""
    
    def __init__(self, scale='small', num_domains=NUM_DOMAINS):
        """
        初始化PVFP框架
        
        Args:
            scale: 网络规模 ('small' 或 'large')
            num_domains: 域的数量
        """
        print(f"\n{'='*70}")
        print(f"初始化PVFP框架")
        print(f"{'='*70}")
        
        self.scale = scale
        self.num_domains = num_domains
        
        # 1. 加载网络拓扑
        print("\n[步骤 1] 加载网络拓扑...")
        self.topo_loader = TopologyLoader(scale=scale)
        self.topology = self.topo_loader.generate_topology()
        
        # 2. 划分域
        print("\n[步骤 2] 划分网络域...")
        self.domains = self.topo_loader.partition_domains(num_domains)
        
        # 3. 初始化并行规则
        print("\n[步骤 3] 初始化VNF并行规则...")
        self.parallel_rules = VNFParallelRules()
        
        # 4. 初始化SFC分解器
        print("\n[步骤 4] 初始化SFC分解器...")
        self.decomposer = SFCDecomposer(self.topology, self.domains)
        
        # 5. 初始化SFC生成器
        print("\n[步骤 5] 初始化SFC生成器...")
        self.sfc_generator = SFCGenerator(self.topology)
        
        # 6. 初始化联邦聚合器
        print("\n[步骤 6] 初始化联邦聚合器...")
        self.aggregator = FederatedAggregator(
            num_domains=num_domains,
            lambda_staleness=LAMBDA_STALENESS,
            delta_base=DELTA_BASE
        )
        
        # 7. 初始化域协调器
        print("\n[步骤 7] 初始化域协调器...")
        self.coordinator = DomainCoordinator(num_domains, self.aggregator)
        
        # 8. 初始化DQN代理（每个域一个）
        print("\n[步骤 8] 初始化域级DQN代理...")
        self.domain_agents = []
        
        # 计算状态和动作空间维度
        sample_sfc = self.sfc_generator.generate_sfc(0)
        sample_env = VNFPlacementEnv(
            self.topology, 
            sample_sfc, 
            self.parallel_rules,
            self.domains[0]
        )
        state_dim = sample_env.get_state_dim()
        action_dim = len(self.domains[0])  # 每个域的节点数作为动作空间
        
        for domain_id in range(num_domains):
            agent = DQNAgent(
                domain_id=domain_id,
                state_dim=state_dim,
                action_dim=len(self.domains[domain_id]),
                buffer_capacity=REPLAY_BUFFER_SIZE
            )
            self.domain_agents.append(agent)
            print(f"  域 {domain_id}: 状态维度={state_dim}, 动作维度={len(self.domains[domain_id])}")
        
        # 训练统计
        self.training_history = {
            'losses': [],
            'rewards': [],
            'latencies': [],
            'aggregation_info': []
        }
        
        print(f"\n{'='*70}")
        print(f"PVFP框架初始化完成!")
        print(f"{'='*70}\n")
    
    def train_domain(self, domain_id, sfc_segments, epochs=LOCAL_EPOCHS):
        """
        训练单个域
        
        Args:
            domain_id: 域ID
            sfc_segments: 分配给该域的SFC段列表 [(sfc_id, vnf_segment), ...]
            epochs: 训练轮数
        
        Returns:
            training_stats: 训练统计信息
        """
        agent = self.domain_agents[domain_id]
        domain_nodes = self.domains[domain_id]
        
        epoch_losses = []
        epoch_rewards = []
        
        print(f"\n[域 {domain_id}] 开始训练 {epochs} 轮...")
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_reward = 0
            num_steps = 0
            
            # 处理每个SFC段
            for sfc_id, vnf_segment in sfc_segments:
                if len(vnf_segment) == 0:
                    continue
                
                # 创建临时SFC请求
                temp_sfc_request = {
                    'id': sfc_id,
                    'source': domain_nodes[0],
                    'destination': domain_nodes[-1],
                    'vnf_sequence': vnf_segment,
                    'bandwidth_requirement': 1.0,
                    'cpu_requirement_per_vnf': VNF_CPU_REQUIREMENT
                }
                
                # 创建环境
                env = VNFPlacementEnv(
                    self.topology,
                    temp_sfc_request,
                    self.parallel_rules,
                    domain_nodes
                )
                
                # Episode循环
                state = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    # 选择动作
                    action = agent.select_action(state, training=True)
                    
                    # 执行动作
                    next_state, reward, done, info = env.step(action)
                    
                    # 存储经验
                    agent.store_transition(state, action, reward, next_state, done)
                    
                    # 训练
                    loss = agent.train(batch_size=BATCH_SIZE)
                    
                    epoch_loss += loss
                    episode_reward += reward
                    num_steps += 1
                    
                    state = next_state
                
                epoch_reward += episode_reward
            
            # 更新epsilon
            if num_steps > 0:
                avg_reward = epoch_reward / len(sfc_segments)
                agent.update_epsilon(avg_reward)
                
                epoch_losses.append(epoch_loss / num_steps)
                epoch_rewards.append(avg_reward)
            
            if (epoch + 1) % LOG_INTERVAL == 0:
                print(f"  Epoch {epoch+1}/{epochs} - "
                      f"Loss: {epoch_losses[-1]:.4f}, "
                      f"Avg Reward: {epoch_rewards[-1]:.2f}, "
                      f"Epsilon: {agent.epsilon:.4f}")
        
        training_stats = {
            'domain_id': domain_id,
            'losses': epoch_losses,
            'rewards': epoch_rewards,
            'avg_loss': np.mean(epoch_losses) if epoch_losses else 0,
            'avg_reward': np.mean(epoch_rewards) if epoch_rewards else 0
        }
        
        return training_stats
    
    def run_federated_training(self, num_sfcs=10, aggregation_rounds=AGGREGATION_EPOCHS):
        """
        执行完整的联邦训练流程
        
        Args:
            num_sfcs: SFC请求数量
            aggregation_rounds: 聚合轮数
        
        Returns:
            training_results: 训练结果
        """
        print(f"\n{'='*70}")
        print(f"开始联邦训练")
        print(f"{'='*70}")
        print(f"聚合轮数: {aggregation_rounds}")
        print(f"每轮本地训练: {LOCAL_EPOCHS} epochs")
        print(f"SFC请求数: {num_sfcs}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for round_idx in range(aggregation_rounds):
            print(f"\n{'#'*70}")
            print(f"聚合轮次 {round_idx + 1}/{aggregation_rounds}")
            print(f"{'#'*70}")
            
            # 生成SFC请求
            sfc_requests = self.sfc_generator.generate_batch_sfcs(num_sfcs)
            
            # 云端分解SFC
            print(f"\n[云端] 分解SFC请求...")
            sfc_segments_by_domain = self.decomposer.decompose_batch_sfcs(
                [(sfc['id'], sfc['vnf_sequence']) for sfc in sfc_requests]
            )
            
            # 各域并行训练
            domain_weights_list = []
            domain_timestamps_list = []
            round_stats = []
            
            for domain_id in range(self.num_domains):
                sfc_segments = sfc_segments_by_domain.get(domain_id, [])
                
                if len(sfc_segments) == 0:
                    print(f"\n[域 {domain_id}] 未分配SFC段，跳过训练")
                    continue
                
                # 训练域
                train_start = time.time()
                stats = self.train_domain(domain_id, sfc_segments, epochs=LOCAL_EPOCHS)
                train_end = time.time()
                
                # 收集权重和时间戳
                weights = self.domain_agents[domain_id].get_weights()
                domain_weights_list.append(weights)
                domain_timestamps_list.append(train_end)
                
                round_stats.append(stats)
                
                print(f"\n[域 {domain_id}] 训练完成 - "
                      f"用时: {train_end - train_start:.2f}s, "
                      f"平均Loss: {stats['avg_loss']:.4f}, "
                      f"平均Reward: {stats['avg_reward']:.2f}")
            
            # 云端聚合
            if len(domain_weights_list) > 0:
                print(f"\n[云端] 执行联邦聚合...")
                global_weights, agg_info = self.aggregator.aggregate(
                    domain_weights_list,
                    domain_timestamps_list
                )
                
                # 下发全局模型
                print(f"\n[云端] 下发全局模型到各域...")
                for domain_id in range(self.num_domains):
                    self.domain_agents[domain_id].set_weights(global_weights)
                
                # 记录统计
                self.training_history['aggregation_info'].append(agg_info)
                
                avg_loss = np.mean([s['avg_loss'] for s in round_stats])
                avg_reward = np.mean([s['avg_reward'] for s in round_stats])
                self.training_history['losses'].append(avg_loss)
                self.training_history['rewards'].append(avg_reward)
                
                print(f"\n[聚合完成] 轮次 {round_idx + 1} - "
                      f"平均Loss: {avg_loss:.4f}, "
                      f"平均Reward: {avg_reward:.2f}")
            
            # 重置拓扑资源
            self.topo_loader.reset_topology()
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"联邦训练完成!")
        print(f"{'='*70}")
        print(f"总用时: {total_time:.2f}秒")
        print(f"平均每轮: {total_time/aggregation_rounds:.2f}秒")
        print(f"{'='*70}\n")
        
        # 保存结果
        self.save_results()
        
        return self.training_history
    
    def evaluate(self, num_sfcs=20):
        """
        评估训练好的模型
        
        Args:
            num_sfcs: 测试SFC数量
        
        Returns:
            evaluation_results: 评估结果
        """
        print(f"\n{'='*70}")
        print(f"模型评估")
        print(f"{'='*70}")
        
        # 生成测试SFC
        test_sfcs = self.sfc_generator.generate_batch_sfcs(num_sfcs)
        
        total_latency = 0
        total_resource = 0
        success_count = 0
        
        for sfc in test_sfcs:
            # 为每个域创建环境并部署
            for domain_id in range(self.num_domains):
                domain_nodes = self.domains[domain_id]
                
                env = VNFPlacementEnv(
                    self.topology,
                    sfc,
                    self.parallel_rules,
                    domain_nodes
                )
                
                state = env.reset()
                done = False
                agent = self.domain_agents[domain_id]
                
                while not done:
                    action = agent.select_action(state, training=False)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                
                metrics = env.get_metrics()
                if metrics['deployment_rate'] == 1.0:
                    total_latency += metrics['total_latency']
                    total_resource += metrics['total_resource_cost']
                    success_count += 1
                    break
        
        avg_latency = total_latency / success_count if success_count > 0 else 0
        avg_resource = total_resource / success_count if success_count > 0 else 0
        success_rate = success_count / num_sfcs
        
        results = {
            'num_test_sfcs': num_sfcs,
            'success_count': success_count,
            'success_rate': success_rate,
            'avg_latency': avg_latency,
            'avg_resource': avg_resource
        }
        
        print(f"\n评估结果:")
        print(f"  测试SFC数: {num_sfcs}")
        print(f"  成功部署: {success_count} ({success_rate*100:.1f}%)")
        print(f"  平均延迟: {avg_latency:.2f} ms")
        print(f"  平均资源开销: {avg_resource:.2f}")
        print(f"{'='*70}\n")
        
        return results
    
    def save_results(self):
        """保存训练结果"""
        os.makedirs(RESULT_SAVE_PATH, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RESULT_SAVE_PATH, f"pvfp_results_{timestamp}.json")
        
        results = {
            'config': {
                'scale': self.scale,
                'num_domains': self.num_domains,
                'learning_rate': LEARNING_RATE,
                'batch_size': BATCH_SIZE,
                'lambda_staleness': LAMBDA_STALENESS,
                'delta_base': DELTA_BASE
            },
            'training_history': {
                'losses': [float(x) for x in self.training_history['losses']],
                'rewards': [float(x) for x in self.training_history['rewards']],
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"[结果保存] {filename}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print(" "*20 + "PVFP - 联邦深度强化学习VNF并行部署")
    print("="*70)
    
    # 创建PVFP框架
    pvfp = PVFPFramework(scale='small', num_domains=4)
    
    # 执行联邦训练
    training_results = pvfp.run_federated_training(
        num_sfcs=10,
        aggregation_rounds=20
    )
    
    # 评估模型
    evaluation_results = pvfp.evaluate(num_sfcs=20)
    
    # 打印聚合统计
    pvfp.aggregator.print_summary()
    
    print("\n" + "="*70)
    print(" "*25 + "实验完成!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
