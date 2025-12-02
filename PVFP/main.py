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
import random
from datetime import datetime
import networkx as nx

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

        # 2.1 在每个域中选取可部署VNF的功能节点，并为其生成部署成本
        self._init_function_nodes()
        
        # 3. 初始化并行规则
        print("\n[步骤 3] 初始化VNF并行规则...")
        self.parallel_rules = VNFParallelRules()
        
        # 4. 初始化SFC分解器（基于功能节点域进行分解）
        print("\n[步骤 4] 初始化SFC分解器...")
        self.decomposer = SFCDecomposer(self.topology, self.function_domains)
        
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
        sample_domain_nodes = self.function_domains.get(0, self.domains[0])
        sample_env = VNFPlacementEnv(
            self.topology, 
            sample_sfc, 
            self.parallel_rules,
            sample_domain_nodes
        )
        state_dim = sample_env.get_state_dim()
        # 所有域使用统一的动作空间维度（所有域功能节点数的最大值），以支持联邦聚合
        max_action_dim = max(len(nodes) for nodes in self.function_domains.values())
        
        for domain_id in range(num_domains):
            domain_nodes = self.function_domains.get(domain_id, self.domains[domain_id])
            agent = DQNAgent(
                domain_id=domain_id,
                state_dim=state_dim,
                action_dim=max_action_dim,
                buffer_capacity=REPLAY_BUFFER_SIZE
            )
            # 记录该域真实可用的动作数量（域内节点数）
            agent.valid_action_dim = len(domain_nodes)
            self.domain_agents.append(agent)
            print(f"  域 {domain_id}: 状态维度={state_dim}, 动作维度={len(domain_nodes)}")
        
        # 训练统计
        self.training_history = {
            'losses': [],
            'rewards': [],
            'latencies': [],
            'aggregation_info': []
        }
        
        os.makedirs(RESULT_SAVE_PATH, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(RESULT_SAVE_PATH, f"training_log_{timestamp}.txt")
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("PVFP 联邦训练日志\n")
            f.write(f"scale={self.scale}, num_domains={self.num_domains}\n")
            f.write("="*60 + "\n\n")

        # 在训练开始前记录当前拓扑和域划分概要
        self._log_topology_summary()

        # 在控制台打印拓扑和NFV模型的详细信息
        self._print_topology_console()

        # 说明训练中reward和loss的定义
        print("\n[说明] Reward定义: 每一步 reward = -(本步新增部署成本 + 链路成本)，即以最小化总成本为目标。")
        print("[说明] Loss定义: DQN的TD误差均方，即 (Q(s,a) - (r + γ max_a' Q_target(s',a')))^2 的平均值。")

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
        # 只在功能节点上部署VNF
        domain_nodes = self.function_domains.get(domain_id, self.domains[domain_id])
        
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
                    # 选择动作（仅在本域有效动作范围内）
                    action = agent.select_action(
                        state,
                        training=True,
                        valid_action_dim=len(domain_nodes)
                    )
                    
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
            
            # 更新epsilon，并记录当前epoch的统计
            if num_steps > 0:
                avg_reward = epoch_reward / len(sfc_segments)
                agent.update_epsilon(avg_reward)
                
                epoch_loss_avg = epoch_loss / num_steps
                epoch_losses.append(epoch_loss_avg)
                epoch_rewards.append(avg_reward)

                # 打印每个epoch的loss，便于观察收敛过程
                print(f"  Epoch {epoch+1}/{epochs} - "
                      f"Loss: {epoch_loss_avg:.4f}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {agent.epsilon:.4f}")
                self._append_log(
                    f"[域 {domain_id}] Epoch {epoch+1}/{epochs} - "
                    f"Loss: {epoch_loss_avg:.4f}, "
                    f"Avg Reward: {avg_reward:.2f}, "
                    f"Epsilon: {agent.epsilon:.4f}"
                )
        
        training_stats = {
            'domain_id': domain_id,
            'losses': epoch_losses,
            'rewards': epoch_rewards,
            'avg_loss': np.mean(epoch_losses) if epoch_losses else 0,
            'avg_reward': np.mean(epoch_rewards) if epoch_rewards else 0
        }
        
        return training_stats
    
    def _append_log(self, text):
        """将文本追加写入训练日志文件"""
        if not hasattr(self, 'log_file'):
            return
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(text + "\n")
        except Exception:
            pass
    
    def _init_function_nodes(self):
        """在每个域内选取功能节点，并为其生成部署成本矩阵"""
        self.function_nodes = set()
        self.function_domains = {}
        # small 规模：沿用按比例选取功能节点的策略
        if self.scale == 'small':
            for domain_id, nodes in self.domains.items():
                num_nodes = len(nodes)
                if num_nodes == 0:
                    self.function_domains[domain_id] = []
                    continue

                # 按比例选取功能节点，至少1个
                num_func = max(1, int(round(FUNCTION_NODE_RATIO * num_nodes)))
                num_func = min(num_func, num_nodes)
                func_nodes = random.sample(list(nodes), num_func)

                self.function_domains[domain_id] = func_nodes
                self.function_nodes.update(func_nodes)
        else:
            # large 规模：全局功能节点总数固定为 5 个，参考 Algorithm2 配置
            all_nodes = list(self.topology.nodes())
            selected_funcs = set()
            domain_funcs = {}

            # 1) 先在每个域中各选 1 个候选功能节点，保证每域至少有 1 个
            for domain_id, nodes in self.domains.items():
                domain_nodes = list(nodes)
                if not domain_nodes:
                    domain_funcs[domain_id] = []
                    continue

                cand = random.choice(domain_nodes)
                selected_funcs.add(cand)
                domain_funcs.setdefault(domain_id, []).append(cand)

            # 2) 若当前全局功能节点数不足 5，则从剩余节点中补足
            remaining = [n for n in all_nodes if n not in selected_funcs]
            need_extra = max(0, 5 - len(selected_funcs))
            if need_extra > 0 and remaining:
                extra_nodes = random.sample(remaining, min(need_extra, len(remaining)))
                for n in extra_nodes:
                    selected_funcs.add(n)
                    # 找到该节点所属的域并加入对应列表
                    for domain_id, nodes in self.domains.items():
                        if n in nodes:
                            domain_funcs.setdefault(domain_id, []).append(n)
                            break

            self.function_nodes = selected_funcs
            # 为每个域记录其功能节点列表
            for domain_id in self.domains.keys():
                self.function_domains[domain_id] = domain_funcs.get(domain_id, [])

        # 为功能节点生成按VNF类型区分的部署成本以及部署成本预算上限
        for node in self.topology.nodes():
            is_func = node in self.function_nodes
            self.topology.nodes[node]['is_function_node'] = is_func
            if is_func:
                deploy_costs = {}
                for vnf_type in VNF_TYPES:
                    # 为便于与 Algorithm2 对齐，大规模网络下部署成本最大值为 10，
                    # 小规模保持原有 MAX_DEPLOYMENT_COST 范围。
                    if self.scale == 'large':
                        deploy_costs[vnf_type] = random.randint(1, 10)
                    else:
                        deploy_costs[vnf_type] = random.randint(1, MAX_DEPLOYMENT_COST)
                self.topology.nodes[node]['deployment_costs'] = deploy_costs
                # 部署成本预算上限（单个SFC内该节点可承受的部署成本总和）
                budget = random.randint(MIN_DEPLOY_BUDGET, MAX_DEPLOY_BUDGET)
                self.topology.nodes[node]['deploy_budget_capacity'] = budget
            else:
                # 非功能节点不携带部署成本和预算
                self.topology.nodes[node]['deployment_costs'] = {}
                self.topology.nodes[node]['deploy_budget_capacity'] = 0

    def _log_topology_summary(self):
        """在日志中记录当前拓扑和域划分概要"""
        if self.topology is None or not hasattr(self, 'log_file'):
            return
        try:
            num_nodes = self.topology.number_of_nodes()
            num_links = self.topology.number_of_edges()
            self._append_log(
                f"[拓扑] scale={self.scale}, 节点数={num_nodes}, 链路数={num_links}"
            )
            for domain_id, nodes in self.domains.items():
                self._append_log(
                    f"[拓扑] 域 {domain_id}: {len(nodes)} 个节点 {nodes}"
                )
                func_nodes = self.function_domains.get(domain_id, [])
                self._append_log(
                    f"[拓扑] 域 {domain_id}: 功能节点 {len(func_nodes)} 个 {func_nodes}"
                )
        except Exception:
            # 拓扑日志失败不影响训练
            pass

    def _print_topology_console(self):
        """在控制台打印拓扑和NFV相关信息，便于对齐NFV模型结构"""
        if self.topology is None:
            return

        print("\n[拓扑详情]" )
        print(f"  scale={self.scale}, 节点数={self.topology.number_of_nodes()}, 链路数={self.topology.number_of_edges()}")

        # 域划分与功能节点
        for domain_id, nodes in self.domains.items():
            func_nodes = self.function_domains.get(domain_id, [])
            print(f"  域 {domain_id}: 节点 {nodes}")
            print(f"    功能节点: {func_nodes}")

        # 功能节点的部署预算与部署成本
        print("\n[功能节点部署预算与部署成本]")
        if hasattr(self, 'function_nodes'):
            for node in sorted(self.function_nodes):
                data = self.topology.nodes[node]
                budget = data.get('deploy_budget_capacity', None)
                deploy_costs = data.get('deployment_costs', {})
                print(f"  节点 {node}: 预算={budget}, 部署成本={deploy_costs}")

        # 链路属性：带宽、成本、时延
        print("\n[链路属性]")
        for u, v in self.topology.edges():
            e = self.topology.edges[(u, v)]
            bw = e.get('bandwidth', None)
            cost = e.get('cost', None)
            delay = e.get('delay', None)
            print(f"  边 ({u}, {v}): 带宽={bw}, 成本={cost}, 时延={delay:.2f}ms")
    
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

                # 将该域本轮的每个epoch loss追加写入文件，便于后续画loss曲线
                losses = stats.get('losses', [])
                if losses:
                    loss_file = os.path.join(
                        RESULT_SAVE_PATH,
                        f"loss_curve_{self.scale}_domain{domain_id}.txt"
                    )
                    try:
                        with open(loss_file, 'a', encoding='utf-8') as f_loss:
                            for epoch_idx, loss_val in enumerate(losses, start=1):
                                f_loss.write(
                                    f"round={round_idx+1}, epoch={epoch_idx}, loss={loss_val:.6f}\n"
                                )
                    except Exception:
                        pass
                
                print(f"\n[域 {domain_id}] 训练完成 - "
                      f"用时: {train_end - train_start:.2f}s, "
                      f"平均Loss: {stats['avg_loss']:.4f}, "
                      f"平均Reward: {stats['avg_reward']:.2f}")
                self._append_log(
                    f"[域 {domain_id}] 训练完成 - "
                    f"用时: {train_end - train_start:.2f}s, "
                    f"平均Loss: {stats['avg_loss']:.4f}, "
                    f"平均Reward: {stats['avg_reward']:.2f}"
                )
            
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
                self._append_log(
                    f"[聚合完成] 轮次 {round_idx + 1} - "
                    f"平均Loss: {avg_loss:.4f}, "
                    f"平均Reward: {avg_reward:.2f}"
                )

                agg_weights_str = ",".join(f"{w:.4f}" for w in agg_info.get('aggregation_weights', []))
                staleness_str = ",".join(f"{s:.4f}" for s in agg_info.get('staleness_factors', []))
                self._append_log(
                    f"[聚合详情] 轮次 {agg_info.get('aggregation_round', round_idx + 1)} - "
                    f"参与域数: {agg_info.get('num_domains', len(domain_weights_list))}, "
                    f"avg_staleness: {agg_info.get('avg_staleness', 0):.4f}, "
                    f"weights: [{agg_weights_str}], "
                    f"staleness: [{staleness_str}]"
                )
            
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
        
        total_latency = 0.0
        total_resource = 0.0
        success_count = 0
        per_sfc_results = []
        
        for sfc in test_sfcs:
            sfc_id = sfc.get('id')
            vnf_sequence = sfc.get('vnf_sequence', [])

            # 初始化当前SFC的评估结果容器
            sfc_result = {
                'sfc_id': sfc_id,
                'source': sfc.get('source'),
                'destination': sfc.get('destination'),
                'vnf_sequence': vnf_sequence,
                'success': False,
                # 对于跨域部署，assigned_domain 将是使用到的域ID列表
                'assigned_domain': None,
                'metrics': None,
                # 映射原始VNF索引 -> 部署节点
                'vnf_placements': None,
                # 记录每个域上的局部部署结果，便于调试
                'domain_details': {},
            }

            # 云端按照当前拓扑状态对该SFC进行一次分段（与训练时一致）
            segments_by_domain = self.decomposer.decompose_sfc(vnf_sequence, sfc_id)

            # 记录每个域对应段在原始VNF序列中的起始索引，保证拼接顺序一致
            domain_start_index = {}
            start_idx = 0
            for did in sorted(segments_by_domain.keys()):
                seg = segments_by_domain[did]
                seg_len = len(seg)
                if seg_len > 0:
                    domain_start_index[did] = start_idx
                    start_idx += seg_len

            # 如果所有域的段都是空的，则无法部署该SFC
            if not domain_start_index:
                per_sfc_results.append(sfc_result)
                continue

            # 聚合该SFC在所有相关域上的部署结果
            sfc_success = True
            used_domains = []
            global_vnf_placements = {}

            agg_total_latency = 0.0
            agg_total_resource = 0.0
            agg_cpu_cost = 0.0
            agg_bw_cost = 0.0
            agg_paths = []
            agg_path_link_costs = []

            for domain_id in sorted(domain_start_index.keys()):
                segment = segments_by_domain.get(domain_id, [])
                if not segment:
                    continue

                used_domains.append(domain_id)
                domain_nodes = self.function_domains.get(domain_id, self.domains[domain_id])

                # 为该域上的SFC段构造局部请求（仅VNF序列不同，源宿暂沿用原SFC）
                local_sfc_request = {
                    'id': sfc_id,
                    'source': sfc.get('source'),
                    'destination': sfc.get('destination'),
                    'vnf_sequence': segment,
                    'bandwidth_requirement': sfc.get('bandwidth_requirement', 1.0),
                    'cpu_requirement_per_vnf': sfc.get('cpu_requirement_per_vnf', VNF_CPU_REQUIREMENT),
                }

                env = VNFPlacementEnv(
                    self.topology,
                    local_sfc_request,
                    self.parallel_rules,
                    domain_nodes
                )

                state = env.reset()
                done = False
                agent = self.domain_agents[domain_id]

                while not done:
                    action = agent.select_action(
                        state,
                        training=False,
                        valid_action_dim=len(domain_nodes)
                    )
                    next_state, reward, done, info = env.step(action)
                    state = next_state

                metrics = env.get_metrics()

                # 该域段未能完成全部VNF部署，则整条SFC视为失败
                if metrics.get('deployment_rate', 0.0) < 1.0:
                    sfc_success = False

                # 将该域段的局部VNF索引映射回原始SFC的全局索引
                start_index = domain_start_index[domain_id]
                local_placements = dict(env.vnf_placements)
                for local_idx, node in local_placements.items():
                    global_idx = start_index + local_idx
                    global_vnf_placements[global_idx] = node

                # 聚合该域段的性能指标
                agg_total_latency += metrics.get('total_latency', 0.0)
                agg_total_resource += metrics.get('total_resource_cost', 0.0)
                agg_cpu_cost += metrics.get('cpu_resource_cost', 0.0)
                agg_bw_cost += metrics.get('bw_resource_cost', 0.0)

                # 聚合路径和链路成本
                paths = metrics.get('paths', []) or []
                path_costs = metrics.get('path_link_costs', []) or []
                agg_paths.extend(paths)
                agg_path_link_costs.extend(path_costs)

                # 记录该域的局部信息
                sfc_result['domain_details'][domain_id] = {
                    'segment': segment,
                    'start_index': start_index,
                    'metrics': metrics,
                    'vnf_placements_local': local_placements,
                }

            # 为跨域段边界补充VNF间路径和链路成本（单域环境中未统计的跨域跳转）
            if len(global_vnf_placements) == len(vnf_sequence) and domain_start_index:
                bandwidth_req = sfc.get('bandwidth_requirement', 1.0)
                # 按原始VNF序列中的起始索引对域排序，依次处理相邻域段的边界
                ordered_domains = sorted(domain_start_index.items(), key=lambda x: x[1])
                for idx in range(1, len(ordered_domains)):
                    prev_did, prev_start = ordered_domains[idx - 1]
                    curr_did, curr_start = ordered_domains[idx]
                    prev_seg = segments_by_domain.get(prev_did, [])
                    curr_seg = segments_by_domain.get(curr_did, [])
                    if not prev_seg or not curr_seg:
                        continue
                    # 前一域段最后一个VNF与后一域段第一个VNF之间的跨域通信
                    prev_last_global = prev_start + len(prev_seg) - 1
                    curr_first_global = curr_start
                    u = global_vnf_placements.get(prev_last_global)
                    v = global_vnf_placements.get(curr_first_global)
                    if u is None or v is None or u == v:
                        continue
                    try:
                        path = nx.shortest_path(self.topology, u, v)
                        path_delay = 0.0
                        path_cost = 0.0
                        for i in range(len(path) - 1):
                            edge = (path[i], path[i+1])
                            if self.topology.has_edge(*edge):
                                edge_data = self.topology.edges[edge]
                                # 链路时延
                                path_delay += edge_data['delay']
                                # 链路成本：ce * b
                                link_cost = edge_data.get('cost', 1)
                                edge_cost = link_cost * bandwidth_req
                                path_cost += edge_cost
                        # 传输延迟（与环境中一致）
                        transmission_delay = BASE_TRANSMISSION_DELAY
                        data_size = PACKET_SIZE + PACKET_HEADER  # bytes
                        transfer_time = data_size * 8 * READ_WRITE_PER_BIT / 1000  # ms
                        extra_latency = path_delay + transmission_delay + transfer_time
                        agg_total_latency += extra_latency
                        agg_bw_cost += path_cost
                        agg_paths.append(path)
                        agg_path_link_costs.append(path_cost)
                    except nx.NetworkXNoPath:
                        # 无路径时仅在延迟上给予惩罚
                        agg_total_latency += 1000

                # 总成本应等于部署成本与链路成本之和
                agg_total_resource = agg_cpu_cost + agg_bw_cost

            # 根据所有相关域的结果，确定该SFC的总体成功与否及聚合指标
            if sfc_success and len(global_vnf_placements) == len(vnf_sequence):
                sfc_result['success'] = True
                sfc_result['assigned_domain'] = used_domains
                sfc_result['vnf_placements'] = global_vnf_placements
                sfc_result['metrics'] = {
                    'total_latency': agg_total_latency,
                    'avg_latency_per_vnf': agg_total_latency / len(vnf_sequence) if vnf_sequence else 0.0,
                    'total_resource_cost': agg_total_resource,
                    'cpu_resource_cost': agg_cpu_cost,
                    'bw_resource_cost': agg_bw_cost,
                    'num_vnfs_deployed': len(global_vnf_placements),
                    'deployment_rate': len(global_vnf_placements) / len(vnf_sequence) if vnf_sequence else 0.0,
                    'paths': agg_paths,
                    'path_link_costs': agg_path_link_costs,
                }

                total_latency += agg_total_latency
                total_resource += agg_total_resource
                success_count += 1

            per_sfc_results.append(sfc_result)

        avg_latency = total_latency / success_count if success_count > 0 else 0.0
        avg_resource = total_resource / success_count if success_count > 0 else 0.0
        success_rate = success_count / num_sfcs if num_sfcs > 0 else 0.0
        
        results = {
            'num_test_sfcs': num_sfcs,
            'success_count': success_count,
            'success_rate': success_rate,
            'avg_latency': avg_latency,
            'avg_resource': avg_resource,
            'per_sfc_results': per_sfc_results,
        }
        
        print(f"\n评估结果:")
        print(f"  测试SFC数: {num_sfcs}")
        print(f"  成功部署: {success_count} ({success_rate*100:.1f}%)")
        print(f"  平均延迟: {avg_latency:.2f} ms")
        print(f"  平均资源开销: {avg_resource:.2f}")
        print(f"{'='*70}\n")
        
        return results
    
    def save_model(self, model_name=None):
        """保存当前全局模型权重到文件"""
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        model_dir = os.path.join(MODEL_SAVE_PATH, self.scale)
        os.makedirs(model_dir, exist_ok=True)

        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"pvfp_model_{self.scale}_{timestamp}.npz"

        model_path = os.path.join(model_dir, model_name)

        # 优先使用聚合器中的全局权重
        global_weights = self.aggregator.get_global_weights()
        if global_weights is None:
            # 如果还没有进行联邦聚合，则从任意一个域代理获取当前权重
            if self.domain_agents:
                global_weights = self.domain_agents[0].get_weights()
            else:
                print("[警告] 没有可保存的模型权重")
                return model_path

        np.savez(model_path, *global_weights)
        print(f"[模型保存] {model_path}")
        return model_path

    def load_model(self, model_path):
        """从文件加载全局模型权重并下发到各域代理"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        data = np.load(model_path, allow_pickle=True)
        # np.savez 默认使用 arr_0, arr_1, ... 作为键
        global_weights = [data[key] for key in sorted(data.files)]

        # 设置聚合器的全局权重，并同步到所有域代理
        self.aggregator.set_global_weights(global_weights)
        for agent in self.domain_agents:
            agent.set_weights(global_weights)

        print(f"[模型加载] {model_path}")
        return global_weights
    
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
    pvfp = PVFPFramework(scale='small', num_domains=NUM_DOMAINS)
    
    # 执行联邦训练
    training_results = pvfp.run_federated_training(
        num_sfcs=10,
        aggregation_rounds=AGGREGATION_EPOCHS
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
