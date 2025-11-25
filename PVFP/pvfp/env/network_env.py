# -*- coding: utf-8 -*-
"""
网络环境实现
定义状态、动作、奖励函数，以及VNF部署的仿真环境
"""

import numpy as np
import networkx as nx
import random
import sys
sys.path.append('../..')
from config import *


class VNFPlacementEnv:
    """VNF部署环境"""
    
    def __init__(self, topology, sfc_request, parallel_rules, domain_nodes=None):
        """
        初始化VNF部署环境
        
        Args:
            topology: 网络拓扑
            sfc_request: SFC请求
            parallel_rules: 并行规则对象
            domain_nodes: 该域包含的节点列表（可选）
        """
        self.topology = topology
        self.sfc_request = sfc_request
        self.parallel_rules = parallel_rules
        self.domain_nodes = domain_nodes if domain_nodes else list(topology.nodes())
        
        # SFC信息
        self.vnf_sequence = sfc_request['vnf_sequence']
        self.source = sfc_request['source']
        self.destination = sfc_request['destination']
        self.bandwidth_req = sfc_request['bandwidth_requirement']
        self.cpu_req_per_vnf = sfc_request.get('cpu_requirement_per_vnf', VNF_CPU_REQUIREMENT)
        
        # 状态空间维度
        self.num_nodes = len(self.domain_nodes)
        self.num_vnfs = len(self.vnf_sequence)
        
        # 动作空间：选择节点部署VNF
        self.action_space_size = self.num_nodes
        
        # 当前状态
        self.current_vnf_idx = 0  # 当前要部署的VNF索引
        self.vnf_placements = {}  # {vnf_idx: node_id}
        self.deployed_vnfs = set()
        
        # 统计信息
        self.total_latency = 0.0
        # 资源开销：CPU利用率和带宽利用率
        self.total_resource_cost = 0.0
        self.cpu_resource_cost = 0.0
        self.bw_resource_cost = 0.0

        # 部署成本预算（单个SFC内部使用，不在SFC之间累计）
        self.node_budget_available = {}
        for node in self.domain_nodes:
            node_data = self.topology.nodes[node]
            budget_cap = node_data.get('deploy_budget_capacity', float('inf'))
            self.node_budget_available[node] = budget_cap

        # 记录VNF间通信路径及其链路成本，便于评估时输出
        self.used_paths = []          # 每次VNF间通信的节点序列
        self.path_link_costs = []     # 对应路径上的链路成本
    
    def get_state_dim(self):
        """
        获取状态空间维度
        
        状态包括：
        - 每个节点的资源状态 (CPU可用率)
        - 每条链路的资源状态 (带宽可用率)
        - 当前VNF的特征
        - 已部署VNF的位置信息
        """
        # 节点资源状态：num_nodes
        # 链路资源状态：num_links (简化为平均可用带宽)
        # VNF特征：VNF类型编码
        # 已部署信息：num_vnfs (binary)
        
        state_dim = self.num_nodes + 1 + len(VNF_TYPES) + self.num_vnfs
        return state_dim
    
    def get_state(self):
        """
        获取当前状态向量
        
        Returns:
            state: 状态向量
        """
        state = []
        
        # 1. 节点资源状态（CPU可用率）
        for node in self.domain_nodes:
            cpu_capacity = self.topology.nodes[node]['cpu_capacity']
            cpu_available = self.topology.nodes[node]['cpu_available']
            cpu_ratio = cpu_available / cpu_capacity if cpu_capacity > 0 else 0
            state.append(cpu_ratio)
        
        # 2. 平均链路带宽可用率
        total_bw_available = 0
        total_bw_capacity = 0
        for edge in self.topology.edges():
            if edge[0] in self.domain_nodes or edge[1] in self.domain_nodes:
                total_bw_available += self.topology.edges[edge]['bandwidth_available']
                total_bw_capacity += self.topology.edges[edge]['bandwidth']
        
        avg_bw_ratio = total_bw_available / total_bw_capacity if total_bw_capacity > 0 else 0
        state.append(avg_bw_ratio)
        
        # 3. 当前VNF类型（one-hot编码）
        if self.current_vnf_idx < self.num_vnfs:
            current_vnf_type = self.vnf_sequence[self.current_vnf_idx]
            vnf_encoding = [0] * len(VNF_TYPES)
            if current_vnf_type in VNF_TYPES:
                vnf_encoding[VNF_TYPES.index(current_vnf_type)] = 1
            state.extend(vnf_encoding)
        else:
            state.extend([0] * len(VNF_TYPES))
        
        # 4. 已部署VNF标记
        deployed_flags = [1 if i in self.deployed_vnfs else 0 
                         for i in range(self.num_vnfs)]
        state.extend(deployed_flags)
        
        return np.array(state, dtype=np.float32)
    
    def reset(self):
        """
        重置环境
        
        Returns:
            initial_state: 初始状态
        """
        self.current_vnf_idx = 0
        self.vnf_placements = {}
        self.deployed_vnfs = set()
        self.total_latency = 0.0
        self.total_resource_cost = 0.0
        self.cpu_resource_cost = 0.0
        self.bw_resource_cost = 0.0

        # 重置本SFC内部的节点部署预算
        self.node_budget_available = {}
        for node in self.domain_nodes:
            node_data = self.topology.nodes[node]
            budget_cap = node_data.get('deploy_budget_capacity', float('inf'))
            self.node_budget_available[node] = budget_cap

        # 重置路径记录
        self.used_paths = []
        self.path_link_costs = []
        
        return self.get_state()
    
    def step(self, action):
        """
        执行动作（在指定节点部署当前VNF）
        
        Args:
            action: 节点索引（在domain_nodes中的索引）
        
        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否完成
            info: 附加信息
        """
        # 验证动作有效性
        if action < 0 or action >= len(self.domain_nodes):
            # 无效动作，给予惩罚
            return self.get_state(), -100, True, {'error': 'invalid_action'}
        
        selected_node = self.domain_nodes[action]

        # 当前VNF类型
        if self.current_vnf_idx < self.num_vnfs:
            current_vnf_type = self.vnf_sequence[self.current_vnf_idx]
        else:
            current_vnf_type = None

        # 节点部署预算检查：本次部署的成本不能超过节点剩余预算
        if current_vnf_type is not None:
            node_data = self.topology.nodes[selected_node]
            deploy_costs = node_data.get('deployment_costs', {})
            deploy_cost = deploy_costs.get(current_vnf_type, 0.0)
            budget_avail = self.node_budget_available.get(selected_node, float('inf'))
            if budget_avail < deploy_cost:
                # 部署预算不足，给予惩罚
                return self.get_state(), -50, True, {'error': 'insufficient_deploy_budget'}
        else:
            deploy_cost = 0.0
            budget_avail = self.node_budget_available.get(selected_node, float('inf'))

        # 检查CPU资源是否充足
        node_cpu_available = self.topology.nodes[selected_node]['cpu_available']
        if node_cpu_available < self.cpu_req_per_vnf:
            # 资源不足，给予惩罚
            return self.get_state(), -50, True, {'error': 'insufficient_resources'}
        
        # 部署VNF
        self.vnf_placements[self.current_vnf_idx] = selected_node
        self.deployed_vnfs.add(self.current_vnf_idx)
        
        # 更新资源
        self.topology.nodes[selected_node]['cpu_used'] += self.cpu_req_per_vnf
        self.topology.nodes[selected_node]['cpu_available'] -= self.cpu_req_per_vnf
        self.topology.nodes[selected_node]['vnfs_deployed'].append(
            (self.sfc_request['id'], self.current_vnf_idx)
        )
        
        # 记录本步之前的总成本
        prev_total_cost = self.total_resource_cost

        # 计算延迟和成本
        latency_increment = self._calculate_latency_increment(self.current_vnf_idx, selected_node)
        self.total_latency += latency_increment

        # 部署成本：当前VNF在所选功能节点上的实例化成本，并从节点预算中扣减
        if current_vnf_type is not None:
            self.cpu_resource_cost += deploy_cost
            self.node_budget_available[selected_node] = budget_avail - deploy_cost

        # 汇总总成本（部署成本 + 链路成本）
        self.total_resource_cost = self.cpu_resource_cost + self.bw_resource_cost

        # 本步新增成本
        step_cost = self.total_resource_cost - prev_total_cost
        
        # 移动到下一个VNF
        self.current_vnf_idx += 1
        
        # 检查是否完成
        done = self.current_vnf_idx >= self.num_vnfs
        
        # 奖励：以“最小化成本”为目标，使用本步新增成本的负值
        reward = -step_cost
        
        next_state = self.get_state()
        info = {
            'latency': latency_increment,
            'total_latency': self.total_latency,
            'vnf_placed': self.current_vnf_idx - 1,
            'node_selected': selected_node,
            'step_cost': step_cost,
            'total_cost': self.total_resource_cost
        }
        
        return next_state, reward, done, info
    
    def _calculate_latency_increment(self, vnf_idx, node):
        """
        计算部署VNF后增加的延迟
        
        延迟包括：
        1. VNF激活延迟（执行时间）
        2. 并行执行延迟（如果有并行）
        3. 通信延迟（节点间传输）
        
        根据论文公式 (13)-(18)
        
        Args:
            vnf_idx: VNF索引
            node: 部署的节点
        
        Returns:
            latency: 延迟 (ms)
        """
        latency = 0.0
        
        # 1. VNF激活/执行延迟
        exec_time = random.uniform(VNF_EXEC_TIME_MIN, VNF_EXEC_TIME_MAX)
        latency += exec_time
        
        # 2. 通信延迟
        if vnf_idx > 0:
            # 与前一个VNF之间的通信延迟
            prev_node = self.vnf_placements.get(vnf_idx - 1)
            if prev_node is not None and prev_node != node:
                # 计算路径延迟
                try:
                    path = nx.shortest_path(self.topology, prev_node, node)
                    path_delay = 0
                    path_cost = 0.0
                    for i in range(len(path) - 1):
                        edge = (path[i], path[i+1])
                        if self.topology.has_edge(*edge):
                            edge_data = self.topology.edges[edge]
                            # 链路时延
                            path_delay += edge_data['delay']
                            # 链路成本：ce * b
                            link_cost = edge_data.get('cost', 1)
                            edge_cost = link_cost * self.bandwidth_req
                            self.bw_resource_cost += edge_cost
                            path_cost += edge_cost
                    
                    # 传输延迟
                    transmission_delay = BASE_TRANSMISSION_DELAY
                    data_size = PACKET_SIZE + PACKET_HEADER  # bytes
                    transfer_time = data_size * 8 * READ_WRITE_PER_BIT / 1000  # ms
                    
                    latency += path_delay + transmission_delay + transfer_time

                    # 记录该次VNF间通信路径及其链路成本
                    self.used_paths.append(path)
                    self.path_link_costs.append(path_cost)
                except nx.NetworkXNoPath:
                    # 无路径，给予高惩罚
                    latency += 1000
        
        # 3. 并行执行可以减少延迟（简化处理）
        # 这里假设如果可以并行，延迟可以部分重叠
        if vnf_idx > 0:
            current_vnf_type = self.vnf_sequence[vnf_idx]
            prev_vnf_type = self.vnf_sequence[vnf_idx - 1]
            
            if self.parallel_rules.is_parallelizable(prev_vnf_type, current_vnf_type):
                # 并行执行，延迟减少20%
                latency *= 0.8
        
        return latency
    
    def get_metrics(self):
        """
        获取性能指标
        
        Returns:
            metrics: 指标字典
        """
        metrics = {
            'total_latency': self.total_latency,
            'avg_latency_per_vnf': self.total_latency / self.num_vnfs if self.num_vnfs > 0 else 0,
            'total_resource_cost': self.total_resource_cost,
            'cpu_resource_cost': self.cpu_resource_cost,
            'bw_resource_cost': self.bw_resource_cost,
            'num_vnfs_deployed': len(self.deployed_vnfs),
            'deployment_rate': len(self.deployed_vnfs) / self.num_vnfs if self.num_vnfs > 0 else 0,
            'paths': list(self.used_paths),
            'path_link_costs': list(self.path_link_costs),
        }
        
        return metrics
    
    def render(self):
        """渲染当前状态（打印信息）"""
        print(f"\n{'='*50}")
        print(f"SFC {self.sfc_request['id']} 部署状态")
        print(f"{'='*50}")
        print(f"VNF序列: {self.vnf_sequence}")
        print(f"已部署: {len(self.deployed_vnfs)}/{self.num_vnfs}")
        print(f"当前VNF: {self.current_vnf_idx}")
        print(f"总延迟: {self.total_latency:.2f} ms")
        print(f"资源开销: {self.total_resource_cost:.2f}")
        print(f"\n部署映射:")
        for vnf_idx, node in sorted(self.vnf_placements.items()):
            print(f"  VNF {vnf_idx} ({self.vnf_sequence[vnf_idx]}) -> 节点 {node}")
        print(f"{'='*50}")


class MultiSFCEnvironment:
    """多SFC环境管理器"""
    
    def __init__(self, topology, parallel_rules, domain_nodes=None):
        """
        初始化多SFC环境
        
        Args:
            topology: 网络拓扑
            parallel_rules: 并行规则
            domain_nodes: 域节点列表
        """
        self.topology = topology
        self.parallel_rules = parallel_rules
        self.domain_nodes = domain_nodes
        
        self.environments = {}  # {sfc_id: env}
        self.sfc_requests = []
    
    def add_sfc_request(self, sfc_request):
        """添加SFC请求"""
        sfc_id = sfc_request['id']
        env = VNFPlacementEnv(
            self.topology,
            sfc_request,
            self.parallel_rules,
            self.domain_nodes
        )
        self.environments[sfc_id] = env
        self.sfc_requests.append(sfc_request)
    
    def add_batch_sfc_requests(self, sfc_requests):
        """批量添加SFC请求"""
        for sfc_request in sfc_requests:
            self.add_sfc_request(sfc_request)
    
    def get_env(self, sfc_id):
        """获取指定SFC的环境"""
        return self.environments.get(sfc_id)
    
    def reset_all(self):
        """重置所有环境"""
        for env in self.environments.values():
            env.reset()
    
    def get_aggregate_metrics(self):
        """获取所有SFC的聚合指标"""
        total_latency = 0
        total_resource = 0
        total_vnfs = 0
        deployed_vnfs = 0
        
        for env in self.environments.values():
            metrics = env.get_metrics()
            total_latency += metrics['total_latency']
            total_resource += metrics['total_resource_cost']
            total_vnfs += env.num_vnfs
            deployed_vnfs += metrics['num_vnfs_deployed']
        
        num_sfcs = len(self.environments)
        
        aggregate_metrics = {
            'num_sfcs': num_sfcs,
            'avg_latency_per_sfc': total_latency / num_sfcs if num_sfcs > 0 else 0,
            'total_latency': total_latency,
            'avg_resource_per_sfc': total_resource / num_sfcs if num_sfcs > 0 else 0,
            'total_resource': total_resource,
            'overall_deployment_rate': deployed_vnfs / total_vnfs if total_vnfs > 0 else 0
        }
        
        return aggregate_metrics
