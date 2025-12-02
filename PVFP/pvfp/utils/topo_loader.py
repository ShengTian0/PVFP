# -*- coding: utf-8 -*-
"""
网络拓扑加载器
生成和加载网络拓扑，支持小规模和大规模配置
"""

import networkx as nx
import numpy as np
import random
import sys
sys.path.append('../..')
from config import *


class TopologyLoader:
    """网络拓扑加载器"""
    
    def __init__(self, scale='small', seed=RANDOM_SEED):
        """
        初始化拓扑加载器
        
        Args:
            scale: 'small' 或 'large'
            seed: 随机种子
        """
        self.scale = scale
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # 根据规模选择配置
        if scale == 'small':
            self.config = SMALL_SCALE
        else:
            self.config = LARGE_SCALE
        
        self.topology = None
    
    def generate_topology(self):
        """
        生成网络拓扑
        
        Returns:
            topology: NetworkX图对象
        """
        num_nodes = self.config['nodes']
        cpu_capacity = self.config['cpu_capacity']
        bandwidth = self.config['bandwidth']

        # 创建随机图（使用Erdős–Rényi模型确保连通性）
        # small 规模：保持原有“按目标链路数反推 p” 的逻辑；
        # large 规模：按照 Algorithm2 实验设置，使用 p=0.03 近似每对节点有 3% 概率连边，
        #             不再强制精确链路数，只要求生成的图连通。
        if self.scale == 'large':
            p = 0.03
            attempts = 0
            max_attempts = 100

            while attempts < max_attempts:
                G = nx.erdos_renyi_graph(num_nodes, p, seed=self.seed + attempts)
                if nx.is_connected(G):
                    break
                attempts += 1

            if attempts >= max_attempts:
                print(f"[警告] 无法生成连通拓扑，使用近似配置")
                G = nx.erdos_renyi_graph(num_nodes, p, seed=self.seed)
                if not nx.is_connected(G):
                    # 退化为最大连通子图，保证后续环境不会因不连通而出错
                    largest_cc = max(nx.connected_components(G), key=len)
                    G = G.subgraph(largest_cc).copy()
        else:
            num_links = self.config['links']
            # 计算连接概率以达到目标链路数
            p = (2 * num_links) / (num_nodes * (num_nodes - 1))
            p = min(p, 1.0)
            
            # 生成图直到获得所需的链路数，且保持连通性
            attempts = 0
            max_attempts = 100
            
            while attempts < max_attempts:
                G = nx.erdos_renyi_graph(num_nodes, p, seed=self.seed + attempts)
                
                # 要求基础图首先是连通的
                if not nx.is_connected(G):
                    attempts += 1
                    continue

                # 调整链路数
                current_links = G.number_of_edges()
                
                if current_links > num_links:
                    # 删除多余的边，但尽量避免破坏连通性
                    edges = list(G.edges())
                    random.shuffle(edges)
                    for edge in edges:
                        if G.number_of_edges() <= num_links:
                            break
                        G.remove_edge(*edge)
                        if not nx.is_connected(G):
                            # 删除该边会导致图不连通，撤销删除
                            G.add_edge(*edge)
                elif current_links < num_links:
                    # 添加边直到达到目标
                    all_possible_edges = [(i, j) for i in range(num_nodes) 
                                         for j in range(i+1, num_nodes)]
                    existing_edges = set(G.edges())
                    possible_new_edges = [e for e in all_possible_edges 
                                         if e not in existing_edges]
                    random.shuffle(possible_new_edges)
                    
                    for edge in possible_new_edges:
                        if G.number_of_edges() >= num_links:
                            break
                        G.add_edge(*edge)
                
                # 若当前图既连通又达到目标链路数，则结束
                if nx.is_connected(G) and G.number_of_edges() == num_links:
                    break
                
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"[警告] 无法生成精确的拓扑，使用近似配置")
                G = nx.erdos_renyi_graph(num_nodes, p, seed=self.seed)
        
        # 为节点添加属性
        for node in G.nodes():
            G.nodes[node]['cpu_capacity'] = cpu_capacity
            G.nodes[node]['cpu_used'] = 0
            G.nodes[node]['cpu_available'] = cpu_capacity
            G.nodes[node]['vnfs_deployed'] = []
        
        # 为链路添加属性
        for edge in G.edges():
            G.edges[edge]['bandwidth'] = bandwidth
            G.edges[edge]['bandwidth_used'] = 0
            G.edges[edge]['bandwidth_available'] = bandwidth
            # 随机延迟 (ms)
            G.edges[edge]['delay'] = random.uniform(1, 5)
            # 传输成本（单位带宽的成本）：
            #  - small 规模：保持随机 1..MAX_LINK_COST
            #  - large  规模：统一为 3，以便与 Algorithm2 实验对齐
            if self.scale == 'large':
                G.edges[edge]['cost'] = 3
            else:
                G.edges[edge]['cost'] = random.randint(1, MAX_LINK_COST)
        
        self.topology = G
        
        print(f"\n[拓扑生成] {self.scale.upper()}规模")
        print(f"  节点数: {G.number_of_nodes()}")
        print(f"  链路数: {G.number_of_edges()}")
        print(f"  每节点CPU: {cpu_capacity} 核")
        print(f"  每链路带宽: {bandwidth} Mbps")
        print(f"  连通性: {'是' if nx.is_connected(G) else '否'}")
        
        return G
    
    def partition_domains(self, num_domains=NUM_DOMAINS):
        """
        将拓扑划分为多个域
        
        Args:
            num_domains: 域的数量
        
        Returns:
            domains: {domain_id: [node_ids]}
        """
        if self.topology is None:
            raise ValueError("请先生成拓扑")
        
        num_nodes = self.topology.number_of_nodes()
        nodes = list(self.topology.nodes())
        
        # 使用社区检测算法或简单划分
        # 这里使用简单的均匀划分
        nodes_per_domain = num_nodes // num_domains
        
        domains = {}
        for i in range(num_domains):
            start_idx = i * nodes_per_domain
            if i == num_domains - 1:
                # 最后一个域包含剩余所有节点
                end_idx = num_nodes
            else:
                end_idx = (i + 1) * nodes_per_domain
            
            domains[i] = nodes[start_idx:end_idx]
        
        print(f"\n[域划分] 共 {num_domains} 个域")
        for domain_id, domain_nodes in domains.items():
            print(f"  域 {domain_id}: {len(domain_nodes)} 个节点 {domain_nodes}")
        
        return domains
    
    def get_topology(self):
        """获取拓扑"""
        return self.topology
    
    def reset_topology(self):
        """重置拓扑资源使用情况"""
        if self.topology is None:
            return
        
        for node in self.topology.nodes():
            self.topology.nodes[node]['cpu_used'] = 0
            self.topology.nodes[node]['cpu_available'] = \
                self.topology.nodes[node]['cpu_capacity']
            self.topology.nodes[node]['vnfs_deployed'] = []
        
        for edge in self.topology.edges():
            self.topology.edges[edge]['bandwidth_used'] = 0
            self.topology.edges[edge]['bandwidth_available'] = \
                self.topology.edges[edge]['bandwidth']
    
    def save_topology(self, filename):
        """保存拓扑到文件"""
        if self.topology is None:
            return
        
        nx.write_gpickle(self.topology, filename)
        print(f"[拓扑保存] {filename}")
    
    def load_topology(self, filename):
        """从文件加载拓扑"""
        self.topology = nx.read_gpickle(filename)
        print(f"[拓扑加载] {filename}")
        return self.topology


class SFCGenerator:
    """SFC请求生成器"""
    
    def __init__(self, topology, seed=RANDOM_SEED):
        """
        初始化SFC生成器
        
        Args:
            topology: 网络拓扑
            seed: 随机种子
        """
        self.topology = topology
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_sfc(self, sfc_id=0, min_length=MIN_SFC_LENGTH, 
                    max_length=MAX_SFC_LENGTH):
        """
        生成单个SFC请求
        
        Args:
            sfc_id: SFC ID
            min_length: 最小长度
            max_length: 最大长度
        
        Returns:
            sfc_request: {
                'id': sfc_id,
                'source': 源节点,
                'destination': 目的节点,
                'vnf_sequence': VNF序列,
                'bandwidth_requirement': 带宽需求
            }
        """
        # 随机选择源和目的节点
        nodes = list(self.topology.nodes())
        source = random.choice(nodes)
        destination = random.choice([n for n in nodes if n != source])

        # 根据规模生成VNF序列
        if len(nodes) >= 50:
            # large 规模：SFC长度固定为4，从前4种服务中无重复抽取
            sfc_length = 4
            service_pool = VNF_TYPES[:4]
            # 若可用服务类型少于4，则退化为随机选择
            if len(service_pool) >= sfc_length:
                vnf_sequence = random.sample(service_pool, k=sfc_length)
            else:
                vnf_sequence = [random.choice(VNF_TYPES) for _ in range(sfc_length)]
        else:
            # small 规模：保持原有随机长度和可重复服务的逻辑
            sfc_length = random.randint(min_length, max_length)
            vnf_sequence = [random.choice(VNF_TYPES) for _ in range(sfc_length)]
        
        # 随机带宽需求 (Mbps)
        bandwidth_requirement = random.uniform(0.5, 2.0)
        
        sfc_request = {
            'id': sfc_id,
            'source': source,
            'destination': destination,
            'vnf_sequence': vnf_sequence,
            'bandwidth_requirement': bandwidth_requirement,
            'cpu_requirement_per_vnf': VNF_CPU_REQUIREMENT
        }
        
        return sfc_request
    
    def generate_batch_sfcs(self, num_sfcs=10, min_length=MIN_SFC_LENGTH,
                           max_length=MAX_SFC_LENGTH):
        """
        批量生成SFC请求
        
        Args:
            num_sfcs: SFC数量
            min_length: 最小长度
            max_length: 最大长度
        
        Returns:
            sfc_requests: SFC请求列表
        """
        sfc_requests = []
        
        for i in range(num_sfcs):
            sfc = self.generate_sfc(i, min_length, max_length)
            sfc_requests.append(sfc)
        
        print(f"\n[SFC生成] 共生成 {num_sfcs} 个SFC请求")
        print(f"  长度范围: [{min_length}, {max_length}]")
        print(f"  平均长度: {np.mean([len(sfc['vnf_sequence']) for sfc in sfc_requests]):.2f}")
        
        return sfc_requests
    
    def print_sfc(self, sfc_request):
        """打印SFC请求详情"""
        print(f"\nSFC {sfc_request['id']}:")
        print(f"  源节点: {sfc_request['source']}")
        print(f"  目的节点: {sfc_request['destination']}")
        print(f"  VNF序列: {sfc_request['vnf_sequence']}")
        print(f"  长度: {len(sfc_request['vnf_sequence'])}")
        print(f"  带宽需求: {sfc_request['bandwidth_requirement']:.2f} Mbps")
