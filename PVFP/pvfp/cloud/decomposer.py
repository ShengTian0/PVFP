# -*- coding: utf-8 -*-
"""
SFC分解算法实现 (Algorithm 1)
将SFC请求按域资源比例分解为多个段
"""

import numpy as np
from typing import List, Dict, Tuple
import sys
sys.path.append('..')


class SFCDecomposer:
    """SFC分解器 - 实现论文Algorithm 1"""
    
    def __init__(self, topology, domains):
        """
        初始化SFC分解器
        
        Args:
            topology: 网络拓扑对象
            domains: 域划分信息 {domain_id: [node_ids]}
        """
        self.topology = topology
        self.domains = domains
        self.num_domains = len(domains)
    
    def get_domain_resources(self) -> Dict[int, Dict[str, float]]:
        """
        获取每个域的资源快照
        
        Returns:
            {domain_id: {'cpu': total_cpu, 'bandwidth': total_bw, 'utilization': cpu_util}}
        """
        domain_resources = {}
        
        for domain_id, node_ids in self.domains.items():
            total_cpu = 0.0
            used_cpu = 0.0
            total_bw = 0.0
            
            # 计算域内所有节点的资源
            for node_id in node_ids:
                node_data = self.topology.nodes[node_id]
                cpu_capacity = node_data.get('cpu_capacity', 0)
                cpu_used = node_data.get('cpu_used', 0)
                
                total_cpu += cpu_capacity
                used_cpu += cpu_used
            
            # 计算域内链路带宽
            for edge in self.topology.edges():
                if edge[0] in node_ids and edge[1] in node_ids:
                    edge_data = self.topology.edges[edge]
                    total_bw += edge_data.get('bandwidth', 0)
            
            # 计算资源利用率
            cpu_util = used_cpu / total_cpu if total_cpu > 0 else 0
            available_cpu = total_cpu - used_cpu
            
            domain_resources[domain_id] = {
                'total_cpu': total_cpu,
                'used_cpu': used_cpu,
                'available_cpu': available_cpu,
                'utilization': cpu_util,
                'bandwidth': total_bw
            }
        
        return domain_resources
    
    def decompose_sfc(self, sfc: List[str], sfc_id: int = 0) -> Dict[int, List[str]]:
        """
        根据论文公式(12)分解SFC到各个域
        
        Algorithm 1核心逻辑:
        |Fi| = |F| * (Σ(rcpu_v) for v in Vi) / (Σ(rcpu_v) for all v)
        
        Args:
            sfc: VNF序列
            sfc_id: SFC请求ID
        
        Returns:
            {domain_id: [VNF segment]}，每个域分配到的VNF列表
        """
        sfc_length = len(sfc)
        domain_resources = self.get_domain_resources()
        
        # 计算总的可用CPU资源
        total_available_cpu = sum(
            res['available_cpu'] for res in domain_resources.values()
        )
        
        if total_available_cpu <= 0:
            # 资源不足，返回空分配
            print(f"[警告] SFC {sfc_id}: 域总可用CPU为0，无法分解")
            return {domain_id: [] for domain_id in self.domains.keys()}
        
        # 计算每个域应该分配的VNF数量（公式12）
        domain_allocations = {}
        allocated_vnfs = 0
        
        for domain_id, resources in domain_resources.items():
            available_cpu = resources['available_cpu']
            
            # 按可用CPU比例分配VNF数量
            ratio = available_cpu / total_available_cpu
            num_vnfs = int(np.floor(sfc_length * ratio))
            
            domain_allocations[domain_id] = num_vnfs
            allocated_vnfs += num_vnfs
        
        # 处理由于取整导致的VNF未完全分配问题
        # 将剩余VNF分配给资源最多的域
        remaining_vnfs = sfc_length - allocated_vnfs
        if remaining_vnfs > 0:
            # 找到可用CPU最多的域
            max_cpu_domain = max(
                domain_resources.keys(),
                key=lambda d: domain_resources[d]['available_cpu']
            )
            domain_allocations[max_cpu_domain] += remaining_vnfs

        # 进一步平衡：如果SFC长度足够长，则尽量保证每个域至少分配到1个VNF，
        # 避免某些域（如日志中的域1）长期为0段而无法参与训练。
        if sfc_length >= len(domain_allocations):
            zero_domains = [d for d, n in domain_allocations.items() if n == 0]
            for d in zero_domains:
                # 从当前分配最多且数量>1的域借出一个VNF
                donor_candidates = [k for k, n in domain_allocations.items() if n > 1]
                if not donor_candidates:
                    break
                donor = max(donor_candidates, key=lambda k: domain_allocations[k])
                domain_allocations[donor] -= 1
                domain_allocations[d] += 1
        
        # 根据分配数量切分SFC
        sfc_segments = {}
        start_idx = 0
        
        for domain_id in sorted(domain_allocations.keys()):
            num_vnfs = domain_allocations[domain_id]
            end_idx = start_idx + num_vnfs
            
            # 提取VNF段
            segment = sfc[start_idx:end_idx]
            sfc_segments[domain_id] = segment
            
            start_idx = end_idx
        
        # 打印分解结果
        print(f"[SFC分解] SFC {sfc_id} (长度={sfc_length}) 分解结果:")
        for domain_id, segment in sfc_segments.items():
            print(f"  域 {domain_id}: {len(segment)} 个VNF - {segment}")
        
        return sfc_segments
    
    def decompose_batch_sfcs(self, sfc_requests: List[Tuple[int, List[str]]]) -> Dict[int, List[Tuple[int, List[str]]]]:
        """
        批量分解多个SFC请求
        
        Args:
            sfc_requests: [(sfc_id, sfc), ...]
        
        Returns:
            {domain_id: [(sfc_id, sfc_segment), ...]}
        """
        domain_sfc_segments = {domain_id: [] for domain_id in self.domains.keys()}
        
        for sfc_id, sfc in sfc_requests:
            segments = self.decompose_sfc(sfc, sfc_id)
            
            for domain_id, segment in segments.items():
                if len(segment) > 0:  # 只添加非空段
                    domain_sfc_segments[domain_id].append((sfc_id, segment))
        
        print(f"\n[批量分解完成] 共 {len(sfc_requests)} 个SFC请求")
        for domain_id, segments in domain_sfc_segments.items():
            print(f"  域 {domain_id}: 分配到 {len(segments)} 个SFC段")
        
        return domain_sfc_segments
    
    def validate_decomposition(self, original_sfc: List[str], 
                              segments: Dict[int, List[str]]) -> bool:
        """
        验证分解的正确性
        
        Args:
            original_sfc: 原始SFC
            segments: 分解后的段
        
        Returns:
            True如果分解正确，False否则
        """
        # 重构SFC
        reconstructed = []
        for domain_id in sorted(segments.keys()):
            reconstructed.extend(segments[domain_id])
        
        # 检查长度
        if len(reconstructed) != len(original_sfc):
            print(f"[验证失败] 长度不匹配: {len(reconstructed)} != {len(original_sfc)}")
            return False
        
        # 检查内容（顺序）
        if reconstructed != original_sfc:
            print(f"[验证失败] 内容不匹配")
            return False
        
        return True
    
    def get_decomposition_statistics(self, segments: Dict[int, List[str]]) -> Dict:
        """
        获取分解统计信息
        
        Args:
            segments: 分解后的段
        
        Returns:
            统计信息字典
        """
        stats = {
            'num_domains_used': sum(1 for seg in segments.values() if len(seg) > 0),
            'segment_lengths': {did: len(seg) for did, seg in segments.items()},
            'total_vnfs': sum(len(seg) for seg in segments.values()),
            'max_segment_length': max(len(seg) for seg in segments.values()) if segments else 0,
            'min_segment_length': min(len(seg) for seg in segments.values() if len(seg) > 0) if segments else 0,
        }
        
        return stats
