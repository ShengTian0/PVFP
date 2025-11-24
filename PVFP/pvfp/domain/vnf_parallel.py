# -*- coding: utf-8 -*-
"""
VNF并行化规则实现 (Rule 1, Rule 2, Rule 3)
实现论文中的三条并行性规则，判断VNF是否可以并行部署
"""

import numpy as np
from typing import Dict, List, Tuple, Set


class VNFParallelRules:
    """VNF并行规则管理器"""
    
    def __init__(self):
        """初始化VNF并行关系配置"""
        # VNF类型
        self.vnf_types = ['NAT', 'LB', 'NIDS', 'Gateway', 'VPN', 'FW', 'Caching']
        
        # Rule 1: 执行顺序依赖关系
        # 如果 (fi, fj) 在此集合中，表示fi必须在fj之前执行（不可并行）
        self.order_dependencies = self._init_order_dependencies()
        
        # Rule 2: 位置依赖关系
        # first: 必须在最前面, mid: 可以在中间, final: 必须在最后
        self.position_constraints = self._init_position_constraints()
        
        # Rule 3: 优先级（用于冲突解决）
        # 数字越大优先级越高
        self.priority_levels = self._init_priority_levels()
    
    def _init_order_dependencies(self) -> Set[Tuple[str, str]]:
        """
        初始化顺序依赖关系
        返回不可并行的VNF对集合
        """
        dependencies = set()
        
        # 示例依赖关系（根据实际VNF特性定义）
        # NAT必须在某些VNF之前
        dependencies.add(('NAT', 'FW'))
        
        # 负载均衡应该在入侵检测之前
        dependencies.add(('LB', 'NIDS'))
        
        # Gateway通常在早期
        dependencies.add(('Gateway', 'VPN'))
        dependencies.add(('Gateway', 'Caching'))
        
        # 防火墙在VPN之前
        dependencies.add(('FW', 'VPN'))
        
        return dependencies
    
    def _init_position_constraints(self) -> Dict[str, str]:
        """
        初始化位置约束
        返回每个VNF的位置要求: 'first', 'mid', 'final'
        """
        constraints = {}
        
        # Gateway通常是第一个
        constraints['Gateway'] = 'first'
        
        # NAT通常在前面
        constraints['NAT'] = 'first'
        
        # Caching通常在最后
        constraints['Caching'] = 'final'
        
        # 其他VNF可以在中间
        constraints['LB'] = 'mid'
        constraints['NIDS'] = 'mid'
        constraints['VPN'] = 'mid'
        constraints['FW'] = 'mid'
        
        return constraints
    
    def _init_priority_levels(self) -> Dict[str, int]:
        """
        初始化优先级等级
        返回每个VNF的优先级（1-10，数字越大优先级越高）
        """
        priorities = {}
        
        # 安全相关VNF优先级最高
        priorities['FW'] = 10
        priorities['NIDS'] = 9
        priorities['VPN'] = 8
        
        # 网络基础VNF
        priorities['Gateway'] = 7
        priorities['NAT'] = 7
        
        # 性能优化VNF
        priorities['LB'] = 6
        priorities['Caching'] = 5
        
        return priorities
    
    def check_rule1_order(self, vnf_i: str, vnf_j: str) -> bool:
        """
        Rule 1: 检查执行顺序依赖
        
        Args:
            vnf_i: 第一个VNF类型
            vnf_j: 第二个VNF类型
        
        Returns:
            True 如果可以并行（无顺序依赖）, False 如果有顺序依赖
        """
        # 检查是否存在顺序依赖
        if (vnf_i, vnf_j) in self.order_dependencies:
            return False  # vnf_i必须在vnf_j之前，不能并行
        if (vnf_j, vnf_i) in self.order_dependencies:
            return False  # vnf_j必须在vnf_i之前，不能并行
        
        return True  # 无顺序依赖，可以并行
    
    def check_rule2_position(self, vnf_i: str, vnf_j: str, 
                            position_i: int, position_j: int,
                            sfc_length: int) -> bool:
        """
        Rule 2: 检查位置依赖约束
        
        Args:
            vnf_i: 第一个VNF类型
            vnf_j: 第二个VNF类型
            position_i: vnf_i在SFC中的位置索引
            position_j: vnf_j在SFC中的位置索引
            sfc_length: SFC总长度
        
        Returns:
            True 如果位置约束允许并行, False 否则
        """
        constraint_i = self.position_constraints.get(vnf_i, 'mid')
        constraint_j = self.position_constraints.get(vnf_j, 'mid')
        
        # 检查first约束
        if constraint_i == 'first' and position_i != 0:
            return False  # first VNF必须在位置0
        if constraint_j == 'first' and position_j != 0:
            return False
        
        # 两个都是first的不能并行（只能有一个在位置0）
        if constraint_i == 'first' and constraint_j == 'first':
            return False
        
        # 检查final约束
        if constraint_i == 'final' and position_i != sfc_length - 1:
            return False  # final VNF必须在最后位置
        if constraint_j == 'final' and position_j != sfc_length - 1:
            return False
        
        # 两个都是final的不能并行（只能有一个在最后位置）
        if constraint_i == 'final' and constraint_j == 'final':
            return False
        
        return True  # 位置约束满足
    
    def check_rule3_priority(self, vnf_i: str, vnf_j: str) -> Tuple[bool, str]:
        """
        Rule 3: 优先级冲突解决
        当存在资源冲突时，根据优先级决定保留哪个VNF
        
        Args:
            vnf_i: 第一个VNF类型
            vnf_j: 第二个VNF类型
        
        Returns:
            (是否有冲突, 优先保留的VNF)
        """
        priority_i = self.priority_levels.get(vnf_i, 5)
        priority_j = self.priority_levels.get(vnf_j, 5)
        
        if priority_i > priority_j:
            return True, vnf_i
        elif priority_j > priority_i:
            return True, vnf_j
        else:
            # 优先级相同，随机选择或按名称排序
            return True, vnf_i if vnf_i < vnf_j else vnf_j
    
    def is_parallelizable(self, vnf_i: str, vnf_j: str,
                         position_i: int = None, position_j: int = None,
                         sfc_length: int = None) -> bool:
        """
        综合判断两个VNF是否可以并行部署
        
        Args:
            vnf_i: 第一个VNF类型
            vnf_j: 第二个VNF类型
            position_i: vnf_i的位置（可选）
            position_j: vnf_j的位置（可选）
            sfc_length: SFC长度（可选）
        
        Returns:
            True 如果可以并行, False 否则
        """
        # Rule 1: 检查顺序依赖
        if not self.check_rule1_order(vnf_i, vnf_j):
            return False
        
        # Rule 2: 如果提供了位置信息，检查位置约束
        if position_i is not None and position_j is not None and sfc_length is not None:
            if not self.check_rule2_position(vnf_i, vnf_j, position_i, position_j, sfc_length):
                return False
        
        # Rule 3在资源分配时使用，这里只做并行性判断
        return True
    
    def get_parallel_groups(self, sfc: List[str]) -> List[List[int]]:
        """
        将SFC分解为可并行执行的VNF组
        
        Args:
            sfc: VNF序列列表
        
        Returns:
            并行组列表，每个组包含可以并行执行的VNF索引
        """
        n = len(sfc)
        parallel_groups = []
        assigned = [False] * n
        
        for i in range(n):
            if assigned[i]:
                continue
            
            # 创建新的并行组
            current_group = [i]
            assigned[i] = True
            
            # 查找可以与当前组并行的VNF
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                
                # 检查是否可以与组内所有VNF并行
                can_parallel = True
                for k in current_group:
                    if not self.is_parallelizable(sfc[k], sfc[j], k, j, n):
                        can_parallel = False
                        break
                
                if can_parallel:
                    current_group.append(j)
                    assigned[j] = True
            
            parallel_groups.append(current_group)
        
        return parallel_groups
    
    def resolve_conflict(self, vnfs: List[str], available_resource: float,
                        required_resource: float) -> List[str]:
        """
        当资源不足时，根据优先级解决冲突
        
        Args:
            vnfs: 冲突的VNF列表
            available_resource: 可用资源
            required_resource: 每个VNF所需资源
        
        Returns:
            根据优先级筛选后的VNF列表
        """
        # 按优先级排序
        sorted_vnfs = sorted(vnfs, 
                           key=lambda v: self.priority_levels.get(v, 5),
                           reverse=True)
        
        selected = []
        used_resource = 0.0
        
        for vnf in sorted_vnfs:
            if used_resource + required_resource <= available_resource:
                selected.append(vnf)
                used_resource += required_resource
            else:
                break
        
        return selected
