# -*- coding: utf-8 -*-
"""
测试并行规则模块
"""

import sys
sys.path.append('..')

from pvfp.domain.vnf_parallel import VNFParallelRules


def test_parallel_rules():
    """测试VNF并行规则"""
    
    print("\n" + "="*70)
    print("测试VNF并行规则")
    print("="*70 + "\n")
    
    rules = VNFParallelRules()
    
    # 测试Rule 1: 顺序依赖
    print("【测试 Rule 1: 顺序依赖】")
    test_pairs = [
        ('NAT', 'FW'),
        ('LB', 'NIDS'),
        ('Gateway', 'VPN'),
        ('LB', 'Caching')
    ]
    
    for vnf1, vnf2 in test_pairs:
        result = rules.check_rule1_order(vnf1, vnf2)
        status = "✓ 可并行" if result else "✗ 不可并行(有顺序依赖)"
        print(f"  {vnf1} <-> {vnf2}: {status}")
    
    # 测试Rule 2: 位置依赖
    print("\n【测试 Rule 2: 位置依赖】")
    sfc_length = 5
    test_positions = [
        ('Gateway', 'NAT', 0, 1),
        ('Gateway', 'Gateway', 0, 0),
        ('Caching', 'FW', 4, 3),
        ('LB', 'NIDS', 2, 3)
    ]
    
    for vnf1, vnf2, pos1, pos2 in test_positions:
        result = rules.check_rule2_position(vnf1, vnf2, pos1, pos2, sfc_length)
        status = "✓ 位置约束满足" if result else "✗ 位置约束冲突"
        print(f"  {vnf1}@{pos1} <-> {vnf2}@{pos2}: {status}")
    
    # 测试Rule 3: 优先级
    print("\n【测试 Rule 3: 优先级冲突解决】")
    test_conflicts = [
        ('FW', 'LB'),
        ('NIDS', 'Caching'),
        ('Gateway', 'NAT'),
        ('VPN', 'LB')
    ]
    
    for vnf1, vnf2 in test_conflicts:
        has_conflict, preferred = rules.check_rule3_priority(vnf1, vnf2)
        priority1 = rules.priority_levels.get(vnf1, 5)
        priority2 = rules.priority_levels.get(vnf2, 5)
        print(f"  {vnf1}(优先级{priority1}) vs {vnf2}(优先级{priority2}): 保留 {preferred}")
    
    # 测试并行组分解
    print("\n【测试 SFC并行组分解】")
    test_sfcs = [
        ['Gateway', 'NAT', 'LB', 'NIDS', 'Caching'],
        ['FW', 'VPN', 'LB', 'Caching'],
        ['NAT', 'LB', 'NIDS', 'FW', 'VPN', 'Caching']
    ]
    
    for sfc in test_sfcs:
        groups = rules.get_parallel_groups(sfc)
        print(f"\n  SFC: {sfc}")
        print(f"  并行组数: {len(groups)}")
        for i, group in enumerate(groups):
            vnfs_in_group = [sfc[idx] for idx in group]
            print(f"    组{i+1}: {vnfs_in_group} (索引: {group})")
    
    print("\n" + "="*70)
    print("测试完成!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_parallel_rules()
