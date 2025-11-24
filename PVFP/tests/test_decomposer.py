# -*- coding: utf-8 -*-
"""
测试SFC分解器
"""

import sys
sys.path.append('..')

from pvfp.utils.topo_loader import TopologyLoader, SFCGenerator
from pvfp.cloud.decomposer import SFCDecomposer


def test_decomposer():
    """测试SFC分解器"""
    
    print("\n" + "="*70)
    print("测试SFC分解器")
    print("="*70 + "\n")
    
    # 1. 生成拓扑
    print("[步骤 1] 生成拓扑...")
    topo_loader = TopologyLoader(scale='small')
    topology = topo_loader.generate_topology()
    
    # 2. 划分域
    print("\n[步骤 2] 划分域...")
    domains = topo_loader.partition_domains(num_domains=4)
    
    # 3. 创建分解器
    print("\n[步骤 3] 创建SFC分解器...")
    decomposer = SFCDecomposer(topology, domains)
    
    # 4. 生成SFC请求
    print("\n[步骤 4] 生成SFC请求...")
    sfc_gen = SFCGenerator(topology)
    sfcs = sfc_gen.generate_batch_sfcs(num_sfcs=5, min_length=4, max_length=8)
    
    # 5. 测试单个SFC分解
    print("\n" + "="*70)
    print("【测试单个SFC分解】")
    print("="*70)
    
    for sfc in sfcs[:2]:
        print(f"\n原始SFC {sfc['id']}:")
        sfc_gen.print_sfc(sfc)
        
        segments = decomposer.decompose_sfc(sfc['vnf_sequence'], sfc['id'])
        
        # 验证分解
        is_valid = decomposer.validate_decomposition(sfc['vnf_sequence'], segments)
        print(f"  分解验证: {'✓ 通过' if is_valid else '✗ 失败'}")
        
        # 统计信息
        stats = decomposer.get_decomposition_statistics(segments)
        print(f"\n  统计信息:")
        print(f"    使用域数: {stats['num_domains_used']}")
        print(f"    VNF总数: {stats['total_vnfs']}")
        print(f"    最大段长度: {stats['max_segment_length']}")
        print(f"    最小段长度: {stats['min_segment_length']}")
    
    # 6. 测试批量分解
    print("\n" + "="*70)
    print("【测试批量SFC分解】")
    print("="*70)
    
    sfc_list = [(sfc['id'], sfc['vnf_sequence']) for sfc in sfcs]
    domain_segments = decomposer.decompose_batch_sfcs(sfc_list)
    
    print("\n各域分配情况:")
    for domain_id, segments in domain_segments.items():
        print(f"\n域 {domain_id}:")
        print(f"  分配到的SFC段数: {len(segments)}")
        total_vnfs = sum(len(seg) for _, seg in segments)
        print(f"  总VNF数: {total_vnfs}")
        
        for sfc_id, segment in segments[:3]:  # 只显示前3个
            print(f"    SFC {sfc_id}: {segment}")
    
    print("\n" + "="*70)
    print("测试完成!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_decomposer()
