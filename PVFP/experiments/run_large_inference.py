# -*- coding: utf-8 -*-
"""大规模网络推理/评估脚本
只加载已训练模型并在固定数量的SFC上进行评估，不进行训练。
"""

import sys
import os
import time

sys.path.append('..')

from main import PVFPFramework
from config import MODEL_SAVE_PATH


def find_latest_model(scale='large'):
    """在MODEL_SAVE_PATH/<scale>/目录下查找最新的npz模型文件"""
    model_dir = os.path.join(MODEL_SAVE_PATH, scale)
    if not os.path.exists(model_dir):
        return None

    candidates = [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith('.npz')
    ]
    if not candidates:
        return None

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def run_large_scale_inference(num_sfcs=30):
    """运行大规模网络的纯推理/评估实验"""
    print("\n" + "=" * 80)
    print(" " * 25 + "大规模网络推理/评估实验")
    print("=" * 80)

    start_time = time.time()

    # 创建PVFP框架（不会训练，只用于构建拓扑、环境和agent结构）
    pvfp = PVFPFramework(scale='large', num_domains=3)

    # 查找并加载最近一次保存的模型
    model_path = find_latest_model(scale='large')
    if model_path is None:
        print("[错误] 未找到已保存的large规模模型，请先运行大规模训练脚本生成模型。")
        return None, None

    print(f"[信息] 加载模型: {model_path}")
    pvfp.load_model(model_path)

    # 评估
    evaluation_results = pvfp.evaluate(num_sfcs=num_sfcs)

    # 打印每个SFC的详细推理结果
    per_sfc_results = evaluation_results.get('per_sfc_results', [])
    print("\n" + "-" * 80)
    print("逐SFC推理结果 (large):")
    print("-" * 80)
    for sfc_res in per_sfc_results:
        sfc_id = sfc_res.get('sfc_id')
        vnf_seq = sfc_res.get('vnf_sequence', [])
        success = sfc_res.get('success', False)
        assigned_domain = sfc_res.get('assigned_domain')
        metrics = sfc_res.get('metrics') or {}
        placements = sfc_res.get('vnf_placements') or {}

        print(f"SFC {sfc_id}:")
        print(f"  源节点: {sfc_res.get('source')} -> 目的节点: {sfc_res.get('destination')}")
        print(f"  VNF序列({len(vnf_seq)}): {vnf_seq}")
        print(f"  部署结果: {'成功' if success else '失败'}")
        if success:
            # assigned_domain 现在可能是单个域ID，也可能是域ID列表
            if isinstance(assigned_domain, (list, tuple)):
                print(f"  参与部署的域: {assigned_domain}")
            else:
                print(f"  部署域: 域 {assigned_domain}")
            print(f"  总延迟: {metrics.get('total_latency', 0):.2f} ms")
            print(f"  平均每VNF延迟: {metrics.get('avg_latency_per_vnf', 0):.2f} ms")
            print(f"  总成本(部署+链路): {metrics.get('total_resource_cost', 0):.2f}")
            print(
                f"  成本构成: 部署成本={metrics.get('cpu_resource_cost', 0):.2f}, "
                f"链路成本={metrics.get('bw_resource_cost', 0):.2f}"
            )
            print(f"  部署率: {metrics.get('deployment_rate', 0)*100:.1f}%")
            # 全局 VNF 到节点的部署映射
            if placements:
                print("  全局部署映射(按原始VNF索引):")
                for vnf_idx, node in sorted(placements.items()):
                    if 0 <= vnf_idx < len(vnf_seq):
                        print(f"    VNF {vnf_idx} ({vnf_seq[vnf_idx]}) -> 节点 {node}")
                    else:
                        print(f"    VNF {vnf_idx} -> 节点 {node}")

                # 基于部署映射构造按VNF顺序的数据包节点链
                ordered_nodes = []
                for vnf_idx in sorted(placements.keys()):
                    ordered_nodes.append(placements[vnf_idx])
                print(f"  VNF链节点序列(按VNF顺序): {ordered_nodes}")

            # VNF间通信路径及其链路成本（聚合后的）
            paths = metrics.get('paths', []) or []
            path_costs = metrics.get('path_link_costs', []) or []
            if paths:
                print("  VNF间路径与链路成本(跨域聚合):")
                for i, path in enumerate(paths):
                    cost = path_costs[i] if i < len(path_costs) else 0.0
                    print(f"    路径 {i}: 节点序列 {path}, 链路成本={cost:.2f}")

            # 逐域打印本SFC分配到的段及其本地域内部署详情
            domain_details = sfc_res.get('domain_details', {}) or {}
            if domain_details:
                print("  按域分段部署详情:")
                for did, info in sorted(domain_details.items()):
                    seg = info.get('segment', [])
                    start_index = info.get('start_index', 0)
                    local_place = info.get('vnf_placements_local', {}) or {}
                    print(f"    域 {did}: 段起始全局索引={start_index}, 段VNF序列={seg}")
                    if local_place:
                        print("      本域局部部署映射(局部idx -> 节点):")
                        for lidx, node in sorted(local_place.items()):
                            if 0 <= lidx < len(seg):
                                print(f"        局部VNF {lidx} ({seg[lidx]}) -> 节点 {node}")
                            else:
                                print(f"        局部VNF {lidx} -> 节点 {node}")
        print("-" * 80)

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"large规模推理/评估总用时: {total_time/60:.2f} 分钟")
    print("=" * 80 + "\n")

    return evaluation_results


if __name__ == "__main__":
    run_large_scale_inference(num_sfcs=30)
