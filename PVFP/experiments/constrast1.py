# -*- coding: utf-8 -*-
"""PVFP 与 Algorithm2 小规模网络成本对比实验脚本

在同一 small 拓扑和同一批 SFC 请求上：
- 使用已训练好的 PVFP 模型进行推理，记录每条 SFC 的总成本（部署+链路）；
- 使用 Algorithm2 在相同拓扑与请求上求解，记录其成本；
- 只对两种方法都成功的 SFC 进行逐条成本对比，并计算各自的平均成本。

注意：不修改任何现有文件，仅作为独立对比脚本使用。
"""

import os
import sys
import time
from typing import List, Dict, Tuple

import numpy as np

# 兼容从 PVFP/experiments 目录运行
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, "..", ".."))
PVFP_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))

if PVFP_DIR not in sys.path:
    sys.path.append(PVFP_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from main import PVFPFramework  # type: ignore
from config import MODEL_SAVE_PATH  # type: ignore
from Algorithm2 import Algorithm2  # type: ignore
from MGraph import MDWGraph, inf  # type: ignore


def find_latest_model(scale: str = "small") -> str:
    """在 MODEL_SAVE_PATH/<scale>/ 目录下查找最新的 .npz 模型文件。"""
    model_dir = os.path.join(MODEL_SAVE_PATH, scale)
    if not os.path.exists(model_dir):
        return ""

    candidates = [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(".npz")
    ]
    if not candidates:
        return ""

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def build_algorithm2_topology_from_pvfp(pvfp: PVFPFramework) -> Tuple[MDWGraph, List[int], np.ndarray, List[float]]:
    """将 PVFP 的 networkx 拓扑转换为 Algorithm2 使用的 MDWGraph 及约束结构。"""
    topo = pvfp.topology
    num_nodes = topo.number_of_nodes()

    # 假定节点编号为 0..N-1
    V = list(range(num_nodes))

    # 邻接矩阵 E：使用 PVFP 链路的 cost 属性作为边权
    E = np.full((num_nodes, num_nodes), inf, dtype=float)
    for i in range(num_nodes):
        E[i][i] = 0.0

    for u, v in topo.edges():
        edge_data = topo.edges[(u, v)]
        cost = float(edge_data.get("cost", 1.0))
        E[u][v] = cost
        E[v][u] = cost

    initial_Graph = MDWGraph(V, E)

    # 功能结点集合 function_V：直接复用 PVFP 中的功能结点
    function_V = sorted(getattr(pvfp, "function_nodes"))

    # 链路资源限制 E_constrains：设为很大，避免资源删除边
    E_constrains = np.zeros((num_nodes, num_nodes), dtype=float)
    LARGE_CAP = 1e9
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if E[i][j] != inf and i != j:
                E_constrains[i][j] = E_constrains[j][i] = LARGE_CAP

    # 功能结点资源限制：设为很大，避免资源不足
    function_V_constrains = [LARGE_CAP for _ in function_V]

    return initial_Graph, function_V, E_constrains, function_V_constrains


def convert_sfc_to_algorithm2_request(
    sfc: Dict,
    pvfp: PVFPFramework,
    function_V: List[int],
) -> Dict:
    """将 PVFP 的 SFC 请求转换为 Algorithm2 期望的 sk 字典结构。"""
    source = int(sfc.get("source"))
    destination = int(sfc.get("destination"))
    vnf_seq: List[str] = list(sfc.get("vnf_sequence", []))
    function_num = len(vnf_seq)

    if function_num == 0:
        # 空 SFC 直接返回不可用请求
        return {}

    # Algorithm2 中 SFC 为 [1, 2, ..., function_num]
    SFC = list(range(1, function_num + 1))

    # 带宽需求 bk：复用 PVFP 的 bandwidth_requirement
    bk = float(sfc.get("bandwidth_requirement", 1.0))

    topo = pvfp.topology

    # 构造部署成本矩阵：行=功能结点，列=功能位置
    # 对于位置 j 的 VNF，使用相应类型在该结点上的部署成本
    function_V_num = len(function_V)
    deployment_cost = np.zeros((function_V_num, function_num), dtype=float)

    for fi, node in enumerate(function_V):
        node_data = topo.nodes[node]
        node_costs = node_data.get("deployment_costs", {})
        for j in range(function_num):
            vnf_type = vnf_seq[j]
            cost = float(node_costs.get(vnf_type, 0.0))
            deployment_cost[fi][j] = cost

    # 为简单起见，设定有 2 个并行功能，从位置 1 开始（若长度不足，则退化为无效请求）
    if function_num >= 2:
        parallel_num = 2
        parallel_position = 1
    else:
        # Algorithm2 在 parallel_num<=1 时不工作，这里直接返回空请求
        return {}

    # alpha 用于时延计算，这里给定一个常数向量即可
    alpha = np.ones(len(function_V), dtype=float)

    sk = {
        "source": source,
        "destination": destination,
        "SFC": SFC,
        "bk": bk,
        "deployment_cost": deployment_cost,
        "parallel_num": parallel_num,
        "parallel_position": parallel_position,
        "function_num": function_num,
        "alpha": alpha,
    }

    return sk


def run_pvfp_on_sfcs(pvfp: PVFPFramework, num_sfcs: int):
    """在 small 拓扑上运行 PVFP 推理，返回使用到的 SFC 列表和评估结果。"""
    # 先生成一批 SFC，请后续 Algorithm2 共用
    test_sfcs = pvfp.sfc_generator.generate_batch_sfcs(num_sfcs)

    # 临时替换 PVFP 的 generate_batch_sfcs，使 evaluate 使用同一批 SFC
    original_generate = pvfp.sfc_generator.generate_batch_sfcs

    def _fixed_generate(n: int):  # n 被忽略，始终返回同一批 SFC
        return test_sfcs

    pvfp.sfc_generator.generate_batch_sfcs = _fixed_generate
    try:
        evaluation_results = pvfp.evaluate(num_sfcs=num_sfcs)
    finally:
        pvfp.sfc_generator.generate_batch_sfcs = original_generate

    return test_sfcs, evaluation_results


def run_algorithm2_on_sfcs(
    algo2: Algorithm2,
    pvfp: PVFPFramework,
    initial_Graph: MDWGraph,
    function_V: List[int],
    E_constrains: np.ndarray,
    function_V_constrains: List[float],
    test_sfcs: List[Dict],
) -> List[Dict]:
    """在与 PVFP 相同拓扑和 SFC 上运行 Algorithm2，返回每条 SFC 的结果信息。"""
    results: List[Dict] = []

    for idx, sfc in enumerate(test_sfcs):
        sk = convert_sfc_to_algorithm2_request(sfc, pvfp, function_V)
        if not sk:
            # 构造失败，视为该条无法处理
            results.append({"success": False, "weight": 0.0})
            continue

        try:
            shortest_path, deploy_server, function_order, weight = algo2.handle_a_request(
                sk,
                initial_Graph,
                function_V,
                E_constrains,
                function_V_constrains,
            )
        except Exception:
            # Algorithm2 在该请求上出现内部错误，视为失败
            results.append({"success": False, "weight": 0.0})
            continue

        success = bool(shortest_path and weight != 0)
        results.append(
            {
                "success": success,
                "weight": float(weight) if success else 0.0,
                "shortest_path": shortest_path,
                "deploy_server": deploy_server,
                "function_order": function_order,
            }
        )

    return results


def run_contrast_experiment(num_sfcs: int = 30):
    """在 small 网络上比较 PVFP 与 Algorithm2 的成本表现。"""
    print("\n" + "=" * 80)
    print(" " * 20 + "PVFP 与 Algorithm2 小规模网络成本对比实验")
    print("=" * 80)

    start_time = time.time()

    # 1. 初始化 PVFP 框架并加载最近的模型
    pvfp = PVFPFramework(scale="small", num_domains=3)

    model_path = find_latest_model(scale="small")
    if not model_path:
        print("[错误] 未找到已保存的 PVFP 模型，请先完成训练。")
        return

    pvfp.load_model(model_path)

    # 2. 在同一批 SFC 上运行 PVFP 推理
    test_sfcs, pvfp_eval = run_pvfp_on_sfcs(pvfp, num_sfcs=num_sfcs)
    pvfp_per_sfc = pvfp_eval.get("per_sfc_results", [])

    # 3. 构建 Algorithm2 需要的拓扑和约束
    initial_Graph, function_V, E_constrains, function_V_constrains = build_algorithm2_topology_from_pvfp(pvfp)

    # 4. 在相同拓扑和 SFC 上运行 Algorithm2
    algo2 = Algorithm2()
    algo2_results = run_algorithm2_on_sfcs(
        algo2,
        pvfp,
        initial_Graph,
        function_V,
        E_constrains,
        function_V_constrains,
        test_sfcs,
    )

    # 5. 汇总并对比：只统计两种方法都成功的 SFC
    print("\n" + "-" * 80)
    print("逐 SFC 成本对比 (仅列出两种方法均成功的请求)：")
    print("-" * 80)

    common_indices: List[int] = []
    for idx in range(min(len(pvfp_per_sfc), len(algo2_results))):
        sfc_res = pvfp_per_sfc[idx]
        a2_res = algo2_results[idx]

        pvfp_success = bool(sfc_res.get("success", False))
        algo2_success = bool(a2_res.get("success", False))

        if not (pvfp_success and algo2_success):
            continue

        metrics = sfc_res.get("metrics") or {}
        pvfp_cost = float(metrics.get("total_resource_cost", 0.0))
        algo2_cost = float(a2_res.get("weight", 0.0))

        sfc_id = sfc_res.get("sfc_id", idx)
        print(
            f"SFC {sfc_id}: PVFP成本={pvfp_cost:.2f}, "
            f"Algorithm2成本={algo2_cost:.2f}"
        )

        common_indices.append(idx)

    if not common_indices:
        print("\n[提示] 在本次实验中，两种方法没有在同一条 SFC 上同时成功，无法计算平均成本对比。")
        return

    # 6. 计算两种方法在共同成功 SFC 上的平均成本
    total_pvfp = 0.0
    total_algo2 = 0.0

    for idx in common_indices:
        sfc_res = pvfp_per_sfc[idx]
        a2_res = algo2_results[idx]
        metrics = sfc_res.get("metrics") or {}
        total_pvfp += float(metrics.get("total_resource_cost", 0.0))
        total_algo2 += float(a2_res.get("weight", 0.0))

    n = len(common_indices)
    avg_pvfp = total_pvfp / n
    avg_algo2 = total_algo2 / n

    print("\n" + "=" * 80)
    print(f"共 {n} 条 SFC 两种方法均成功部署：")
    print(f"  PVFP 平均成本:      {avg_pvfp:.2f}")
    print(f"  Algorithm2 平均成本: {avg_algo2:.2f}")
    print("=" * 80)

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"对比实验总用时: {total_time/60:.2f} 分钟")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_contrast_experiment(num_sfcs=30)
