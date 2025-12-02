# -*- coding: utf-8 -*-
"""PVFP 与 Algorithm2 统一成本配置的大规模网络对比实验脚本 (constrast1_1_large)

实验设置：
- 使用 large 拓扑和同一批 SFC 请求；
- 将所有链路成本统一为 3；
- 将所有功能节点的部署成本预算上限统一为 9999；
- 分别在所有 VNF 部署成本统一为 {10, 13, 15, 17, 19} 时，比较 PVFP 与 Algorithm2 的总成本；
- 只对两种方法都成功的 SFC 进行逐条成本对比，并计算各自的平均成本。

不修改任何现有文件，仅作为独立对比脚本使用。
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
from config import MODEL_SAVE_PATH, VNF_TYPES  # type: ignore
from Algorithm2 import Algorithm2  # type: ignore
from MGraph import MDWGraph, inf  # type: ignore


UNIFORM_LINK_COST = 5.0
UNIFORM_BUDGET_CAP = 9999.0
DEPLOY_COST_VALUES = [1.0, 3.0, 5.0, 7.0, 9.0]

# 是否在推理/对比过程中打印详细日志
VERBOSE = True


def find_latest_model(scale: str = "large") -> str:
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


def apply_uniform_costs(
    pvfp: PVFPFramework,
    link_cost: float,
    budget_cap: float,
    deploy_cost_val: float,
) -> None:
    """在现有 PVFP 拓扑上施加统一的链路成本和部署成本设置。"""
    topo = pvfp.topology

    # 1) 所有边的链路成本统一为 link_cost
    for u, v in topo.edges():
        topo.edges[(u, v)]["cost"] = float(link_cost)

    # 2) 所有功能节点的部署预算上限统一为 budget_cap，部署成本统一为 deploy_cost_val
    for node in topo.nodes():
        node_data = topo.nodes[node]
        if not node_data.get("is_function_node", False):
            # 非功能节点，不参与 VNF 部署
            continue

        # 预算上限
        node_data["deploy_budget_capacity"] = float(budget_cap)

        # 各 VNF 类型的部署成本统一
        deploy_costs = node_data.get("deployment_costs", {})
        for vnf_type in VNF_TYPES:
            deploy_costs[vnf_type] = float(deploy_cost_val)
        node_data["deployment_costs"] = deploy_costs


def print_uniform_cost_snapshot(pvfp: PVFPFramework, deploy_cost_val: float) -> None:
    """调试输出：查看统一成本设置是否已经写入拓扑。

    仅用于对比脚本本身的可视化，不影响算法逻辑。
    """
    if not VERBOSE:
        return

    topo = pvfp.topology
    print("\n[统一成本后-功能节点部署预算与部署成本]")
    for node in sorted(topo.nodes()):
        data = topo.nodes[node]
        if not data.get("is_function_node", False):
            continue
        budget = data.get("deploy_budget_capacity", 0)
        costs = data.get("deployment_costs", {})
        print(f"  节点 {node}: 预算={budget}, 部署成本={costs}")

    print("\n[统一成本后-部分链路属性]")
    shown = 0
    for u, v in topo.edges():
        edge = topo.edges[(u, v)]
        cost = edge.get("cost", None)
        delay = edge.get("delay", None)
        print(f"  边 ({u}, {v}): 成本={cost}, 时延={delay:.2f}ms")
        shown += 1
        if shown >= 5:
            break


def build_algorithm2_topology_from_pvfp(
    pvfp: PVFPFramework,
    link_cost: float,
    budget_cap: float,
) -> Tuple[MDWGraph, List[int], np.ndarray, List[float]]:
    """将 PVFP 的 networkx 拓扑转换为 Algorithm2 使用的 MDWGraph 及约束结构。

    为满足实验要求，这里显式将所有链路权重设为 link_cost，
    并将所有功能节点的资源上限 (function_V_constrains) 设为 budget_cap。
    """
    topo = pvfp.topology
    # 注意：large 拓扑的节点编号可能是 0..49 的稀疏子集，
    # 不能简单用节点数量作为矩阵大小，否则会在 E[u][v] 处越界。
    nodes = sorted(topo.nodes())
    if not nodes:
        raise ValueError("拓扑中没有任何节点，无法构建 Algorithm2 的图结构")

    max_node_id = max(nodes)
    matrix_size = max_node_id + 1

    # Algorithm2 中默认假设节点编号可直接作为矩阵下标，
    # 因此这里令 V = [0, 1, ..., max_node_id]，矩阵大小与之对应。
    V = list(range(matrix_size))

    # 邻接矩阵 E：所有存在的边权重均为 link_cost
    E = np.full((matrix_size, matrix_size), inf, dtype=float)
    for i in range(matrix_size):
        E[i][i] = 0.0

    for u, v in topo.edges():
        E[u][v] = float(link_cost)
        E[v][u] = float(link_cost)

    initial_Graph = MDWGraph(V, E)

    # 功能结点集合 function_V：直接复用 PVFP 中的功能结点
    function_V = sorted(getattr(pvfp, "function_nodes"))

    # 链路资源限制 E_constrains：给足够大的容量，避免因带宽约束删边
    E_constrains = np.zeros((matrix_size, matrix_size), dtype=float)
    LARGE_CAP = 1e9
    for i in range(matrix_size):
        for j in range(i + 1, matrix_size):
            if E[i][j] != inf and i != j:
                E_constrains[i][j] = E_constrains[j][i] = LARGE_CAP

    # 功能结点资源限制：统一为 budget_cap
    function_V_constrains = [float(budget_cap) for _ in function_V]

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


def evaluate_pvfp_on_given_sfcs(pvfp: PVFPFramework, test_sfcs: List[Dict]) -> Dict:
    """在给定的一批 SFC 上运行 PVFP 推理，返回评估结果。"""
    original_generate = pvfp.sfc_generator.generate_batch_sfcs

    def _fixed_generate(n: int):  # n 被忽略，始终返回同一批 SFC
        return test_sfcs

    pvfp.sfc_generator.generate_batch_sfcs = _fixed_generate
    try:
        evaluation_results = pvfp.evaluate(num_sfcs=len(test_sfcs))
    finally:
        pvfp.sfc_generator.generate_batch_sfcs = original_generate

    return evaluation_results


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


def run_uniform_cost_experiments_large(num_sfcs: int = 30):
    """在统一成本设置下，针对 large 拓扑比较 PVFP 与 Algorithm2。"""
    if VERBOSE:
        print("\n" + "=" * 80)
        print(" " * 15 + "PVFP 与 Algorithm2 统一成本配置大规模网络对比实验 (constrast1_1_large)")
        print("=" * 80)

    start_time = time.time()

    # 1. 初始化 PVFP 框架并加载最近的 large 模型
    pvfp = PVFPFramework(scale="large", num_domains=3)

    model_path = find_latest_model(scale="large")
    if not model_path:
        print("[错误] 未找到已保存的 PVFP large 模型，请先完成大规模网络训练。")
        return

    pvfp.load_model(model_path)

    # 2. 生成一批固定的 SFC 请求，供所有成本设置共用
    test_sfcs = pvfp.sfc_generator.generate_batch_sfcs(num_sfcs)

    summary_results = []  # (deploy_cost, num_common, avg_pvfp, avg_algo2)

    for deploy_cost in DEPLOY_COST_VALUES:
        if VERBOSE:
            print("\n" + "-" * 80)
            print(f"部署成本统一为 {deploy_cost:.1f} 时的对比结果：")
            print("-" * 80)

        # 重置拓扑的 CPU/带宽使用情况，避免前一轮评估占用资源
        if hasattr(pvfp, "topo_loader"):
            pvfp.topo_loader.reset_topology()

        # 2.1 在 PVFP 拓扑上施加统一成本设置
        apply_uniform_costs(
            pvfp,
            link_cost=UNIFORM_LINK_COST,
            budget_cap=UNIFORM_BUDGET_CAP,
            deploy_cost_val=deploy_cost,
        )

        # 打印当前统一成本配置的快照，便于验证设置是否生效
        print_uniform_cost_snapshot(pvfp, deploy_cost)

        # 2.2 在同一批 SFC 上运行 PVFP 推理
        pvfp_eval = evaluate_pvfp_on_given_sfcs(pvfp, test_sfcs)
        pvfp_per_sfc = pvfp_eval.get("per_sfc_results", [])

        # 2.3 构建 Algorithm2 所需拓扑与约束（基于当前 PVFP 拓扑）
        initial_Graph, function_V, E_constrains, function_V_constrains = build_algorithm2_topology_from_pvfp(
            pvfp,
            link_cost=UNIFORM_LINK_COST,
            budget_cap=UNIFORM_BUDGET_CAP,
        )

        # 2.4 在相同拓扑和 SFC 上运行 Algorithm2
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

        # 2.5 汇总并对比：只统计两种方法都成功的 SFC
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
            if VERBOSE:
                print(
                    f"SFC {sfc_id}: PVFP成本={pvfp_cost:.2f}, "
                    f"Algorithm2成本={algo2_cost:.2f}"
                )

            common_indices.append(idx)

        if not common_indices:
            if VERBOSE:
                print("\n[提示] 在本次成本设置下，两种方法没有在同一条 SFC 上同时成功，无法计算平均成本对比。")
            summary_results.append((deploy_cost, 0, 0.0, 0.0))
            continue

        # 2.6 计算两种方法在共同成功 SFC 上的平均成本
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

        if VERBOSE:
            print("\n" + "=" * 80)
            print(f"部署成本={deploy_cost:.1f} 时，共 {n} 条 SFC 两种方法均成功部署：")
            print(f"  PVFP 平均成本:      {avg_pvfp:.2f}")
            print(f"  Algorithm2 平均成本: {avg_algo2:.2f}")
            print("=" * 80)

        summary_results.append((deploy_cost, n, avg_pvfp, avg_algo2))

    # 3. 汇总各个部署成本配置下的结果
    if VERBOSE:
        print("\n" + "#" * 80)
        print("各统一部署成本配置下的大规模网络总体结果汇总：")
        print("(部署成本, 共同成功SFC数, PVFP平均成本, Algorithm2平均成本)")
        for deploy_cost, n, avg_pvfp, avg_algo2 in summary_results:
            print(
                f"  c={deploy_cost:.1f}: N_common={n}, "
                f"PVFP_avg={avg_pvfp:.2f}, Algo2_avg={avg_algo2:.2f}"
            )
        print("#" * 80)

        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"大规模网络对比实验总用时: {total_time/60:.2f} 分钟")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    run_uniform_cost_experiments_large(num_sfcs=30)
