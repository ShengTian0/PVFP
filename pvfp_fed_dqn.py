#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pvfp_fed_dqn.py
复现论文: Parallel Placement of Virtualized Network Functions via FDRL
实现要点：
 - 多域（K domains）仿真环境
 - 每个域运行局部 DQN（3 层全连接 600 units，ReLU）
 - 云端执行延迟加权聚合 (Eq.11)
 - 软更新、资源导向 replay buffer、reward-based epsilon (Eq.22)
全部注释为中文（简体）。
适配：Python 3.12 + PyTorch 2.x
"""

import random
import math
import copy
import time
from collections import deque, namedtuple
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 配置（可按论文 Table II 设置）
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 网络/仿真参数（论文中默认值或建议值）
NUM_DOMAINS = 3                   # K，论文中实验采用 3 域
MAX_SFC_LEN = 10                   # 最大 SFC 长度（论文用 10）
VNF_TYPES = ['NAT','LB','NIDS','GW','VPN','FW','Caching']  # 7 种 VNF
NODE_CPU_SMALL = 30                # small-scale 每节点 CPU cores（论文小图）
NODE_CPU_LARGE = 20                # large-scale
LINK_BW_SMALL = 4.0                # Mbps
LINK_BW_LARGE = 2.0                # Mbps

# DQN & FDRL 超参（论文中推荐或实验值）
LEARNING_RATE = 5e-4               # α = 5e-4（论文选择该值进行比较）:contentReference[oaicite:6]{index=6}
BATCH_SIZE = 128                   # batch size = 128（论文后续实验使用）:contentReference[oaicite:7]{index=7}
GAMMA = 0.99
REPLAY_CAPACITY_BASE = 128         # 基础经验回放大小（可按 domain 资源缩放）
SOFT_TAU = 0.01                    # φ 用于软更新 (Eq.21)（论文 φ 很小）
LAMBDA_STALENESS = 5              # λ 用于时延衰减 (Eq.10)（论文取 5）:contentReference[oaicite:8]{index=8}
BASE_AGG_WEIGHT = 0.9              # δ 基线权重（论文采用 0.9）:contentReference[oaicite:9]{index=9}
EPSILON_MIN = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 经验单元
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# -----------------------------
# 环境：简化的多域网络仿真
# -----------------------------
class Node:
    """节点（主机/服务器），包含可用 CPU"""
    def __init__(self, node_id:int, cpu_capacity: int):
        self.id = node_id
        self.cpu_capacity = cpu_capacity
        self.cpu_used = 0  # 当前已占用核心数

class Link:
    """链路，包含带宽"""
    def __init__(self, u:int, v:int, bw: float):
        self.u = u; self.v = v; self.bw = bw
        # 简化：不建模更细粒度排队/传输，使用常数延迟参数

class DomainEnv:
    """
    域仿真环境：包含若干节点/链路、SFC 段 Fi。
    state: 包括节点 cpu 剩余率、链路 bw 剩余率、当前 Fi 的 VNF 资源需求描述等（向量化）
    action: (parallelization matrix, 部署矩阵) — 在实现中简化为：
            - 选择并行组（由 idx 列表表示）
            - 为每个 VNF 选择目标 node id
    reward: - (activation + parallel_exec + communication) 按论文 (Eq.7,18)
    """
    def __init__(self, domain_id:int, num_nodes:int=4, node_cpu:int=30, link_bw:float=4.0):
        self.id = domain_id
        self.nodes = [Node(i, node_cpu) for i in range(num_nodes)]
        # 简化的全连图链路集合（仅保存 bw）
        self.links = [Link(i, j, link_bw) for i in range(num_nodes) for j in range(i+1, num_nodes)]
        # 用于演示的固定参数（按论文描述）
        self.t_dup = 1.0   # packet duplication time per instance (ms)
        self.t_mer = 1.0   # packet merging time per instance (ms)
        self.tr_per_vnf = (5.0, 10.0)  # VNF 执行时间范围 (ms)
        self.trans_delay = 20.0  # transmission delay 对齐（ms）
        # 经验缓冲区大小基于 domain 资源（此处用节点数作为 proxy）
        self.replay_capacity = int(REPLAY_CAPACITY_BASE * (num_nodes / 4.0))
        if self.replay_capacity < 64:
            self.replay_capacity = 64
        # 维护简单运行时指标
        self.cpu_total = sum(n.cpu_capacity for n in self.nodes)

    def get_state_vector(self, sfc_segment: List[str]):
        """
        固定长度 state 向量：
        - 节点 CPU 利用率（固定长度）
        - 链路 BW 利用率（固定长度）
        - VNF CPU 需求（固定 MAX_SFC_LEN）
        - VNF BW 需求（固定 MAX_SFC_LEN）
        """

        # 1. 节点 CPU 利用率
        node_cpu_util = [n.cpu_used / n.cpu_capacity for n in self.nodes]

        # 2. 链路 BW 利用率（简化为 0，长度固定）
        link_bw_util = [0.0 for _ in self.links]

        # 3. VNF CPU / BW 需求（固定到 MAX_SFC_LEN）
        vnf_cpu = []
        vnf_bw = []
        for v in sfc_segment:
            vnf_cpu.append(random.randint(1, 3))
            vnf_bw.append(round(random.uniform(0.1, 0.3), 3))

        # ---- 关键：补齐到固定长度 ----
        while len(vnf_cpu) < MAX_SFC_LEN:
            vnf_cpu.append(0.0)
            vnf_bw.append(0.0)

        # ---- 关键：截断防止越界 ----
        vnf_cpu = vnf_cpu[:MAX_SFC_LEN]
        vnf_bw = vnf_bw[:MAX_SFC_LEN]

        # 拼接成向量
        state = np.array(node_cpu_util + link_bw_util + vnf_cpu + vnf_bw, dtype=np.float32)
        return state

    def reset_usage(self):
        for n in self.nodes:
            n.cpu_used = 0

    def estimate_latency_for_action(self, sfc_segment: List[str], parallel_groups: List[List[int]], deploy_nodes: List[int]) -> Tuple[float,float,float]:
        """
        根据论文公式估算 activation latency, parallel exec latency, communication latency (ms)
        - activation：假设每个实例启动时间与 VNF 无关但可设为常数（例如 taf=2ms）
        - parallel exec：基于 VNF 执行时间与并行实例的dup/mer成本
        - communication：基于路径 hop 数及 transmission delay，取各并行路径的 max
        注意：此处为简化版模拟，保留与论文一致的组成与相对大小。
        """
        # Activation latency：每个部署实例加 taf（取 2ms）
        taf = 2.0
        activation = 0.0
        for node_id in deploy_nodes:
            activation += taf

        # Parallel execution latency
        # 对每个 VNF：取随机执行时间 t_exec ∈ [5,10] ms（论文），
        # 若某组并行 n > 1，则增加 dup/mer 成本 (n-1)*(t_dup + t_mer)
        pe = 0.0
        # 执行时间按顺序或并行分组计算：并行组内取 max 执行时间
        for group in parallel_groups:
            # group: vnf indices in this parallel set
            # 对每个 vnf 随机抽 t_exec
            t_execs = [random.uniform(*self.tr_per_vnf) for _ in group]
            group_exec = max(t_execs)
            # 并行开销
            n = max(1, len(group))
            tpf = (n-1)*(self.t_dup + self.t_mer)
            pe += group_exec + tpf

        # Communication latency：假设每路径包括 hops（部署到 node）的传输延迟之和 + waiting twc
        # 简化：假设平均 hop 为 2 （论文里会受拓扑影响），twc 取 1ms
        avg_hops = 2
        tuv = self.trans_delay
        twc = 1.0
        # 依据并行路径数量（分支）取最大路径延迟
        num_paths = max(1, len(parallel_groups))
        co = max([(avg_hops * tuv + twc) for _ in range(num_paths)])
        return activation, pe, co

# -----------------------------
# DQN 网络（3 层 600 units）
# -----------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim:int, output_dim:int):
        super(QNetwork, self).__init__()
        # 论文中网络结构：3 个全连接层，每层 600 units，ReLU
        # 输出维度对应 action space 大小（在本实现中，我们用参数化 action，输出作 value）
        self.net = nn.Sequential(
            nn.Linear(input_dim, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# 经验回放（资源导向容量）
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity:int):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size:int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# -----------------------------
# DQN Agent（每个 domain 一个）
# -----------------------------
class DQNAgent:
    def __init__(self, domain_env: DomainEnv, input_dim:int, action_dim:int,
                 lr=LEARNING_RATE, gamma=GAMMA, batch_size=BATCH_SIZE):
        self.env = domain_env
        self.input_dim = input_dim
        self.action_dim = action_dim  # 简化：action_dim 为离散动作数量的近似
        self.gamma = gamma
        self.batch_size = batch_size

        # 预测网络和目标网络
        self.q_net = QNetwork(input_dim, action_dim).to(DEVICE)
        self.target_q = QNetwork(input_dim, action_dim).to(DEVICE)
        self.target_q.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(self.env.replay_capacity)

        # epsilon 初始值（论文用 reward-based 自适应 epsilon，初始化取 1.0）
        self.epsilon = 1.0
        # 记录上一步 reward，用于 Eq.22 计算
        self.prev_reward = -1e9

    def select_action(self, state_vec: np.ndarray):
        """
        选择动作：
         - 由于论文中 action 包括 parallelization 矩阵和部署向量，本实现采用参数化策略：
           * 枚举若干候选并行划分与部署（简化）：随机或基于贪心选择
         - 这里 DQN 输出值对应若干候选策略的 Q 值，action_dim 表示候选数量
        """
        if random.random() < self.epsilon:
            # 随机探索：随机挑选索引
            return random.randint(0, self.action_dim-1)
        else:
            # 利用：用网络评估
            s = torch.from_numpy(state_vec).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                qvals = self.q_net(s)
            return int(torch.argmax(qvals, dim=1).item())

    def push_transition(self, *args):
        self.replay.push(*args)

    def update(self):
        if len(self.replay) < 8:
            return None
        transitions = self.replay.sample(self.batch_size)
        batch = Transition(*transitions)

        # 转张量
        state_batch = torch.tensor(np.stack(batch.state)).float().to(DEVICE)
        action_batch = torch.tensor(batch.action).long().unsqueeze(1).to(DEVICE)
        reward_batch = torch.tensor(batch.reward).float().unsqueeze(1).to(DEVICE)
        next_state_batch = torch.tensor(np.stack(batch.next_state)).float().to(DEVICE)
        done_batch = torch.tensor(batch.done).float().unsqueeze(1).to(DEVICE)

        # Q(s,a)
        q_values = self.q_net(state_batch).gather(1, action_batch)

        # target: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_q(next_state_batch).max(1)[0].unsqueeze(1)
            target = reward_batch + self.gamma * next_q * (1.0 - done_batch)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def soft_update_target(self, tau=SOFT_TAU):
        # 软更新：θ_target = τ θ_local + (1-τ) θ_target (Eq.21)
        for target_param, param in zip(self.target_q.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def adapt_epsilon_by_reward(self, cur_reward: float):
        # 论文自适应 epsilon：ε = R_t / R_{t-1} （Eq.22）
        # 为避免分母为 0 或负激增，加入小常数与截断
        if abs(self.prev_reward) < 1e-6:
            new_eps = max(EPSILON_MIN, 0.5)
        else:
            # 若 reward 负值（论文 reward 为 -latency），保持比例
            ratio = cur_reward / (self.prev_reward + 1e-9)
            # 映射到 (EPSILON_MIN, 1]
            new_eps = float(np.clip(ratio, -1.0, 1.0))
            new_eps = max(EPSILON_MIN, min(1.0, abs(new_eps)))
        self.prev_reward = cur_reward
        self.epsilon = new_eps

# -----------------------------
# 云端：联邦聚合服务
# -----------------------------
class FederatedServer:
    def __init__(self, base_model: QNetwork, lambda_staleness=LAMBDA_STALENESS, base_delta=BASE_AGG_WEIGHT):
        self.global_model = copy.deepcopy(base_model).to(DEVICE)
        self.lambda_staleness = lambda_staleness
        self.base_delta = base_delta
        # 服务器时间戳（记录收到第一个模型的时刻）
        self.reference_recv_time = None

    def aggregate(self, client_models: List[Tuple[float, QNetwork]]):
        """
        client_models: 列表每项 (recv_time, model)
         - 按论文 Eq.10/11，计算 ς(t_i - t_1) = 1/(ti - t1 + 1)^λ
           然后 δ_i = ς * δ (base)
         - Θ(t+1) = (1 - sum δi) Θ(t) + sum δi θi(t)
        """
        if len(client_models) == 0:
            return

        # 以第一个收到为 t1
        t1 = client_models[0][0]
        # 计算各自权重
        weights = []
        for (ti, _) in client_models:
            staleness = 1.0 / ( (ti - t1 + 1.0) ** self.lambda_staleness )
            weights.append(staleness * self.base_delta)

        sum_weights = sum(weights)
        # new_global = (1 - sum δi) * old_global + sum δi * θ_i
        new_state = {}
        # 将 global model state dict 作为 base
        global_state = self.global_model.state_dict()
        # 先 scale current global by (1 - sum_weights)
        for k, v in global_state.items():
            new_state[k] = v.clone() * (1.0 - sum_weights)

        # 累积各客户端模型
        for w, (_, client_model) in zip(weights, client_models):
            c_state = client_model.state_dict()
            for k, v in c_state.items():
                new_state[k] = new_state[k] + v.clone() * w

        # 更新 global model
        self.global_model.load_state_dict(new_state)

    def distribute(self):
        # 返回 global model 的深拷贝，给各 domain 初始化/更新本地模型
        return copy.deepcopy(self.global_model)

# -----------------------------
# 工具：生成候选动作（并行划分 & 部署方案）
# -----------------------------
def generate_candidate_policies(sfc_len:int, domain_env:DomainEnv, num_candidates:int=16):
    """
    生成一组候选策略（并行分组 + 部署节点）
    并行分组用简单启发：将连续段合并或拆分
    部署节点：随机选择 domain 的节点
    返回一个候选集，索引即 action id
    """
    candidates = []
    for _ in range(num_candidates):
        # 随机并行分组：将 sfc_len 个元素随机分配到 1~min(L, sfc_len) 组
        groups = []
        remaining = list(range(sfc_len))
        random.shuffle(remaining)
        num_groups = random.randint(1, min(4, sfc_len))
        sizes = [1] * num_groups
        # 随机把其余分配
        for i in range(sfc_len - num_groups):
            sizes[random.randrange(num_groups)] += 1
        idx = 0
        for sz in sizes:
            groups.append(sorted(remaining[idx: idx + sz]))
            idx += sz
        # 部署节点：为每 VNF 选 node id
        node_choices = [random.choice([n.id for n in domain_env.nodes]) for _ in range(sfc_len)]
        candidates.append((groups, node_choices))
    return candidates

# -----------------------------
# 主训练循环：FDRL-based training (Algorithm 2)
# -----------------------------
def train_fed_dqn(num_epochs=50, local_epochs=5, sfc_requests=50):
    """
    高层训练流程（简化/可扩展）：
    - 按 epoch（aggregation episode）循环
    - cloud: 分配初始全局模型 Θ(t) -> 下发到各 domain
    - 各域并行进行 local_epochs 的本地训练（使用 DQN）
    - 各域上传模型和接收时间戳 (模拟时间差)
    - cloud 执行延迟加权聚合 -> 更新 global
    - 返回 global
    """
    # 初始化 K 个域环境（以小规模示例）
    domains = [DomainEnv(i, num_nodes=4, node_cpu=NODE_CPU_SMALL, link_bw=LINK_BW_SMALL) for i in range(NUM_DOMAINS)]

    # 估算 state_dim：node cpu util + link bw util + vnf_cpu + vnf_bw
    # 这里以每域 4 nodes，links 为 n*(n-1)/2，vnf 每条两个属性（cpu,bw）
    sample_state = domains[0].get_state_vector(['f']*MAX_SFC_LEN)
    state_dim = sample_state.shape[0]
    # action_dim: 我们为每个域生成 M 候选策略，并将 action_dim 设为 M
    ACTION_CANDIDATES = 16

    # 初始化本地 agents
    agents = [DQNAgent(domains[i], input_dim=state_dim, action_dim=ACTION_CANDIDATES) for i in range(NUM_DOMAINS)]

    # 初始化 federated server（使用任意模型作为 base）
    server = FederatedServer(base_model=agents[0].q_net, lambda_staleness=LAMBDA_STALENESS, base_delta=BASE_AGG_WEIGHT)

    # training loop
    for epoch in range(num_epochs):
        print(f"===== 聚合回合 Epoch {epoch+1}/{num_epochs} =====")
        # cloud: 生成 SFC 请求集合并进行初步分解（Algorithm 1）
        # 简化：每个 epoch 生成 sfc_requests 个 SFC，随机长度 4~MAX_SFC_LEN
        sfc_list = []
        for _ in range(sfc_requests):
            l = random.randint(4, MAX_SFC_LEN)
            # 生成随机 VNF 类型序列（论文 7 种）
            sfc = [random.choice(VNF_TYPES) for _ in range(l)]
            sfc_list.append(sfc)

        # 预分解 SFC 到每个 domain（简化采用均匀分配或基于剩余 cpu）
        # 使用论文 Eq.12 思路：按照 domain 可用 CPU 比例分配 VNF 个数
        # 计算每个 domain 当前可用 CPU总和
        cpu_avails = [d.cpu_total - sum(n.cpu_used for n in d.nodes) for d in domains]
        cpu_sum = sum(cpu_avails) + 1e-9
        # 分配每条 SFC 的 segments（此处对每条 sfc 分配同一策略：按比例切片）
        sfc_segments_per_domain = []  # list of list: 每条 sfc 的 domain segments
        for sfc in sfc_list:
            length = len(sfc)
            counts = []
            rem = length
            for i in range(NUM_DOMAINS-1):
                cnt = int(math.floor((cpu_avails[i]/cpu_sum)*length))
                counts.append(cnt)
                rem -= cnt
            counts.append(rem)
            # 可能出现 0 长 segment，处理
            segments = []
            idx = 0
            for cnt in counts:
                segments.append(sfc[idx: idx+max(0,cnt)])
                idx += max(0,cnt)
            # 若有空段，均匀补齐（保证总和等于 length）
            # 简化：若有空段则把剩余分配到最后
            sfc_segments_per_domain.append(segments)

        # 模拟本地训练：各 domain 并行（顺序模拟）
        client_uploads = []
        for di, agent in enumerate(agents):
            env = agent.env
            # 每个 domain 执行 local_epochs 次训练（每个 epoch 处理若干 sfc segments）
            local_losses = []
            local_rewards = []
            # domain 重置使用
            env.reset_usage()
            # 为 domain 生成候选策略集（统一）
            candidates = generate_candidate_policies(MAX_SFC_LEN, env, num_candidates=ACTION_CANDIDATES)
            for le in range(local_epochs):
                # 随机取一条 SFC 的 segment 分配给该域（若为空则跳过）
                for sfc_segments in sfc_segments_per_domain:
                    seg = sfc_segments[di]
                    if len(seg) == 0:
                        continue
                    # 获取 state
                    state_vec = env.get_state_vector(seg)
                    # 选择动作（action id）
                    a_id = agent.select_action(state_vec)
                    groups, deploy_nodes = candidates[a_id]
                    # 估算延迟项
                    activation, pe, co = env.estimate_latency_for_action(seg, groups, deploy_nodes)
                    total_latency = activation + pe + co
                    reward = - total_latency
                    # next_state：简单用同态 state（真实应受资源变化影响）
                    next_state = env.get_state_vector(seg)
                    done = False
                    # push transition
                    agent.push_transition(state_vec, a_id, reward, next_state, done)
                    # 训练更新
                    loss = agent.update()
                    if loss is not None:
                        local_losses.append(loss)
                    local_rewards.append(reward)
                # 每个 local epoch 后做软更新 target 网络（论文建议每 N epochs 做软更新，此处每 epoch）
                agent.soft_update_target(tau=SOFT_TAU)
            # 更新 epsilon（基于 reward, Eq.22）
            # 取本地平均 reward
            if len(local_rewards) > 0:
                avg_reward = float(np.mean(local_rewards))
                agent.adapt_epsilon_by_reward(avg_reward)
            else:
                avg_reward = -1000.0
            # 模拟上传模型和时间戳（这里用 time.time() + 随机延迟表示收模型时间）
            simulated_recv_time = time.time() + random.uniform(0, 1.0)  # 模拟不同 domain 上传延迟
            client_uploads.append((simulated_recv_time, copy.deepcopy(agent.q_net)))
            print(f" Domain {di}: local_loss_avg={np.mean(local_losses) if local_losses else None}, avg_reward={avg_reward:.3f}, epsilon={agent.epsilon:.3f}")
        # cloud 聚合（Algorithm 2，Line 11-14）
        # 按收到时间排序
        client_uploads.sort(key=lambda x: x[0])
        server.aggregate(client_uploads)
        # 分发更新后的 global model 到每个 domain（并替换本地 q_net）
        new_global = server.distribute()
        for agent in agents:
            agent.q_net.load_state_dict(new_global.state_dict())
            # 同步 target 网络
            agent.target_q.load_state_dict(new_global.state_dict())
        print(f" -> 完成聚合并下发全局模型 Epoch {epoch+1}")
    print("训练完成。")
    return server

# -----------------------------
# 如果作为脚本运行
# -----------------------------
if __name__ == "__main__":
    server = train_fed_dqn(num_epochs=6, local_epochs=3, sfc_requests=30)
    print("示例训练运行结束。可按需修改参数重新执行。")
