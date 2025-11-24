# -*- coding: utf-8 -*-
"""
PVFP配置文件
包含所有超参数和网络配置
"""

# ==================== 网络拓扑配置 ====================
# 小规模网络配置
SMALL_SCALE = {
    'nodes': 12,
    'links': 15,
    'cpu_capacity': 30,  # 每个节点的CPU核心数
    'bandwidth': 4,      # 每条链路带宽 (Mbps)
}

# 大规模网络配置
LARGE_SCALE = {
    'nodes': 35,
    'links': 79,
    'cpu_capacity': 20,
    'bandwidth': 2,
}

# ==================== VNF配置 ====================
# 7类VNF类型
VNF_TYPES = ['NAT', 'LB', 'NIDS', 'Gateway', 'VPN', 'FW', 'Caching']

# VNF执行时间 (ms)
VNF_EXEC_TIME_MIN = 5
VNF_EXEC_TIME_MAX = 10

# VNF资源需求
VNF_CPU_REQUIREMENT = 1  # 每个VNF需要的CPU核心数
VNF_MEMORY_REQUIREMENT = 512  # MB

# ==================== SFC配置 ====================
MAX_SFC_LENGTH = 10  # SFC最大长度
MIN_SFC_LENGTH = 3   # SFC最小长度

# ==================== DQN超参数 ====================
# 网络结构
DQN_HIDDEN_LAYERS = 3
DQN_NEURONS_PER_LAYER = 600
DQN_ACTIVATION = 'relu'

# 训练超参数
LEARNING_RATE = 5e-4  # 论文使用的学习率
BATCH_SIZE = 128
GAMMA = 0.99  # 折扣因子
TAU = 0.01    # 软更新参数 φ

# 经验回放
REPLAY_BUFFER_SIZE = 128  # 可根据域资源调整
MIN_REPLAY_SIZE = 64

# 探索策略
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
USE_ADAPTIVE_EPSILON = True  # 使用基于奖励的自适应ε

# ==================== 联邦学习配置 ====================
NUM_DOMAINS = 4  # 域的数量
AGGREGATION_EPOCHS = 100  # 全局聚合轮数
LOCAL_EPOCHS = 10  # 每个域的本地训练轮数

# 时滞参数
LAMBDA_STALENESS = 5  # 时滞衰减指数 λ
DELTA_BASE = 0.9      # 基础聚合权重 δ

# ==================== 延迟计算配置 ====================
BASE_TRANSMISSION_DELAY = 20  # 基准传输延迟 (ms)
PACKET_SIZE = 500             # 数据包大小 (bytes)
PACKET_HEADER = 60            # 包头大小 (bytes)
READ_WRITE_PER_BIT = 0.08     # 每比特读写时间 (ms)

# ==================== 实验配置 ====================
NUM_EXPERIMENTS = 30  # 独立实验次数
RANDOM_SEED = 42

# 日志和保存
LOG_INTERVAL = 10
SAVE_INTERVAL = 50
MODEL_SAVE_PATH = './logs/models/'
RESULT_SAVE_PATH = './logs/results/'

# ==================== GPU配置 ====================
USE_GPU = True
GPU_MEMORY_FRACTION = 0.8
