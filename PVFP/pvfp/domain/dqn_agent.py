# -*- coding: utf-8 -*-
"""
域级DQN代理实现
包含经验回放、自适应ε-greedy、软目标更新等机制
"""

import numpy as np
import tensorflow as tf
from collections import deque
import random
import sys
sys.path.append('../..')
from config import *

# TensorFlow 2.x 兼容性处理
if hasattr(tf, 'compat'):
    tf = tf.compat.v1
    tf.disable_eager_execution()

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        """
        初始化回放缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        添加经验到缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        从缓冲区随机采样一批经验
        
        Args:
            batch_size: 批次大小
        
        Returns:
            (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 这里返回原生Python列表，在DQNAgent.train中统一转换为NumPy数组，
        # 便于对状态做展平和堆叠处理，避免形状不一致导致错误
        return list(states), list(actions), list(rewards), list(next_states), list(dones)
    
    def size(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)


class DQNNetwork:
    """DQN网络：3层全连接，每层600个神经元"""
    
    def __init__(self, state_dim, action_dim, learning_rate=LEARNING_RATE, name='dqn'):
        """
        初始化DQN网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            learning_rate: 学习率
            name: 网络名称
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.name = name
        
        # 构建网络
        self._build_network()
    
    def _build_network(self):
        """构建3层全连接网络，每层600个神经元"""
        with tf.variable_scope(self.name):
            # 输入层
            self.state_input = tf.placeholder(tf.float32, [None, self.state_dim], name='state')
            
            # 第一层：600个神经元，ReLU激活
            layer1 = tf.keras.layers.Dense(
                units=DQN_NEURONS_PER_LAYER,
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(),
                name='layer1'
            )(self.state_input)
            
            # 第二层：600个神经元，ReLU激活
            layer2 = tf.keras.layers.Dense(
                units=DQN_NEURONS_PER_LAYER,
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(),
                name='layer2'
            )(layer1)
            
            # 第三层：600个神经元，ReLU激活
            layer3 = tf.keras.layers.Dense(
                units=DQN_NEURONS_PER_LAYER,
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(),
                name='layer3'
            )(layer2)
            
            # 输出层：Q值
            self.q_values = tf.keras.layers.Dense(
                units=self.action_dim,
                activation=None,
                kernel_initializer=tf.keras.initializers.VarianceScaling(),
                name='q_values'
            )(layer3)
            
            # 用于训练的占位符
            self.action = tf.placeholder(tf.int32, [None], name='action')
            self.target_q = tf.placeholder(tf.float32, [None], name='target_q')
            
            # 计算选择的动作的Q值
            action_one_hot = tf.one_hot(self.action, self.action_dim)
            q_value_selected = tf.reduce_sum(self.q_values * action_one_hot, axis=1)
            
            # 损失函数：均方误差
            self.loss = tf.reduce_mean(tf.square(self.target_q - q_value_selected))
            
            # 优化器：Adam
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
    
    def get_trainable_variables(self):
        """获取可训练的变量"""
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class DQNAgent:
    """域级DQN代理"""
    
    def __init__(self, domain_id, state_dim, action_dim, 
                 buffer_capacity=REPLAY_BUFFER_SIZE):
        """
        初始化DQN代理
        
        Args:
            domain_id: 域ID
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            buffer_capacity: 经验回放缓冲区容量
        """
        self.domain_id = domain_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 创建会话
        config = tf.ConfigProto()
        if USE_GPU:
            config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRACTION
        self.sess = tf.Session(config=config)
        
        # 创建预测网络和目标网络
        self.predict_net = DQNNetwork(state_dim, action_dim, name=f'predict_net_{domain_id}')
        self.target_net = DQNNetwork(state_dim, action_dim, name=f'target_net_{domain_id}')
        
        # 软更新操作：θ_target ← φ * θ_predict + (1-φ) * θ_target
        self.update_target_ops = self._build_soft_update_ops()
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # 探索参数
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.use_adaptive_epsilon = USE_ADAPTIVE_EPSILON
        
        # 奖励历史（用于自适应ε）
        self.reward_history = []
        self.prev_avg_reward = 0
        
        # 训练统计
        self.training_step = 0
        self.episode_rewards = []
        
        # 初始化变量
        self.sess.run(tf.global_variables_initializer())
        self._hard_update_target_network()
    
    def _build_soft_update_ops(self):
        """构建软更新操作"""
        predict_vars = self.predict_net.get_trainable_variables()
        target_vars = self.target_net.get_trainable_variables()
        
        update_ops = []
        for pred_var, target_var in zip(predict_vars, target_vars):
            # θ_target ← τ * θ_predict + (1-τ) * θ_target
            updated_value = TAU * pred_var + (1 - TAU) * target_var
            update_ops.append(target_var.assign(updated_value))
        
        return update_ops
    
    def _normalize_state(self, state):
        """将环境状态转换为固定长度的一维向量"""
        state_arr = np.asarray(state, dtype=np.float32).flatten()
        if state_arr.size < self.state_dim:
            padded = np.zeros(self.state_dim, dtype=np.float32)
            padded[:state_arr.size] = state_arr
            state_arr = padded
        elif state_arr.size > self.state_dim:
            state_arr = state_arr[:self.state_dim]
        return state_arr
    
    def _hard_update_target_network(self):
        """硬更新：完全复制预测网络到目标网络"""
        predict_vars = self.predict_net.get_trainable_variables()
        target_vars = self.target_net.get_trainable_variables()
        
        for pred_var, target_var in zip(predict_vars, target_vars):
            self.sess.run(target_var.assign(pred_var))
    
    def soft_update_target_network(self):
        """软更新目标网络"""
        self.sess.run(self.update_target_ops)
    
    def select_action(self, state, training=True):
        """
        使用ε-greedy策略选择动作
        
        Args:
            state: 当前状态
            training: 是否在训练模式
        
        Returns:
            选择的动作
        """
        if training and np.random.rand() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(0, self.action_dim)
        else:
            # 利用：选择Q值最大的动作
            state_vec = self._normalize_state(state).reshape(1, -1)
            q_values = self.sess.run(
                self.predict_net.q_values,
                feed_dict={self.predict_net.state_input: state_vec}
            )
            return np.argmax(q_values[0])
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转移到经验回放缓冲区"""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self, batch_size=BATCH_SIZE):
        """
        从经验回放缓冲区采样并训练
        
        Args:
            batch_size: 批次大小
        
        Returns:
            loss: 训练损失
        """
        # 检查缓冲区大小
        if self.replay_buffer.size() < MIN_REPLAY_SIZE:
            return 0.0
            
        # 确保不采样超过缓冲区大小
        current_batch_size = min(batch_size, self.replay_buffer.size())
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(current_batch_size)
        
        # 确保状态数据格式正确：逐个展平并堆叠成矩阵
        try:
            states = np.vstack([self._normalize_state(s).reshape(1, -1) for s in states])
            next_states = np.vstack([self._normalize_state(s).reshape(1, -1) for s in next_states])
        except Exception as e:
            print(f"状态数据格式错误: {e}")
            if len(states) > 0:
                print(f"样本状态示例形状: {np.array(states[0]).shape}")
            return 0.0

        # 将动作、奖励和终止标记转换为NumPy数组
        actions = np.asarray(actions, dtype=np.int32)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        # 计算目标Q值：Q_target = r + γ * max(Q_target(s', a'))
        next_q_values = self.sess.run(
            self.target_net.q_values,
            feed_dict={self.target_net.state_input: next_states}
        )
        
        max_next_q = np.max(next_q_values, axis=1)
        target_q = rewards + GAMMA * max_next_q * (1 - dones)
        
        # 训练预测网络
        _, loss = self.sess.run(
            [self.predict_net.train_op, self.predict_net.loss],
            feed_dict={
                self.predict_net.state_input: states,
                self.predict_net.action: actions,
                self.predict_net.target_q: target_q
            }
        )
        
        # 软更新目标网络
        self.soft_update_target_network()
        
        self.training_step += 1
        
        return loss
    
    def update_epsilon(self, episode_reward):
        """
        更新探索率ε
        
        Args:
            episode_reward: 当前episode的总奖励
        """
        self.reward_history.append(episode_reward)
        
        if self.use_adaptive_epsilon and len(self.reward_history) > 1:
            # 基于奖励的自适应ε：ε_i = R_i_t / R_i_{t-1}
            current_avg_reward = np.mean(self.reward_history[-10:])
            
            if self.prev_avg_reward != 0:
                self.epsilon = abs(current_avg_reward / self.prev_avg_reward)
                self.epsilon = np.clip(self.epsilon, self.epsilon_min, EPSILON_START)
            
            self.prev_avg_reward = current_avg_reward
        else:
            # 标准指数衰减
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_weights(self):
        """获取网络权重"""
        predict_vars = self.predict_net.get_trainable_variables()
        weights = self.sess.run(predict_vars)
        return weights
    
    def set_weights(self, weights):
        """设置网络权重"""
        predict_vars = self.predict_net.get_trainable_variables()
        for var, weight in zip(predict_vars, weights):
            self.sess.run(var.assign(weight))
        
        # 同步更新目标网络
        self._hard_update_target_network()
    
    def save_model(self, path):
        """保存模型"""
        saver = tf.train.Saver()
        saver.save(self.sess, path)
    
    def load_model(self, path):
        """加载模型"""
        saver = tf.train.Saver()
        saver.load(self.sess, path)
    
    def close(self):
        """关闭会话"""
        self.sess.close()
