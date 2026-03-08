import torch
import torch.nn as nn
import numpy as np
import os
from networks import PolicyNetwork, ValueNetwork
from config import cfg

# 实例化网络-根据观测决策-存储经验-用PPO算法更新网络参数-保存/加载模型
class MAPPOAgent:
    """
    多智能体近端策略优化 (MAPPO) 智能体类
    采用 CTDE (中心化训练，分布式执行) 范式
    """

    def __init__(self):
        # 1. 初始化基础参数
        self.n_agents = cfg.N_AGENTS  # 机器人数量

        # 2. 初始化神经网络
        # 策略网络 (Actor)：每个机器人一个，负责根据局部观测输出动作概率
        self.policy_nets = [
            PolicyNetwork(cfg.OBS_DIM, cfg.ACT_DIM, cfg.HIDDEN_DIM)
            for _ in range(cfg.N_AGENTS)
        ]
        # 价值网络 (Critic)：全局共享一个，负责根据全局状态评估状态价值
        self.value_net = ValueNetwork(cfg.GLOBAL_OBS_DIM, cfg.HIDDEN_DIM)

        # 3. 初始化优化器
        # 每个策略网络一个独立的优化器
        self.policy_optims = [
            torch.optim.Adam(net.parameters(), lr=cfg.LR)
            for net in self.policy_nets
        ]
        # 价值网络的优化器
        self.value_optim = torch.optim.Adam(self.value_net.parameters(), lr=cfg.LR)

        # 4. 经验回放缓冲区 (Buffer)
        # 用于存储训练过程中的 (s, a, r, s') 数据
        self.buffer = []

    def get_action(self, local_obs, deterministic=False):
        """
        根据局部观测获取动作

        参数:
            local_obs: 每个机器人的局部观测数组
            deterministic: 是否使用确定性策略 (True=选概率最大的, False=随机采样)

        返回:
            actions: 动作数组
            log_probs: 对应动作的对数概率 (用于PPO更新)
        """
        actions = []
        log_probs = []

        # 遍历每个机器人，独立决策
        for i in range(self.n_agents):
            # 将numpy观测转换为torch张量
            obs = torch.FloatTensor(local_obs[i])

            # 通过策略网络前向传播，得到动作概率分布
            probs = self.policy_nets[i](obs)

            if deterministic:
                # 测试/演示阶段：选择概率最大的动作 (贪婪策略)
                action = torch.argmax(probs).item()
            else:
                # 训练阶段：按照概率分布随机采样 (探索)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()

            # 记录动作和对应的对数概率
            actions.append(action)
            # 加 1e-10 防止 log(0) 报错
            log_probs.append(torch.log(probs[action] + 1e-10).item())

        return np.array(actions), np.array(log_probs)

    def store_transition(self, transition):
        """
        存储一条经验数据到缓冲区

        参数:
            transition: 元组 (局部观测, 全局观测, 动作, 动作概率, 奖励, 是否终止)
        """
        self.buffer.append(transition)

    def update(self):
        """
        MAPPO核心更新逻辑：利用缓冲区数据更新策略网络和价值网络

        返回:
            policy_loss_avg: 平均策略损失
            value_loss: 价值损失
        """
        # 如果缓冲区为空，不更新
        if len(self.buffer) == 0:
            return 0.0, 0.0

        # ==========================================
        # 第一步：解析缓冲区数据
        # ==========================================
        # 将 buffer 中的数据按列解压
        # batch[0] = 所有局部观测, batch[1] = 所有全局观测...
        batch = list(zip(*self.buffer))

        local_obs = np.array(batch[0])  # shape: [batch_size, n_agents, obs_dim]
        global_obs = np.array(batch[1])  # shape: [batch_size, global_obs_dim]
        actions = np.array(batch[2])  # shape: [batch_size, n_agents]
        old_log_probs = np.array(batch[3])  # shape: [batch_size, n_agents]
        rewards = np.array(batch[4])  # shape: [batch_size, n_agents]
        dones = np.array(batch[5])  # shape: [batch_size] (True/False)
        batch_size = len(dones)

        # ==========================================
        # 第二步：计算 GAE (广义优势估计)
        # 目的：估计每个动作的“优势” (A = Q - V)，即比平均状态好多少
        # ==========================================
        with torch.no_grad():  # 这部分不需要计算梯度
            # 1. 用价值网络计算每个状态的价值 V(s)
            values = self.value_net(torch.FloatTensor(global_obs)).numpy().flatten()
            # 2. 计算下一个状态的价值 V(s')，最后一个状态的 next_value 设为 0
            next_values = np.append(values[1:], 0)

        # 初始化优势数组和回报数组
        advantages = np.zeros(batch_size, dtype=np.float32)
        returns = np.zeros(batch_size, dtype=np.float32)
        last_adv = 0

        # 从后往前 (Backward) 计算 GAE
        for t in reversed(range(batch_size)):
            # 计算 TD 误差 (Delta): r + gamma * V(s') - V(s)
            # 这里 rewards[t].mean() 是因为我们用全局奖励来训练中心化的Critic
            delta = rewards[t].mean() + cfg.GAMMA * next_values[t] * (1 - dones[t]) - values[t]

            # 递推计算优势: A_t = delta_t + (gamma * lambda) * A_{t+1}
            last_adv = delta + cfg.GAMMA * cfg.GAE_LAMBDA * (1 - dones[t]) * last_adv
            advantages[t] = last_adv

            # 计算回报 (Return): G_t = A_t + V(s_t)
            returns[t] = advantages[t] + values[t]

        # 优势归一化 (Normalization)：让优势均值为0，方差为1，大幅提升训练稳定性
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ==========================================
        # 第三步：更新策略网络 (Actor)
        # ==========================================
        adv_tensor = torch.FloatTensor(advantages)  # 转换为tensor
        policy_loss_total = 0.0

        # 遍历每个机器人，独立更新其策略网络
        for i in range(self.n_agents):
            # 提取当前机器人对应的数据
            obs_i = torch.FloatTensor(local_obs[:, i, :])  # 当前机器人的观测序列
            act_i = torch.LongTensor(actions[:, i])  # 当前机器人的动作序列
            old_logp_i = torch.FloatTensor(old_log_probs[:, i])  # 旧策略下的对数概率

            # 1. 用新策略重新计算动作概率
            probs = self.policy_nets[i](obs_i)
            dist = torch.distributions.Categorical(probs)
            new_logp_i = dist.log_prob(act_i)  # 新策略下的对数概率

            # 2. 计算 PPO 损失 (核心：Clipped Surrogate Objective)
            # 计算概率比 r(theta) = pi_new / pi_old
            ratio = torch.exp(new_logp_i - old_logp_i)

            # Surrogate 1: 原始目标
            surr1 = ratio * adv_tensor
            # Surrogate 2: 裁剪后的目标 (防止更新步长太大)
            surr2 = torch.clamp(ratio, 1 - cfg.EPS_CLIP, 1 + cfg.EPS_CLIP) * adv_tensor

            # PPO 损失：取两者的最小值，然后取负号 (因为梯度上升变下降)
            loss = -torch.min(surr1, surr2).mean()

            # 3. 反向传播更新参数
            self.policy_optims[i].zero_grad()  # 清空旧梯度
            loss.backward()  # 计算梯度
            torch.nn.utils.clip_grad_norm_(self.policy_nets[i].parameters(), 0.5)  # 梯度裁剪，防止梯度爆炸
            self.policy_optims[i].step()  # 更新网络参数

            policy_loss_total += loss.item()

        # ==========================================
        # 第四步：更新价值网络 (Critic)
        # ==========================================
        returns_tensor = torch.FloatTensor(returns)  # 目标回报 (Ground Truth)

        # 1. 价值网络前向预测
        value_pred = self.value_net(torch.FloatTensor(global_obs)).flatten()

        # 2. 计算均方误差损失 (MSE)：让预测值 V(s) 接近实际回报 G_t
        value_loss = nn.MSELoss()(value_pred, returns_tensor)

        # 3. 反向传播更新
        self.value_optim.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_optim.step()

        # ==========================================
        # 第五步：清空缓冲区，准备下一回合
        # ==========================================
        self.buffer = []

        return policy_loss_total / self.n_agents, value_loss.item()

    def save_models(self):
        """保存模型权重到本地文件夹"""
        # 创建文件夹 (如果不存在)
        os.makedirs(cfg.MODEL_PATH, exist_ok=True)

        # 保存每个机器人的策略网络
        for i in range(self.n_agents):
            torch.save(
                self.policy_nets[i].state_dict(),
                f"{cfg.MODEL_PATH}/policy_agent_{i}.pth"
            )

        # 保存全局价值网络
        torch.save(
            self.value_net.state_dict(),
            f"{cfg.MODEL_PATH}/value_net.pth"
        )
        print(f"✅ 模型已保存至: {cfg.MODEL_PATH}")

    def load_models(self):
        """从本地加载模型权重"""
        # 加载每个机器人的策略网络
        for i in range(self.n_agents):
            self.policy_nets[i].load_state_dict(
                torch.load(f"{cfg.MODEL_PATH}/policy_agent_{i}.pth")
            )
            self.policy_nets[i].eval()  # 切换到评估模式 (禁用 Dropout/BatchNorm)

        # 加载全局价值网络
        self.value_net.load_state_dict(
            torch.load(f"{cfg.MODEL_PATH}/value_net.pth")
        )
        self.value_net.eval()
        print("✅ 模型加载成功")