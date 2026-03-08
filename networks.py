# 深度强化学习多机器人任务分配 - 网络模块（Actor-Critic）(已加详细注释)
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    策略网络（Actor）：机器人的“大脑”
    功能：输入局部观测（我在哪？任务在哪？），输出每个动作的执行概率
    """

    def __init__(self, obs_dim, act_dim, hidden_dim):
        """
        初始化策略网络结构

        参数:
            obs_dim: 输入观测的维度 (输入层大小)
            act_dim: 输出动作的维度 (输出层大小，即有多少种动作可选)
            hidden_dim: 隐藏层神经元数量
        """
        super().__init__()

        # 定义一个序列化的网络容器 (Sequential)
        self.net = nn.Sequential(
            # 第一层：全连接层 (Linear)
            # 作用：将高维的观测数据映射到隐藏层空间
            nn.Linear(obs_dim, hidden_dim),

            # 层归一化 (LayerNorm)
            # 【关键技巧】作用：让每一层输入的数据分布保持稳定，大幅加速训练收敛，防止梯度消失/爆炸
            nn.LayerNorm(hidden_dim),

            # 激活函数 (ReLU)
            # 作用：引入非线性，让神经网络可以拟合复杂的函数
            nn.ReLU(),

            # 第二层：全连接层 (隐藏层 -> 隐藏层)
            # 作用：进一步提取和组合特征
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),  # 再次激活

            # 第三层：输出层 (隐藏层 -> 动作维度)
            # 作用：将隐藏层特征映射为具体的“动作分数” (Logits)
            nn.Linear(hidden_dim, act_dim)
        )

    def forward(self, x):
        """
        前向传播函数 (核心推理逻辑)

        参数:
            x: 输入的观测张量 (Observation)

        返回:
            probs: 每个动作的概率分布 (总和为1)
        """
        # 1. 通过网络层得到原始分数 (Logits)
        logits = self.net(x)

        # 2. 通过 Softmax 函数将分数转换为概率
        # 为什么用 Softmax？
        # - 它能把任意实数转换成 (0, 1) 之间的正数
        # - 所有输出的总和为 1，符合概率分布的定义
        return F.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """
    价值网络（Critic）：全局的“评委”
    功能：输入全局状态（所有人在哪？所有任务在哪？），输出当前局面的“好坏程度” (标量分数)
    """

    def __init__(self, global_obs_dim, hidden_dim):
        """
        初始化价值网络结构

        参数:
            global_obs_dim: 全局观测的维度 (通常比局部观测大很多)
            hidden_dim: 隐藏层神经元数量
        """
        super().__init__()

        self.net = nn.Sequential(
            # 第一层：输入层 (全局观测 -> 隐藏层)
            nn.Linear(global_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 同样使用层归一化
            nn.ReLU(),

            # 第二层：隐藏层
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            # 第三层：输出层
            # 注意：输出维度是 1！
            # 因为价值网络只需要输出一个数字：V(s)，表示当前状态有多好
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        前向传播函数

        参数:
            x: 输入的全局状态张量 (Global State)

        返回:
            value: 状态价值标量 (没有Softmax，因为是回归问题，不是分类问题)
        """
        # 直接输出网络结果即可，不需要 Softmax
        # 因为价值可以是任意实数 (可以是负数，表示局面很糟)
        return self.net(x)