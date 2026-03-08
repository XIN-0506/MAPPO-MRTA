# 深度强化学习多机器人任务分配 - 环境模块 (已修复seed + 详细注释)
import numpy as np
from config import cfg


class MultiRobotTaskEnv:
    """
    多机器人异构任务分配环境类
    符合标准 Gym 接口风格 (reset, step)
    """

    def __init__(self):
        """
        初始化环境参数
        """
        # 1. 基础网格参数
        self.grid_size = cfg.GRID_SIZE  # 网格世界大小 (10x10)
        self.n_agents = cfg.N_AGENTS  # 机器人数量 (2个)
        self.n_tasks = cfg.N_TASKS  # 任务数量 (4个)

        # 2. 环境状态变量 (在 reset 中会被具体赋值)
        self.agents_pos = None  # 机器人位置数组 shape: [n_agents, 2]
        self.tasks_pos = None  # 任务位置数组 shape: [n_tasks, 2]
        self.tasks_done = None  # 任务完成状态数组 shape: [n_tasks] (True/False)

        # 3. 静态障碍物 (固定在地图中心 [5, 5])
        self.obstacles = np.array([[5, 5]])

        # 4. 【核心异构】任务类型预设
        # 0 = 紧急任务 (高优先级), 1 = 普通任务 (低优先级)
        # 这里设定前2个是紧急任务，后2个是普通任务
        self.task_types = np.array([0, 0, 1, 1])

    def reset(self, seed=None):
        """
        重置环境：在每回合开始时调用，生成新的随机场景

        参数:
            seed: 随机种子 (可选，用于复现结果)

        返回:
            local_obs: 每个机器人的局部观测
            global_obs: 全局观测 (用于Critic)
        """
        # 【关键修复】如果传入了seed，则设置numpy的随机种子
        if seed is not None:
            np.random.seed(seed)

        # 1. 生成不重叠的随机位置
        # 原理：把10x10的网格展平成0-99的数字，随机抽取不放回
        total_entities = self.n_agents + self.n_tasks
        all_pos = np.random.choice(self.grid_size ** 2, total_entities, replace=False)

        # 2. 分配位置
        # 前 n_agents 个位置给机器人
        self.agents_pos = np.array([
            [p // self.grid_size, p % self.grid_size]  # 将一维数字转回二维坐标 (x, y)
            for p in all_pos[:self.n_agents]
        ])
        # 剩下的位置给任务
        self.tasks_pos = np.array([
            [p // self.grid_size, p % self.grid_size]
            for p in all_pos[self.n_agents:]
        ])

        # 3. 重置任务完成状态：全部设为 False (未完成)
        self.tasks_done = np.zeros(self.n_tasks, dtype=bool)

        # 4. 返回初始观测
        return self._get_obs()

    def step(self, actions):
        """
        环境步进核心函数：执行动作，改变环境状态，返回奖励

        参数:
            actions: 机器人的动作数组 shape: [n_agents]
                     (0=上, 1=下, 2=左, 3=右, 4=不动)

        返回:
            next_obs: 下一个观测
            rewards: 奖励数组
            done: 是否终止 (True=所有任务完成/回合结束)
            info: 额外信息 (空字典，符合gym标准)
        """
        # 初始化奖励数组
        rewards = np.zeros(self.n_agents)

        # 定义动作对应的移动向量
        # 索引0: [-1, 0] -> 向上 (x减1)
        # 索引1: [1, 0]  -> 向下 (x加1)
        # 索引2: [0, -1] -> 向左 (y减1)
        # 索引3: [0, 1]  -> 向右 (y加1)
        # 索引4: [0, 0]  -> 不动
        moves = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])

        # ==========================================
        # 第一步：移动机器人 (核心异构逻辑)
        # ==========================================
        for i in range(self.n_agents):
            # 【核心异构】获取当前机器人的速度系数
            # R1-快速型: speed=1.0, 大概率移动成功
            # R2-负载型: speed=0.7, 有30%概率原地踏步
            speed = cfg.AGENT_TYPES[i]["speed"]

            # 模拟速度差异：生成随机数，如果小于speed则成功移动
            if np.random.rand() < speed:
                new_pos = self.agents_pos[i] + moves[actions[i]]
            else:
                new_pos = self.agents_pos[i]  # 移动失败，原地不动

            # 边界约束：防止机器人走出网格 (0到9)
            self.agents_pos[i] = np.clip(new_pos, 0, self.grid_size - 1)

        # ==========================================
        # 第二步：检测碰撞
        # ==========================================
        collision = self._check_collision()

        # ==========================================
        # 第三步：计算奖励 (Reward Shaping)
        # ==========================================
        done = False  # 回合终止标记

        for i in range(self.n_agents):
            # 1. 时间惩罚 (每一步都扣一点)
            # 目的：鼓励机器人尽快完成任务，不要磨洋工
            rewards[i] -= 0.1

            # 2. 碰撞惩罚
            if collision[i]:
                rewards[i] -= 2.0  # 撞墙或撞机器人扣大分

            # 3. 任务完成奖励 (核心)
            # 遍历所有任务，看看是否到达了某个未完成的任务点
            for j in range(self.n_tasks):
                if not self.tasks_done[j]:
                    # 计算机器人i到任务j的欧氏距离
                    dist = np.linalg.norm(self.agents_pos[i] - self.tasks_pos[j])

                    # 如果距离小于阈值 (0.8)，认为到达并完成了任务
                    if dist < 0.8:
                        self.tasks_done[j] = True  # 标记任务为完成

                        # 【异构奖励】奖励 = 基础奖励 + 优先级权重
                        # 紧急任务 (priority=3) 奖励更高
                        priority = cfg.TASK_TYPES[self.task_types[j]]["priority"]
                        rewards[i] += 10.0 + priority * 5.0
                        break  # 一个机器人一步只完成一个任务

        # ==========================================
        # 第四步：检查回合终止条件
        # ==========================================
        if self.tasks_done.all():
            done = True
            # 团队协作奖励：所有任务都完成了，大家都加分
            rewards += 5.0

        # 返回 (观测, 奖励, 是否结束, 调试信息)
        return self._get_obs(), rewards, done, {}

    def _get_obs(self):
        """
        构建观测空间 (局部观测 + 全局观测)

        返回:
            local_obs: 每个机器人看到的局部信息
                       [自身x, 自身y, 自身类型ID, 任务1信息, 任务2信息...]
            global_obs: 所有局部观测的拼接 (用于中心化Critic)
        """
        local_obs = []

        for i in range(self.n_agents):
            # 1. 自身信息 (Self State)
            # [x坐标, y坐标, 机器人类型ID (0或1)]
            self_info = np.array([self.agents_pos[i][0], self.agents_pos[i][1], i])

            # 2. 所有任务信息 (Task State)
            task_info = []
            for j in range(self.n_tasks):
                t_type = self.task_types[j]
                task_info.extend([
                    self.tasks_pos[j][0],  # 任务x坐标
                    self.tasks_pos[j][1],  # 任务y坐标
                    cfg.TASK_TYPES[t_type]["priority"],  # 任务优先级
                    1.0 if self.tasks_done[j] else 0.0  # 任务是否完成 (1.0=完成, 0.0=未完成)
                ])

            # 3. 拼接自身信息和任务信息，作为当前机器人的局部观测
            local_obs.append(np.concatenate([self_info, task_info]))

        # 4. 全局观测 = 把所有机器人的局部观测展平成一个一维向量
        return np.array(local_obs), np.array(local_obs).flatten()

    def _check_collision(self):
        """
        检测碰撞：撞障碍物 或 机器人之间互撞

        返回:
            collisions: 布尔数组，True表示该机器人发生了碰撞
        """
        collisions = np.zeros(self.n_agents, dtype=bool)

        # 1. 检测撞障碍物
        for i, pos in enumerate(self.agents_pos):
            for obs in self.obstacles:
                if np.array_equal(pos, obs):
                    collisions[i] = True

        # 2. 检测机器人之间互撞
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if np.array_equal(self.agents_pos[i], self.agents_pos[j]):
                    collisions[i] = True
                    collisions[j] = True

        return collisions