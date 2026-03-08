# 核心需求：在一个10*10的网络世界里，有2个异构机器人（一个快，一个慢），4个异构任务（2个紧急高优先级，2个普通低优先级）
#          中间有1个障碍物，让机器人通过强化学习自主学会分工合作，高效完成所有任务，紧急任务优先处理，还不能撞墙，撞到其他机器人。

# 深度强化学习多机器人任务分配 - 配置模块 (已修复)
class Config:
    # 环境参数
    GRID_SIZE = 10
    N_AGENTS = 2
    N_TASKS = 4

    # 机器人异构定义
    AGENT_TYPES = [
        {"speed": 1.0, "name": "R1-快速型", "color": "#1E90FF"},
        {"speed": 0.7, "name": "R2-负载型", "color": "#DC143C"}
    ]

    # 任务异构定义
    TASK_TYPES = [
        {"priority": 3.0, "workload": 1, "name": "紧急", "color": "#9D4EDD"},
        {"priority": 1.0, "workload": 2, "name": "普通", "color": "#38B000"}
    ]

    # 算法参数
    OBS_DIM = 3 + N_TASKS * 4
    ACT_DIM = 5
    GLOBAL_OBS_DIM = OBS_DIM * N_AGENTS
    HIDDEN_DIM = 128
    LR = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    EPS_CLIP = 0.2
    MAX_EPISODES = 600
    MAX_STEPS = 100

    # 【关键修复】模型保存路径
    MODEL_PATH = "./mappo_mrta_models"


cfg = Config()