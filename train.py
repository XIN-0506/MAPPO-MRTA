# 纯训练脚本：只负责训练和保存模型 (已加详细注释)
import torch
import numpy as np
from config import cfg
from environment import MultiRobotTaskEnv
from agent import MAPPOAgent


def main():
    """
    主训练函数：标准的深度强化学习训练循环
    """
    # ==========================================
    # 第一步：初始化
    # ==========================================

    # 1. 设置全局随机种子 (保证实验结果可复现)
    np.random.seed(42)  # numpy随机种子
    torch.manual_seed(42)  # PyTorch随机种子

    # 2. 实例化环境和智能体
    env = MultiRobotTaskEnv()  # 创建多机器人任务分配环境
    agent = MAPPOAgent()  # 创建MAPPO智能体

    # 3. 打印训练配置信息
    print("=" * 60)
    print("🚀 开始MAPPO多机器人任务分配训练 (纯训练模式)")
    print(f"训练回合: {cfg.MAX_EPISODES} | 机器人: {cfg.N_AGENTS} | 任务: {cfg.N_TASKS}")
    print("=" * 60)

    # ==========================================
    # 第二步：训练循环 (核心)
    # ==========================================

    # 外层循环：每一个 Episode (回合)
    for episode in range(cfg.MAX_EPISODES):
        # 1. 重置环境，开始新的一回合
        # 获取初始观测：局部观测(给Actor) + 全局观测(给Critic)
        local_obs, global_obs = env.reset()

        # 初始化本回合的累计奖励和终止标记
        total_reward = 0.0
        done = False

        # 内层循环：每一个 Step (步)
        for step in range(cfg.MAX_STEPS):
            # 2. 智能体根据观测决策 (Act)
            # 输入：局部观测
            # 输出：执行的动作数组，以及动作对应的对数概率(用于后续PPO更新)
            actions, log_probs = agent.get_action(local_obs)

            # 3. 环境执行动作，返回新状态 (Step)
            # 输入：动作数组
            # 输出：新观测, 奖励, 是否终止, 调试信息
            next_obs, rewards, done, _ = env.step(actions)
            next_local_obs, next_global_obs = next_obs

            # 4. 存储经验到缓冲区 (Store)
            # 这是为了实现 Experience Replay / Batch Update
            # 存储元组：(s_local, s_global, a, log_p, r, done)
            agent.store_transition((
                local_obs, global_obs, actions, log_probs, rewards, done
            ))

            # 5. 更新累计奖励，准备下一步
            total_reward += rewards.mean()  # 这里取所有机器人奖励的平均值
            local_obs, global_obs = next_local_obs, next_global_obs

            # 6. 如果回合结束 (done=True)，跳出内层循环
            if done:
                break

        # ==========================================
        # 第三步：更新网络 (Learn)
        # ==========================================
        # 注意：这里是回合结束后才更新 (Monte Carlo风格)
        # 如果缓冲区有数据，就调用 update() 进行 PPO 更新
        if len(agent.buffer) > 0:
            agent.update()

        # ==========================================
        # 第四步：打印训练日志 (Log)
        # ==========================================
        # 每 50 个回合打印一次信息，方便监控训练进度
        if episode % 50 == 0:
            task_done = env.tasks_done.sum()  # 统计本回合完成了几个任务
            print(f"[回合 {episode:4d}/{cfg.MAX_EPISODES}] 平均奖励: {total_reward:6.2f} | 完成任务: {task_done}/{cfg.N_TASKS}")

    # ==========================================
    # 第五步：保存模型 (Save)
    # ==========================================
    print("=" * 60)
    print("🏆 训练完成！正在保存模型...")
    agent.save_models()  # 调用 agent 的保存方法
    print("=" * 60)


if __name__ == "__main__":
    # 程序入口
    main()