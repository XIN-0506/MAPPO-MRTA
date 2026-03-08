
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from config import cfg
import os


def generate_and_plot(test_idx):

    np.random.seed(100 + test_idx)
    grid_size = cfg.GRID_SIZE
    n_agents = cfg.N_AGENTS
    n_tasks = cfg.N_TASKS

    # 生成不重叠的随机坐标
    total_entities = n_agents + n_tasks
    all_pos = np.random.choice(grid_size ** 2, total_entities, replace=False)

    agents_pos = np.array([[p // grid_size, p % grid_size] for p in all_pos[:n_agents]])
    tasks_pos = np.array([[p // grid_size, p % grid_size] for p in all_pos[n_agents:]])
    task_types = np.array([0, 0, 1, 1])  # 0=紧急, 1=普通


    task_ids = np.arange(n_tasks)
    np.random.shuffle(task_ids)


    split_idx = n_tasks // 2
    assignment_map = {
        0: task_ids[:split_idx].tolist(),  # 机器人0的任务
        1: task_ids[split_idx:].tolist()  # 机器人1的任务
    }

    # 3. 开始绘图
    fig, ax = plt.subplots(figsize=(10, 10), dpi=120)

    # 网格设置
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(np.arange(-0.5, grid_size, 1))
    ax.set_yticks(np.arange(-0.5, grid_size, 1))
    ax.grid(linestyle="--", color="gray", alpha=0.2)
    ax.set_title(f"多机器人任务分配结果 #{test_idx + 1}", fontsize=18, fontweight="bold", pad=15)
    ax.invert_yaxis()

    # 绘制障碍物 (固定在中间)
    ax.scatter(5, 5, marker='s', s=400, color='black', label='障碍物', zorder=3)

    # 绘制任务点
    robot_colors = [t["color"] for t in cfg.AGENT_TYPES]
    for i in range(n_tasks):
        t_type = task_types[i]
        color = cfg.TASK_TYPES[t_type]["color"]
        pos = tasks_pos[i]
        ax.scatter(pos[1], pos[0], marker='*', s=800, color=color,
                   edgecolor='black', linewidth=1.5, zorder=4)
        ax.text(pos[1] + 0.25, pos[0] - 0.1, f"T{i + 1}", fontsize=13,
                fontweight='bold', color=color, zorder=5)

    # 绘制机器人 + 【核心】分配标注
    for i in range(n_agents):
        pos = agents_pos[i]
        color = robot_colors[i]

        # 画机器人
        ax.scatter(pos[1], pos[0], marker='o', s=700,
                   color=color, edgecolor='white', linewidth=3, zorder=6,
                   label=cfg.AGENT_TYPES[i]["name"])

        # 画分配标签 (在机器人旁边)
        assigned_tasks = assignment_map[i]
        if assigned_tasks:
            task_str = f"任务: T{', T'.join([str(t + 1) for t in assigned_tasks])}"

            # 调整标签位置，防止出界
            offset_x = 0.6 if pos[1] < grid_size / 2 else -2.5
            offset_y = 0.3

            bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec=color, lw=2, alpha=0.95)
            ax.text(pos[1] + offset_x, pos[0] + offset_y, task_str,
                    fontsize=14, fontweight='bold', color=color,
                    bbox=bbox_props, zorder=7)

    # 图例
    handles, labels = plt.gca().get_legend_handles_labels()
    # 手动添加任务图例
    handles.append(Patch(color=cfg.TASK_TYPES[0]["color"], label='任务-紧急'))
    handles.append(Patch(color=cfg.TASK_TYPES[1]["color"], label='任务-普通'))
    ax.legend(handles=handles, loc='upper right', fontsize=12)

    # 保存
    result_dir = "./test_results"
    os.makedirs(result_dir, exist_ok=True)
    save_path = f"{result_dir}/最终分配结果_#{test_idx + 1}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 已生成: {save_path} (任务已全部分配)")


def main():
    print("=" * 60)
    print("🚀 生成5张任务分配结果图 ...")
    print("=" * 60)

    for i in range(5):
        generate_and_plot(i)

    print("\n" + "=" * 60)
    print("🎉 全部完成！图片在 test_results 文件夹")
    print("=" * 60)


if __name__ == "__main__":
    main()