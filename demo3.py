import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import heapq
import argparse

# 创建解析器对象
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument("number", type=int, help="run-times")

# 解析参数
args = parser.parse_args()

save_path = os.path.join(os.getcwd(), "simulation_results")
os.makedirs(save_path, exist_ok=True)

def generate_advanced_trajectory_animation():
    # 初始化动画场景
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Cruise Missile Operation Simulation - Terrain Evasion & Precision Strike")

    # 随机选择目标位置和起始位置
    start_position = (random.randint(5, 15), random.randint(5, 15))
    target_position = (random.randint(85, 95), random.randint(85, 95))
    
    # 设置更多的地形障碍物
    obstacle_positions = []
    num_obstacles = 20  # 增加障碍物数量
    obstacle_radius = 5  # 障碍物半径增大
    np.random.seed(42)  # 为了结果可重复
    for _ in range(num_obstacles):
        x = random.randint(20, 80)
        y = random.randint(20, 80)
        obstacle_positions.append((x, y))
    
    # 将障碍物映射到网格地图上
    grid_size = 1  # 网格大小
    grid_width = int(100 / grid_size)
    grid_height = int(100 / grid_size)
    grid = np.zeros((grid_height, grid_width))

    # 标记障碍物在网格地图上，考虑障碍物半径
    for obs in obstacle_positions:
        obs_x = int(obs[0] / grid_size)
        obs_y = int(obs[1] / grid_size)
        # 将障碍物半径内的网格标记为不可通行
        for i in range(-obstacle_radius, obstacle_radius + 1):
            for j in range(-obstacle_radius, obstacle_radius + 1):
                x = obs_x + i
                y = obs_y + j
                if 0 <= x < grid_width and 0 <= y < grid_height:
                    if np.hypot(i, j) <= obstacle_radius:
                        grid[y, x] = 1  # 障碍物标记为1

    # 设置初始标记和轨迹
    target_marker, = ax.plot([], [], 'ro', markersize=10, label="Target")
    missile_marker, = ax.plot([], [], 'bo', markersize=8, label="Missile")
    trajectory_line, = ax.plot([], [], 'b-', lw=2, label="Trajectory")
    explosion_marker, = ax.plot([], [], 'yo', markersize=15, label="Explosion")

    # 绘制地形障碍物，显示为圆形区域
    for obs in obstacle_positions:
        circle = plt.Circle(obs, obstacle_radius, color='gray', alpha=0.5)
        ax.add_patch(circle)

    # 显示图例
    ax.legend(loc="upper left")

    # 使用 A* 算法生成避障轨迹
    def heuristic(a, b):
        """计算启发式函数，这里使用欧几里得距离。"""
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def astar(grid, start, goal):
        """在给定的网格上使用 A* 算法寻找路径。"""
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),          (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                data.append(start)
                return data[::-1]

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + np.hypot(i, j)
                if 0 <= neighbor[0] < grid.shape[0]:
                    if 0 <= neighbor[1] < grid.shape[1]:
                        if grid[neighbor[0]][neighbor[1]] == 1:
                            continue
                    else:
                        continue
                else:
                    continue

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

        return []

    # 将起始和目标位置转换为网格坐标
    start_node = (int(start_position[1] / grid_size), int(start_position[0] / grid_size))
    goal_node = (int(target_position[1] / grid_size), int(target_position[0] / grid_size))

    path = astar(grid, start_node, goal_node)

    if not path:
        print("No path found!")
        return

    # 将网格路径转换为实际坐标
    x_data = [node[1] * grid_size + grid_size / 2 for node in path]
    y_data = [node[0] * grid_size + grid_size / 2 for node in path]

    # 平滑路径
    from scipy.ndimage import gaussian_filter1d
    x_data = gaussian_filter1d(x_data, sigma=2)
    y_data = gaussian_filter1d(y_data, sigma=2)

    def init():
        target_marker.set_data([], [])
        missile_marker.set_data([], [])
        trajectory_line.set_data([], [])
        explosion_marker.set_data([], [])
        return target_marker, missile_marker, trajectory_line, explosion_marker

    def update(frame):
        # 设置目标位置
        if frame == 0:
            target_marker.set_data(*target_position)
            return target_marker, missile_marker, trajectory_line, explosion_marker

        # 更新轨迹和巡飞弹的位置
        missile_x = x_data[frame] if frame < len(x_data) else target_position[0]
        missile_y = y_data[frame] if frame < len(y_data) else target_position[1]
        
        missile_marker.set_data(missile_x, missile_y)
        trajectory_line.set_data(x_data[:frame+1], y_data[:frame+1])

        # 在目标处引爆
        if frame >= len(x_data) - 1:
            explosion_marker.set_data(target_position)
        
        return target_marker, missile_marker, trajectory_line, explosion_marker

    # 创建动画，增加帧率
    anim = FuncAnimation(fig, update, frames=len(x_data), init_func=init, blit=True, repeat=False)

    # 保存为 GIF 动画，增加 fps
    gif_path = os.path.join(save_path, f"result_{args.number}.gif")
    anim.save(gif_path, writer=PillowWriter(fps=20))
    print(f"Animation saved at: {gif_path}")

# 运行生成 GIF 动画
generate_advanced_trajectory_animation()
