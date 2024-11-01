import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# 获取命令行参数
if len(sys.argv) != 2:
    print("Usage: python3 view.py <EventNumber>")
    sys.exit(1)

event_number = sys.argv[1]  # 获取命令行传入的数字参数

# 构造文件名
#input_file = f'/Users/fengy/ANNIESofts/Analysis/ProjectionComplete/plots/Event{event_number}output.txt'
#input_file = f'/Users/fengy/ANNIESofts/Analysis/ProjectionComplete/MC_plots/Event{event_number}_MCoutput.txt'


#input_file = f'/Users/fengy/ANNIESofts/Analysis/ProjectionComplete/MC_plots/4410stepTest_w/Event{event_number}_MCoutput.txt'

input_file = f'/Users/fengy/ANNIESofts/Analysis/ProjectionComplete/MC_plots/8.4410stepTest_SamePos_^3w_cap=1/Event{event_number}_MCoutput.txt'

# 读取数据
with open(input_file, 'r') as file:
    result = json.load(file)

# 提取所有的diff value，用于归一化
diff_values = [elem[6] for elem in result]  # 第7个元素是diff value
diff_values = np.array(diff_values)

# 归一化diff value
norm_diff_values = (diff_values - np.min(diff_values)) / (np.max(diff_values) - np.min(diff_values))

# 创建3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colorMin = 0.0

# 遍历每个粒子的数据
for i, elem in enumerate(result):
    # 读取粒子的x, y, z位置 (第0, 1, 2个元素)
    x, y, z = elem[0], elem[1], elem[2]
    
    # 读取粒子的x, y, z方向 (第3, 4, 5个元素)
    dir_x, dir_y, dir_z = elem[3], elem[4], elem[5]
    
    # 获取归一化的diff value（第6个元素）并计算箭头长度
    normalized_value = norm_diff_values[i]
    arrow_length = 1 - normalized_value 

    thisColorValue = 0
    if(arrow_length > colorMin):
        thisColorValue = (arrow_length - colorMin) / (1 - colorMin)
    
    # 计算颜色，使用rainbow colormap
    color = cm.rainbow(thisColorValue)
    
    # 绘制箭头，起点是x, y, z，方向是dir_x, dir_y, dir_z
    ax.quiver(x, y, z, dir_x, dir_y, dir_z, length=arrow_length, color=color, arrow_length_ratio=0.2)

# 设置颜色条
norm = plt.Normalize(vmin=colorMin, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=norm)
sm.set_array([])

# 设置颜色条
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Diff value (normalized)')

# 设置图像标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Particle Positions and Directions for Event {event_number}')

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 2])

# 显示图像
plt.show()

