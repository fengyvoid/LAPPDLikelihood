import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 初始的Muon位置和方向
muon_start_position = [-0.374149027601086, -0.5679029804241634, 1.2]
mu_direction = np.array([0.29269516, 0.11321445, 0.94947987])

# 步数和步长设定
x_step = 1
x_step_size = 0.05
y_step = 1
y_step_size = 0.05
theta_step = 2
theta_step_size = np.radians(3)  # 转换成弧度
phi_step = 2
phi_step_size = np.radians(3)  # 转换成弧度
arrow_length = 0.2  # 箭头长度

# 定义旋转矩阵
def rotate_vector(vector, theta, phi):
    """旋转向量 vector, theta 是绕y轴旋转的角度，phi 是绕x轴旋转的角度"""
    # 绕y轴旋转 (theta)
    rot_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
    # 绕x轴旋转 (phi)
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(phi), -np.sin(phi)],
                      [0, np.sin(phi), np.cos(phi)]])
    
    # 先绕y轴旋转，再绕x轴旋转
    rotated_vector = rot_x @ (rot_y @ vector)
    return rotated_vector

# 创建3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 遍历x和y坐标，调整Muon起始位置
for x_offset in range(-x_step, x_step + 1):
    for y_offset in range(-y_step, y_step + 1):
        # 计算新的x, y坐标
        x_at_z = muon_start_position[0] + x_offset * x_step_size
        y_at_z = muon_start_position[1] + y_offset * y_step_size
        z_at_z = muon_start_position[2]
        new_start_position = [x_at_z, y_at_z, z_at_z]

        # 遍历theta和phi，调整Muon方向
        for theta_offset in range(-theta_step, theta_step + 1):
            for phi_offset in range(-phi_step, phi_step + 1):
                # 计算新的theta和phi
                theta = theta_offset * theta_step_size
                phi = phi_offset * phi_step_size
                
                # 旋转初始Muon方向
                new_mu_direction = rotate_vector(mu_direction, theta, phi)
                new_mu_direction = new_mu_direction / np.linalg.norm(new_mu_direction)  # 归一化

                # 绘制箭头，箭头起点为start position，方向为muon direction
                ax.quiver(new_start_position[0], new_start_position[1], new_start_position[2], 
                          new_mu_direction[0], new_mu_direction[1], new_mu_direction[2], 
                          length=arrow_length, color='b', arrow_length_ratio=0.2)

# 设置X, Y, Z轴的范围
ax.set_xlim([-1, 0])
ax.set_ylim([-1, 0])
ax.set_zlim([0, 2])

# 添加红色箭头，表示最初的Muon位置和方向，箭头长度为0.5
ax.quiver(muon_start_position[0], muon_start_position[1], muon_start_position[2], 
          mu_direction[0], mu_direction[1], mu_direction[2], 
          length=0.5, color='r', arrow_length_ratio=0.2)

# 设置图像标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Muon Start Positions and Directions')

# 显示图像
plt.show()

