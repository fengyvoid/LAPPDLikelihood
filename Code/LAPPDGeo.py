import numpy as np

# LAPPD 网格生成函数
def LAPPD_Grid_position(LAPPD_Center_Position, LAPPD_direction):
    L_grid_y = 0.00691
    L_grid_x = 0.00691
    steps_y = 14
    steps_x = 14

    n = LAPPD_direction / np.linalg.norm(LAPPD_direction)
    reference_vec = np.array([0, 1, 0])
    sx = np.cross(n, reference_vec)
    sx /= np.linalg.norm(sx)

    sy = np.cross(n, sx)
    sy /= np.linalg.norm(sy)

    steps_y_total = 2 * steps_y
    y_positions = [sy * L_grid_y * (steps_y - 0.5) - sy * L_grid_y * i for i in range(steps_y_total)]
    positions = np.zeros((steps_y_total, 2 * steps_x, 3))
    
    for i, y_vec in enumerate(y_positions):
        x_start = -sx * L_grid_x * (steps_x - 0.5)
        for j in range(2 * steps_x):
            x_offset = x_start + sx * L_grid_x * j
            pos = LAPPD_Center_Position + y_vec + x_offset
            positions[i, j, :] = pos

    return positions