import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# 定义矩阵大小N（N需是2的整数倍）
N = 50

def polarizer_seeds(pixel_num):
    polarizer_matrix = torch.tensor([[0, torch.pi / 4], [3 * torch.pi / 4, torch.pi / 2]])    # 原始偏振矩阵（单位：弧度）
    repeat_count = int(pixel_num / 2)# 重复次数
    return polarizer_matrix.repeat(repeat_count, repeat_count)

polarizer_array = polarizer_seeds(N)

print(polarizer_array.shape)

# 自定义颜色
colors = ['#729ef3', '#f9dc5c', '#9ee493', '#f27a7d']  # 蓝、黄、绿、红
cmap = ListedColormap(colors)

# 生成对应的整数索引矩阵（便于着色）
color_indices = (polarizer_array.numpy() / (np.pi / 4)).astype(int) % 4

# 绘图
plt.figure(figsize=(6, 6))
plt.imshow(color_indices, cmap=cmap, interpolation='none')

# 标注颜色和角度对应关系的colorbar
cbar = plt.colorbar(ticks=[0, 1, 2, 3])
cbar.ax.set_yticklabels(['0', 'π/4', 'π/2', '3π/4'])

# 图形修饰
plt.title("Polarizer Array (Repeated)")
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.show()
