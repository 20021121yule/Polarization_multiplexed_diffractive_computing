import torch
from ONN import *
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 复数域上的线性变换
def PGenerate(width, unitSize, dis, wavelength):
    k = 2 * np.pi * wavelength  # 角波数

    l1 = width[0] * (np.arange(1, unitSize[0] + 1) - (unitSize[0] + 1) / 2)
    l2 = width[1] * (np.arange(1, unitSize[1] + 1) - (unitSize[1] + 1) / 2)

    x1, y1 = np.meshgrid(l1, l1)
    x2, y2 = np.meshgrid(l2, l2)

    # 转换为列向量和行向量
    x1 = x1.ravel()  # 将 x1 转为行向量
    y1 = y1.ravel()  # 将 y1 转为行向量
    x2 = x2.ravel()[:, None]  # 将 x2 转为列向量
    y2 = y2.ravel()[:, None]  # 将 y2 转为列向量

    # 计算空间距离
    r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + dis ** 2)

    P = np.exp(-1j * k * r) / (r ** 2)

    # 归一化
    P /= np.linalg.norm(P, 2)

    return torch.tensor(P, dtype=torch.cfloat)  # 返回 PyTorch 张量


# 不同传播距离自然对应不同的复变换
dis_1 = 0.0001
dis_2 = 0.0005

# 我们目前的复变换是传播矩阵
trans_matrix_xx = PGenerate((pixel_size, pixel_size), (N, N), dis_1, wavelength)
trans_matrix_yy = PGenerate((pixel_size, pixel_size), (N, N), dis_2, wavelength)

# 数据集需要的个数
dataset_num = 2000
input_tensor = 6 * torch.rand(dataset_num, 1, N, N).to(torch.cfloat)
ground_truth = torch.zeros(dataset_num, 1, N, N).to(torch.cfloat)
# 规定第一个是x方向偏振，那么第二个是y方向偏振，以此类推
for i in range(dataset_num):
    if (i // 2) % 2 == 0:

        ground_truth[i, 0, :, :] = torch.matmul(trans_matrix_xx,
                                                input_tensor[i, 0, :, :].reshape(-1, 1).to(torch.cfloat)).reshape(N, N)

    else:
        ground_truth[i, 0, :, :] = torch.matmul(trans_matrix_yy,
                                                input_tensor[i, 0, :, :].reshape(-1, 1).to(torch.cfloat)).reshape(N, N)

# 显示传播后的强度图
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(torch.abs(ground_truth[0, 0]).cpu().numpy(), cmap='gray')
plt.title('Ground Truth @ dis_1')
plt.subplot(1, 2, 2)
plt.imshow(torch.abs(ground_truth[2, 0]).cpu().numpy(), cmap='gray')
plt.title('Ground Truth @ dis_2')
plt.colorbar()
plt.show()


np.save('input_tensor.npy', input_tensor.cpu().detach().numpy())
np.save('ground_truth.npy', ground_truth.cpu().detach().numpy())


print("数据处理完成，输入与对应传播后的复振幅已保存为 input_tensor.npy 和 ground_truth.npy。")
