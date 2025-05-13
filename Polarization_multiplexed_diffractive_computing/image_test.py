import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 基础参数
pixel_size = 3.0e-3  # 像素宽度，单位：米
wavelength = 532.8e-9  # 波长，单位：米
dz = 0.01  # 层间间距，单位：米
N = 20  # 图像尺寸

def PGenerate(width, unitSize, dis, wavelength):
    """
    生成散射矩阵 P
    width: 每个像素在空间中的真实宽度 (pixel_size, pixel_size)
    unitSize: 像素数量 (N, N)
    dis: 层间距离 dz
    """
    k = 2 * np.pi / wavelength  # 角波数

    # 空间坐标（以中心为原点）
    l1 = width[0] * (np.arange(1, unitSize[0] + 1) - (unitSize[0] + 1) / 2)
    l2 = width[1] * (np.arange(1, unitSize[1] + 1) - (unitSize[1] + 1) / 2)

    # 生成网格
    x1, y1 = np.meshgrid(l1, l1)  # 输入平面
    x2, y2 = np.meshgrid(l2, l2)  # 输出平面

    x1, y1 = x1.ravel(), y1.ravel()             # [N*N]
    x2, y2 = x2.ravel()[:, None], y2.ravel()[:, None]  # [N*N, 1]

    # 距离矩阵（每个输入点与输出点之间的传播路径）
    r = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + dis**2)  # [N*N, N*N]
    P = np.exp(-1j * k * r) / (r**2)

    # 归一化
    P /= np.linalg.norm(P, 2)
    return torch.tensor(P, dtype=torch.cfloat)

# 加载图像为灰度张量
def load_image_to_tensor(path, N):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像：{path}")
    image = cv2.resize(image, (N, N))
    image_tensor = torch.from_numpy(image).float() / 255.0  # 归一化
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # [B, C, H, W]
    return image_tensor

# ================== 主程序 ===================

# 图像路径
image_path = "/Users/yule/Desktop/test.jpg"
input_tensor = load_image_to_tensor(image_path, N)  # [1, 1, N, N]
input_field = input_tensor[0, 0]  # [N, N]，实数图像
input_field = input_field.to(torch.cfloat)  # 转复数光场

# 生成传播矩阵 P
P = PGenerate((pixel_size, pixel_size), (N, N), dz, wavelength)  # [N*N, N*N]

# 展平输入为列向量并传播
input_vector = input_field.reshape(N*N, 1)  # [N*N, 1]
output_vector = torch.matmul(P, input_vector)  # [N*N, 1]
output_field = output_vector.reshape(N, N)  # [N, N]

# 显示传播后的模值图像
plt.imshow(torch.abs(output_field).cpu().numpy(), cmap='gray')
plt.title("Propagated Field via P Matrix")
plt.axis("off")
plt.show()
