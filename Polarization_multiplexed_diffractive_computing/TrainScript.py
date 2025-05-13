import torch.optim as optim
from ONN import *
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# -------------------load data----------------------------
class CustomComplexDataset(Dataset):
    def __init__(self, input_file, target_file):
        self.input_data = np.load(input_file, mmap_mode='r')  # [N, 1, H, W]
        self.target_data = np.load(target_file, mmap_mode='r')  # [N, 1, H, W]

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_sample = torch.tensor(self.input_data[idx], dtype=torch.cfloat)
        target_sample = torch.tensor(self.target_data[idx], dtype=torch.cfloat)
        return input_sample, target_sample


# ========== 主程序 ==========
if __name__ == '__main__':

    # 加载数据集
    dataset = CustomComplexDataset("input_tensor.npy", "ground_truth.npy")
    train_loader = DataLoader(dataset, batch_size=batch, shuffle=False)

    # 构建网络
    net = OpticalNetwork(layer, N, pixel_size, wavelength)


    # 自定义复数MSE损失函数
    # 后续训练是在偏振自由度上平均的，所以这里只需要考虑归一化即可
    def complex_mse(output, target):
        return torch.mean((output.real - target.real) ** 2 + (output.imag - target.imag) ** 2)


    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    num_epochs = 1000

    global_step = 0  # 全局 step 传入网络控制偏振方向
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0.0
        batch_count = 0

        for input_tensor, target_tensor in train_loader:
            input_tensor = input_tensor.to(torch.cfloat)
            target_tensor = target_tensor.to(torch.cfloat)

            optimizer.zero_grad()
            pred = net(input_tensor, global_step)
            loss = complex_mse(pred, target_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() # 本质上是在偏振自由度上平均了
            batch_count += 1
            global_step += 1

        avg_loss = total_loss / batch_count
        if epoch % 10 == 0:
         print(f"Epoch [{epoch}/{num_epochs}], Average Loss = {avg_loss:.6f}")

    # ========== 可视化结果 ==========
    with torch.no_grad():
        net.eval()

        # 使用训练最后一个 batch 的数据做测试
        input_tensor_vis = input_tensor.to(torch.cfloat)
        target_tensor_vis = target_tensor.to(torch.cfloat)

        # 这里使用 global_step - 1 表示最后一次偏振方向（与训练中保持一致）
        pred = net(input_tensor_vis, global_step - 1)

        # 抽取第一个样本
        pred_sample = pred[0]
        target_sample = target_tensor_vis[0, 0]

        # 获取模值图
        pred_abs = torch.abs(pred_sample).cpu().numpy()
        target_abs = torch.abs(target_sample).cpu().numpy()
        error_abs = np.abs(pred_abs - target_abs)

        # 获取相位图
        pred_phase = torch.angle(pred_sample).cpu().numpy()
        target_phase = torch.angle(target_sample).cpu().numpy()
        error_phase = np.angle(np.exp(1j * (pred_phase - target_phase)))  # wrap 到 [-π, π]

        # 可视化模值和误差图
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(target_abs, cmap='gray')
        plt.title("Target |E|")
        plt.colorbar()
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(pred_abs, cmap='gray')
        plt.title("Predicted |E|")
        plt.colorbar()
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(error_abs, cmap='hot')
        plt.title("Amplitude Error |ΔE|")
        plt.colorbar()
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # 可视化相位误差图
        plt.figure(figsize=(5, 5))
        plt.imshow(error_phase, cmap='bwr', vmin=-np.pi, vmax=np.pi)
        plt.title("Phase Error Δϕ")
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        plt.show()
