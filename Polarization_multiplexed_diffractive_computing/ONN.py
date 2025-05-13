import torch
import torch.nn as nn

layer = 6  # 网络层数，注意在此项目当中不包括起偏器,包括入射层和最后输出层，调制层层数 = layer - 2
N = 30  # 图像尺寸 (像素数)，这里是 50×50 的图像
pixel_size = 2 * 532.8e-6  # 像素宽度，单位：米，满足论文中要求的pixel_size = 2 * wavelength
wavelength = 532.8e-6  # 波长，单位：米，由于尺寸需求，这里就选毫米波了
dz = 0.001  # 层间间距，单位：米

batch = 2 # 输入批次

def polarizer_seeds(pixel_num):
    polarizer_matrix = torch.tensor([[0.0, torch.pi / 4],
                                     [3 * torch.pi / 4, torch.pi / 2]])  # 原始偏振矩阵（单位：弧度）
    repeat_count = int(pixel_num / 2)  # 重复次数
    return polarizer_matrix.repeat(repeat_count, repeat_count)


class OpticalNetwork(nn.Module):
    def __init__(self, layerNum, N, pixel_size, wavelength):
        super(OpticalNetwork, self).__init__()

        self.layerNum = layerNum
        self.N = N
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.k = 2.0 * torch.pi / wavelength  # 波矢（k）
        self.dz = dz  # 层间间距

        # 生成偏振种子，polarization_seeds
        self.polarizer_seeds = polarizer_seeds(self.N)

        self.matrix_cos2theta = torch.cos(self.polarizer_seeds) ** 2
        self.matrix_sin2theta = torch.sin(self.polarizer_seeds) ** 2
        self.matrix_sinthetacostheta = torch.sin(self.polarizer_seeds) * torch.cos(self.polarizer_seeds)

        # 生成调制层参数 O
        self.O = nn.ParameterList([
            nn.Parameter(torch.rand(N, N))  # 每层随机初始化一个 N×N 的矩阵作为调制层
            for _ in range(self.layerNum - 2)
        ])

        # 生成调制层参数 Amp
        self.Amp = nn.ParameterList([
            nn.Parameter(torch.rand(N, N))  # 每层随机初始化一个 N×N 的矩阵作为调制层
            for _ in range(self.layerNum - 2)
        ])

        # 计算波矢k_x和k_y
        k_x = 2 * torch.pi / (pixel_size * N) * torch.arange(-N / 2, N / 2)
        k_y = 2 * torch.pi / (pixel_size * N) * torch.arange(-N / 2, N / 2)
        k_x, k_y = torch.meshgrid(k_x, k_y, indexing='ij')
        k_z = torch.sqrt(self.k ** 2 - k_x ** 2 - k_y ** 2)  # 计算k_z（垂直方向的波矢）

        # 检查k_z中的NaN（如果有NaN，说明计算出了不合理的波矢）
        nan_mask = torch.isnan(k_z)
        if torch.any(nan_mask):
            print('WARNING: NAN pixels detected')

        # 构造传播函数H
        self.H = torch.exp(1j * k_z * dz)  # 利用角谱法计算传播函数

    def forward(self, X, epoch):
        # 偏振处理
        xx = torch.zeros_like(X, dtype=torch.cfloat)
        if epoch % 2 == 0:        X = torch.cat([X, xx], dim=1) # 如果epoch为偶数，那么输入我们认为沿着x方向偏振
        else: X = torch.cat([xx, X], dim=1) # epoch为奇数，沿y方向偏振

        # 传播计算
        X = self.propagation(X)
        X = self.imaging_layer(X, epoch)

        return X

    def AST(self, input):
        # 角谱定理：傅里叶变换、频谱移动、传播
        opt_field_fft = torch.fft.fft2(input, dim=(-2, -1))  # 2D傅里叶变换
        opt_field_fftshift = torch.fft.fftshift(opt_field_fft, dim=(-2, -1))  # 中心化频谱
        opt_field_fftshift = opt_field_fftshift * self.H  # 使用传播函数H进行传播
        opt_field_ifftshift = torch.fft.ifftshift(opt_field_fftshift, dim=(-2, -1))  # 逆频谱移动
        opt_field_ifft = torch.fft.ifft2(opt_field_ifftshift, dim=(-2, -1))  # 逆傅里叶变换，得到传播后的光场

        return opt_field_ifft

    def apply_polarizer_array(self, input):
        # 注意注册的时候要注册为复数形式的float
        output = torch.zeros(input.shape, dtype=torch.cfloat, device=input.device)

        # 输出x偏振
        output[:, 0] = self.matrix_cos2theta * input[:, 0] + self.matrix_sinthetacostheta * input[:, 1]
        # 输出y偏振
        output[:, 1] = self.matrix_sinthetacostheta * input[:, 0] + self.matrix_sin2theta * input[:, 1]

        return output

    def polarize_operator(self, input):
        output = self.AST(input)  # 先传播一个间距dz
        output = self.apply_polarizer_array(output) # 输出就是经过起偏器

        return output

    def propagation(self, X):
        for kk in range(self.layerNum - 2):
            X = self.AST(X)
            phase = self.O[kk]
            complex_modulation = self.Amp[kk] * torch.exp(1j * phase)
            X = complex_modulation * X

            if kk == int((self.layerNum - 2) / 2):
                X = self.apply_polarizer_array(X)

        return X

    def imaging_layer(self, input, epoch):
        # 最后一层也进行传播
        input = self.AST(input)

        # 在最后一层加入的偏振片
        if epoch % 2 == 0:
            angle = torch.tensor(0.0)  # 偏振片角度
        else: angle = torch.tensor(torch.pi / 2)

        polarizer_matrix = torch.tensor([[angle, angle],
                                         [angle, angle]])  # 原始偏振矩阵（单位：弧度）
        repeat_count = int(self.N / 2)  # 重复次数
        polarizer_matrix = polarizer_matrix.repeat(repeat_count, repeat_count)

        matrix_cos2theta = torch.cos(polarizer_matrix) ** 2
        matrix_sin2theta = torch.sin(polarizer_matrix) ** 2
        matrix_sinthetacostheta = torch.sin(polarizer_matrix) * torch.cos(polarizer_matrix)

        output = torch.zeros(input.shape, dtype=torch.cfloat, device=input.device)
        output[:, 0] = matrix_cos2theta * input[:, 0] + matrix_sinthetacostheta * input[:, 1]    # 输出x偏振
        output[:, 1] = matrix_sinthetacostheta * input[:, 0] + matrix_sin2theta * input[:, 1]    # 输出y偏振

        # 将两个方向偏振组合起来
        output_comb = output[:, 0] * torch.cos(angle) + output[:, 1] * torch.sin(angle)

        return output_comb
