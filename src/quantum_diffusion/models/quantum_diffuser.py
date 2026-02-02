import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import DDPMScheduler
from .unet_1d import UNet1D
from core.diffusion_sde import RealTimeSDE, ImaginaryTimeSDE

class QuantumDiffuser(nn.Module):
    """量子相空间Diffusion模型（统一实时/虚时接口）
    核心功能：
    1. 实时动力学：学习量子演化的噪声模型
    2. 虚时采样：从经典分布生成量子准概率分布
    """
    def __init__(
        self,
        system: "Base1DQuantumSystem",
        unet_config: dict = None,
        num_train_timesteps: int = 1000
    ):
        super().__init__()
        self.system = system
        self.device = system.device
        self.hbar = system.hbar

        # 噪声调度器（DDPMScheduler兼容实时/虚时）
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
            prediction_type="epsilon"  # 预测噪声ε
        )

        # 1D U-Net（默认配置）
        unet_config = unet_config or {
            "in_channels": 2,
            "out_channels": 2,
            "hidden_channels": 64,
            "num_blocks": 3,
            "time_emb_dim": 16
        }
        self.unet = UNet1D(**unet_config).to(self.device)

        # 绑定SDE
        self.real_time_sde = RealTimeSDE(system).to(self.device)
        self.imaginary_time_sde = ImaginaryTimeSDE(system).to(self.device)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """前向传播：预测噪声（Diffusers标准接口）
        x: (batch_size, 2, 1) → (Re(α), Im(α))
        timesteps: (batch_size,)
        """
        return self.unet(x, timesteps.unsqueeze(-1).float())

    def _prepare_data(self, batch_size: int) -> torch.Tensor:
        """生成训练数据：相空间坐标α（从经典分布采样）"""
        # 经典分布采样（均匀分布覆盖相空间主要区域）
        q = torch.randn((batch_size, 1), device=self.device) * 2.0
        p = torch.randn((batch_size, 1), device=self.device) * 2.0
        alpha = q + 1j * p  # (batch_size, 1) 复数
        # 转换为模型输入格式：(batch_size, 2, 1)
        return torch.cat([q.unsqueeze(1), p.unsqueeze(1)], dim=1).float()

    def train_real_time_step(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """实时动力学训练单步"""
        # 1. 生成数据
        x_clean = self._prepare_data(batch_size)  # (batch_size, 2, 1)
        alpha_clean = x_clean[:, 0, :] + 1j * x_clean[:, 1, :]  # (batch_size, 1)

        # 2. 正向Diffusion：添加噪声
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (batch_size,), device=self.device)
        noise = torch.randn_like(x_clean)
        x_noisy = self.noise_scheduler.add_noise(x_clean, noise, timesteps)

        # 3. 计算物理约束目标
        # 从噪声样本恢复复数α
        alpha_noisy = x_noisy[:, 0, :] + 1j * x_noisy[:, 1, :]
        # 计算SDE漂移项（物理约束）
        drift = self.real_time_sde.drift(alpha_noisy.squeeze(-1), timesteps.float() / self.noise_scheduler.num_train_timesteps)
        drift = torch.cat([drift.real.unsqueeze(1), drift.imag.unsqueeze(1)], dim=1).unsqueeze(-1)

        # 4. 目标：噪声 + 物理约束修正
        target = noise + 0.1 * drift  # 平衡噪声预测与物理约束

        return x_noisy, timesteps, target

    def train_imaginary_time_step(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """虚时采样训练单步"""
        # 1. 生成经典Boltzmann分布样本
        beta_classical = 1.0  # 温度倒数
        q = torch.randn((batch_size, 1), device=self.device)
        p = torch.randn((batch_size, 1), device=self.device)
        # Boltzmann权重采样
        H_classical = self.system.classical_hamiltonian(q + 1j * p)
        weight = torch.exp(-beta_classical * H_classical)
        mask = torch.rand_like(weight) < weight / weight.max()
        while not mask.all():
            q[~mask] = torch.randn_like(q[~mask])
            p[~mask] = torch.randn_like(p[~mask])
            H_classical[~mask] = self.system.classical_hamiltonian(q[~mask] + 1j * p[~mask])
            weight[~mask] = torch.exp(-beta_classical * H_classical[~mask])
            mask[~mask] = torch.rand_like(weight[~mask]) < weight[~mask] / weight.max()
        x_clean = torch.cat([q.unsqueeze(1), p.unsqueeze(1)], dim=1).float()

        # 2. 正向Diffusion：经典→量子过渡
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (batch_size,), device=self.device)
        noise = torch.randn_like(x_clean)
        x_noisy = self.noise_scheduler.add_noise(x_clean, noise, timesteps)

        # 3. 量子分布目标（从解析解获取）
        alpha_clean = x_clean[:, 0, :] + 1j * x_clean[:, 1, :]
        target_quantum = self.system.s_order.quasiprobability_dist(alpha_clean.squeeze(-1))
        target_quantum = target_quantum.unsqueeze(1).unsqueeze(1)  # 适配模型输出

        return x_noisy, timesteps, target_quantum

    def sample_imaginary_time(self, num_samples: int) -> torch.Tensor:
        """虚时采样：生成量子准概率分布样本"""
        self.eval()
        # 1. 从噪声初始化
        x = torch.randn((num_samples, 2, 1), device=self.device)

        # 2. 反向去噪过程
        self.noise_scheduler.set_timesteps(self.noise_scheduler.num_train_timesteps)
        with torch.no_grad():
            for t in reversed(self.noise_scheduler.timesteps):
                # 预测噪声
                noise_pred = self(x, torch.full((num_samples,), t, device=self.device))
                # 去噪步骤
                x = self.noise_scheduler.step(noise_pred, t, x).prev_sample

        # 3. 转换为复数相空间坐标
        q = x[:, 0, 0].cpu().numpy()
        p = x[:, 1, 0].cpu().numpy()
        alpha = q + 1j * p
        return alpha, q, p

    def train_loop(
        self,
        num_epochs: int,
        batch_size: int,
        lr: float = 1e-4,
        mode: str = "real_time"  # "real_time" 或 "imaginary_time"
    ):
        """训练循环"""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()

            # 生成训练数据
            if mode == "real_time":
                x_noisy, timesteps, target = self.train_real_time_step(batch_size)
            elif mode == "imaginary_time":
                x_noisy, timesteps, target = self.train_imaginary_time_step(batch_size)
            else:
                raise ValueError(f"不支持的训练模式：{mode}")

            # 模型预测
            noise_pred = self(x_noisy, timesteps)
            loss = criterion(noise_pred, target)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 日志输出
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

        return self
        