import torch
import numpy as np
from systems.harmonic_oscillator import HarmonicOscillator
from models.quantum_diffuser import QuantumDiffuser
from utils.visualization import plot_quantum_distribution
from utils.metrics import kl_divergence

def simulate_imaginary_time_sampling(
    model_path: str = "models/imaginary_time_diffuser.pth",
    num_samples: int = 10000,
    n_quantum: int = 0,  # 目标量子数
    output_path: str = "results/imaginary_time_sampling.png"
):
    """虚时采样模拟：生成量子准概率分布"""
    # 1. 加载模型和配置
    checkpoint = torch.load(model_path)
    system = HarmonicOscillator(**checkpoint["system_kwargs"]).to("cuda" if torch.cuda.is_available() else "cpu")
    diffuser = QuantumDiffuser(
        system=system,
        num_train_timesteps=checkpoint["num_train_timesteps"]
    ).to(system.device)
    diffuser.load_state_dict(checkpoint["model_state_dict"])

    # 2. 虚时采样
    print(f"开始虚时采样（{num_samples}个样本）")
    alpha_samples, q_samples, p_samples = diffuser.sample_imaginary_time(num_samples=num_samples)

    # 3. 计算精确解（用于对比）
    q_grid = np.linspace(-3, 3, 100)
    p_grid = np.linspace(-3, 3, 100)
    exact_dist = system.exact_quasiprobability(q_grid, p_grid, n=n_quantum)

    # 4. 计算采样分布
    sampled_dist, _, _ = np.histogram2d(q_samples, p_samples, bins=50, density=True)
    # 调整维度匹配
    sampled_dist = np.resize(sampled_dist, exact_dist.shape)

    # 5. 精度评估
    kl = kl_divergence(exact_dist, sampled_dist)
    print(f"KL散度（精确解 vs 采样）：{kl:.4f}")

    # 6. 可视化
    plot_quantum_distribution(
        q_samples,
        p_samples,
        exact_dist=exact_dist,
        q_grid=q_grid,
        p_grid=p_grid,
        title=f"Quantum Distribution (s={system.s}, KL={kl:.4f})",
        output_path=output_path
    )
    print(f"采样结果保存至：{output_path}")

    return alpha_samples, q_samples, p_samples

if __name__ == "__main__":
    simulate_imaginary_time_sampling(
        model_path="models/imaginary_time_diffuser.pth",
        num_samples=10000,
        n_quantum=0  # 基态
    )
    