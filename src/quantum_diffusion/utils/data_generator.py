import torch
import numpy as np
from systems.harmonic_oscillator import HarmonicOscillator

def generate_quantum_data(
    system: HarmonicOscillator,
    num_samples: int = 10000,
    q_range: tuple = (-3, 3),
    p_range: tuple = (-3, 3),
    n_quantum: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成量子准概率分布的训练数据（基于精确解）
    返回:
        X: (num_samples, 2) → (Re(α), Im(α))
        y: (num_samples,) → 准概率分布值
    """
    # 均匀采样相空间
    q = np.random.uniform(q_range[0], q_range[1], num_samples)
    p = np.random.uniform(p_range[0], p_range[1], num_samples)
    alpha = (q / np.sqrt(2 * system.hbar)) + 1j * (p / np.sqrt(2 * system.hbar))

    # 计算精确分布值
    alpha_tensor = torch.tensor(alpha, dtype=torch.complex64, device=system.device)
    y = system.s_order.quasiprobability_dist(alpha_tensor, n=n_quantum).cpu().numpy()

    # 转换为模型输入格式
    X = np.stack([q, p], axis=1)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (num_samples, 2, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)

    return X_tensor, y_tensor
    