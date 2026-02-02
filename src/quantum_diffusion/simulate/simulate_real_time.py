import torch
import numpy as np
from systems.harmonic_oscillator import HarmonicOscillator
from models.quantum_diffuser import QuantumDiffuser
from utils.visualization import plot_real_time_evolution

def simulate_real_time_evolution(
    model_path: str = "models/real_time_diffuser.pth",
    num_samples: int = 1000,
    t_max: float = 10.0,
    num_time_steps: int = 20,
    output_path: str = "results/real_time_evolution.png"
):
    """实时动力学模拟：预测量子分布演化"""
    # 1. 加载模型和配置
    checkpoint = torch.load(model_path)
    system = HarmonicOscillator(**checkpoint["system_kwargs"]).to("cuda" if torch.cuda.is_available() else "cpu")
    diffuser = QuantumDiffuser(
        system=system,
        num_train_timesteps=checkpoint["num_train_timesteps"]
    ).to(system.device)
    diffuser.load_state_dict(checkpoint["model_state_dict"])
    diffuser.eval()

    # 2. 生成初始样本（从经典分布采样）
    q0 = torch.randn((num_samples, 1), device=system.device) * 0.5
    p0 = torch.randn((num_samples, 1), device=system.device) * 0.5
    alpha0 = q0 + 1j * p0
    initial_samples = torch.cat([q0.unsqueeze(1), p0.unsqueeze(1)], dim=1).float()

    # 3. 模拟时间演化
    times = np.linspace(0, t_max, num_time_steps)
    evolution_samples = [initial_samples.cpu().numpy()]

    with torch.no_grad():
        for t in times[1:]:
            # 时间步长
            dt = t - times[times < t][-1]
            # 正向SDE演化
            current_samples = torch.tensor(evolution_samples[-1], device=system.device).float()
            alpha_current = current_samples[:, 0, :] + 1j * current_samples[:, 1, :]
            # 模型预测噪声
            timestep = torch.full((num_samples,), int(t / t_max * checkpoint["num_train_timesteps"]), device=system.device)
            noise_pred = diffuser(current_samples, timestep)
            # 去噪更新
            updated_samples = diffuser.noise_scheduler.step(
                noise_pred, timestep, current_samples
            ).prev_sample
            evolution_samples.append(updated_samples.cpu().numpy())

    # 4. 可视化结果
    plot_real_time_evolution(
        evolution_samples,
        times,
        s=system.s,
        output_path=output_path
    )
    print(f"演化结果保存至：{output_path}")

    return evolution_samples, times

if __name__ == "__main__":
    simulate_real_time_evolution(
        model_path="models/real_time_diffuser.pth",
        num_samples=1000,
        t_max=10.0,
        num_time_steps=20
    )