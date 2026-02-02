import torch
from systems.harmonic_oscillator import HarmonicOscillator
from models.quantum_diffuser import QuantumDiffuser

def train_real_time_diffuser(
    num_epochs: int = 5000,
    batch_size: int = 256,
    lr: float = 1e-4,
    save_path: str = "models/real_time_diffuser.pth",
    **system_kwargs
):
    """训练实时动力学Diffusion模型
    system_kwargs: 量子系统参数（如m, omega, dissipation等）
    """
    # 1. 初始化量子系统
    system = HarmonicOscillator(**system_kwargs).to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"初始化谐振子系统：m={system.m}, omega={system.omega}, s={system.s}")
    print(f"耗散配置：{system.dissipation}")

    # 2. 初始化Diffusion模型
    diffuser = QuantumDiffuser(
        system=system,
        num_train_timesteps=1000
    ).to(system.device)

    # 3. 训练循环
    print(f"开始训练（{num_epochs}轮，批次大小{batch_size}）")
    diffuser.train_loop(
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        mode="real_time"
    )

    # 4. 保存模型
    torch.save({
        "model_state_dict": diffuser.state_dict(),
        "system_kwargs": system_kwargs,
        "num_train_timesteps": diffuser.noise_scheduler.num_train_timesteps
    }, save_path)
    print(f"模型保存至：{save_path}")

    return diffuser

if __name__ == "__main__":
    # 训练示例：带振幅衰减的Wigner分布实时演化
    train_real_time_diffuser(
        num_epochs=5000,
        batch_size=256,
        lr=1e-4,
        m=1.0,
        omega=1.0,
        s=0.0,  # Wigner分布
        dissipation={
            "type": "amplitude_damping",
            "gamma": 0.1,
            "non_markovian": False
        }
    )