import torch
from systems.harmonic_oscillator import HarmonicOscillator
from models.quantum_diffuser import QuantumDiffuser

def train_imaginary_time_diffuser(
    num_epochs: int = 5000,
    batch_size: int = 256,
    lr: float = 1e-4,
    save_path: str = "models/imaginary_time_diffuser.pth",
    **system_kwargs
):
    """训练虚时采样Diffusion模型（从Boltzmann→量子分布）"""
    # 1. 初始化量子系统
    system = HarmonicOscillator(**system_kwargs).to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"初始化谐振子系统：m={system.m}, omega={system.omega}, s={system.s}")
    print(f"目标量子分布：{'Wigner' if s==0 else 'Husimi Q' if s==-1 else 'Glauber P'}")

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
        mode="imaginary_time"
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
    # 训练示例：从Boltzmann→Husimi Q分布（s=-1）
    train_imaginary_time_diffuser(
        num_epochs=5000,
        batch_size=256,
        lr=1e-4,
        m=1.0,
        omega=1.0,
        s=-1.0,  # Husimi Q分布
        dissipation={
            "type": "amplitude_damping",
            "gamma": 0.05  # 弱耗散
        }
    )