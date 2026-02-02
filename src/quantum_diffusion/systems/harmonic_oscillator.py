import torch
import numpy as np
from .base_system import Base1DQuantumSystem

class HarmonicOscillator(Base1DQuantumSystem):
    """一维量子谐振子系统
    哈密顿量：H = p²/(2m) + (1/2)mω²q²
    相空间变量：α = (q + ip)/√(2ħ) → q = Re(α)√(2ħ), p = Im(α)√(2ħ)
    """
    def __init__(
        self,
        m: float = 1.0,  # 质量
        omega: float = 1.0,  # 角频率
        s: float = 0.0,  # s-序参数
        dissipation: dict = None,
        hbar: float = 1.0,
        device: str = "cuda"
    ):
        super().__init__(s=s, dissipation=dissipation, hbar=hbar, device=device)
        self.m = m
        self.omega = omega

    def classical_hamiltonian(self, alpha: torch.Tensor) -> torch.Tensor:
        """经典哈密顿量（相空间α表示）"""
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)
        # α → (q, p) 转换
        q = torch.real(alpha) * np.sqrt(2 * self.hbar)
        p = torch.imag(alpha) * np.sqrt(2 * self.hbar)
        # 谐振子哈密顿量
        return p**2 / (2 * self.m) + 0.5 * self.m * self.omega**2 * q**2

    def dissipation_terms(self, alpha: torch.Tensor, t: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """通用耗散项（支持马尔可夫/非马尔可夫）
        配置示例：
        dissipation = {
            "type": "amplitude_damping",  # 振幅衰减
            "gamma": 0.1,  # 耗散强度
            "non_markovian": False,  # 是否非马尔可夫
            "tau_c": 1.0  # 非马尔可夫记忆时间
        }
        """
        batch_size = alpha.shape[0]
        default_diss = {
            "type": "amplitude_damping",
            "gamma": 0.1,
            "non_markovian": False,
            "tau_c": 1.0
        }
        diss = {**default_diss, **self.dissipation}

        gamma = diss["gamma"]
        if diss["non_markovian"] and t is not None:
            # 非马尔可夫耗散：指数记忆核
            gamma = gamma * torch.exp(-t / diss["tau_c"]).to(self.device)

        if diss["type"] == "amplitude_damping":
            # 振幅衰减：D1=-γ/2 α, D2=γ/4
            D1 = -0.5 * gamma * alpha
            D2 = 0.25 * gamma * torch.ones((batch_size, 1, 1), device=self.device)
        elif diss["type"] == "phase_damping":
            # 相位衰减：D1=0, D2=γ/2 |α|²
            D1 = torch.zeros_like(alpha)
            D2 = 0.5 * gamma * torch.abs(alpha)**2.unsqueeze(-1).unsqueeze(-1)
        elif diss["type"] == "thermal":
            # 热库耗散：D1=γ(⟨n⟩+1)α, D2=γ⟨n⟩
            n_th = diss.get("n_th", 0.5)  # 热占据数
            D1 = gamma * (n_th + 1) * alpha
            D2 = gamma * n_th * torch.ones((batch_size, 1, 1), device=self.device)
        else:
            # 无耗散
            D1 = torch.zeros_like(alpha)
            D2 = torch.zeros((batch_size, 1, 1), device=self.device)

        return D1, D2

    def exact_quasiprobability(self, q: np.ndarray, p: np.ndarray, n: int = 0) -> np.ndarray:
        """谐振子准概率分布精确解（用于验证）"""
        q_tensor = torch.tensor(q, dtype=torch.float32, device=self.device)
        p_tensor = torch.tensor(p, dtype=torch.float32, device=self.device)
        alpha = (q_tensor / np.sqrt(2 * self.hbar)) + 1j * (p_tensor / np.sqrt(2 * self.hbar))
        dist = self.s_order.quasiprobability_dist(alpha, n=n)
        return dist.cpu().numpy()
        