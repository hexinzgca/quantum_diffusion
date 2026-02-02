import torch
import numpy as np
from .quantum_operators import SOrderedProbability

class BaseSDE:
    """SDE基类（定义通用接口）"""
    def __init__(self, system: "Base1DQuantumSystem"):
        self.system = system
        self.s_order = SOrderedProbability(s=system.s, hbar=system.hbar)
        self.hbar = system.hbar
        self.device = system.device

    def drift(self, alpha: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """漂移项μ(α,t)"""
        raise NotImplementedError

    def diffusion(self, alpha: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """扩散项Σ(α,t)"""
        raise NotImplementedError

    def forward_step(self, alpha: torch.Tensor, t: torch.Tensor, dt: float) -> torch.Tensor:
        """正向SDE单步演化"""
        z = torch.randn_like(alpha, device=self.device)
        mu = self.drift(alpha, t)
        sigma = self.diffusion(alpha, t)
        return alpha + mu * dt + sigma * np.sqrt(dt) * z

class RealTimeSDE(BaseSDE):
    """实时动力学SDE（描述量子系统的时间演化）
    基于Fokker-Planck方程等价转换，兼容任意耗散
    """
    def drift(self, alpha: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """漂移项：融合哈密顿动力学+耗散漂移"""
        # 1. 哈密顿动力学贡献
        H_classical = self.system.classical_hamiltonian(alpha)
        H_quantum = self.s_order.weyl_symbol(alpha, H_classical)
        grad_H = torch.autograd.grad(H_quantum.sum(), alpha.conj(), create_graph=True)[0]
        ham_drift = grad_H / (1j * self.hbar)

        # 2. 耗散漂移贡献
        D1, _ = self.system.dissipation_terms(alpha, t)
        
        return ham_drift + D1

    def diffusion(self, alpha: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """扩散项：从耗散矩阵推导（保证正半定）"""
        _, D2 = self.system.dissipation_terms(alpha, t)
        # 扩散矩阵开方（保证数值稳定性）
        sqrt_D2 = torch.sqrt(torch.clamp(D2, min=1e-8))  # 避免负数值
        return sqrt_D2.squeeze(-1).squeeze(-1)

class ImaginaryTimeSDE(BaseSDE):
    """虚时采样SDE（从经典分布→量子分布）
    核心：通过噪声退火实现量子分布采样
    """
    def __init__(self, system: "Base1DQuantumSystem", beta_schedule: str = "linear"):
        super().__init__(system)
        self.beta_schedule = beta_schedule
        self.T = 1.0  # 虚时总长度

    def _get_beta(self, tau: torch.Tensor) -> torch.Tensor:
        """噪声强度调度（虚时τ∈[0,T]）"""
        if self.beta_schedule == "linear":
            return 1.0 - tau / self.T  # 从1线性衰减到0
        elif self.beta_schedule == "exponential":
            return torch.exp(-5 * tau / self.T)  # 指数衰减
        else:
            raise ValueError(f"不支持的调度策略：{self.beta_schedule}")

    def drift(self, alpha: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """虚时漂移项：量子势能梯度+耗散修正"""
        # 1. 量子势能梯度（虚时演化核心）
        H_classical = self.system.classical_hamiltonian(alpha)
        H_quantum = self.s_order.weyl_symbol(alpha, H_classical)
        grad_H = torch.autograd.grad(H_quantum.sum(), alpha, create_graph=True)[0]
        
        # 2. 噪声强度调度
        beta = self._get_beta(tau)
        return -beta * grad_H  # 虚时方向：能量最小化

    def diffusion(self, alpha: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """虚时扩散项：随虚时衰减的高斯噪声"""
        beta = self._get_beta(tau)
        _, D2 = self.system.dissipation_terms(alpha)
        base_diff = torch.sqrt(torch.clamp(D2, min=1e-8)).squeeze(-1).squeeze(-1)
        return base_diff * torch.sqrt(beta)

    def reverse_step(self, alpha: torch.Tensor, tau: torch.Tensor, dt: float) -> torch.Tensor:
        """反向SDE单步（去噪过程）"""
        z = torch.randn_like(alpha, device=self.device)
        mu = self.drift(alpha, tau)
        sigma = self.diffusion(alpha, tau)
        # 虚时步长为负（从T→0）
        return alpha + mu * (-dt) + sigma * np.sqrt(dt) * z